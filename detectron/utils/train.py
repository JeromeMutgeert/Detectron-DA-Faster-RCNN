# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################
#
# Based on:
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Utilities driving the train_net binary"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from shutil import copyfile
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import logging
import numpy as np
import os
import re
import time

from caffe2.python import memonger,net_drawer
from caffe2.python import workspace

from detectron.core.config import cfg
from detectron.core.config import get_output_dir
from detectron.datasets.roidb import combined_roidb_for_training
from detectron.modeling import model_builder
import detectron.modeling.PADA as pada
from detectron.utils import lr_policy
from detectron.utils.training_stats import TrainingStats
import detectron.utils.env as envu
import detectron.utils.net as nu


def blob_summary(blobs=None):
    if blobs is None:
        blobs = workspace.Blobs()
        print()
        print("Current blobs in the workspace:\n{}".format('\n'.join(blobs)))
        print()
        for blob in blobs:
            print("Fetched {}:\n{}".format(blob,workspace.FetchBlob(blob)))
            print()
        # blobs = ['conv1','conv1_w']
        # blobs = ['gpu_0/'+b for b in blobs]
    else:
        blobs = ['gpu_0/'+b for b in blobs]
    print()
    for blob in blobs:
        
        b = workspace.FetchBlob(blob)
        shape = b.shape
        b = np.array(b.astype(float)).reshape(-1)
        order = np.argsort(b)
        step = max(1,len(b)//10)
        idxs = np.arange(step//2,len(b),step)
        percentiles = b[order[idxs]]
        hi = b[order[-1]]
        lo = b[order[0]]
        abs_mean,mean,std,zeros = [np.format_float_scientific(v,precision=2) for v in [np.abs(b).mean(), b.mean(),b.std(),sum(b == 0.0)/len(b)]]
        print(" {} {} ({}): abs mean:{} mean:{} std:{} zeros:{} min-5-15-...-85-95-max percentiles: {} ".format(blob, shape, len(b),
               abs_mean,mean,std,zeros,' '.join([np.format_float_scientific(p,precision=2) for p in [lo] + list(percentiles)+ [hi]])))
        print()


def print_conf_matrix(conf_matrix):
    shades = 'M987654321. '[::-1]
    import detectron.datasets.dummy_datasets as dummy_datasets
    classes = np.array(dummy_datasets.get_coco_dataset().classes.values(),dtype=str)
    print()
    # header:
    for c in classes:
        print(c[0],end='')
    print(' <- pr.; gt:')
    # body:
    for c_gt,class_gt in enumerate(classes):
        for p in conf_matrix[c_gt,:]:
            if p >= 1.1:
                print('woah',p)
            else:
                try:
                    idx = int(p / 0.1) + int(p > 0.)
                    # print(p,idx,len(shades))
                    print(shades[idx],end='')
                except:
                    print('whaow!',idx,p)
        print(' : {} (sum: {})'.format(class_gt,conf_matrix[c_gt,:].sum()))
    print()
    

def train_model():
    """Model training loop."""
    start_time = time.time()
    
    model, weights_file, start_iter, checkpoints, output_dir = create_model()
    if 'final' in checkpoints:
        # The final model was found in the output directory, so nothing to do
        return checkpoints

    setup_model_for_training(model, weights_file, output_dir)
    training_stats = TrainingStats(model)
    CHECKPOINT_PERIOD = int(cfg.TRAIN.SNAPSHOT_ITERS / cfg.NUM_GPUS)
    
    if model.train and cfg.TRAIN.PADA:
        if not hasattr(model,'class_weight_db'):
            model.class_weight_db = pada.ClassWeightDB()
        model.class_weight_db.setup(model.roi_data_loader)
    if cfg.TRAIN.DA_FADE_IN:
        model.da_fade_in = pada.DAScaleFading(cfg.SOLVER.MAX_ITER)
    
    # if cfg.INTERRUPTING:
    #     source_set_size = len(model.roi_data_loader._roidb)
    #     if cfg.TRAIN.DOMAIN_ADAPTATION:
    #         source_ims_per_batch = cfg.NUM_GPUS * (cfg.TRAIN.IMS_PER_BATCH//2)
    #     else:
    #         source_ims_per_batch = cfg.NUM_GPUS * cfg.TRAIN.IMS_PER_BATCH
    #     CHECKPOINT_PERIOD = int(1.0 + source_set_size / (source_ims_per_batch * cfg.NUM_GPUS))
    #     print("Checkpoint period, and interruption, set for after {} batches".format(CHECKPOINT_PERIOD))

    for cur_iter in range(start_iter, cfg.SOLVER.MAX_ITER):
        # print('iter:',cur_iter)
        # print(model.roi_data_loader._cur,list(model.roi_data_loader._perm)[:10])
        
        if model.roi_data_loader.has_stopped():
            handle_critical_error(model, 'roi_data_loader failed')
        training_stats.IterTic()
        lr = model.UpdateWorkspaceLr(cur_iter, lr_policy.get_lr_at_iter(cur_iter))
        workspace.RunNet(model.net.Proto().name)
        if cur_iter == start_iter:
            nu.print_net(model)
            # blob_summary(['conv{}_{}'.format(i,j) for i in [1,2,3,4,5] for j in [1,2,3] if not ((j==3) and (i < 3))])
        training_stats.IterToc()
        training_stats.UpdateIterStats()
        training_stats.LogIterStats(cur_iter, lr)
        
        if (cur_iter) % (training_stats.LOG_PERIOD*50) == 0:
            print_conf_matrix(model.class_weight_db.conf_matrix)
            pool2 = workspace.FetchBlob('gpu_0/rois').astype(float)
            print(pool2[:,0])
            
            # print('pool2 max: {}'.format(pool2.max()))
            # blob_summary(['conv3_1_w','conv3_1_w_grad','conv3_1_b','conv5_3','da_fc7','da_conv_2','dc_ip3','dc_ip3_w','dc_ip2_w_grad'])
        
        
        if cfg.INTERRUPTING and time.time() - start_time > cfg.THRESH_TIME:
            checkpoints[cur_iter] = os.path.join(
                output_dir, 'model_iter{}.pkl'.format(cur_iter)
            )
            nu.save_model_to_weights_file(checkpoints[cur_iter], model,cur_iter=cur_iter)
        
            # stop this process and restart to continue form the checkpoint.
            model.roi_data_loader.shutdown()
            
            if cfg.TRAIN.DOMAIN_ADAPTATION:
                # triggers target data loader to stop:
                with open('./TargetDataLoaderProcess/read.txt','w') as f:
                    f.write(str(0))
                    f.flush()
                    os.fsync(f.fileno())
                    
            # wait a bit for it to stop:
            time.sleep(5)
            
            # enqueue new job:
            os.system('sbatch run.job')
            
            return checkpoints
        
        if (cur_iter + 1) % CHECKPOINT_PERIOD == 0 and cur_iter > start_iter:
            checkpoints[cur_iter] = os.path.join(
                output_dir, 'model_iter{}.pkl'.format(cur_iter)
            )
            nu.save_model_to_weights_file(checkpoints[cur_iter], model, cur_iter=cur_iter)

        if cur_iter == start_iter + training_stats.LOG_PERIOD:
            # Reset the iteration timer to remove outliers from the first few
            # SGD iterations
            training_stats.ResetIterTimer()
          
        v = training_stats.iter_total_loss+model.class_weight_db.avg_pada_weight
        if training_stats.iter_total_loss > 4:
            print('Loss is high: {}'.format(training_stats.iter_total_loss))
            # pool2 = workspace.FetchBlob('gpu_0/pool2').astype(float)
            # print('pool2 max: {}'.format(pool2.max()))
            # blob_summary(['conv3_1_w','conv3_1_w_grad','conv3_1_b','conv5_3','da_fc7','da_conv_2','dc_ip3','dc_ip3_w','dc_ip2_w_grad'])
        
        if np.isnan(v) or v == np.infty or v == -np.infty:
            nu.print_net(model)
            blobs = workspace.Blobs()
            print()
            print("Current blobs in the workspace:\n{}".format('\n'.join(blobs)))
            print()
            for blob in blobs:
                print("Fetched {}:\n{}".format(blob,workspace.FetchBlob(blob)))
                print()
            # blob_summary(['conv3_1_w','conv3_1_b','conv5_3','da_fc7','da_conv_2','dc_ip3','dc_ip3_w','dc_ip2_w_grad'])
            blob_summary()
            handle_critical_error(model, 'Loss is {}'.format(v))

    # Save the final model
    checkpoints['final'] = os.path.join(output_dir, 'model_final.pkl')
    nu.save_model_to_weights_file(checkpoints['final'], model, cur_iter=cur_iter)
    # Shutdown data loading threads
    model.roi_data_loader.shutdown()
    return checkpoints


def handle_critical_error(model, msg):
    logger = logging.getLogger(__name__)
    logger.critical(msg)
    model.roi_data_loader.shutdown()
    raise Exception(msg)


def create_model():
    """Build the model and look for saved model checkpoints in case we can
    resume from one.
    """
    logger = logging.getLogger(__name__)
    start_iter = 0
    checkpoints = {}
    output_dir = get_output_dir(cfg.TRAIN.DATASETS, training=True)
    weights_file = cfg.TRAIN.WEIGHTS
    if cfg.TRAIN.AUTO_RESUME:
        # Check for the final model (indicates training already finished)
        final_path = os.path.join(output_dir, 'model_final.pkl')
        if os.path.exists(final_path):
            logger.info('model_final.pkl exists; no need to train!')
            return None, None, None, {'final': final_path}, output_dir

        if cfg.TRAIN.COPY_WEIGHTS:
            copyfile(
                weights_file,
                os.path.join(output_dir, os.path.basename(weights_file)))
            logger.info('Copy {} to {}'.format(weights_file, output_dir))

        # Find the most recent checkpoint (highest iteration number)
        files = os.listdir(output_dir)
        for f in files:
            iter_string = re.findall(r'(?<=model_iter)\d+(?=\.pkl)', f)
            if len(iter_string) > 0:
                checkpoint_iter = int(iter_string[0])
                if checkpoint_iter >= start_iter:
                    # Start one iteration immediately after the checkpoint iter
                    start_iter = checkpoint_iter + 1
                    resume_weights_file = f

        if start_iter > 0:
            # Override the initialization weights with the found checkpoint
            weights_file = os.path.join(output_dir, resume_weights_file)
            logger.info(
                '========> Resuming from checkpoint {} at start iter {}'.
                format(weights_file, start_iter)
            )

    logger.info('Building model: {}'.format(cfg.MODEL.TYPE))
    model = model_builder.create(cfg.MODEL.TYPE, train=True)
    if cfg.MEMONGER:
        optimize_memory(model)
    # Performs random weight initialization as defined by the model
    workspace.RunNetOnce(model.param_init_net)
    return model, weights_file, start_iter, checkpoints, output_dir


def optimize_memory(model):
    """Save GPU memory through blob sharing."""
    for device in range(cfg.NUM_GPUS):
        namescope = 'gpu_{}/'.format(device)
        losses = [namescope + l for l in model.losses]
        model.net._net = memonger.share_grad_blobs(
            model.net,
            losses,
            set(model.param_to_grad.values()),
            namescope,
            share_activations=cfg.MEMONGER_SHARE_ACTIVATIONS
        )


def setup_model_for_training(model, weights_file, output_dir):
    """Loaded saved weights and create the network in the C2 workspace."""
    logger = logging.getLogger(__name__)
    if cfg.TRAIN.DOMAIN_ADAPTATION:
        add_model_da_training_inputs(model)
    else:
        add_model_training_inputs(model)

    if weights_file:
        # Override random weight initialization with weights from a saved model
        nu.initialize_gpu_from_weights_file(model, weights_file, gpu_id=0)
    # Even if we're randomly initializing we still need to synchronize
    # parameters across GPUs
    nu.broadcast_parameters(model)
    workspace.CreateNet(model.net)

    logger.info('Outputs saved to: {:s}'.format(os.path.abspath(output_dir)))
    dump_proto_files(model, output_dir)

    # from IPython import display
    # graph = net_drawer.GetPydotGraphMinimal(model.net.Proto().op,"da-frcnn",rankdir='LR')
    # png = graph.create(format='png')
    # with open('graph.png','w') as f:
    #     f.write(png)
    #     f.flush()
    # print(graph)
    # import pydot
    # print(pydot.graph_from_dot_data(graph))
    # (graph2,) = pydot.graph_from_dot_data(str(graph))
    # png = graph2.create_png()
    # png = graph.create_png()
    # import matplotlib.pyplot as plt
    # plt.imshow('graph.png')
    # plt.show()
    
    # Start loading mini-batches and enqueuing blobs
    model.roi_data_loader.register_sigint_handler()
    # Jerome: TODO: set back to True:
    model.roi_data_loader.start(prefill=False)
    return output_dir


def add_model_training_inputs(model):
    """Load the training dataset and attach the training inputs to the model."""
    logger = logging.getLogger(__name__)
    logger.info('Loading dataset: {}'.format(cfg.TRAIN.DATASETS))
    roidb = combined_roidb_for_training(
        cfg.TRAIN.DATASETS, cfg.TRAIN.PROPOSAL_FILES
    )
    logger.info('{:d} roidb entries'.format(len(roidb)))
    model_builder.add_training_inputs(model, source_roidb=roidb)

def add_model_da_training_inputs(model):
    """Load the training dataset and attach the training inputs to the model."""
    logger = logging.getLogger(__name__)
    logger.info('Loading source dataset: {}'.format(cfg.TRAIN.SOURCE_DATASETS))
    source_roidb = combined_roidb_for_training(
        cfg.TRAIN.SOURCE_DATASETS, cfg.TRAIN.SOURCE_PROPOSAL_FILES, True
    )
    logger.info('{:d} source roidb entries'.format(len(source_roidb)))
    
    logger.info('Loading target dataset: {}'.format(cfg.TRAIN.TARGET_DATASETS))
    target_roidb = combined_roidb_for_training(
        cfg.TRAIN.TARGET_DATASETS, cfg.TRAIN.TARGET_PROPOSAL_FILES, False
    )
    if cfg.TRAIN.PADA:
        # add indices for the target images for updating their class weights
        for i,rois in enumerate(target_roidb):
            rois['im_idx'] = i
            
    logger.info('{:d} target roidb entries'.format(len(target_roidb)))
    # roidb = source_roidb+target_roidb
    model_builder.add_training_inputs(model, source_roidb=source_roidb, target_roidb=target_roidb)

def dump_proto_files(model, output_dir):
    """Save prototxt descriptions of the training network and parameter
    initialization network."""
    with open(os.path.join(output_dir, 'net.pbtxt'), 'w') as fid:
        fid.write(str(model.net.Proto()))
    with open(os.path.join(output_dir, 'param_init_net.pbtxt'), 'w') as fid:
        fid.write(str(model.param_init_net.Proto()))
