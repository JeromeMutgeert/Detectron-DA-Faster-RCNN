#!/usr/bin/env python

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

"""Utilities for training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import datetime
import numpy as np

from caffe2.python import utils as c2_py_utils

from detectron.core.config import cfg
import detectron.datasets.dummy_datasets as dummy_datasets
from detectron.utils.logging import log_json_stats
from detectron.utils.logging import SmoothedValue
from detectron.utils.timer import Timer
import detectron.utils.net as nu


class TrainingStats(object):
    """Track vital training statistics."""

    def __init__(self, model):
        # Window size for smoothing tracked values (with median filtering)
        self.WIN_SZ = 1
        # Output logging period in SGD iterations
        self.LOG_PERIOD = 1
        self.smoothed_losses_and_metrics = {
            key: SmoothedValue(self.WIN_SZ)
            for key in model.losses + model.metrics
        }
        self.losses_and_metrics = {
            key: 0
            for key in model.losses + model.metrics
        }
        self.smoothed_total_loss = SmoothedValue(self.WIN_SZ)
        self.smoothed_mb_qsize = SmoothedValue(self.WIN_SZ)
        self.iter_total_loss = np.nan
        self.iter_timer = Timer()
        self.model = model

    def IterTic(self):
        self.iter_timer.tic()

    def IterToc(self):
        return self.iter_timer.toc(average=False)

    def ResetIterTimer(self):
        self.iter_timer.reset()

    def UpdateIterStats(self):
        """Update tracked iteration statistics."""
        for k in self.losses_and_metrics.keys():
            if k in self.model.losses:
                self.losses_and_metrics[k] = nu.sum_multi_gpu_blob(k)
            else:
                self.losses_and_metrics[k] = nu.average_multi_gpu_blob(k)
        for k, v in self.smoothed_losses_and_metrics.items():
            v.AddValue(self.losses_and_metrics[k])
        self.iter_total_loss = np.sum(
            np.array([self.losses_and_metrics[k] for k in self.model.losses])
        )
        self.smoothed_total_loss.AddValue(self.iter_total_loss)
        self.smoothed_mb_qsize.AddValue(
            self.model.roi_data_loader._minibatch_queue.qsize()
        )
        

    def LogIterStats(self, cur_iter, lr):
        """Log the tracked statistics."""
        if (cur_iter % self.LOG_PERIOD == 0 or
                cur_iter == cfg.SOLVER.MAX_ITER - 1):
            stats = self.GetStats(cur_iter, lr)
            log_json_stats(stats)

    def GetStats(self, cur_iter, lr):
        eta_seconds = self.iter_timer.average_time * (
            cfg.SOLVER.MAX_ITER - cur_iter
        )
        eta = str(datetime.timedelta(seconds=int(eta_seconds)))
        mem_stats = c2_py_utils.GetGPUMemoryUsageStats()
        mem_usage = np.max(mem_stats['max_by_gpu'][:cfg.NUM_GPUS])
        stats = dict(
            iter=cur_iter,
            lr=float(lr),
            time=self.iter_timer.average_time,
            loss=self.smoothed_total_loss.GetMedianValue(),
            eta=eta,
            mb_qsize=int(
                np.round(self.smoothed_mb_qsize.GetMedianValue())
            ),
            mem=int(np.ceil(mem_usage / 1024 / 1024)),
        )
        if cfg.TRAIN.DA_FADE_IN:
            stats['da_weight'] = self.model.da_fade_in.get_weight()
        if cfg.TRAIN.PADA:
            
            stats['avg_pada_weight'] = self.model.class_weight_db.get_avg_pada_weight()
            stats['total_detects'] = self.model.class_weight_db.total_sum_softmax.sum() / 2
            stats['KL_div'] = self.model.class_weight_db.get_KL_to_init()
            stats['accuracy_fg'] = self.model.class_weight_db.fg_acc.get()
            stats['acc_fg_weighted'] = self.model.class_weight_db.weighted_fg_acc.get()
            
            target_dist = self.model.class_weight_db.get_dist()
            print('target_dist: {}'.format(list(target_dist)))
            class_weights = self.model.class_weight_db.class_weights
            print('class_weights: {}'.format(list(class_weights)))
            
            classes = np.array(dummy_datasets.get_coco_dataset().classes.values(),dtype=str)
            
            for dist in [target_dist,class_weights]:
                order = np.argsort(dist)[::-1]
                o_target_dist = target_dist[order]
                o_classes = classes[order]
                cwo = class_weights[order]
                print("dist tops: ",end='')
                for prob,w,c in list(zip(o_target_dist,cwo,o_classes))[:5]:
                    print("{}:{:.3f} ({:.3f})".format(c,prob,w),end=';  ')
                print()
            print()
            
        for k, v in self.smoothed_losses_and_metrics.items():
            stats[k] = v.GetMedianValue()
        # for k,v in stats.items():
        #     print(k,v)
        
        
        
        
        return stats
