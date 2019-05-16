
"""Perform analysis on detections on a dataset."""

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
# from __future__ import unicode_literals


import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import argparse
# import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import os
# import pprint
# import sys
# import time



# from caffe2.python import workspace

# from detectron.core.config import assert_and_infer_cfg
# from detectron.core.config import cfg
# from detectron.core.config import merge_cfg_from_file
# from detectron.core.config import merge_cfg_from_list
# from detectron.core.test_engine import run_inference
from detectron.datasets.dummy_datasets import get_coco_dataset
from detectron.utils.io import load_object
from detectron.utils.logging import setup_logging
# import detectron.utils.c2 as c2_utils



# c2_utils.import_detectron_ops()

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
# cv2.ocl.setUseOpenCL(False)


def parse_args():
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    # parser.add_argument(
    #     '--vis', dest='vis', help='visualize detections', action='store_true'
    # )
    # parser.add_argument(
    #     '--multi-gpu-testing',
    #     dest='multi_gpu_testing',
    #     help='using cfg.NUM_GPUS for inference',
    #     action='store_true'
    # )
    parser.add_argument(
        '--class_weights',
        dest='class_weights',
        help='class distribution file by summed softmax outputs (of the source set)',
        default=None,
        type=str,
        nargs=1
    )
    parser.add_argument(
        '--voc_class_weights',
        dest='voc_class_weights',
        help='class distribution file by summed softmax outputs for the target set',
        default=None,
        type=str,
        nargs=3
    )
    parser.add_argument(
        '--target_class_weights',
        dest='target_class_weights',
        help='class distribution file by summed softmax outputs for the target set',
        default=None,
        type=str,
        nargs=1
    )
    parser.add_argument(
        '--target_divergence',
        dest='target_divergence',
        help='the desired KL-divergence between source and voc',
        default=None,
        type=float,
        nargs=1
    )
    parser.add_argument(
        '--features',
        dest='feats',
        help='feature vector collection for t-sne visualisation',
        default=None,
        type=str,
        nargs=1
    )
    parser.add_argument(
        '--do_val_set',
        dest='do_val_set',
        action='store_true'
    )
    # # This allows to set all fields of cfg. Use to set TEST.WEIGTHS and optionally override NUM_GPUS:
    # parser.add_argument(
    #     'opts',
    #     help='See detectron/core/config.py for all options',
    #     default=None,
    #     nargs=argparse.REMAINDER
    # )
    # if len(sys.argv) == 1:
    #     parser.print_help()
    #     sys.exit(1)
    return parser.parse_args()


def plt_bars(counts,labels,counts2=None,reversed=True,ax=None,eps=np.finfo(float).eps,figsize=(30,20),log=None,**kwargs):
    if ax is None:
        fig,ax = plt.subplots(figsize=figsize)
    order = counts.argsort()
    if counts2 is not None:
        ratios = -(counts2 * np.log(counts/(counts2 + eps) + eps) + (1 - counts2) * np.log((1 - counts)/(1 - counts2 + eps) + eps))
        print(ratios[1:].sum())
        order = ratios.argsort()
    if reversed:
        order = order[::-1]
    x = np.arange(len(labels))
    ax.bar(x,counts[order],**kwargs)
    if counts2 is not None:
        ax.bar(x,counts2[order],alpha=.4,**kwargs)
    plt.xticks(x,labels[order],rotation=90)
    if log:
        ax.set_yscale('log')
    
coco_classes = np.array(get_coco_dataset().classes.values(),dtype=str)
def plot_dists(coco_dist, yisual_dist, source_name,target_name):
    plt_bars(coco_dist,coco_classes,yisual_dist)
    plt.title('Class distributions of detections: KL( {} || {} ) = {} )'.format(target_name,source_name,KL_div(yisual_dist,coco_dist)))
    plt.legend([source_name,target_name])
    plt.show()

def plot_dist(coco_dist,name=None,log=None):
    plt_bars(coco_dist,coco_classes,log=log)
    if name is not None:
        plt.legend([name])
    plt.show()
    

def KL_div(target,source,eps=np.finfo(float).eps):
    # We take KL(target||source), the distance of the source distribution from the pespective of the target distribution.
    # assert len(target.shape) == 1 or len(source.shape) == 1
    if len(target.shape) == len(source.shape):
        return -(target * np.log(source/(target + eps) + eps)).sum()
    elif len(target.shape) == 2:
        assert len(source.shape) == 1
        return -(target * np.log(source[None,:]/(target + eps) + eps)).sum(axis=1)
    elif len(source.shape) == 2:
        assert len(target.shape) == 1
        target = target[None,:]
        return -(target * np.log(source/(target + eps) + eps)).sum(axis=1)
    else:
        assert False,"Weird shapes received"

def get_dist(wts):
    dist = wts.sum(axis=0)
    dist /= dist.sum()
    return dist
    
# def plot_dist(dist,dist2=None):
#     fig,ax = plt.subplots()
#     plt_bars(dist, coco_classes, dist2, ax=ax)
#     plt.show()




if __name__ == '__main__':
    # workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    logger = setup_logging(__name__)
    args = parse_args()
    logger.info('Called with args:')
    logger.info(args)
    assert args.class_weights is not None, 'class_weights file required'
    print(str(args.class_weights[0]))
    
    
    
    
    coco_wts = load_object(args.class_weights[0])
    print(coco_wts.shape)
    coco_dist = get_dist(coco_wts)
    print("coco size: ",coco_wts.sum())
    
    assert args.voc_class_weights is not None, 'voc dist files needed'
    voc_files = args.voc_class_weights
    
    if args.do_val_set:
        voc_files = ['collecting/test/voc_2007_test/generalized_rcnn/class_weights.pkl',
                     'collecting/test/voc_2012_val/generalized_rcnn/class_weights.pkl']
    voc_wts = np.concatenate([load_object(vocfile) for vocfile in voc_files], axis=0)
    
    
    print(voc_wts.shape)
    voc_dist = get_dist(voc_wts)
    print('voc size:',voc_wts.sum())
    
    assert args.target_class_weights is not None
    # source_wts = wts
    # wts = load_object(args.target_class_weights[0])
    # print('voc weights overloaded with targets')
    yisual_wts = load_object(args.target_class_weights[0])
    yisual_dist = get_dist(yisual_wts)
    print('yisual size:',yisual_wts.sum())
    
    sets = [(coco_wts,'coco'), (voc_wts, 'voc'), (yisual_wts, 'yisual')]
    pairs = [(sets[0],sets[1]),(sets[0],sets[2]),(sets[1],sets[2])]
    # for set1,set2 in pairs:
    #     for s1,s2 in [(set1,set2),(set2,set1)]:
    #         wts_source,source_name = s1
    #         wts_target,target_name = s2
    #         dist_source, dist_target = get_dist(wts_source), get_dist(wts_target)
    #
    #         kl_div = KL_div(dist_target,dist_source)
    #         print("KL({}||{}) = {}".format(target_name,source_name,kl_div))
    #         # plot_dists(dist_source,dist_target,source_name,target_name)
    
    # source_dist = get_dist(source_wts)
    #
    # # # plot_dists:
    # plt_bars(source_dist,coco_classes,get_dist(wts))
    # plt.title('Class distribution of detections')
    # plt.legend(['coco train','yisual train'])
    # plt.show()
    #
    # voc_dist = get_dist(wts)
    # print(KL_div(voc_dist,source_dist))
    # # print(KL_div(voc_dist,source_dist))
    #
    # detects = wts.sum(axis=1)
    # voc_mean_detect = detects.mean()
    # source_detects = source_wts.sum(axis=1)
    # source_mean_detect = source_detects.mean()
    # print('mean amount of detections per image: source: {}  target: {}'.format(source_mean_detect,voc_mean_detect))
    
    
    # Filtering VOC
    
    
    
    def remove_from_dist(dist,subdists,portions):
        aportions = (1 - portions)
        return (dist[None,:]/aportions[:,None]) - (portions/aportions)[:,None] * subdists
        return dist/aportion - (portion/aportion) * subdist
        new_dist = (dist - portion * subdist)
        # 'new_dist/(1-portion)' would be a correct answer. The following answer has more numerical stability: re-normalizing.
        return new_dist / new_dist.sum()
    
    
    def score_fn(proposed,prop_weight):
        """The function to be maximized by (greedy) subset selection"""
        return np.log(KL_div(proposed,coco_dist)) - np.log(KL_div(yisual_dist,proposed))
        # return KL_div(proposed,coco_dist)/divergences_coco[-1] - KL_div(yisual_dist,proposed)/divergences[-1] #+ prop_weight/total_weight
        # return prop_weight**2 *KL_div(proposed,coco_dist)/KL_div(yisual_dist,proposed)
    
    # normalise per-image:
    im_weights = voc_wts.sum(axis=1)
    total_weight = im_weights.sum()
    min_weight = .05 * total_weight
    im_dists = voc_wts / im_weights[:, None]
    
    # scores = [ KL_div(yisual_dist,im_dist) - KL_div(im_dist,coco_dist) for im_dist in im_dists] #
    
    # order = np.argsort(scores)[::-1]
    
    N = int(len(im_weights)*1.15)
    
    weights = [total_weight]
    divergences = [KL_div(yisual_dist,voc_dist)]
    divergences_coco = [KL_div(voc_dist,coco_dist)]
    dists = [voc_dist[:]]
    dists = np.empty((N+1,81),dtype=np.float32)
    dists[0,:] = voc_dist[:]
    shown = False
    
    
    start_score = score_fn(voc_dist,total_weight)
    # for i in order:
    #     im_dist = im_dists[i]
    #     im_weight = im_weights[i]
    #     voc_dist = remove_from_dist(voc_dist,im_dist,im_weight/total_weight)
    #     total_weight -= im_weight
    #     weights.append(total_weight)
    #     divergences.append(KL_div(yisual_dist,voc_dist))
    #     divergences_coco.append(KL_div(voc_dist,coco_dist))
    #     dists.append(voc_dist[:])
    #
    #     if total_weight < weights[0]/2 and not shown: # if half-way:
    #         plot_dists(coco_dist,voc_dist,'voc_sub','coco')
    #         plot_dists(coco_dist,yisual_dist,'coco','yisual')
    #         plot_dists(voc_dist,yisual_dist,'voc_sub','yisual')
    #         shown = True
    
    removed = np.full(len(im_weights), False, dtype=bool)
    choices = []
    remains = len(im_weights)
    ns = [remains]
    undos = 0
    prev_i = -1
    best_score = -np.infty
    for n in range(N):
        if n % 100 == 0:
            print(n)
            if undos != 0:
                print('undos:',undos)
                undos = 0
        
        prop_dists = remove_from_dist(voc_dist,im_dists,im_weights/total_weight)
        scores = (score_fn(prop_dists,total_weight - im_weights) - score_fn(voc_dist,total_weight)) / np.abs(im_weights) #normalized by im_weights
        
        if total_weight < min_weight:
            scores[~removed] = -np.infty
        
        i = np.argmax(scores)
        if i == prev_i:
            break
        prev_i = i
        
        remains += 1 if removed[i] else -1
        undos += removed[i]
        if removed[i]:
            print('undo ({})'.format(undos))
        if remains == 0:
            break
        choices.append(i)
        removed[i] = ~removed[i] #flip, on or off.
        im_dist = im_dists[i]
        im_weight = im_weights[i]
        im_weights[i] = -im_weights[i] # flip, on or off. Removing with Negative im weight equals adding again.
        # voc_dist = remove_from_dist(voc_dist,im_dist,im_weight/total_weight)
        voc_dist = prop_dists[i]
        voc_dist /= voc_dist.sum() # for numeric stability
        total_weight -= im_weight
        weights.append(total_weight)
        divergences.append(KL_div(yisual_dist,voc_dist))
        divergences_coco.append(KL_div(voc_dist,coco_dist))
        # print(len(voc_dist[:]))
        dists[n+1,:] = voc_dist[:]
        ns.append(remains)
        
        score = score_fn(voc_dist,total_weight)
        if score > best_score:
            best_score = score
            best_dist = voc_dist
            best_removed = removed[:]
        else:
            print('nope',n)
            

        # if total_weight < weights[0]/2 and not shown: # if half-way:
        #     plot_dists(coco_dist,voc_dist,'coco','voc_sub')
        #     plot_dists(coco_dist,yisual_dist,'coco','yisual')
        #     plot_dists(voc_dist,yisual_dist,'voc_sub','yisual')
        #     shown = True
    
    removed = best_removed
    voc_dist = best_dist
    
    # if args.do_val_set:
    #     np.save('voc_subset_val.npy',~removed)
    # else:
    #     np.save('voc_subset.npy',~removed)
    
    # Some score analysis for the log-KL-divergence-sum-score:
    diff = best_score-start_score
    factorsum = np.exp(diff)
    coco_div_improve = KL_div(voc_dist,coco_dist)/KL_div(dists[0,:],coco_dist)
    yisual_div_improve = KL_div(yisual_dist,dists[0,:])/KL_div(yisual_dist,voc_dist)
    print('Score impovement: {} -> {}, diff: {} (log-space), {} (factor space), KL(voc||coco)) *= {}, KL(yisual||voc) /= {}'.format(
        start_score,best_score,diff,factorsum,coco_div_improve,yisual_div_improve))
    
    print(remains,float(remains)/len(im_weights))
    plot_dists(coco_dist,voc_dist,'coco','voc_sub')
    plot_dists(coco_dist,yisual_dist,'coco','yisual')
    plot_dists(voc_dist,yisual_dist,'voc_sub','yisual')
    
    plt.figure(figsize=(30,20))
    plt.plot(weights,divergences)
    plt.plot(weights,divergences_coco)
    ns = (np.array(ns))*weights[0]/im_dists.shape[0]
    plt.plot(ns,divergences)
    plt.plot(ns,divergences_coco)
    nops = np.linspace(weights[0],weights[-1],len(weights))
    plt.plot(nops,ns/weights[0])
    plt.plot(nops,divergences)
    plt.plot(nops,divergences_coco)
    plt.plot([weights[0],weights[-1]],[KL_div(yisual_dist,coco_dist)]*2)
    plt.ylim(0,1)
    plt.show()
    
    # plt.figure(figsize=(30,20))
    # nimgs = np.arange(len(weights))[::-1] + 1
    # plt.plot(nimgs,divergences)
    # plt.plot(nimgs,divergences_coco)
    # plt.plot([nimgs[0],nimgs[-1]],[KL_div(yisual_dist,coco_dist)]*2)
    # plt.show()
        
    
    
    
    
    
    
    
    
    
    
    
    