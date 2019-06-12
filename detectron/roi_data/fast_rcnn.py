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

"""Construct minibatches for Fast R-CNN training. Handles the minibatch blobs
that are specific to Fast R-CNN. Other blobs that are generic to RPN, etc.
are handled by their respecitive roi_data modules.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import numpy as np
import numpy.random as npr

from detectron.core.config import cfg
import detectron.modeling.FPN as fpn
import detectron.roi_data.keypoint_rcnn as keypoint_rcnn_roi_data
import detectron.roi_data.mask_rcnn as mask_rcnn_roi_data
import detectron.utils.blob as blob_utils
import detectron.utils.boxes as box_utils

logger = logging.getLogger(__name__)


def get_fast_rcnn_blob_names(is_training=True):
    """Fast R-CNN blob names."""
    # rois blob: holds R regions of interest, each is a 5-tuple
    # (batch_idx, x1, y1, x2, y2) specifying an image batch index and a
    # rectangle (x1, y1, x2, y2)
    blob_names = ['rois']
    if is_training:
        # labels_int32 blob: R categorical labels in [0, ..., K] for K
        # foreground classes plus background
        blob_names += ['labels_int32']
    if is_training:
        # bbox_targets blob: R bounding-box regression targets with 4
        # targets per class
        blob_names += ['bbox_targets']
        # bbox_inside_weights blob: At most 4 targets per roi are active
        # this binary vector sepcifies the subset of active targets
        blob_names += ['bbox_inside_weights']
        blob_names += ['bbox_outside_weights']
    if is_training and cfg.MODEL.MASK_ON:
        # 'mask_rois': RoIs sampled for training the mask prediction branch.
        # Shape is (#masks, 5) in format (batch_idx, x1, y1, x2, y2).
        blob_names += ['mask_rois']
        # 'roi_has_mask': binary labels for the RoIs specified in 'rois'
        # indicating if each RoI has a mask or not. Note that in some cases
        # a *bg* RoI will have an all -1 (ignore) mask associated with it in
        # the case that no fg RoIs can be sampled. Shape is (batchsize).
        blob_names += ['roi_has_mask_int32']
        # 'masks_int32' holds binary masks for the RoIs specified in
        # 'mask_rois'. Shape is (#fg, M * M) where M is the ground truth
        # mask size.
        blob_names += ['masks_int32']
    if is_training and cfg.MODEL.KEYPOINTS_ON:
        # 'keypoint_rois': RoIs sampled for training the keypoint prediction
        # branch. Shape is (#instances, 5) in format (batch_idx, x1, y1, x2,
        # y2).
        blob_names += ['keypoint_rois']
        # 'keypoint_locations_int32': index of keypoint in
        # KRCNN.HEATMAP_SIZE**2 sized array. Shape is (#instances). Used in
        # SoftmaxWithLoss.
        blob_names += ['keypoint_locations_int32']
        # 'keypoint_weights': weight assigned to each target in
        # 'keypoint_locations_int32'. Shape is (#instances). Used in
        # SoftmaxWithLoss.
        blob_names += ['keypoint_weights']
        # 'keypoint_loss_normalizer': optional normalization factor to use if
        # cfg.KRCNN.NORMALIZE_BY_VISIBLE_KEYPOINTS is False.
        blob_names += ['keypoint_loss_normalizer']
    if cfg.FPN.FPN_ON and cfg.FPN.MULTILEVEL_ROIS:
        # Support for FPN multi-level rois without bbox reg isn't
        # implemented (... and may never be implemented)
        k_max = cfg.FPN.ROI_MAX_LEVEL
        k_min = cfg.FPN.ROI_MIN_LEVEL
        # Same format as rois blob, but one per FPN level
        for lvl in range(k_min, k_max + 1):
            blob_names += ['rois_fpn' + str(lvl)]
        blob_names += ['rois_idx_restore_int32']
        if is_training:
            if cfg.MODEL.MASK_ON:
                for lvl in range(k_min, k_max + 1):
                    blob_names += ['mask_rois_fpn' + str(lvl)]
                blob_names += ['mask_rois_idx_restore_int32']
            if cfg.MODEL.KEYPOINTS_ON:
                for lvl in range(k_min, k_max + 1):
                    blob_names += ['keypoint_rois_fpn' + str(lvl)]
                blob_names += ['keypoint_rois_idx_restore_int32']
    if is_training and cfg.TRAIN.DOMAIN_ADAPTATION:
        
        # DA: group rois into groups suitable for slicing:
        # 0: unsup_source           | da
        # 1: unsup_target           |
        # 2: sup_unsup_source       |  | sup
        # 3: sup_source                |
        
        # PADA: group rois into groups suitable for slicing:
        # 0: unsup_source           | da
        # 1: unsup_target           |    | eval
        # 2: sup_unsup_source       |  | sup
        # 3: sup_source                |
        
        
        blob_names += ['dc_label']
        # blob_names += ['da_rois'] # merged with and replaced by 'rois', da_rois selected by slice on:
        blob_names += ['da_start','da_end']
        # blob_names += ['label_mask']  # replaced by slice on sup_ (followed by cls_reg_source_indices for PADA)
        blob_names += ['sup_start','sup_end']
        # blob_names += ['source_labels_int32']
        # blob_names += ['source_bbox_targets']
        # blob_names += ['source_bbox_inside_weights']
        # blob_names += ['source_bbox_outside_weights']
        blob_names += ['da_label_wide']
        
        if cfg.TRAIN.PADA:
            blob_names += ['pada_roi_weights']
            blob_names += ['avg_pada_weight'] #scalar
            # blob_names += ['da_dc_mask']
            blob_names += ['eval_start','eval_end']
            # blob_names += ['cls_reg_source_inidces']
            # blob_names += ['test_indices']
        
        # edits:
        # 'rois' is overloaded with all rois now.
        # da_rois -> da_indices
        # label_mask -> cls_reg_indices, cls_reg_source_indices
        # da_dc_mask # was equal to label_mask, only in int instead of bool.
        # () -> cls_reg_target_indices
        
    return blob_names


def add_fast_rcnn_blobs(blobs, im_scales, roidb):
    """Add blobs needed for training Fast R-CNN style models."""
    # Sample training RoIs from each image and append them to the blob lists
    
    # # by Jerome: make sure the roi batch is equally divisible by IMGS_PER_BATCH by decreasing the rois per img to
    # # the amount of the image in the batch with the least of them
    # min_rois_per_img = min(len(entry['gt_classes']) for entry in roidb)
    # if min_rois_per_img < int(cfg.TRAIN.BATCH_SIZE_PER_IM):
    #     print('rois per img pruned to', min_rois_per_img)
    
    # split 1:
    # rois -> da_indices | cls_reg_indices
    # split 2:
    # cls_reg rois ->  cls_reg_{source|target}_indices
    
    # if cfg.TRAIN.DOMAIN_ADAPTATION:
    # split_1_offset = 0
    # split_2_offset = 0
    #
    # da_sources = 0
    # da_targets = 0
    # sup_sources = 0
    
    blobs['groups'] = []
    # blobs['set'] = []
    
    for im_i, entry in enumerate(roidb):
        # if True or (cfg.TRAIN.DOMAIN_ADAPTATION and entry['is_source']) or not cfg.TRAIN.DOMAIN_ADAPTATION:
        if cfg.TRAIN.DOMAIN_ADAPTATION:
            frcn_blobs = _sample_da_rois(entry, im_scales[im_i], im_i)
            # b = frcn_blobs
            # b['da_indices'] += split_1_offset
            # b['cls_reg_indices'] += split_1_offset
            # b['sup_source_indices'] += split_1_offset
            # split_1_offset += len(b['rois'])
            #
            # # da_sources += len(b['da_indices'])
            #
            # if cfg.TRAIN.PADA:
            #     b['cls_reg_source_indices'] += split_2_offset
            #     b['cls_reg_target_indices'] += split_2_offset
            #     split_2_offset += len(b['cls_reg_indices'])
            #
            # # not needed but well:
            # frcn_blobs = b
            
        else:
            frcn_blobs = _sample_rois(entry, im_scales[im_i], im_i)
            
        for k, v in frcn_blobs.items():
            if len(v):
                blobs[k].append(v)
        
        # if cfg.TRAIN.DOMAIN_ADAPTATION:
        #     da_blobs = _sample_da_rois(entry, im_scales[im_i], im_i)
        #     for k, v in da_blobs.items():
        #         blobs[k].append(v)
        
    # Concat the training blob lists into tensors
    for k, v in blobs.items():
        if isinstance(v, list) and len(v) > 0:
            blobs[k] = np.concatenate(v)
            # # DEBUG support for 'set':
            # try:
            #     blobs[k] = np.concatenate(v)
            # except:
            #     blobs[k] = set.union(*v)
    
    # Add FPN multilevel training RoIs, if configured
    if cfg.FPN.FPN_ON and cfg.FPN.MULTILEVEL_ROIS:
        _add_multilevel_rois(blobs)
    
    if cfg.TRAIN.DOMAIN_ADAPTATION:
        
        groups = blobs['groups']
        
        
        chunks = np.empty(4,dtype=object)
        chunks[:] = [[] for _ in range(4)]
        sup_chunks = np.empty(2,dtype=object)
        sup_chunks[:] = [[] for _ in range(2)]
        i_sup = 0
        for i,g in enumerate(groups):
            chunks[g].append(i)
            if g >= 2: # if sup:
                sup_chunks[g-2].append(i_sup)
                i_sup += 1
        
        # flatten chuncks, get order:
        order = np.empty(len(groups),dtype=np.int32)
        i = 0
        for group in chunks:
            order[i:i+len(group)] = group
            i += len(group)
        
        blobs['rois'] = blobs['rois'][order]
        
        # DEBUG:
        # ordered_groups = groups[order]
        
        # DA: group rois into groups suitable for slicing:
        # 0: unsup_source           | da
        # 1: unsup_target           |
        # 2: sup_unsup_source       |  | sup
        # 3: sup_source                |
        
        # PADA: group rois into groups suitable for slicing:
        # 0: unsup_source           | da
        # 1: unsup_target           |    | eval
        # 2: sup_unsup_source       |  | sup
        # 3: sup_source                |
        
        da_start =             0
        eval_start =                        len(chunks[0])
        eval_end = sup_start = eval_start + len(chunks[1])
        da_end =               eval_end   + len(chunks[2])
        sup_end =              da_end     + len(chunks[3])
        
        # # DEBUG
        # print('chunks: ',end='')
        # for c in chunks:
        #     print(len(c),end=' ')
        # print()
        
        sup_order = np.empty(sup_end - sup_start,dtype=np.int32)
        sup_order[:len(sup_chunks[0]) ] = sup_chunks[0]
        sup_order[ len(sup_chunks[0]):] = sup_chunks[1]
        
        blob_names = []
        blob_names += ['labels_int32']
        blob_names += ['bbox_targets']
        blob_names += ['bbox_inside_weights']
        blob_names += ['bbox_outside_weights']
        for name in blob_names:
            blobs[name] = blobs[name][sup_order]
        
        
        dc_label = np.zeros(da_end,dtype=np.int32)
        dc_label[da_start:eval_start] = 1
        dc_label[eval_start:eval_end] = 0
        dc_label[eval_end:da_end] = 1
        blobs['dc_label'] = dc_label
        
        if cfg.TRAIN.PADA:
            # also reorder the pada_roi_weights that aligns with the da slice:
            # da slice at:
            groups = groups[groups != 3]
            inds = np.array([0,eval_start,sup_start])
            order = np.empty(da_end,dtype=np.int32)
            for i,g in enumerate(groups):
                order[inds[g]] = i
                inds[g] += 1
            
            blobs['pada_roi_weights'] = blobs['pada_roi_weights'][order]
        
        
        del blobs['groups'] # this info was only needed for the re-ordering above.
        
        # submit slicing info instead:
        blob_names = []
        blob_names += ['da_start','da_end']
        blob_names += ['sup_start','sup_end']
        if cfg.TRAIN.PADA:
            blob_names += ['eval_start','eval_end']
            
        blobs['da_start'] = da_start
        blobs['da_end'] = da_end
        blobs['sup_start'] = sup_start
        blobs['sup_end'] = sup_end
        if cfg.TRAIN.PADA:
            blobs['eval_start'] = eval_start
            blobs['eval_end'] = eval_end
        if cfg.MODEL.CONV_BODY[:6] == "VGG16.":
            for name in blob_names:
                blobs[name] = np.array([blobs[name]]+[-1 if name[-3:] == 'end' else 0]*1,dtype=np.int32) # for VGG16
        else:
            for name in blob_names:
                blobs[name] = np.array([blobs[name]]+[-1 if name[-3:] == 'end' else 0]*3,dtype=np.int32)
        
        
        # # DEBUG code:
        # l = []
        # labs = blobs['labels_int32']
        # box_t = blobs['bbox_targets']
        # biw = blobs['bbox_inside_weights']
        # bow = blobs['bbox_outside_weights']
        # wts = list(blobs['pada_roi_weights'].copy())
        # # wts = [1] * 512
        # i = 0
        # for g,roi in zip(ordered_groups,blobs['rois']):
        #     if g < 2: # if unsup:
        #         l.append((tuple(roi),wts.pop(0)))
        #     else:
        #         w = wts.pop(0) if g == 2 else None
        #         l.append((tuple(roi),labs[i],tuple(box_t[i]),tuple(biw[i]),tuple(bow[i]),w))
        #         i += 1
        # orig_set = blobs['set']
        # for tup in l:
        #     if tup not in orig_set:
        #         print('AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH')
        # assert len(wts) == 0
        # del blobs['set']
        
        if cfg.TRAIN.PADA:
            # Cancel out the large gradients from loss-averaging over small roi proposal batches.
            # loss-averaging for regular batches means dividing the instance losses by the regular batch size.
            # we correct small batch loss-averaging by multiplying by the small batch size, and dividing by the regular batch size.
            regular_rois_batch_size = cfg.TRAIN.IMS_PER_BATCH * cfg.TRAIN.BATCH_SIZE_PER_IM
            pada_weights = blobs['pada_roi_weights']
            rois = len(pada_weights)
            pada_weights *= rois/regular_rois_batch_size # lower pada_weights means lower learning rate.
            blobs['pada_roi_weights'] = pada_weights
        
        # blobs['test_indices'] = np.arange(10,dtype=np.int32)
        # blobs['da_indices'] = (np.arange(10,dtype=np.int32) + 5)
        
        # edits:
        # 'rois' is overloaded with all rois now.
        # da_rois -> da_indices
        # label_mask -> cls_reg_indices, cls_reg_source_indices
        # da_dc_mask # was equal to label_mask, only in int instead of bool.
        # () -> cls_reg_target_indices
        
        # edits:
        # 'rois' is overloaded with all rois now.
        # all_rois = blobs['rois']
        # # all_selection = blobs['roi_indices']
        # rois = all_rois[all_selection]
        # blobs['rois'] = rois
        # roi_indices = np.arange(len(all_rois))
        # roi_indices[all_selection] = np.arange(len(all_selection))
        #
        # # da_rois -> da_indices
        # blobs['da_indices'] = roi_indices[blobs['da_indices']]
        # label_mask -> cls_reg_indices, cls_reg_source_indices
        # da_dc_mask # was equal to label_mask, only in int instead of bool.
        # () -> cls_reg_target_indices
        
    
    # Perform any final work and validity checks after the collating blobs for
    # all minibatch images
    valid = True
    if cfg.MODEL.KEYPOINTS_ON:
        valid = keypoint_rcnn_roi_data.finalize_keypoint_minibatch(blobs, valid)

    return valid


def _sample_rois(roidb, im_scale, batch_idx): #max_rois = int(cfg.TRAIN.BATCH_SIZE_PER_IM)):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    # rois_per_image = min(max_rois,int(cfg.TRAIN.BATCH_SIZE_PER_IM))
    rois_per_image = int(cfg.TRAIN.BATCH_SIZE_PER_IM)
    fg_rois_per_image = int(np.round(cfg.TRAIN.FG_FRACTION * rois_per_image))
    max_overlaps = roidb['max_overlaps']

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(max_overlaps >= cfg.TRAIN.FG_THRESH)[0]
    # Guard against the case when an image has fewer than fg_rois_per_image
    # foreground RoIs
    fg_rois_per_this_image = np.minimum(fg_rois_per_image, fg_inds.size)
    # Sample foreground regions without replacement
    if fg_inds.size > 0:
        fg_inds = npr.choice(
            fg_inds, size=fg_rois_per_this_image, replace=False
        )

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where(
        (max_overlaps < cfg.TRAIN.BG_THRESH_HI) &
        (max_overlaps >= cfg.TRAIN.BG_THRESH_LO)
    )[0]
    # Compute number of background RoIs to take from this image (guarding
    # against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = np.minimum(bg_rois_per_this_image, bg_inds.size)
    # Sample background regions selection
    if bg_inds.size > 0:
        bg_inds = npr.choice(
            bg_inds, size=bg_rois_per_this_image, replace=False
        )

    # The indices that we're selecting (both fg and bg)
    keep_inds = np.append(fg_inds, bg_inds)
    # Label is the class each RoI has max overlap with
    sampled_labels = roidb['max_classes'][keep_inds]
    sampled_labels[fg_rois_per_this_image:] = 0  # Label bg RoIs with class 0
    sampled_boxes = roidb['boxes'][keep_inds]
    bbox_targets, bbox_inside_weights = _expand_bbox_targets(
        roidb['bbox_targets'][keep_inds, :]
    )
    bbox_outside_weights = np.array(
        bbox_inside_weights > 0, dtype=bbox_inside_weights.dtype
    )
    
    # NOT: (is already done by the corrected pada weights)
    # correct for batch averaging for smaller batches to cancel the lr boost.
    # bbox_outside_weights = np.array(bbox_outside_weights,dtype=np.float32) * len(bbox_outside_weights) / cfg.TRAIN.BATCH_SIZE_PER_IM

    # Scale rois and format as (batch_idx, x1, y1, x2, y2)
    sampled_rois = sampled_boxes * im_scale
    repeated_batch_idx = batch_idx * blob_utils.ones((sampled_rois.shape[0], 1))
    sampled_rois = np.hstack((repeated_batch_idx, sampled_rois))

    # Base Fast R-CNN blobs
    blob_dict = dict(
        labels_int32=sampled_labels.astype(np.int32, copy=False),
        rois=sampled_rois,
        bbox_targets=bbox_targets,
        bbox_inside_weights=bbox_inside_weights,
        bbox_outside_weights=bbox_outside_weights
    )

    # Optionally add Mask R-CNN blobs
    if cfg.MODEL.MASK_ON:
        mask_rcnn_roi_data.add_mask_rcnn_blobs(
            blob_dict, sampled_boxes, roidb, im_scale, batch_idx
        )

    # Optionally add Keypoint R-CNN blobs
    if cfg.MODEL.KEYPOINTS_ON:
        keypoint_rcnn_roi_data.add_keypoint_rcnn_blobs(
            blob_dict, roidb, fg_rois_per_image, fg_inds, im_scale, batch_idx
        )
    
    # # optionally add Domain Adaptive R-CNN blobs
    # if cfg.TRAIN.DOMAIN_ADAPTATION:
    #     if roidb['is_source']:
    #         blob_dict['label_mask'] = np.full(blob_dict['labels_int32'].shape, True)
    #         blob_dict['source_labels_int32'] = blob_dict['labels_int32']
    #         blob_dict['source_bbox_targets'] = blob_dict['bbox_targets']
    #         blob_dict['source_bbox_inside_weights'] = blob_dict['bbox_inside_weights']
    #         blob_dict['source_bbox_outside_weights'] = blob_dict['bbox_outside_weights']
    #         blob_dict['da_label_wide'] = np.ones((1,1,200, 400), dtype=np.int32)
    #     else:
    #         blob_dict['label_mask'] = np.full(blob_dict['labels_int32'].shape, False)
    #         blob_dict['da_label_wide'] = np.zeros((1,1,200, 400), dtype=np.int32)

    return blob_dict

def _sample_da_rois(roidb, im_scale, batch_idx):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    # rois_per_image = min(max_rois,int(cfg.TRAIN.BATCH_SIZE_PER_IM))
    is_source = roidb['is_source']
    if is_source:
        rois_per_image = int(cfg.TRAIN.BATCH_SIZE_PER_IM)
        fg_rois_per_image = int(np.round(cfg.TRAIN.FG_FRACTION * rois_per_image))
        max_overlaps = roidb['max_overlaps']
    
        # Select foreground RoIs as those with >= FG_THRESH overlap
        fg_inds = np.where(max_overlaps >= cfg.TRAIN.FG_THRESH)[0]
        # Guard against the case when an image has fewer than fg_rois_per_image
        # foreground RoIs
        fg_rois_per_this_image = np.minimum(fg_rois_per_image, fg_inds.size)
        # Sample foreground regions without replacement
        if fg_inds.size > 0:
            fg_inds = npr.choice(
                fg_inds, size=fg_rois_per_this_image, replace=False
            )
    
        # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
        bg_inds = np.where(
            (max_overlaps < cfg.TRAIN.BG_THRESH_HI) &
            (max_overlaps >= cfg.TRAIN.BG_THRESH_LO)
        )[0]
        # Compute number of background RoIs to take from this image (guarding
        # against there being fewer than desired)
        bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
        bg_rois_per_this_image = np.minimum(bg_rois_per_this_image, bg_inds.size)
        # Sample background regions selection
        if bg_inds.size > 0:
            bg_inds = npr.choice(
                bg_inds, size=bg_rois_per_this_image, replace=False
            )
    
        # The indices that we're selecting (both fg and bg)
        # DA: This is the supervised selection:
        keep_inds = np.append(fg_inds, bg_inds)
        
        # Label is the class each RoI has max overlap with
        sampled_labels = roidb['max_classes'][keep_inds]
        sampled_labels[fg_rois_per_this_image:] = 0  # Label bg RoIs with class 0
        # sampled_boxes = roidb['boxes'][keep_inds]
        bbox_targets, bbox_inside_weights = _expand_bbox_targets(
            roidb['bbox_targets'][keep_inds, :]
        )
        bbox_outside_weights = np.array(
            bbox_inside_weights > 0, dtype=bbox_inside_weights.dtype
        )
        
        # correct for batch averaging for smaller batches to cancel the lr boost.
        bbox_outside_weights = np.array(bbox_outside_weights,dtype=np.float32) * len(bbox_outside_weights) / cfg.TRAIN.BATCH_SIZE_PER_IM
        
        
        rpn_boxes = roidb['rpn_boxes']  # as int, the len.
        total_boxes = len(max_overlaps)
        rpn_rois_per_image = min(cfg.TRAIN.BATCH_SIZE_PER_IM,rpn_boxes)
        selected = np.full(total_boxes,False,dtype=bool)
        
        # box_list = gt_box_list + rpn_box_list
        # the last boxes are rpn_boxes so they start at:
        unsup_start = total_boxes - rpn_boxes
        unsup_end = unsup_start + rpn_rois_per_image
        print('unsup_range: ({},{})'.format(unsup_start,unsup_end))
        selected[unsup_start : unsup_end] = True # unsupervised selection
        
        # DA: group rois into groups suitable for slicing:
        # 0: unsup_source           | da
        # 1: unsup_target           |
        # 2: sup_unsup_source       |  | sup
        # 3: sup_source                |
        
        # PADA: group rois into groups suitable for slicing:
        # 0: unsup_source           | da
        # 1: unsup_target           |    | eval
        # 2: sup_unsup_source       |  | sup
        # 3: sup_source                |
    
        groups = np.full(total_boxes,3)      # default to sup
        groups[unsup_start : unsup_end] = 0  # now topK is default unsup
        unsup_sups = selected[keep_inds]
        groups[keep_inds[unsup_sups]] = 2    # set intersection to sup_unsup
        selected[keep_inds] = True # supervised selection
        all_inds = np.nonzero(selected)[0]
        groups = groups[all_inds]            # take selection of used only
        
        sampled_boxes = roidb['boxes'][all_inds]
        
        
    else: #if target image:
        
        sampled_labels = np.array([])
        bbox_targets = []
        bbox_inside_weights = []
        bbox_outside_weights = []
        
        sampled_boxes = roidb['boxes'][:cfg.TRAIN.BATCH_SIZE_PER_IM]
        groups = np.full(len(sampled_boxes),1)
        
    
    # DA: add a roi selection for DA that is not based on the GT boxes:
    
    # else:
    #     # selected[keep_inds] = True # supervised selection # no supervision on targets...
    #     all_inds = np.nonzero(selected)[0]
    #     groups = np.full(len(all_inds),1)
    
    # full_select = np.arange(len(all_inds))
    #
    # # get da_indices and cls_reg_indices
    # to_all_inds = np.empty(total_boxes,dtype=np.int32)
    # to_all_inds[all_inds] = full_select
    # da_indices = to_all_inds[np.arange(unsup_start,unsup_end)]
    #
    # if is_source:
    #     cls_reg_indices = to_all_inds[keep_inds]
    # else:
    #     # Target set predictions not needed.
    #     cls_reg_indices = []  # Note: this will be overwritten when doing PADA
    #
    # sup_source_indices = cls_reg_indices[:]
    #
    # if cfg.TRAIN.PADA:
    #
    #     # get cls_reg_source_indices and cls_reg_target_indices
    #     if is_source:
    #         cls_reg_source_indices = np.arange(len(cls_reg_indices),dtype=np.int32) # select all from the pre-selection
    #         cls_reg_target_indices = []
    #     else:
    #         # for statistics gathering :
    #         cls_reg_indices = full_select
    #         cls_reg_target_indices = full_select
    #
    #         cls_reg_source_indices = []
    
    
    


    # Scale rois and format as (batch_idx, x1, y1, x2, y2)
    sampled_rois = sampled_boxes * im_scale
    repeated_batch_idx = batch_idx * blob_utils.ones((sampled_rois.shape[0], 1))
    sampled_rois = np.hstack((repeated_batch_idx, sampled_rois))

    # Base Fast R-CNN blobs
    blob_dict = dict(
        labels_int32=sampled_labels.astype(np.int32, copy=False),
        rois=sampled_rois,
        bbox_targets=bbox_targets,
        bbox_inside_weights=bbox_inside_weights,
        bbox_outside_weights=bbox_outside_weights,
        # da_indices=da_indices,
        # cls_reg_indices=cls_reg_indices,
        # sup_source_indices=sup_source_indices,
        groups=groups
    )
    
   
    
    
    
            
            
        
    
    
    # Optionally add Mask R-CNN blobs
    if cfg.MODEL.MASK_ON:
        mask_rcnn_roi_data.add_mask_rcnn_blobs(
            blob_dict, sampled_boxes, roidb, im_scale, batch_idx
        )

    # # Optionally add Keypoint R-CNN blobs
    # if cfg.MODEL.KEYPOINTS_ON:
    #     keypoint_rcnn_roi_data.add_keypoint_rcnn_blobs(
    #         blob_dict, roidb, fg_rois_per_image, fg_inds, im_scale, batch_idx
    #     )
    
        
    # edits:
    # 'rois' is overloaded with all rois now.
    # da_rois -> da_indices
    # label_mask -> cls_reg_indices, cls_reg_source_indices
    # da_dc_mask # was equal to label_mask, only in int instead of bool.
    # () -> cls_reg_target_indices
    
    if is_source:
        # blob_dict['label_mask'] = np.full(blob_dict['labels_int32'].shape, True)
        # blob_dict['source_labels_int32'] = blob_dict['labels_int32']
        # blob_dict['source_bbox_targets'] = blob_dict['bbox_targets']
        # blob_dict['source_bbox_inside_weights'] = blob_dict['bbox_inside_weights']
        # blob_dict['source_bbox_outside_weights'] = blob_dict['bbox_outside_weights']
        blob_dict['da_label_wide'] = np.ones((1,1,200, 400), dtype=np.int32)
    else:
        # blob_dict['label_mask'] = np.full(blob_dict['labels_int32'].shape, False)
        blob_dict['da_label_wide'] = np.zeros((1,1,200, 400), dtype=np.int32)
    
    #  # Base Fast R-CNN blobs
    # blob_dict = dict(
    #     da_rois=sampled_rois
    # )
    
    # # add Domain Adaptive R-CNN blobs
    # if is_source:
    #     blob_dict['dc_label'] = np.expand_dims(np.ones(rpn_rois_per_image, dtype=np.int32), axis=1)
    # else:
    #     blob_dict['dc_label'] = np.expand_dims(np.zeros(rpn_rois_per_image, dtype=np.int32), axis=1)
    
    if cfg.TRAIN.PADA:
        blob_dict['pada_roi_weights'] = roidb['pada_roi_weights']
        # blob_dict['da_dc_mask'] = np.full(rois_per_image,is_source.astype(bool))
        # for splitting with the second 'Gather pair':
        # blob_dict['cls_reg_source_indices'] = cls_reg_source_indices
        # blob_dict['cls_reg_target_indices'] = cls_reg_target_indices
    
    
    # PADA: group rois into groups suitable for slicing:
    # 0: unsup_source           | da
    # 1: unsup_target           |    | eval
    # 2: sup_unsup_source       |  | sup
    # 3: sup_source                |
    
    # # DEBUG code:
    # s = set()
    # labs = sampled_labels[:]
    # box_t = bbox_targets[:]
    # biw = bbox_inside_weights[:]
    # bow = bbox_outside_weights[:]
    # wts = list(blob_dict['pada_roi_weights'].copy())
    # # wts = [1] * 256
    # print(len(wts))
    # print(len(groups))
    # print(len(groups[groups != 3]))
    # print(is_source)
    # print()
    # i = 0
    # for g,roi in zip(groups,blob_dict['rois']):
    #     if g < 2: # if unsup:
    #         s.add((tuple(roi),wts.pop(0)))
    #     else:
    #         w = wts.pop(0) if g == 2 else None
    #         s.add((tuple(roi),labs[i],tuple(box_t[i]),tuple(biw[i]),tuple(bow[i]),w))
    #         i += 1
    #
    # blob_dict['set'] = s
    
    return blob_dict

def _expand_bbox_targets(bbox_target_data):
    """Bounding-box regression targets are stored in a compact form in the
    roidb.

    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets). The loss weights
    are similarly expanded.

    Returns:
        bbox_target_data (ndarray): N x 4K blob of regression targets
        bbox_inside_weights (ndarray): N x 4K blob of loss weights
    """
    num_bbox_reg_classes = cfg.MODEL.NUM_CLASSES
    if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG:
        num_bbox_reg_classes = 2  # bg and fg

    clss = bbox_target_data[:, 0]
    bbox_targets = blob_utils.zeros((clss.size, 4 * num_bbox_reg_classes))
    bbox_inside_weights = blob_utils.zeros(bbox_targets.shape)
    inds = np.where(clss > 0)[0]
    for ind in inds:
        cls = int(clss[ind])
        start = 4 * cls
        end = start + 4
        bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
        bbox_inside_weights[ind, start:end] = (1.0, 1.0, 1.0, 1.0)
    return bbox_targets, bbox_inside_weights


def _add_multilevel_rois(blobs):
    """By default training RoIs are added for a single feature map level only.
    When using FPN, the RoIs must be distributed over different FPN levels
    according the level assignment heuristic (see: modeling.FPN.
    map_rois_to_fpn_levels).
    """
    lvl_min = cfg.FPN.ROI_MIN_LEVEL
    lvl_max = cfg.FPN.ROI_MAX_LEVEL

    def _distribute_rois_over_fpn_levels(rois_blob_name):
        """Distribute rois over the different FPN levels."""
        # Get target level for each roi
        # Recall blob rois are in (batch_idx, x1, y1, x2, y2) format, hence take
        # the box coordinates from columns 1:5
        target_lvls = fpn.map_rois_to_fpn_levels(
            blobs[rois_blob_name][:, 1:5], lvl_min, lvl_max
        )
        # Add per FPN level roi blobs named like: <rois_blob_name>_fpn<lvl>
        fpn.add_multilevel_roi_blobs(
            blobs, rois_blob_name, blobs[rois_blob_name], target_lvls, lvl_min,
            lvl_max
        )

    _distribute_rois_over_fpn_levels('rois')
    if cfg.MODEL.MASK_ON:
        _distribute_rois_over_fpn_levels('mask_rois')
    if cfg.MODEL.KEYPOINTS_ON:
        _distribute_rois_over_fpn_levels('keypoint_rois')
