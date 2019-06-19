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

"""Various network "heads" for classification and bounding box prediction.

The design is as follows:

... -> RoI ----\                               /-> box cls output -> cls loss
                -> RoIFeatureXform -> box head
... -> Feature /                               \-> box reg output -> reg loss
       Map

The Fast R-CNN head produces a feature representation of the RoI for the purpose
of bounding box classification and regression. The box output module converts
the feature representation into classification and regression predictions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from detectron.core.config import cfg
from detectron.utils.c2 import const_fill
from detectron.utils.c2 import gauss_fill
from detectron.utils.net import get_group_gn
import detectron.utils.blob as blob_utils


# ---------------------------------------------------------------------------- #
# Fast R-CNN outputs and losses
# ---------------------------------------------------------------------------- #

def add_fast_rcnn_outputs(model, blob_in, dim):
    """Add RoI classification and bounding box regression output ops."""
    
    if cfg.TRAIN.DOMAIN_ADAPTATION:
        blob_in = model.net.Copy(blob_in,'feats_copy_sup')
        blob_in = model.net.Slice([blob_in,'sup_start','sup_end'],'sup_source_feats')
    
    # Box classification layer
    model.FC(
        blob_in,
        'cls_score',
        dim,
        model.num_classes,
        weight_init=gauss_fill(0.01),
        bias_init=const_fill(0.0)
    )
    
    if model.train and cfg.TRAIN.PADA:
        # model.net.Gather(['cls_score','cls_reg_source_indices'],['sup_cls_score'])
        # model.PADAbyGradientWeightingLayer('sup_cls_score','pada_cls_score','labels_int32')
        model.PADAbyGradientWeightingLayer('cls_score','pada_cls_score','labels_int32')
    
    if not model.train:  # == if test
        # Only add softmax when testing; during training the softmax is combined
        # with the label cross entropy loss for numerical stability
        model.Softmax('cls_score', 'cls_prob', engine='CUDNN')
    # Box regression layer
    num_bbox_reg_classes = (
        2 if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG else model.num_classes
    )
    
    # if model.train and cfg.TRAIN.PADA and not cfg.MODEL.CLS_AGNOSTIC_BBOX_REG:
        # The class-specific bbox predictors are independant of each other, so no pada weighting needed for this last layer.
        # blob_in = model.PADAbyGradientWeightingLayer(blob_in, 'pada_weighted_feats', 'source_labels_int32')
        # blob_in = blob_weighted
        
    model.FC(
        blob_in,
        'bbox_pred',
        dim,
        num_bbox_reg_classes * 4,
        weight_init=gauss_fill(0.001),
        bias_init=const_fill(0.0)
    )
    
    if model.train and cfg.TRAIN.PADA and not cfg.MODEL.CLS_AGNOSTIC_BBOX_REG:
        # model.net.Gather(['bbox_pred','cls_reg_source_indices'],['sup_bbox_pred'])
        # model.PADAbyGradientWeightingLayer('sup_bbox_pred','pada_bbox_pred','labels_int32')
        model.PADAbyGradientWeightingLayer('bbox_pred','pada_bbox_pred','labels_int32')
        


def add_fast_rcnn_losses(model):
    """Add losses for RoI classification and bounding box regression."""
    if False and cfg.TRAIN.DOMAIN_ADAPTATION:
        scores = 'cls_score'
        box_preds = 'bbox_pred'
        if cfg.TRAIN.PADA:
            scores = 'pada_' + scores
            box_preds = 'pada_' + box_preds
        # model.MaskingInput([scores, 'label_mask'], ['source_cls_score'])
        # model.MaskingInput([box_preds, 'label_mask'], ['source_bbox_pred'])
        
        cls_prob, loss_cls = model.net.SoftmaxWithLoss(
            ['source_cls_score', 'source_labels_int32'], ['cls_prob', 'loss_cls'],
            scale=model.GetLossScale()
        )
        loss_bbox = model.net.SmoothL1Loss(
            [
                'source_bbox_pred', 'source_bbox_targets', 'source_bbox_inside_weights',
                'source_bbox_outside_weights'
            ],
            'loss_bbox',
            scale=model.GetLossScale()
        )
        model.Accuracy(['cls_prob', 'source_labels_int32'], 'accuracy_cls')
        
        def update_conf_matrix(inputs,outputs):
            cls_prob = inputs[0].data
            labels = inputs[1].data
            # print(cls_prob.shape)
            # print(labels.shape)
            
            model.class_weight_db.update_confusion_matrix(cls_prob,labels)
            
        model.net.Python(update_conf_matrix)(['cls_prob','source_labels_int32'],[],name='UpdateConfusionMatrix')
        
            
            
        
    else:
        
        scores = 'cls_score'
        box_preds = 'bbox_pred'
        if cfg.TRAIN.PADA:
            scores = 'pada_' + scores
            box_preds = 'pada_' + box_preds
            
        cls_prob, loss_cls = model.net.SoftmaxWithLoss(
            [scores, 'labels_int32'], ['cls_prob',     'loss_cls'],
            scale=model.GetLossScale()
        )
        loss_bbox = model.net.SmoothL1Loss(
            [
                box_preds, 'bbox_targets',    'bbox_inside_weights',
                'bbox_outside_weights'
            ],
            'loss_bbox',
            scale=model.GetLossScale()
        )
        model.Accuracy(['cls_prob', 'labels_int32'], 'accuracy_cls')
    
    if cfg.TRAIN.PADA:
        def update_conf_matrix(inputs,outputs):
            cls_prob = inputs[0].data
            labels = inputs[1].data
            # print(cls_prob.shape)
            # print(labels.shape)
            
            model.class_weight_db.update_confusion_matrix(cls_prob,labels)
            
        model.net.Python(update_conf_matrix)(['cls_prob','labels_int32'],[],name='UpdateConfusionMatrix')
    
    loss_gradients = blob_utils.get_loss_gradients(model,   [loss_cls, loss_bbox])
    model.AddLosses(['loss_cls', 'loss_bbox'])
    model.AddMetrics('accuracy_cls')
    return loss_gradients


# ---------------------------------------------------------------------------- #
# Box heads
# ---------------------------------------------------------------------------- #

def add_roi_2mlp_head(model, blob_in, dim_in, spatial_scale):
    """Add a ReLU MLP with two hidden layers."""
    hidden_dim = cfg.FAST_RCNN.MLP_HEAD_DIM
    roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
    roi_feat = model.RoIFeatureTransform(
        blob_in,
        'roi_feat',
        blob_rois='rois',
        method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
        resolution=roi_size,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale
    )
    model.FC(roi_feat, 'fc6', dim_in * roi_size * roi_size, hidden_dim)
    model.Relu('fc6', 'fc6')
    model.FC('fc6', 'fc7', hidden_dim, hidden_dim)
    model.Relu('fc7', 'fc7')
    return 'fc7', hidden_dim


def add_roi_Xconv1fc_head(model, blob_in, dim_in, spatial_scale):
    """Add a X conv + 1fc head, as a reference if not using GroupNorm"""
    hidden_dim = cfg.FAST_RCNN.CONV_HEAD_DIM
    roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
    roi_feat = model.RoIFeatureTransform(
        blob_in,
        'roi_feat',
        blob_rois='rois',
        method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
        resolution=roi_size,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale
    )

    current = roi_feat
    for i in range(cfg.FAST_RCNN.NUM_STACKED_CONVS):
        current = model.Conv(
            current, 'head_conv' + str(i + 1), dim_in, hidden_dim, 3,
            stride=1, pad=1,
            weight_init=('MSRAFill', {}),
            bias_init=('ConstantFill', {'value': 0.}),
            no_bias=0)
        current = model.Relu(current, current)
        dim_in = hidden_dim

    fc_dim = cfg.FAST_RCNN.MLP_HEAD_DIM
    model.FC(current, 'fc6', dim_in * roi_size * roi_size, fc_dim)
    model.Relu('fc6', 'fc6')
    return 'fc6', fc_dim


def add_roi_Xconv1fc_gn_head(model, blob_in, dim_in, spatial_scale):
    """Add a X conv + 1fc head, with GroupNorm"""
    hidden_dim = cfg.FAST_RCNN.CONV_HEAD_DIM
    roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
    roi_feat = model.RoIFeatureTransform(
        blob_in, 'roi_feat',
        blob_rois='rois',
        method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
        resolution=roi_size,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale
    )

    current = roi_feat
    for i in range(cfg.FAST_RCNN.NUM_STACKED_CONVS):
        current = model.ConvGN(
            current, 'head_conv' + str(i + 1), dim_in, hidden_dim, 3,
            group_gn=get_group_gn(hidden_dim),
            stride=1, pad=1,
            weight_init=('MSRAFill', {}),
            bias_init=('ConstantFill', {'value': 0.}))
        current = model.Relu(current, current)
        dim_in = hidden_dim

    fc_dim = cfg.FAST_RCNN.MLP_HEAD_DIM
    model.FC(current, 'fc6', dim_in * roi_size * roi_size, fc_dim)
    model.Relu('fc6', 'fc6')
    return 'fc6', fc_dim
