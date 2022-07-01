from .single_stage_rbbox import SingleStageDetectorRbbox
from ..registry import DETECTORS

from mmdet.models.builder import DETECTORS, build_backbone, build_neck
from mmdet.core import bbox2result, dbbox2result
from mmdet.apis.env import get_root_logger

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.C3.correlation import CorrelationModule

@DETECTORS.register_module
class RetinaNetRbbox(SingleStageDetectorRbbox):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head=None,
                 rbbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(RetinaNetRbbox, self).__init__(backbone, neck, bbox_head, rbbox_head,
                                             train_cfg, test_cfg, pretrained)

@DETECTORS.register_module
class RetinaNetOBBEarlyFusion(RetinaNetRbbox):
    """Implementation of `RetinaNet <https://arxiv.org/abs/1708.02002>`_"""

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 rbbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(RetinaNetOBBEarlyFusion, self).__init__(backbone, neck, bbox_head, rbbox_head,
                                             train_cfg, test_cfg, pretrained)
        self.bbox_head_cfg = rbbox_head

    def process_userinput(self, img, **kwargs):
        x = self.extract_feat(img)
        return x

    def check_UEL(self, kwargs, bbox_head_cfg):
        kwargs.pop('img_metas')
        if 'Userinput' not in bbox_head_cfg['type']:
            kwargs.pop('gt_userinputs')
            kwargs.update({'UEL': False})
        else:
            kwargs.update({'UEL': True})
        return kwargs

    def forward_bbox_head(self, x, img_metas, gt_bboxes, gt_masks, gt_labels, gt_bboxes_ignore):
        losses = dict()

        if self.with_bbox:
            bbox_outs = self.bbox_head(x)
            bbox_loss_inputs = bbox_outs + (gt_bboxes, gt_labels, img_metas, self.train_cfg)
            # TODO: make if flexible to add the bbox_head
            bbox_losses = self.bbox_head.loss(
                *bbox_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            losses.update(bbox_losses)
        if self.with_rbbox:
            rbbox_outs = self.rbbox_head(x)
            rbbox_loss_inputs = rbbox_outs + (gt_bboxes, gt_masks, gt_labels, img_metas, self.train_cfg)
            rbbox_losses = self.rbbox_head.loss(
                *rbbox_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            losses.update(rbbox_losses)
        return losses
        
    def forward_train(self,
                img,
                img_metas,
                gt_bboxes,
                gt_masks,
                gt_labels,
                gt_bboxes_ignore=None,
                **kwargs):
        kwargs.update({'img_metas': img_metas})
        x = self.process_userinput(img, **kwargs)

        kwargs = self.check_UEL(kwargs, self.bbox_head_cfg)
        losses = self.forward_bbox_head(x, img_metas, gt_bboxes,
                                              gt_masks, gt_labels, gt_bboxes_ignore)
        return losses

    def simple_test(self, img, img_metas, rescale=False, **kwargs):
        kwargs.update({'img_metas': img_metas})
        x = self.process_userinput(img, **kwargs)
        rbbox_outs = self.rbbox_head(x)
        rbbox_inputs = rbbox_outs + (img_metas, self.test_cfg, rescale)
        rbbox_list =self.rbbox_head.get_bboxes(*rbbox_inputs)
        rbbox_results = [
            dbbox2result(det_rbboxes, det_labels, self.rbbox_head.num_classes)
            for det_rbboxes, det_labels in rbbox_list
        ]
        return rbbox_results[0]


@DETECTORS.register_module
class RetinaNetOBBLateFusion(RetinaNetOBBEarlyFusion):
    """Implementation of `RetinaNet <https://arxiv.org/abs/1708.02002>`_"""

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 rbbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 userinput_backbone=None,
                 userinput_neck=None):
        super(RetinaNetOBBLateFusion, self).__init__(backbone, neck, bbox_head, rbbox_head,
                                             train_cfg, test_cfg, pretrained)

        self.rbbox_head_cfg = rbbox_head
        self.logger = get_root_logger()

        self.logger.info(str(self.rbbox_head_cfg))
        self.with_userinput_neck = False


        if userinput_backbone is not None:
            self.logger.info(str(userinput_backbone))
            self.userinput_backbone = build_backbone(userinput_backbone)
            if userinput_neck is not None:
                self.logger.info(str(userinput_neck))
                self.with_userinput_neck = True
                self.userinput_neck = build_neck(userinput_neck)

            if self.with_userinput_neck:
                if isinstance(self.neck, nn.Sequential):
                    for m in self.neck:
                        m.init_weights()
                else:
                    self.userinput_neck.init_weights()
        
        self.conv = nn.Conv2d(in_channels=self.rbbox_head_cfg['in_channels']*2, out_channels=self.rbbox_head_cfg['in_channels'], kernel_size=1)

    def extract_userinput_feat(self, userinput):
        x = self.userinput_backbone(userinput)
        if self.with_userinput_neck:
            x = self.userinput_neck(x)
        return x

    def process_userinput(self, img, **kwargs):
        img, user_input = img[:, :3, :, :], img[:, 3:, :, :]
        f = self.extract_feat(img)
        f_userinput = self.extract_userinput_feat(user_input)
        f = [self.conv(torch.cat((feat, user_input_feat), 1)) for feat, user_input_feat in zip(f, f_userinput)]
        return f


@DETECTORS.register_module
class RetinaNetOBBC3Det(RetinaNetOBBLateFusion):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head=None,
                 rbbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 userinput_backbone=None,
                 userinput_neck=None):
        super(RetinaNetOBBC3Det, self).__init__(backbone, neck, bbox_head, rbbox_head,
                                             train_cfg, test_cfg, pretrained, userinput_backbone, userinput_neck)

        self.num_classes = self.rbbox_head_cfg['num_classes'] - 1
        self.cor = CorrelationModule(self.num_classes)
        self.conv = nn.Conv2d(in_channels=self.rbbox_head_cfg['in_channels']*2+self.rbbox_head_cfg['num_classes']-1, 
                                out_channels=self.rbbox_head_cfg['in_channels'], kernel_size=1)

    def process_userinput(self, img, **kwargs):
        gt_userinputs, img_meta = kwargs['gt_userinputs'], kwargs['img_meta']
        img, user_input = img[:, :3, :, :], img[:, 3:, :, :]
        f = self.extract_feat(img)
        f_wh = [(feat.shape[2], feat.shape[3]) for feat in f]
        f_userinput = [F.interpolate(user_input, size=wh) for wh in f_wh]
        f = self.cor(f, f_userinput, gt_userinputs, img_metas)

        f_userinput = self.extract_userinput_feat(user_input)
        f = [self.conv(torch.cat((feat, user_input_feat), 1)) for feat, user_input_feat in zip(f, f_userinput)]
        return f
