from .two_stage_rbbox import TwoStageDetectorRbbox
from ..registry import DETECTORS

from mmdet.models.builder import DETECTORS, build_backbone, build_neck
from mmdet.core import bbox2roi, bbox2result, build_assigner, build_sampler, dbbox2result
from mmdet.apis.env import get_root_logger

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.C3.correlation import CorrelationModule


@DETECTORS.register_module
class FasterRCNNOBB(TwoStageDetectorRbbox):

    def __init__(self,
                 backbone,
                 rpn_head,
                 bbox_roi_extractor,
                 bbox_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 shared_head=None,
                 pretrained=None):
        super(FasterRCNNOBB, self).__init__(
            backbone=backbone,
            neck=neck,
            shared_head=shared_head,
            rpn_head=rpn_head,
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)


@DETECTORS.register_module
class FasterRCNNOBBEarlyFusion(FasterRCNNOBB):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self,
                 backbone,
                 rpn_head,
                 bbox_roi_extractor,
                 bbox_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 shared_head=None,
                 pretrained=None):
        super(FasterRCNNOBBEarlyFusion, self).__init__(
            backbone=backbone,
            neck=neck,
            shared_head=shared_head,
            rpn_head=rpn_head,
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)
        self.bbox_head_cfg = bbox_head

    def forward_rpn(self, x, img_meta, losses, gt_bboxes, gt_bboxes_ignore, proposals=None):
        # RPN forward and loss
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            rpn_loss_inputs = rpn_outs + (gt_bboxes, img_meta,
                                          self.train_cfg.rpn)
            rpn_losses = self.rpn_head.loss(
                *rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            losses.update(rpn_losses)

            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            proposal_inputs = rpn_outs + (img_meta, proposal_cfg)
            proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
        else:
            proposal_list = proposals
        return proposal_list

    def forward_bbox_head(self, x, img, gt_bboxes, gt_labels, gt_bboxes_ignore, gt_masks, proposal_list, kwargs):
        losses = {}
        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            bbox_assigner = build_assigner(self.train_cfg.rcnn.assigner)
            bbox_sampler = build_sampler(
                self.train_cfg.rcnn.sampler, context=self)
            num_imgs = img.size(0)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        # bbox head forward and loss
        if self.with_bbox:
            rois = bbox2roi([res.bboxes for res in sampling_results])
            # TODO: a more flexible way to decide which feature maps to use
            bbox_feats = self.bbox_roi_extractor(
                x[:self.bbox_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                bbox_feats = self.shared_head(bbox_feats)
            cls_score, bbox_pred = self.bbox_head(bbox_feats)

            ## rbbox
            rbbox_targets = self.bbox_head.get_target(
                sampling_results, gt_masks, gt_labels, self.train_cfg.rcnn)

            if kwargs['UEL']:
                loss_bbox = self.bbox_head.loss(cls_score, bbox_pred, rois,
                                            *rbbox_targets, img_metas=kwargs['img_meta'],
                                            gt_userinputs=kwargs['gt_userinputs'])
            else:
                loss_bbox = self.bbox_head.loss(cls_score, bbox_pred,
                                                *rbbox_targets)
            losses.update(loss_bbox)
        return losses

    def check_UEL(self, kwargs):
        # kwargs.pop('img_meta')
        if 'Userinput' not in self.bbox_head_cfg['type']:
            kwargs.pop('gt_userinputs')
            kwargs.update({'UEL': False})
        else:
            kwargs.update({'UEL': True})
        return kwargs

    def process_userinput(self, img, **kwargs):
        x = self.extract_feat(img)
        return x

    def forward_train(self,
                      img,
                      img_meta,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C + Number of Classes, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        kwargs.update({'img_meta': img_meta})
        x = self.process_userinput(img, **kwargs)

        losses = dict()

        proposal_list = self.forward_rpn(x, img_meta, losses, gt_bboxes, gt_bboxes_ignore, proposals)

        kwargs = self.check_UEL(kwargs)
        roi_losses = self.forward_bbox_head(x, img, gt_bboxes, gt_labels, gt_bboxes_ignore, gt_masks, proposal_list, kwargs)
        losses.update(roi_losses)

        return losses

    
    def simple_test(self, img, img_meta, proposals=None, rescale=False, **kwargs):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        kwargs.update({'img_meta': img_meta})
        x = self.process_userinput(img, **kwargs)

        if proposals is None:
            proposal_list = self.simple_test_rpn(x, img_meta, self.test_cfg.rpn)
        else:
            proposal_list = proposals

        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_meta, proposal_list, self.test_cfg.rcnn, rescale=rescale)
        bbox_results = dbbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.simple_test_mask(
                x, img_meta, det_bboxes, det_labels, rescale=rescale)
            return bbox_results, segm_results

@DETECTORS.register_module
class FasterRCNNOBBLateFusion(FasterRCNNOBBEarlyFusion):

    def __init__(self,
                 backbone,
                 rpn_head,
                 bbox_roi_extractor,
                 bbox_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 shared_head=None,
                 pretrained=None,
                 userinput_backbone=None,
                 userinput_neck=None):
        super(FasterRCNNOBBLateFusion, self).__init__(
            backbone=backbone,
            neck=neck,
            shared_head=shared_head,
            rpn_head=rpn_head,
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)
        self.bbox_head_cfg = bbox_head
        self.rpn_head_cfg = rpn_head
        self.logger = get_root_logger()

        if userinput_backbone is not None:
            self.logger.info(str(userinput_backbone))
            self.userhint_backbone = build_backbone(userinput_backbone)
        if userinput_neck is not None:
            self.logger.info(str(userinput_neck))
            self.with_userinput_neck = True
            self.userhint_neck = build_neck(userinput_neck)
        self.conv = nn.Conv2d(in_channels=self.rpn_head_cfg['in_channels']*2, out_channels=self.rpn_head_cfg['in_channels'], kernel_size=1)

    def extract_userinput_feat(self, userinput):
        x = self.userhint_backbone(userinput)
        if self.with_userinput_neck:
            x = self.userhint_neck(x)
        return x

    def init_weights(self, pretrained=None):
        super(FasterRCNNOBBLateFusion, self).init_weights(pretrained)

    def process_userinput(self, img, **kwargs):
        img, user_input = img[:, :3, :, :], img[:, 3:, :, :]
        f = self.extract_feat(img)
        f_userinput = self.extract_userinput_feat(user_input)
        f = [self.conv(torch.cat((feat, user_input_feat), 1)) for feat, user_input_feat in zip(f, f_userinput)]
        return f


@DETECTORS.register_module
class FasterRCNNOBBC3Det(FasterRCNNOBBLateFusion):

    def __init__(self,
                 backbone,
                 rpn_head,
                 bbox_roi_extractor,
                 bbox_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 shared_head=None,
                 pretrained=None,
                 userinput_backbone=None,
                 userinput_neck=None):
        super(FasterRCNNOBBC3Det, self).__init__(
            backbone=backbone,
            neck=neck,
            shared_head=shared_head,
            rpn_head=rpn_head,
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            userinput_backbone=userinput_backbone,
            userinput_neck=userinput_neck)
        self.num_classes = self.bbox_head_cfg['num_classes'] - 1
        self.cor = CorrelationModule(self.num_classes)
        self.conv = nn.Conv2d(in_channels=self.rpn_head_cfg['in_channels']*2+self.num_classes, out_channels=self.rpn_head_cfg['in_channels'], kernel_size=1)

    def process_userinput(self, img, **kwargs):
        gt_userinputs, img_meta = kwargs['gt_userinputs'], kwargs['img_meta']
        img, user_input = img[:, :3, :, :], img[:, 3:, :, :]
        # Get img feature F_I
        f = self.extract_feat(img)
        # Get C3 feature F_C3
        f_wh = [(feat.shape[2], feat.shape[3]) for feat in f]
        f_userinput = [F.interpolate(user_input, size=wh) for wh in f_wh]
        f = self.cor(f, f_userinput, gt_userinputs, img_meta)
        # Get Late Fusion feature F_LF
        f_userinput = self.extract_userinput_feat(user_input)
        f = [self.conv(torch.cat((feat, user_input_feat), 1)) for feat, user_input_feat in zip(f, f_userinput)]
        return f