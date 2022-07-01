import torch
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

from mmdet.core import delta2dbbox, accuracy, RotBox2Polys, hbb2obb_v2 

class UEL:
    def __init__(self, mark_size, num_classes, target_means, target_stds, loss_userinput, reg_class_agnostic, obb=False):
        self.mark_size = mark_size
        self.loss_userinput = loss_userinput
        self.target_means = target_means
        self.target_stds = target_stds
        self.num_classes = num_classes
        self.reg_class_agnostic = reg_class_agnostic
        self.obb = obb

    def get_pos_inds_per_img(self, pos_inds, num_img):
        max_num_obj_per_img = pos_inds.size(0) // num_img
        pos_inds_per_img = pos_inds.reshape(-1, max_num_obj_per_img).nonzero()
        pos_inds_per_img = np.cumsum([len(pos_inds_per_img[pos_inds_per_img[:, 0] == img_idx][:, 1]) for img_idx in range(num_img)])
        return pos_inds_per_img

    def wh2xy(self, bboxes):
        """
        :param bboxes: (x_ctr, y_ctr, w, h) (n, 4)
        :return: out_bboxes: (xmin, ymin, xmax, ymax) (n, 4)
        """
        xmins = bboxes[..., 0] - (bboxes[..., 2] - 1) / 2.0
        ymins = bboxes[..., 1] - (bboxes[..., 3] - 1) / 2.0
        xmaxs = bboxes[..., 0] + (bboxes[..., 2] - 1) / 2.0
        ymaxs = bboxes[..., 1] + (bboxes[..., 3] - 1) / 2.0

        xmins = np.expand_dims(xmins, axis=1)
        ymins = np.expand_dims(ymins, axis=1)
        xmaxs = np.expand_dims(xmaxs, axis=1)
        ymaxs = np.expand_dims(ymaxs, axis=1)

        return np.hstack((xmins, ymins, xmaxs, ymaxs))

    def xy2poly(self, bboxes):
        return np.array([[bbox[0], bbox[1], bbox[0], bbox[3], bbox[2], bbox[3], bbox[2], bbox[1]] for bbox in bboxes])
        
    def compute_iou(self, gt_userinputs, det_bboxes, gt_userinput_labels, gt_userinput_bboxes, obb):
        if obb:
            det_bboxes = det_bboxes[:, :8].astype(np.double)
        else:
            det_bboxes = self.xy2poly(det_bboxes)

        gt_userinputs = gt_userinputs[gt_userinputs != -1].cpu().numpy().astype(np.int)
        gt_userinput_bboxes = gt_userinput_bboxes[:len(gt_userinputs)].cpu().numpy().astype(np.double)
        gt_userinput_labels = gt_userinput_labels[:len(gt_userinputs)].cpu().numpy().astype(np.int)

        uh_score_idx = []

        # Get the size of user inputs 
        gb_wh = np.array([self.mark_size * 2, self.mark_size * 2])
        gb_wh = np.tile(gb_wh, (gt_userinput_bboxes.shape[0],1))
        if len(gt_userinput_bboxes) == 0:
            return np.array([]), np.array([])
        gt_userinput_bboxes = self.xy2poly(self.wh2xy(np.append(gt_userinput_bboxes, gb_wh, axis=1)))

        # Merge all the same class user inputs for fast computation
        userinput_poly = []
        for uidx in np.unique(gt_userinput_labels):
            userinput_poly.append([uidx, unary_union(MultiPolygon([Polygon(ubbox.reshape(-1, 2)) for ubbox in gt_userinput_bboxes[gt_userinput_labels == uidx]]))])

        # Find intersection - Equation 4
        for uidx, ubbox in userinput_poly:
            for m, dbbox in enumerate(det_bboxes):
                dbbox = Polygon(dbbox.reshape(-1,2))
                area = ubbox.intersection(dbbox).area
                if area > 0:
                    uh_score_idx.append([uidx, m])
                        
        uh_score_idx = np.array(uh_score_idx)
        if len(uh_score_idx) > 0:
            return uh_score_idx[:, 0], uh_score_idx[:, 1]
        else:
            return np.array([]), np.array([])

    def get_overlap_objs(self, num_img, pos_inds, pos_inds_per_img, labels, label_weights, bbox_pred, rois, cls_score, gt_userinputs, img_metas):
        userinput_cls_score, userinput_labels, userinput_labels_weight = [], [], []

        target_dim = 5
        if self.reg_class_agnostic:
            pos_bbox_pred = bbox_pred.view(
                bbox_pred.size(0), target_dim)[pos_inds.type(torch.bool)]
        else:
            pos_bbox_pred = bbox_pred.view(
                bbox_pred.size(0), -1,
                target_dim)[pos_inds.type(torch.bool),
                    labels[pos_inds.type(torch.bool)]]

        obbs = hbb2obb_v2(rois[:, 1:][pos_inds])
        for img_idx in range(num_img):
            current_pos = pos_inds_per_img[img_idx]
            prev_pos = 0 if img_idx == 0 else pos_inds_per_img[img_idx-1]

            det_bboxes = delta2dbbox(obbs[prev_pos:current_pos], pos_bbox_pred[prev_pos:current_pos], \
                                        self.target_means, self.target_stds, img_metas[0]['img_shape'])
            det_bboxes = RotBox2Polys(det_bboxes.detach().cpu())
            det_scores = cls_score[pos_inds][prev_pos:current_pos]

            if len(det_bboxes) == 0:
                continue

            # 0: userinput label, 1: bbox, 2: index
            gt_userinput_per_img, gt_bbox_per_img, gt_label_per_img = gt_userinputs[0][img_idx].to(torch.long), gt_userinputs[1][img_idx], gt_userinputs[2][img_idx]
            userinput_ids, score_ids = self.compute_iou(gt_userinput_per_img, det_bboxes, gt_label_per_img, gt_bbox_per_img, self.obb)

            userinput_labels.append(labels[pos_inds][prev_pos:current_pos][score_ids])
            userinput_labels_weight.append(label_weights[pos_inds][prev_pos:current_pos][score_ids])
            userinput_cls_score.append(cls_score[pos_inds][prev_pos:current_pos][score_ids])
        return userinput_labels, userinput_labels_weight, userinput_cls_score
    
    def loss(self, img_metas, labels, label_weights, bbox_pred, rois, cls_score, gt_userinputs):
        losses = dict()

        pos_inds = labels > 0
        num_img = len(img_metas)
        pos_inds_per_img = self.get_pos_inds_per_img(pos_inds, num_img)
        userinput_labels, userinput_labels_weight, userinput_cls_score = self.get_overlap_objs(num_img, pos_inds, 
                            pos_inds_per_img, labels, label_weights, bbox_pred, rois, cls_score, gt_userinputs, img_metas)

        if len(userinput_labels) != 0:
            userinput_labels = torch.cat(userinput_labels)
            userinput_labels_weight = torch.cat(userinput_labels_weight)
            userinput_cls_score = torch.cat(userinput_cls_score)

        if len(userinput_labels) != 0:
            losses['rbbox_loss_userinput'] = self.loss_userinput(
                userinput_cls_score, userinput_labels, userinput_labels_weight)
            losses['rbbox_acc_userinput'] = accuracy(userinput_cls_score, userinput_labels)
        else:
            losses['rbbox_loss_userinput'] = torch.tensor(0.0, requires_grad=False).float().to(cls_score.device) 
            losses['rbbox_acc_userinput'] = torch.tensor(0.0).float().to(cls_score.device) 
        return losses