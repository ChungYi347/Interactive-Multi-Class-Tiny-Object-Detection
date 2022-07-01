from __future__ import division

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init

from mmdet.models.builder import NECKS


@NECKS.register_module
class CorrelationModule(nn.Module):
    def __init__(self, num_classes):
        super(CorrelationModule, self).__init__()
        self.num_classes = num_classes
	
    def gaussian_2d(self, shape, centre, sigma=1.0):
        """Generate heatmap with single 2D gaussian."""
        xs = np.arange(0.5, shape[1] + 0.5, step=1.0, dtype=np.float64)
        ys = np.expand_dims(np.arange(0.5, shape[0] + 0.5, step=1.0, dtype=np.float64), -1)
        alpha = -0.5 / (sigma**2)
        heatmap = np.exp(alpha * ((xs - centre[0])**2 + (ys - centre[1])**2))
        return heatmap

    def forward(self, x, x_userinput, gt_userinputs, img_meta):
        """
        This module, Class-wise Collated Correlation Module (C3), calculates correlation maps class-wise
        as shown in the section 4.1.3 (See https://arxiv.org/abs/2203.15266).
        1. Draw each individual user input
        2. Interpolate all user inputs for matching FPN
        3. Calculate templates - eq 1 in the paper
        4. Calculate correlation maps - eq 2 in the paper
        5. Collate all the cor maps
        Args:
            x: Image featrues (num_outs x batch size x channel x width x height) 
                - num_outs: NECK in configureation
                - batch size: imgs_per_gpu
            gt_userinputs: Information of user inputs (0: user input idx, 1: bboxes, 2: labels)
            img_meta: Image meta information
        Returns:
            Concatenated image features and correlation maps
        """
        # gt_userinputs: 0: userinput idx, 1: bboxes, 2: labels
        
        img_shape = img_meta[0]['img_shape'][:2]
        uh_shape = tuple(gt_userinputs[0].shape)
        uh_per_point = np.zeros((uh_shape + img_shape))

        # 1. Draw each individual user input
        for img_idx in range(len(gt_userinputs[0])):
            for n, (idx, center) in enumerate(zip(gt_userinputs[0][img_idx], gt_userinputs[1][img_idx])):
                if idx == torch.tensor(-1):
                    continue
                uh_per_point[img_idx][n] = torch.tensor(self.gaussian_2d(img_shape, center.cpu().numpy(), 3))

        # 2. Interpolate all user inputs for matching FPN
        uh_per_point = torch.from_numpy(uh_per_point).cuda()
        f_wh = [(feat.shape[2], feat.shape[3]) for feat in x]
        x_userinput = [F.interpolate(uh_per_point, size=wh) for wh in f_wh]

        # 3. Normalize user inputs and Conduct Global Sum Pooling after multipyling to make Temaplates - Equation (1) in paper
        normalized_userinput = []
        for feat, user_input_feat in zip(x, x_userinput):
            # Normalize
            normalized_userinput_feat = []
            for b in range(user_input_feat.shape[0]):
                for n in range(user_input_feat.shape[1]):
                    user_input_feat_wh = user_input_feat[b, n, :, :].shape
                    normalized_userinput_feat.append(user_input_feat[b, n, :, :].unsqueeze(0).unsqueeze(0) 
                                                    / (torch.sum(user_input_feat[b, n, :, :]) + 1e-8))
            user_input_feat = torch.cat(normalized_userinput_feat).view(feat.shape[0], -1, user_input_feat_wh[0], user_input_feat_wh[1])
            
            # Calculate Templates 
            normalized_userinput_feat = []
            for n in range(user_input_feat.shape[1]):
                # Templates
                normalized_userinput_feat.append(torch.sum((feat * user_input_feat[:, n, :, :].unsqueeze(1)), keepdim=True, dim=[2,3]).float())
            normalized_userinput.append(normalized_userinput_feat)

        # res x max_point x B x C x 1 x 1

        # 4. Make correlation maps by conducting Conv operation - Equation (2) in the paper
        combined = []
        # Resolution x Class x B x Channel x 1 x 1
        for feat, class_feat in zip(x, normalized_userinput):
            feats = []
            # Each resolution
            for n in range(feat.shape[0]):
                templates = []
                for userinput_feat in class_feat:
                    # Cor_map
                    template = F.conv2d(torch.unsqueeze(feat[n], 0), torch.unsqueeze(userinput_feat[n], 0)) 
                    templates.append(template)
                feats.append(torch.cat(templates, 1))
            combined.append(torch.cat(feats, 0))

        # 5. Collate the user inputs per class - Equation (3) in the paper
        cor_maps = [torch.zeros((x[n].shape[0], self.num_classes, x[n].shape[2], x[n].shape[3])).cuda() for n in range(len(x))]
        for res_idx in range(len(x)):
            f_shape = x[n].shape
            for img_idx in range(x[n].shape[0]):
                for class_idx in range(self.num_classes):
                    cor_map_per_img = combined[res_idx][img_idx]
                    uh_labels = gt_userinputs[2][img_idx]
                    
                    if torch.any(uh_labels == class_idx+1):
                        cor_map_per_img = cor_map_per_img[uh_labels == class_idx+1]
                        cor_maps[res_idx][img_idx][class_idx] = torch.max(cor_map_per_img, 0)[0].cuda()
        # res x B x max_point x norm_w x norm_h

        x = [torch.cat((feat, userinput_feat), 1) for feat, userinput_feat in zip(x, cor_maps)]
        return x