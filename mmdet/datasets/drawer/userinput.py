from multiprocessing import Manager
from mmcv.parallel import DataContainer as DC
import torch
import numpy as np

class UserinputDrawer:
    def __init__(self, *args, **kwargs):
        self.mark_size = kwargs['mark_size']
        self.CLASSES = kwargs['classes']
        self.noc = kwargs['noc'] 
        self.max_user_input = 20

        manager = Manager()
        self.points = manager.dict()
        self.iter = 0

    def init_points_list(self, data):
        points_list, userinput_bbox_list = np.empty((0, ), int), np.empty((0, 2), float)
        bbox_count = min(len(data['gt_bboxes'].data), len(data['gt_labels'].data), self.max_user_input)
        if bbox_count > 0:
            num_points = np.random.randint(bbox_count)
            points_list = np.random.permutation(len(data['gt_bboxes'].data))[:num_points]
        return points_list, userinput_bbox_list

    def fill_insufficient_userinput(self, data, points_list, userinput_bbox_list):
        """Fill insufficient userinput for fixing shape."""
        insufficient_userinput = self.max_user_input - len(points_list)
        userinput_label_list = data['gt_labels'].data[points_list]
        userinput_label_list = np.append(userinput_label_list, np.ones((insufficient_userinput)) * -1)

        userinput_bbox_list = np.append(userinput_bbox_list, np.ones((insufficient_userinput * 2)) * -1).reshape(-1, 2)
        points_list = np.append(points_list, np.ones((insufficient_userinput)) * -1)
        return (points_list, userinput_bbox_list, userinput_label_list)
    
    def draw_points(self, data, points_list, img_shape, user_inputs, userinput_bbox_list=None):
        for point_idx in points_list:
            bbox, label = data['gt_bboxes'].data[point_idx], data['gt_labels'].data[point_idx]
            user_input, center_point = self.draw_point(bbox, img_shape)
            user_inputs[label-1].append(user_input)
            if userinput_bbox_list is not None:
                userinput_bbox_list = np.append(userinput_bbox_list, center_point)

        if userinput_bbox_list is not None:
            return user_inputs, userinput_bbox_list
        else:
            return user_inputs

    def prepare_user_input(self, data, idx):
        user_inputs, img_shape = self.prepare_empty_input(data)
        points_list, userinput_bbox_list = self.init_points_list(data)
        user_inputs, userinput_bbox_list = self.draw_points(data, points_list, img_shape, user_inputs, userinput_bbox_list)
        
        user_inputs = torch.FloatTensor([np.amax(uh, axis=0) for uh in user_inputs])

        bboxes, labels = data['gt_bboxes'].data[points_list], data['gt_labels'].data[points_list]

        data['img'] = torch.cat((data['img'].data, user_inputs), 0)  \
                if isinstance(data['img'], DC) else [torch.cat((data['img'][0].data, user_inputs), 0)]

        data['gt_userinputs'] = self.fill_insufficient_userinput(data, points_list, userinput_bbox_list)
        return data
    
    def prepare_noc_point(self, data, user_inputs, idx, img_shape):
        bbox_count = min(len(data['gt_labels'].data), len(data['gt_bboxes'].data))
        if bbox_count > 0 and len(self.points[idx]) < bbox_count:
            bbox_idx = np.random.randint(bbox_count)
            while bbox_idx in self.points[idx] and len(self.points[idx]) < bbox_count:
                bbox_idx = np.random.randint(bbox_count)
            self.points[idx] += [bbox_idx]
        
        user_inputs = self.draw_points(data, self.points[idx], img_shape, user_inputs)
        return user_inputs
        
    def prepare_noc(self, data, idx):
        user_inputs, img_shape = self.prepare_empty_input(data)

        if idx not in self.points:
            self.points[idx] = []

        if self.iter != 0:
            user_inputs = self.prepare_noc_point(data, user_inputs, idx, img_shape)

            points_list = self.points[idx]
            bboxes = data['gt_bboxes'].data[points_list]

            if bboxes.shape[-1] == 4:
                userinput_bbox_list = [((bbox[2] + bbox[0])//2, (bbox[3] + bbox[1])//2) for bbox in bboxes]
            else:
                userinput_bbox_list = [((bbox[4] + bbox[0])//2, (bbox[5] + bbox[1])//2) for bbox in bboxes]
            data['gt_userinputs'] = self.fill_insufficient_userinput(data, points_list, userinput_bbox_list)
        else:
            data['gt_userinputs'] = (np.ones((self.max_user_input)) * -1, np.ones((self.max_user_input, 2)) * -1, np.ones((self.max_user_input)) * -1)
        user_inputs = torch.FloatTensor([np.amax(uh, axis=0) for uh in user_inputs])

        data['img'] = torch.cat((data['img'].data, user_inputs), 0) \
                if isinstance(data['img'], DC) else [torch.cat((data['img'][0].data, user_inputs), 0)]
        return data

    def prepare_empty_input(self, data):
        img_meta = data['img_meta']
        img_shape = img_meta.data['pad_shape'] \
                        if isinstance(img_meta, DC) else img_meta[0].data['pad_shape']
        user_inputs = [[np.zeros(img_shape[:2], dtype=np.float64)] for _ in range(len(self.CLASSES))]
        return user_inputs, img_shape

    def gaussian_2d(self, shape, centre, sigma=1.0):
        """Generate heatmap with single 2D gaussian."""
        xs = np.arange(0.5, shape[1] + 0.5, step=1.0, dtype=np.float64)
        ys = np.expand_dims(np.arange(0.5, shape[0] + 0.5, step=1.0, dtype=np.float64), -1)
        alpha = -0.5 / (sigma**2)
        heatmap = np.exp(alpha * ((xs - centre[0])**2 + (ys - centre[1])**2))
        return heatmap

    def draw_point(self, bbox, img_shape):
        r = self.mark_size
        bbox = bbox.to(torch.int)
        if len(bbox) == 4:
            cx, cy = (bbox[2] + bbox[0])//2, (bbox[3] + bbox[1])//2
        else:
            cx, cy = (bbox[4] + bbox[0])//2, (bbox[5] + bbox[1])//2
        user_input = self.gaussian_2d(img_shape, [cx.numpy(), cy.numpy()], self.mark_size)
        return user_input, [cx, cy]
    