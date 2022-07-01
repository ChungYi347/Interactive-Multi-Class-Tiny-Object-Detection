import numpy as np
from pycocotools.coco import COCO

from .DOTA2 import DOTA2Dataset_v2
from .drawer.userinput import UserinputDrawer
from mmcv.parallel import DataContainer as DC
from .utils import to_tensor

class DOTA2Dataset_v2_Tiny(DOTA2Dataset_v2):
    def __init__(self, *args, **kwargs):
        super(DOTA2Dataset_v2_Tiny, self).__init__(*args, **kwargs)
        self.bbox_type = 'OBB'

    CLASSES = ('plane',  'bridge', 'small-vehicle', 'large-vehicle',
                'ship', 'storage-tank','swimming-pool', 'helicopter', )

class DOTA2Dataset_v2_Tiny_UserInput(DOTA2Dataset_v2_Tiny):
    def __init__(self, *args, **kwargs):
        mark_size = kwargs.pop('mark_size')
        noc = kwargs.pop('noc', False) 
        super(DOTA2Dataset_v2_Tiny_UserInput, self).__init__(*args, **kwargs)
        self.drawer = UserinputDrawer(mark_size=mark_size, classes=self.CLASSES, noc=noc)
        print(self.CLASSES)

    def prepare_train_img(self, idx):
        try:
            data = super().prepare_train_img(idx)
            data = self.drawer.prepare_user_input(data, idx)
        except:
            data = None
        return data

    def _prepare_test_img(self, idx):
        data = super().prepare_test_img(idx)
        assert len(data['img']) == 1, 'The input image should be 1.'

        ann = self.get_ann_info(idx)
        gt_bboxes = ann['bboxes']
        gt_labels = ann['labels']

        data['gt_labels'] = DC(to_tensor(gt_labels))
        data['gt_bboxes'] = DC(to_tensor(gt_bboxes))

        if self.drawer.noc:
            data = self.drawer.prepare_noc(data, idx)
        else:
            data = self.drawer.prepare_user_input(data, idx)

        data.pop('gt_labels')
        data.pop('gt_bboxes')
        return data

    def prepare_test_img(self, idx):
        # try:
            data = self._prepare_test_img(idx)
        # except:
            # print(idx)
            # data = None
            return data