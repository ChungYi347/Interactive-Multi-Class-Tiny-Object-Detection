"""
This script is to filter out 10 classes of large objects
and get only 8 classes of tiny objects from 18 classes from DOTA2.
"""

import os
import os.path as osp
import argparse
import shutil
from collections import Counter
import json

CLASSES = ('plane', 'baseball-diamond',
            'bridge', 'ground-track-field',
            'small-vehicle', 'large-vehicle',
            'ship', 'tennis-court',
            'basketball-court', 'storage-tank',
            'soccer-ball-field', 'roundabout',
            'harbor', 'swimming-pool',
            'helicopter', 'container-crane',
            'airport', 'helipad')

tiny_objects = ['plane', 'bridge', 'small-vehicle', 'large-vehicle', 
                'ship', 'storage-tank', 'swimming-pool', 'helicopter']

def make_tiny_objects_dataset(datapath):
    for data_type in ['train1024', 'val1024', 'test1024']:
        data = json.load(open(osp.join(datapath, data_type, f'DOTA2_{data_type}.json')))

        convered_label_info = {
            0: 1,  2: 2, 4: 3, 5: 4, 
            6: 5, 9: 6, 13: 7, 14: 8, 
        }

        # image & anno
        anno = [i for i in data['annotations'] if CLASSES[i['category_id']-1] in tiny_objects]
        anno = []
        for i in data['annotations']:
            if CLASSES[i['category_id']-1] in tiny_objects:
                i.update({'category_id': convered_label_info[i['category_id']-1]})
                anno.append(i)

        cate = []
        for i in data['categories']:
            if i['id']-1 in convered_label_info.keys():
                i.update({'id':convered_label_info[i['id']-1]})
                cate.append(i)

        data.update({'images': data['images'], 'annotations': anno, 'categories': cate})
        f = open(osp.join(datapath, data_type, f'DOTA2_{data_type}_tiny.json'), 'w')
        json.dump(data, f)

    for data_type in ['val1024', 'test1024']:
        # splited image name
        data = json.load(open(osp.join(datapath, data_type, f'DOTA2_{data_type}_tiny.json')))
        f = open(osp.join(datapath, data_type, f'DOTA2_{data_type}_tiny.txt'), 'w')
        f.write('\n'.join(list(set([i['file_name'] for i in data['images']]))))

        # total image name
        data = json.load(open(osp.join(datapath, data_type, f'DOTA2_{data_type}_tiny.json')))
        f = open(osp.join(datapath, data_type, f'DOTA2_{data_type}_tiny_nms.txt'), 'w')
        f.write('\n'.join(list(set([i['file_name'].split('__')[0] for i in data['images']]))))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='split dota1')
    parser.add_argument('--datapath', default='/path/to/dataset')
    args = parser.parse_args()
    datapath = args.datapath
    make_tiny_objects_dataset(datapath)
