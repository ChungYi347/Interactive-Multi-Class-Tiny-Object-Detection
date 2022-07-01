"""
The script will make original train and val folder to train_old and val_old. 
It will make and devide the original dataset to new train, val, and test dataset (70%, 10%, and 20% proportions). 
The detailed information is here (5.1. Datasets section in https://arxiv.org/abs/2203.15266)

The dataset will be changed like this:

Original DOTA-v2.0 | Splited DOTA-v2.0 | DOTA-Tiny
train              | train_old         | train (70%)
val                | val_old           | val (10%)
                   |                   | test (20%)
"""

import os
import os.path as osp
import argparse
import shutil

def make_new_folder(datapath, ids):
    for folder_name in ['train', 'val', 'test']:
        shutil.move(osp.join(datapath, folder_name), osp.join(datapath, folder_name+'_old'))
        os.mkdir(os.path.join(datapath, folder_name))
        for im_label in ['images', 'labelTxt']:
            os.mkdir(os.path.join(datapath, folder_name, im_label))
        move_files(datapath, folder_name, ids[folder_name])

def move_files(datapath, data_type, ids):
    for id in ids:
        shutil.copyfile(osp.join(id[0]+'_old', 'images', id[1]), osp.join(datapath, data_type, 'images', id[1]))
        shutil.copyfile(osp.join(id[0]+'_old', 'labelTxt', osp.splitext(id[1])[0]+'.txt'), osp.join(datapath, data_type, 'labelTxt', osp.splitext(id[1])[0]+'.txt'))

def split(datapath):
    dataset_dir = os.listdir(datapath)
    assert 'train_old' not in os.listdir(datapath) and \
           'val_old' not in os.listdir(datapath) and \
           'test_old' not in os.listdir(datapath), \
           'There are already splited folders.'
        
    train_path, val_path = osp.join(datapath, 'train'), osp.join(datapath, 'val')
    train_ids, val_ids = [(train_path, i) for i in os.listdir(osp.join(train_path, 'images'))], [(val_path, i) for i in os.listdir(osp.join(val_path, 'images'))]

    ids = train_ids + val_ids
    total_num = len(ids)
    train_num, val_num = int(0.7 * total_num), int(0.8 * total_num)

    ids = {
      'train': ids[:train_num], 
      'val': ids[train_num: val_num], 
      'test': ids[val_num:]
    }
    make_new_folder(datapath, ids)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='split dota2')
    parser.add_argument('--datapath', default='/path/to/dataset')
    args = parser.parse_args()
    datapath = args.datapath
    split(datapath)
