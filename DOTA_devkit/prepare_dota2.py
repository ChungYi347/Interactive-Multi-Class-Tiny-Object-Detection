"""
The script is to split large size images (e.g., 4096 x 2048) 
to small size images (e,g, 1024 x 1024) with gap.
"""

import utils as util
import os
import ImgSplit_multi_process
import SplitOnlyImage_multi_process
import shutil
from multiprocessing import Pool
from DOTA2COCO import DOTA2COCOTest, DOTA2COCOTrain
import argparse

wordname_16 = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 
                'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
                'basketball-court', 'storage-tank',  'soccer-ball-field', 
                'roundabout', 'harbor', 'swimming-pool', 'helicopter', 
                'container-crane', 'airport', 'helipad']

def parse_args():
    parser = argparse.ArgumentParser(description='prepare dota1')
    parser.add_argument('--srcpath', default=r'/path/to/source')
    parser.add_argument('--dstpath', default=r'/path/to/destination',
                        help='prepare data')
    args = parser.parse_args()

    return args

def single_copy(src_dst_tuple):
    shutil.copyfile(*src_dst_tuple)
def filecopy(srcpath, dstpath, num_process=32):
    pool = Pool(num_process)
    filelist = util.GetFileFromThisRootDir(srcpath)

    name_pairs = []
    for file in filelist:
        basename = os.path.basename(file.strip())
        dstname = os.path.join(dstpath, basename)
        name_tuple = (file, dstname)
        name_pairs.append(name_tuple)

    pool.map(single_copy, name_pairs)

def singel_move(src_dst_tuple):
    shutil.move(*src_dst_tuple)

def filemove(srcpath, dstpath, num_process=32):
    pool = Pool(num_process)
    filelist = util.GetFileFromThisRootDir(srcpath)

    name_pairs = []
    for file in filelist:
        basename = os.path.basename(file.strip())
        dstname = os.path.join(dstpath, basename)
        name_tuple = (file, dstname)
        name_pairs.append(name_tuple)

    pool.map(filemove, name_pairs)

def getnamelist(srcpath, dstfile):
    filelist = util.GetFileFromThisRootDir(srcpath)
    with open(dstfile, 'w') as f_out:
        for file in filelist:
            basename = util.mybasename(file)
            f_out.write(basename + '\n')

def prepare(srcpath, dstpath):
    """
    :param srcpath: train, val, test
          train --> trainval1024, val --> trainval1024, test --> test1024
    :return:
    """
    if not os.path.exists(os.path.join(dstpath)):
        os.mkdir(os.path.join(dstpath))
    if not os.path.exists(os.path.join(dstpath, 'test1024')):
        os.mkdir(os.path.join(dstpath, 'test1024'))
    if not os.path.exists(os.path.join(dstpath, 'train1024')):
        os.mkdir(os.path.join(dstpath, 'train1024'))
    if not os.path.exists(os.path.join(dstpath, 'val1024')):
        os.mkdir(os.path.join(dstpath, 'val1024'))

    split_train = ImgSplit_multi_process.splitbase(os.path.join(srcpath, 'train'),
                       os.path.join(dstpath, 'train1024'),
                      gap=200,
                      subsize=1024,
                      num_process=32
                      )
    split_train.splitdata(1)

    split_val = ImgSplit_multi_process.splitbase(os.path.join(srcpath, 'val'),
                       os.path.join(dstpath, 'val1024'),
                      gap=200,
                      subsize=1024,
                      num_process=32
                      )
    split_val.splitdata(1)

    split_test = ImgSplit_multi_process.splitbase(os.path.join(srcpath, 'test'),
                       os.path.join(dstpath, 'test1024'),
                      gap=200,
                      subsize=1024,
                      num_process=32
                      )
    split_test.splitdata(1)

    DOTA2COCOTrain(os.path.join(dstpath, 'train1024'), os.path.join(dstpath, 'train1024', 'DOTA2_train1024.json'), wordname_16, difficult='-1')
    DOTA2COCOTrain(os.path.join(dstpath, 'val1024'), os.path.join(dstpath, 'val1024', 'DOTA2_val1024.json'), wordname_16, difficult='-1')
    DOTA2COCOTrain(os.path.join(dstpath, 'test1024'), os.path.join(dstpath, 'test1024', 'DOTA2_test1024.json'), wordname_16)

if __name__ == '__main__':
    args = parse_args()
    srcpath = args.srcpath
    dstpath = args.dstpath
    prepare(srcpath, dstpath)