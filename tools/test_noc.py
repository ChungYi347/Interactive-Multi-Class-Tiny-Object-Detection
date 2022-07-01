import argparse
import os
import os.path as osp
import shutil
import tempfile
import json
import numpy as np

import os, sys
sys.path.insert(0, os.getcwd())

import mmcv
import torch
import torch.distributed as dist
from mmcv.runner import load_checkpoint, get_dist_info
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel

from mmdet.apis import init_dist
from mmdet.core import results2json, coco_eval, OBBDetComp4, HBBDet2Comp4, HBBDet2Poly
from mmdet.datasets import build_dataloader, get_dataset
from mmdet.models import build_detector
import time

from DOTA_devkit.ResultMerge_multi_process import py_cpu_nms_poly_fast, py_cpu_nms
from utils_noc import save_result, read_gt, voc_eval, mergebase_parallel, mergebase_parallel_cell
from test import multi_gpu_test, single_gpu_test

classnames = None

def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        choices=['proposal', 'proposal_fast', 'bbox', 'segm', 'keypoints'],
        help='eval types')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument('--log_dir', help='log the inference speed')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--exp', type=int)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    exp = args.exp

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)

    if "UserInput" in cfg.data.test.type:
        cfg.data.test['noc'] = True

    dataset = get_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    classnames = dataset.CLASSES
    bbox_type = dataset.bbox_type

    result_dic = {}

    init_point = 1
    max_points = 21
    for point in range(init_point, max_points):
        dataset.drawer.iter = point
        print(f'{point} iteration')
        if not distributed:
            model = MMDataParallel(model, device_ids=[0])
            outputs = single_gpu_test(model, data_loader, args.show, args.log_dir)
        else:
            model = MMDistributedDataParallel(model.cuda(), device_ids=[torch.cuda.current_device()])
            outputs = multi_gpu_test(model, data_loader, args.tmpdir)

        rank, _ = get_dist_info()
        data_path = osp.dirname(args.checkpoint)

        if exp is None:
            noc_path = f'{data_path}/noc_{point}'
        else:
            noc_path = f'{data_path}/exp_{exp}/noc_{point}'
        os.makedirs(noc_path, exist_ok=True)

        if 'Userinput' in cfg.dataset_type:
            userinput_path = osp.join(noc_path, f'{rank}.json')
            json.dump(dict(dataset.drawer.points), open(userinput_path, 'w'))

        if args.out and rank == 0:
            print('\nwriting results to {}'.format(args.out))
            mmcv.dump(outputs, args.out)
            # mmcv.dump(cor_maps, args.out.replace('.pkl', f'_cor_maps_{point}.pkl'))
            eval_types = args.eval
            if eval_types:
                print('Starting evaluate {}'.format(' and '.join(eval_types)))
                if eval_types == ['proposal_fast']:
                    result_file = args.out
                    coco_eval(result_file, eval_types, dataset.coco)
                else:
                    if not isinstance(outputs[0], dict):
                        result_file = args.out + f'_{point}.json'
                        results2json(dataset, outputs, result_file)

                        if bbox_type == 'HBB':
                            hbb_results_dict = HBBDet2Poly(dataset, outputs)
                            current_thresh = 0.1
                            outputs = mergebase_parallel_cell(hbb_results_dict, py_cpu_nms_poly_fast, current_thresh)
                        elif bbox_type == 'OBB':
                            obb_results_dict = OBBDetComp4(dataset, outputs)
                            current_thresh = 0.1
                            outputs = mergebase_parallel(obb_results_dict, py_cpu_nms_poly_fast, current_thresh)
                        
                        if point == init_point:
                            recs = read_gt(cfg.data.test.ann_file.replace('.json', '_nms.txt'), \
                                    osp.dirname(cfg.data.test.ann_file).replace('_1024', '').replace('1024', '') + r'/labelTxt/{:s}.txt')

                        classaps, tps, fps = [], [], []
                        map, mtps, mfps = 0, 0, 0
                        for classname in classnames:
                            print('classname:', classname)
                            rec, prec, ap, fp, tp = voc_eval(outputs[classname],
                                recs,
                                classname,
                                ovthresh=0.5,
                                use_07_metric=True)
                            map = map + ap
                            mfps += fp
                            mtps += tp
                            print('ap: ', ap)
                            classaps.append(ap)
                            tps.append(tp)
                            fps.append(fp)

                        map = map/len(classnames)
                        print('map:', map)
                        classaps.append(map)
                        mtps /= len(classnames)
                        mfps /= len(classnames)
                        tps.append(mtps)
                        fps.append(mfps)
                        classaps = 100*np.array(classaps)
                        print('classaps: ', classaps)

                        result_dic[f'point_{point}'] = {
                            'map': list(classaps),
                            'tp': list(tps),
                            'fp': list(fps)
                        } 
                        with open(osp.join(osp.dirname(args.out), 'result.json'), 'w') as f:
                            json.dump(result_dic, f)
                        
                    else:
                        for name in outputs[0]:
                            print('\nEvaluating {}'.format(name))
                            outputs_ = [out[name] for out in outputs]
                            result_file = args.out + '.{}.json'.format(name)
                            results2json(dataset, outputs_, result_file)
                            coco_eval(result_file, eval_types, dataset)

    
if __name__ == '__main__':
    main()
