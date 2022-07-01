import os
import os.path as osp

import mmcv
import numpy as np
import torch
import torch.distributed as dist
from mmcv.runner import Hook, obj_from_dict
from mmcv.parallel import scatter, collate
from pycocotools.cocoeval import COCOeval
from torch.utils.data import Dataset

from .coco_utils import results2json, fast_eval_recall
from .mean_ap import eval_map
from mmdet import datasets
from mmdet.core.evaluation.dota_utils import OBBDetComp4, HBBDet2Poly
from tools.utils_noc import read_gt, voc_eval, mergebase_parallel, mergebase_parallel_cell
from DOTA_devkit.ResultMerge_multi_process import py_cpu_nms_poly_fast

import time
import warnings
from math import inf

class DistEvalHook(Hook):

    def __init__(self, dataset, interval=1, rule='less', key_indicator='loss'):
        if isinstance(dataset, Dataset):
            self.dataset = dataset
        elif isinstance(dataset, dict):
            self.dataset = obj_from_dict(dataset, datasets,
                                         {'test_mode': True})
        else:
            raise TypeError(
                'dataset must be a Dataset object or a dict, not {}'.format(
                    type(dataset)))
        self.interval = interval
        self.rule = rule
        self.rule_map = {'greater': lambda x, y: x > y, 'less': lambda x, y: x < y}
        self.init_value_map = {'greater': -inf, 'less': inf}
        self.compare_func = self.rule_map[rule]
        self.key_indicator = 'loss'

    def losses(self, runner, results_loss):
        raise NotImplementedError

    def before_run(self, runner):
        if runner.meta is None:
            warnings.warn('runner.meta is None. Creating a empty one.')
            runner.meta = dict()
        runner.meta.setdefault('hook_msgs', dict())
        runner.meta['hook_msgs'] = {
            'best_score': self.init_value_map[self.rule],
            'last_ckpt': '',
        }

    def save_best_checkpoint(self, runner, key_score):
        best_score = runner.meta['hook_msgs']['best_score']
        print(self.compare_func, key_score, best_score)
        if self.compare_func(key_score, best_score):
            best_score = key_score
            runner.meta['hook_msgs']['best_score'] = best_score
            last_ckpt = runner.meta['hook_msgs']['last_ckpt']
            if last_ckpt == '':
                last_ckpt = f'epoch_{runner.epoch}.pth'
            runner.meta['hook_msgs']['best_ckpt'] = last_ckpt
            mmcv.symlink(
                last_ckpt,
                osp.join(runner.work_dir, f'best.pth'))
            runner.logger.info(f'Now best checkpoint is epoch_{runner.epoch}.pth.'
                             f'Best {self.key_indicator} is {best_score:0.4f}')

    def after_val_epoch(self, runner):
        if runner.epoch != 0 and runner.epoch % self.interval == 0:
            runner.log_buffer.average()
            self.save_best_checkpoint(runner, runner.log_buffer.output[self.key_indicator])

            runner.model.eval()
            results = [None for _ in range(len(self.dataset))]
            if runner.rank == 0:
                prog_bar = mmcv.ProgressBar(len(self.dataset))
            for idx in range(runner.rank, len(self.dataset), runner.world_size):
                data = self.dataset[idx]
                
                data_gpu = scatter(
                    collate([data], samples_per_gpu=1),
                    [torch.cuda.current_device()])[0]

                # compute output
                with torch.no_grad():
                    result = runner.model(
                        return_loss=False, rescale=False, **data_gpu)
                    results[idx] = result
                    
                batch_size = runner.world_size
                if runner.rank == 0:
                    for _ in range(batch_size):
                        prog_bar.update()

            if runner.rank == 0:
                print('\n')
                dist.barrier()
                for i in range(1, runner.world_size):
                    tmp_file = osp.join(runner.work_dir, 'temp_{}.pkl'.format(i))
                    tmp_results = mmcv.load(tmp_file)
                    for idx in range(i, len(results), runner.world_size):
                        results[idx] = tmp_results[idx]
                    os.remove(tmp_file)
                self.evaluate(runner, results)
            else:
                tmp_file = osp.join(runner.work_dir,
                                    'temp_{}.pkl'.format(runner.rank))
                mmcv.dump(results, tmp_file)
                dist.barrier()
            dist.barrier()

    def after_train_epoch(self, runner):
        if runner._epoch == 0 or not self.every_n_epochs(runner, self.interval):
            return
        runner.model.eval()
        results = [None for _ in range(len(self.dataset))]
        if runner.rank == 0:
            prog_bar = mmcv.ProgressBar(len(self.dataset))
        for idx in range(runner.rank, len(self.dataset), runner.world_size):
            data = self.dataset[idx]
            data_gpu = scatter(
                collate([data], samples_per_gpu=1),
                [torch.cuda.current_device()])[0]

            # compute output
            with torch.no_grad():
                result = runner.model(
                    return_loss=False, rescale=True, **data_gpu)
            results[idx] = result

            batch_size = runner.world_size
            if runner.rank == 0:
                for _ in range(batch_size):
                    prog_bar.update()

        if runner.rank == 0:
            print('\n')
            dist.barrier()
            for i in range(1, runner.world_size):
                tmp_file = osp.join(runner.work_dir, 'temp_{}.pkl'.format(i))
                tmp_results = mmcv.load(tmp_file)
                for idx in range(i, len(results), runner.world_size):
                    results[idx] = tmp_results[idx]
                os.remove(tmp_file)
            self.evaluate(runner, results)
        else:
            tmp_file = osp.join(runner.work_dir,
                                'temp_{}.pkl'.format(runner.rank))
            mmcv.dump(results, tmp_file)
            dist.barrier()
        dist.barrier()

    def evaluate(self):
        raise NotImplementedError


class DistEvalmAPHook(DistEvalHook):

    def losses(self, runner, results_loss):
        losses = {
            'loss_rpn_cls': np.mean([torch.tensor(i['loss_rpn_cls']).cpu().numpy() for i in results_loss if type(i) != type(None)]),
            'loss_rpn_bbox': np.mean([torch.tensor(i['loss_rpn_bbox']).cpu().numpy() for i in results_loss if type(i) != type(None)]),
            'loss_cls': np.mean([torch.tensor(i['loss_cls']).cpu().numpy() for i in results_loss if type(i) != type(None)]),
            'acc': np.mean([torch.tensor(i['acc']).cpu().numpy() for i in results_loss if type(i) != type(None)]),
            'loss_bbox': np.mean([torch.tensor(i['loss_bbox']).cpu().numpy() for i in results_loss if type(i) != type(None)]),
        }
        for key in losses:
            val = float('{:.3f}'.format(losses[key]))
            key = '{}'.format(key)
            runner.log_buffer.output[key] = val
        runner.log_buffer.ready = True
        return losses

    def evaluate(self, runner, results):
        gt_bboxes = []
        gt_labels = []
        gt_ignore = [] if self.dataset.with_crowd else None
        for i in range(len(self.dataset)):
            ann = self.dataset.get_ann_info(i)
            bboxes = ann['bboxes']
            labels = ann['labels']
            if gt_ignore is not None:
                ignore = np.concatenate([
                    np.zeros(bboxes.shape[0], dtype=np.bool),
                    np.ones(ann['bboxes_ignore'].shape[0], dtype=np.bool)
                ])
                gt_ignore.append(ignore)
                bboxes = np.vstack([bboxes, ann['bboxes_ignore']])
                if 'labels_ignore' in ann.keys():
                    labels = np.concatenate([labels, ann['labels_ignore']])
                else:
                    labels = np.concatenate([labels, []])
            gt_bboxes.append(bboxes)
            gt_labels.append(labels)
        # If the dataset is VOC2007, then use 11 points mAP evaluation.
        if hasattr(self.dataset, 'year') and self.dataset.year == 2007:
            ds_name = 'voc07'
        else:
            ds_name = self.dataset.CLASSES
        mean_ap, eval_results = eval_map(
            results,
            gt_bboxes,
            gt_labels,
            gt_ignore=gt_ignore,
            scale_ranges=None,
            iou_thr=0.5,
            dataset=ds_name,
            print_summary=True)
        runner.log_buffer.output['mAP'] = mean_ap
        for c, eval_result in zip(ds_name, eval_results):
            runner.log_buffer.output[f'{c}_ap'] = float(eval_result['ap'])
        runner.log_buffer.ready = True


class CocoDistEvalRecallHook(DistEvalHook):

    def __init__(self,
                 dataset,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=np.arange(0.5, 0.96, 0.05)):
        super(CocoDistEvalRecallHook, self).__init__(dataset)
        self.proposal_nums = np.array(proposal_nums, dtype=np.int32)
        self.iou_thrs = np.array(iou_thrs, dtype=np.float32)

    def evaluate(self, runner, results):
        # the official coco evaluation is too slow, here we use our own
        # implementation instead, which may get slightly different results
        ar = fast_eval_recall(results, self.dataset.coco, self.proposal_nums,
                              self.iou_thrs)
        for i, num in enumerate(self.proposal_nums):
            runner.log_buffer.output['AR@{}'.format(num)] = ar[i]
        runner.log_buffer.ready = True


class CocoDistEvalmAPHook(DistEvalHook):

    def evaluate(self, runner, results):
        tmp_file = osp.join(runner.work_dir, 'temp_0.json')
        results2json(self.dataset, results, tmp_file)

        res_types = ['bbox',
                     'segm'] if runner.model.module.with_mask else ['bbox']
        cocoGt = self.dataset.coco
        cocoDt = cocoGt.loadRes(tmp_file)
        imgIds = cocoGt.getImgIds()
        for res_type in res_types:
            iou_type = res_type
            cocoEval = COCOeval(cocoGt, cocoDt, iou_type)
            cocoEval.params.imgIds = imgIds
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            metrics = ['mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l']
            for i in range(len(metrics)):
                key = '{}_{}'.format(res_type, metrics[i])
                val = float('{:.3f}'.format(cocoEval.stats[i]))
                runner.log_buffer.output[key] = val
            runner.log_buffer.output['{}_mAP_copypaste'.format(res_type)] = (
                '{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
                '{ap[4]:.3f} {ap[5]:.3f}').format(ap=cocoEval.stats[:6])
        runner.log_buffer.ready = True
        os.remove(tmp_file)

class PascalDistEvalmAPHook(DistEvalHook):
    def __init__(self, dataset, interval=1):
        super(PascalDistEvalmAPHook, self).__init__(dataset, interval)
        self.dataset_cfg = dataset

    def losses(self, runner, results_loss):
        losses = {
            'loss_rpn_cls': np.mean([torch.tensor(i['loss_rpn_cls']).cpu().numpy() for i in results_loss if type(i) != type(None)]),
            'loss_rpn_bbox': np.mean([torch.tensor(i['loss_rpn_bbox']).cpu().numpy() for i in results_loss if type(i) != type(None)]),
            'loss_cls': np.mean([torch.tensor(i['loss_cls']).cpu().numpy() for i in results_loss if type(i) != type(None)]),
            'acc': np.mean([torch.tensor(i['acc']).cpu().numpy() for i in results_loss if type(i) != type(None)]),
            'loss_bbox': np.mean([torch.tensor(i['loss_bbox']).cpu().numpy() for i in results_loss if type(i) != type(None)]),
        }
        for key in losses:
            val = float('{:.3f}'.format(losses[key]))
            key = '{}'.format(key)
            runner.log_buffer.output[key] = val
        runner.log_buffer.ready = True
        return losses

    def evaluate(self, runner, outputs):
        classnames = self.dataset.CLASSES
        bbox_type = self.dataset.bbox_type

        if bbox_type == 'HBB':
            hbb_results_dict = HBBDet2Poly(self.dataset, outputs)
            current_thresh = 0.1
            outputs = mergebase_parallel_cell(hbb_results_dict, py_cpu_nms_poly_fast, current_thresh)
        elif bbox_type == 'OBB':
            obb_results_dict = OBBDetComp4(self.dataset, outputs)
            current_thresh = 0.1
            outputs = mergebase_parallel(obb_results_dict, py_cpu_nms_poly_fast, current_thresh)
        
        recs = read_gt(self.dataset_cfg.ann_file.replace('.json', '_nms.txt'), \
                osp.dirname(self.dataset_cfg.ann_file).replace('_1024', '').replace('1024', '') + r'/labelTxt/{:s}.txt')

        # Calculate ap per class
        classaps = []
        for classname in classnames:
            rec, prec, ap, fp, tp = voc_eval(outputs[classname],
                recs,
                classname,
                ovthresh=0.5,
                use_07_metric=True)
            classaps.append(ap)

            key = '{}_{}'.format(classname, 'ap')
            val = float('{:.3f}'.format(ap))
            runner.log_buffer.output[key] = val

        # Calculate map
        map = sum(classaps)/len(classnames)
        classaps.append(map)
        classaps = 100*np.array(classaps)

        print(runner.mode)

        key = '{}'.format('map')
        val = float('{:.3f}'.format(map))
        runner.log_buffer.output[key] = val

        runner.log_buffer.ready = True
        # os.remove(tmp_file)