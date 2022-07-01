import numpy as np
import re
import json

from multiprocessing import Pool
from functools import partial
from DOTA_devkit.ResultMerge_multi_process import nmsbynamedict, poly2origpoly
from DOTA_devkit import polyiou

def save_result(result_dic, result_file, eval_types, dataset, idx, cate=False):
    cocoEval = coco_eval(result_file, eval_types, dataset.coco, noc=True)
    mean_ap = cocoEval.stats[1].item()  # stats[0] records AP@[0.5:0.95]
    print('total: ', mean_ap)
    result_dic[f'total_{idx}'] = mean_ap
    if cate:
        for cat_id in dataset.cat_ids:
            cocoEval = coco_eval(result_file, eval_types, dataset.coco, noc=True, catIds=[cat_id])
            mean_ap = cocoEval.stats[1].item()
            result_dic[f'cat_id_{idx}_{cat_id}'] = mean_ap
            print(f'cat_id {cat_id}: ', mean_ap)

def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def parse_gt(filename):
    """
    :param filename: ground truth file to parse
    :return: all instances in a picture
    """
    objects = []
    with  open(filename, 'r') as f:
        while True:
            line = f.readline()
            if line:
                splitlines = line.strip().split(' ')
                object_struct = {}
                if (len(splitlines) < 9):
                    continue
                object_struct['name'] = splitlines[8]
                object_struct['difficult'] = 0
                object_struct['bbox'] = [float(splitlines[0]),
                                         float(splitlines[1]),
                                         float(splitlines[2]),
                                         float(splitlines[3]),
                                         float(splitlines[4]),
                                         float(splitlines[5]),
                                         float(splitlines[6]),
                                         float(splitlines[7])]
                object_struct['image'] = filename
                objects.append(object_struct)
            else:
                break
    return objects


def read_gt(imagesetfile, annopath):
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    recs = {}
    for i, imagename in enumerate(imagenames):
        recs[imagename] = parse_gt(annopath.format(imagename))

    return recs

def voc_eval(outputs,
             recs,
             classname,
             ovthresh=0.5,
             use_07_metric=False):

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in recs:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                    'difficult': difficult,
                                    'det': det}

    # import pdb; pdb.set_trace()
    image_ids = [x[0] for x in outputs]
    confidence = np.array([float(x[1]) for x in outputs])
    BB = np.array([[float(z) for z in x[2:]] for x in outputs])

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)

    ## note the usage only in numpy not for list
    if len(confidence) == 0 or len(BB) == 0 or len(outputs) == 0:
         return 0, 0, 0, 0, 0

    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d].split('.')[0]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        ## compute det bb with each BBGT
        if BBGT.size > 0:
            # compute overlaps
            # intersection

            # 1. calculate the overlaps between hbbs, if the iou between hbbs are 0, the iou between obbs are 0, too.
            # pdb.set_trace()
            BBGT_xmin =  np.min(BBGT[:, 0::2], axis=1)
            BBGT_ymin = np.min(BBGT[:, 1::2], axis=1)
            BBGT_xmax = np.max(BBGT[:, 0::2], axis=1)
            BBGT_ymax = np.max(BBGT[:, 1::2], axis=1)
            bb_xmin = np.min(bb[0::2])
            bb_ymin = np.min(bb[1::2])
            bb_xmax = np.max(bb[0::2])
            bb_ymax = np.max(bb[1::2])

            ixmin = np.maximum(BBGT_xmin, bb_xmin)
            iymin = np.maximum(BBGT_ymin, bb_ymin)
            ixmax = np.minimum(BBGT_xmax, bb_xmax)
            iymax = np.minimum(BBGT_ymax, bb_ymax)
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb_xmax - bb_xmin + 1.) * (bb_ymax - bb_ymin + 1.) +
                   (BBGT_xmax - BBGT_xmin + 1.) *
                   (BBGT_ymax - BBGT_ymin + 1.) - inters)

            overlaps = inters / uni

            BBGT_keep_mask = overlaps > 0
            BBGT_keep = BBGT[BBGT_keep_mask, :]
            BBGT_keep_index = np.where(overlaps > 0)[0]
            # pdb.set_trace()
            def calcoverlaps(BBGT_keep, bb):
                overlaps = []
                for index, GT in enumerate(BBGT_keep):

                    overlap = polyiou.iou_poly(polyiou.VectorDouble(BBGT_keep[index]), polyiou.VectorDouble(bb))
                    overlaps.append(overlap)
                return overlaps
            if len(BBGT_keep) > 0:
                overlaps = calcoverlaps(BBGT_keep, bb)

                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)
                # pdb.set_trace()
                jmax = BBGT_keep_index[jmax]
        if ovmax > ovthresh:
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)

    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap, fp[-1], tp[-1]


def mergesinglecell(nms, nms_thresh, outputs, class_name):
    nameboxdict = {}
    for splitline in outputs[class_name]:
        splitline = splitline.split(' ')
        oriname = splitline[0]

        confidence = splitline[1]
        det = list(map(float, splitline[2:]))
        if (oriname not in nameboxdict):
            nameboxdict[oriname] = []
        nameboxdict[oriname].append(det)
    nameboxnmsdict = nmsbynamedict(nameboxdict, nms, nms_thresh)

    result = []
    for imgname in nameboxnmsdict:
        for det in nameboxnmsdict[imgname]:
            confidence = det[-1]
            bbox = det[0:-1]
            outline = imgname + ' ' + str(confidence) + ' ' + ' '.join(map(str, bbox))
            result.append([imgname, str(confidence)] + list(map(str, bbox)))
    return {'cls': class_name, 'dets': result}

def mergesingle(nms, nms_thresh, outputs, class_name):
    nameboxdict = {}
    for splitline in outputs[class_name]:
        splitline = splitline.split(' ')
        subname = splitline[0]
        splitname = subname.split('__')
        oriname = splitname[0]
        pattern1 = re.compile(r'__\d+___\d+')
        x_y = re.findall(pattern1, subname)
        x_y_2 = re.findall(r'\d+', x_y[0])
        x, y = int(x_y_2[0]), int(x_y_2[1])

        pattern2 = re.compile(r'__([\d+\.]+)__\d+___')

        rate = re.findall(pattern2, subname)[0]

        confidence = splitline[1]
        poly = list(map(float, splitline[2:]))
        origpoly = poly2origpoly(poly, x, y, rate)
        det = origpoly
        det.append(confidence)
        det = list(map(float, det))
        if (oriname not in nameboxdict):
            nameboxdict[oriname] = []
        nameboxdict[oriname].append(det)
    nameboxnmsdict = nmsbynamedict(nameboxdict, nms, nms_thresh)

    result = []
    for imgname in nameboxnmsdict:
        for det in nameboxnmsdict[imgname]:
            confidence = det[-1]
            bbox = det[0:-1]
            outline = imgname + ' ' + str(confidence) + ' ' + ' '.join(map(str, bbox))
            result.append([imgname, str(confidence)] + list(map(str, bbox)))
    return {'cls': class_name, 'dets': result}


def mergebase_parallel(outputs, nms, nms_thresh):
    pool = Pool(16)

    mergesingle_fn = partial(mergesingle, nms, nms_thresh, outputs)
    result = pool.map(mergesingle_fn, list(outputs.keys()))

    return {res['cls']: res['dets'] for res in result}

def mergebase_parallel_cell(outputs, nms, nms_thresh):
    pool = Pool(16)

    mergesingle_fn = partial(mergesinglecell, nms, nms_thresh, outputs)
    result = pool.map(mergesingle_fn, list(outputs.keys()))

    return {res['cls']: res['dets'] for res in result}


def vis_result(dataset):
    if dataset.noc_vis:
        for im_id in list(dataset.points.keys()):
            img_info = dataset.img_infos[im_id]
            img = Image.fromarray(mmcv.imread(osp.join(dataset.img_prefix, img_info['filename'])))

            colors = list(ImageColor.colormap.keys())

            ann = dataset.get_ann_info(im_id)
            gt_bboxes = ann['bboxes']
            gt_labels = ann['labels']

            draw = ImageDraw.Draw(img)
            
            for n, dets in enumerate(outputs[im_id]):
                if len(dets) == 0:
                    continue
                for det in dets:
                    # draw.rectangle(det[:4], ImageColor.colormap[colors[n+1]])
                    draw.rectangle(det[:4], outline=ImageColor.colormap['red'])

            for point in dataset.points[im_id]:
                bbox = gt_bboxes[point]
                cls = gt_labels[point]
                # draw.rectangle(bbox, ImageColor.colormap[colors[cls]])
                draw.rectangle(bbox, outline=ImageColor.colormap['blue'])

            img.save(f'noc_img/{im_id}_{dataset.iter}.png')

    with open(osp.join(osp.dirname(args.out), 'result.json'), 'w') as f:
        json.dump(result_dic, f)