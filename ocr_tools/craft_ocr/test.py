"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import sys
import os
import time
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from PIL import Image

import cv2
from skimage import io
import numpy as np
import pandas as pd
import ocr_tools.craft_ocr.craft_utils as craft_utils
import ocr_tools.craft_ocr.imgproc as imgproc
import ocr_tools.craft_ocr.file_utils as file_utils
import json
import zipfile
from collections import OrderedDict
from mask_rcnn.inference import Mask_RCNN_detector
from ocr_tools.craft_ocr.craft import CRAFT
from ocr_tools.craft_ocr.deeptext.model import Model
from ocr_tools.craft_ocr.deeptext.utils import CTCLabelConverter, AttnLabelConverter
from utils.barcode import read_barcode

from collections import defaultdict as dd

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")


def centroids(boxes):
    return [((box[0][0]+box[2][0])/2, (box[0][1]+box[2][1])/2) for box in boxes]


def sort_boxes(boxes):

    centers = centroids(boxes)
    arrs = sorted(boxes, key = lambda x: x[0][1])
    sorted_boxes = []
    i = 0

    if len(arrs) > 1:
        for idx in range(1, len(arrs)):

            if not(arrs[idx][0][1] - 3 <= centers[idx - 1][1] <= arrs[idx][2][1] + 3) :
                if idx < len(arrs) - 1:
                    sorted_boxes.extend(sorted(arrs[i: idx], key=lambda x: x[0][0]))
                    i = idx
                    continue
                if idx == len(arrs) - 1 and idx - 1 == i:
                    sorted_boxes.extend(sorted(arrs[idx - 1: ], key = lambda x: x[0][0]))
                    break
                else:
                    sorted_boxes.extend(sorted(arrs[i:], key=lambda x: x[0][0]))
                    break

    else:
        sorted_boxes.extend(boxes)
    return sorted_boxes


def order_hash_table(centers, chap_coord, polys):
    chap_coord = np.array([np.array(box).astype(np.int32).reshape((-1)) for box in chap_coord])
    groups = dd(list)
    # groups = []
    visited = set()
    for i, chapter in enumerate(chap_coord):
        left_x, left_y, right_x, right_y = chapter[0], chapter[1], chapter[4], chapter[5]

        for j, center in enumerate(centers):

            x, y = center[0], center[1]

            if (left_x - 8 < x < right_x + 8) and (left_y - 8 < y < right_y + 8):
                visited.add(j)
                groups[i].append(polys[j])

        groups[i] = sorted(groups[i], key = lambda x: x[0][0])
    return groups


def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None, args=None):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, args.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=args.mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    if args.show_time : print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return polys



def start_craft(args, ROOT):

    # load net
    net = CRAFT()     # initialize

    mask_rcnn = Mask_RCNN_detector(args.device, ROOT)

    """ model configuration """
    if 'CTC' in args.Prediction:
        converter = CTCLabelConverter(args.character)
    else:
        converter = AttnLabelConverter(args.character)
    args.num_class = len(converter.character)

    if args.rgb:
        args.input_channel = 3
    ocr_model = Model(args)
    ocr_model = torch.nn.DataParallel(ocr_model).to(args.device)
    ocr_model.load_state_dict(torch.load(args.saved_model, map_location=args.device))



    print('Loading weights from checkpoint (' + args.trained_model + ')')
    net.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location=args.device)))

    if args.cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False

    net.eval()

    # LinkRefiner
    refine_net = None
    if args.refine:
        from ocr_tools.craft_ocr.refinenet import RefineNet
        refine_net = RefineNet()
        print('Loading weights of refiner from checkpoint (' + args.refiner_model + ')')
        if args.cuda:
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model)))
            refine_net = refine_net.cuda()
            refine_net = torch.nn.DataParallel(refine_net)
        else:
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model, map_location='cpu')))

        refine_net.eval()
        args.poly = True

    t = time.time()

    # load data
    dataframe = pd.DataFrame(columns=['name', 'characters'])

    for dirpath, dirnames, filenames in os.walk(args.test_folder):
        if filenames:
            folder_name = os.path.split(dirpath)[-1]

            for k, filename in enumerate(filenames):
                image_path = os.path.join(dirpath, filename)
                print("Test image {:d}/{:d}: {:s}".format(k, len(filenames), image_path), end='\r')
                image_loaded = imgproc.loadImage(image_path)
                
                # if folder_name == 'size':
                #     cropped_images = mask_rcnn.detection(image_loaded, args.cuda)
                # else:
                #     cropped_images = [image_loaded]

                
                for image in [image_loaded]:
                    save_path = os.path.join(ROOT, 'imagenet/data/')
                    cv2.imwrite(save_path + filename,image)
                    barcode_result = read_barcode(image)
                    chapters_coord = test_net(net, image, args.text_threshold,
                                                    args.link_threshold, args.low_text,
                                                    args.cuda, args.poly, refine_net, args=args)

                    polys= test_net(net, image, args.text_threshold,
                                                        args.link_threshold, args.low_text,
                                                        args.cuda, args.poly, refine_net=None, args=args)

                    centers = centroids(polys)
                    chapters_coord = sort_boxes(chapters_coord)
                    groups = order_hash_table(centers, chapters_coord, polys)

                    text, name = file_utils.saveResult(image_path, image[:,:,::-1], groups, model=ocr_model, converter=converter, args=args)
                    print(text)
                    df = pd.DataFrame(np.array([[folder_name + '_' + str(name), ' '.join(text), barcode_result]]), columns=['name', 'characters', 'barcode'])
                    dataframe = dataframe.append(df, ignore_index=False)

    if not os.path.isdir('output'):
        os.mkdir('output')
    dataframe.to_csv(os.path.join('output', 'out.csv'), line_terminator='\n', index=False)

    print("elapsed time : {}s".format(time.time() - t))