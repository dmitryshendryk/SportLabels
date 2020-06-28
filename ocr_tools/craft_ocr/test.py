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

import pytesseract
from pytesseract import Output


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

    return boxes, polys, ret_score_text



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
        from refinenet import RefineNet
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
            k=0
            for filename in filenames:
                k += 1
                image_path = os.path.join(dirpath, filename)
                print("Test image {:d}/{:d}: {:s}".format(k, len(filenames), image_path), end='\r')
                image = imgproc.loadImage(image_path)

                cropped_images = mask_rcnn.detection(image, args.cuda)

                
                for image in cropped_images:
                    print(os.path.join(ROOT,'imagenet/data/' + filename))
                    cv2.imwrite(os.path.join(ROOT,'imagenet/data/' + filename),image)
                    barcode_result = read_barcode(image)
                    bboxes, polys, score_text = test_net(net, image, args.text_threshold,
                                                        args.link_threshold, args.low_text,
                                                        args.cuda, args.poly, refine_net, args=args)
                    text, name = file_utils.saveResult(image_path, image[:,:,::-1], polys, model=ocr_model, converter=converter, args=args)
                    tesseract_text = pytesseract.image_to_string(image, output_type=Output.DICT)
                    print(text)
                    print(str(tesseract_text['text']))
                    df = pd.DataFrame(np.array([[folder_name + '_' + str(name), text, str(tesseract_text['text']), barcode_result]]), columns=['name', 'characters', 'teseract_characters', 'barcode'])
                    dataframe = dataframe.append(df, ignore_index=False)

    if not os.path.isdir('output'):
        os.mkdir('output')
    dataframe.to_csv(os.path.join('output', 'out.csv'), index=False)

    print("elapsed time : {}s".format(time.time() - t))
