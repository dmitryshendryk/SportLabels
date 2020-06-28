# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import cv2
import ocr_tools.craft_ocr.imgproc
from copy import deepcopy
from PIL import Image
from operator import itemgetter
from ocr_tools.craft_ocr.PerspectiveTransform import four_point_transform
from ocr_tools.craft_ocr.deeptext.demo import demo


# borrowed from https://github.com/lengstrom/fast-style-transfer/blob/master/src/utils.py
def get_files(img_dir):
    imgs, masks, xmls = list_files(img_dir)
    return imgs, masks, xmls


def sort_boxes(boxes):
    boxes = np.array([np.array(box).astype(np.int32).reshape((-1)) for box in boxes])
    arrs = sorted(boxes, key=itemgetter(1))
    sorted_boxes = []

    while True:

        if not arrs:
            break

        boxxes = [arrs[0]]

        if len(arrs) > 1:
            for j in range(1, len(arrs)):
                above = boxxes[-1][1]
                below = (boxxes[-1][7] + above) / 2
                aim_cell = arrs[j][1]

                if above <= aim_cell <= below:
                    boxxes.append(arrs[j])

                else:
                    break

            boxxes = sorted(boxxes, key=itemgetter(0))
            sorted_boxes.extend(boxxes)
            boxxes.clear()
            arrs = arrs[j:]

        else:
            sorted_boxes.extend(boxxes)
            break

    return np.array(sorted_boxes)


def list_files(in_path):
    img_files = []
    mask_files = []
    gt_files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        for file in filenames:
            filename, ext = os.path.splitext(file)
            ext = str.lower(ext)
            if ext == '.jpg' or ext == '.jpeg' or ext == '.gif' or ext == '.png' or ext == '.pgm':
                img_files.append(os.path.join(dirpath, file))
            elif ext == '.bmp':
                mask_files.append(os.path.join(dirpath, file))
            elif ext == '.xml' or ext == '.gt' or ext == '.txt':
                gt_files.append(os.path.join(dirpath, file))
            elif ext == '.zip':
                continue
    # img_files.sort()
    # mask_files.sort()
    # gt_files.sort()
    return img_files, mask_files, gt_files


def saveResult(img_file, img, boxes, model, converter, args=None):
        """
        save text detection result one by one
        Args:
            img_file (str): image file name
            img (array): raw image context
            boxes (array): array of result file
                Shape: [num_detections, 4] for BB output / [num_detections, 4] for QUAD output
        Return:
            text of picture, picture's name
        """

        img = np.array(img)
        img_copy = deepcopy(img)
        # make result file list
        filename, _ = os.path.splitext(os.path.basename(img_file))

        bboxes = []
        boxes = sort_boxes(boxes)

        for i, box in enumerate(boxes):
            poly = box.reshape(-1, 2)
            warped = four_point_transform(img_copy, poly)
            img_ = Image.fromarray(warped, 'RGB')
            bboxes.append(img_)

        text = demo(args, bboxes, model, converter)

        return text, filename