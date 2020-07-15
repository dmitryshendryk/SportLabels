from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg 
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog


import json
import os
import sys
import random 
import PIL
import matplotlib.pyplot as plt 
import cv2
import numpy as np

# from utils.pre_processing import shi_tomashi, get_destination_points, unwarp
from utils.percpective import find_rectangle_size
# ROOT = os.path.abspath('../')
# CONFIG = 'code/mask_rcnn/config'
# WEIGHTS = 'code/mask_rcnn/weights'



class Mask_RCNN_detector():
    def __init__(self, device, ROOT):
        print("Load Mask RCNN", end='\n')
        cfg = get_cfg()
        cfg.merge_from_file(os.path.join(ROOT, 'mask_rcnn/config', "mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.DATALOADER.NUM_WORKERS = 2
        cfg.MODEL.DEVICE = device
        cfg.SOLVER.IMS_PER_BATCH = 2
        cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
        cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon)

        cfg.MODEL.WEIGHTS = os.path.join(ROOT, 'mask_rcnn/weights', "mask_rcnn_model_final.pth")
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model
        
        self.predictor = DefaultPredictor(cfg)

    def detection(self, im, device):

        outputs = self.predictor(im)
        print("Process Mask RCNN and cropp images", end='\n')


        res = []
        for bbox in outputs["instances"].pred_boxes.tensor.numpy():
          
            print(len(bbox),bbox)
            img = im.copy()
            # img = img[abs(int(bbox[1]-10)):abs(int(bbox[3]+10)),abs(int(bbox[0]-10)):abs(int(bbox[2]+10)) ]
            img = img[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2]) ]
            # img = find_rectangle_size(img)
            res.append(img)
        print("Images to process {}".format(len(res)))
        return res