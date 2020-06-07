from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg 
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.evaluation import DatasetEvaluators

import json
import os
import sys
import random 
import PIL
import matplotlib.pyplot as plt 
import cv2
import numpy as np

ROOT = os.path.abspath('../')
DATA_FOLDER = 'mask_rcnn/data'
CONFIG = 'mask_rcnn/config'
WEIGHTS = 'mask_rcnn/weights'
DEVICE = 'cuda'

sys.path.append(ROOT)

from dataset import get_carplate_dicts


class Trainer(DefaultTrainer):

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
       
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
       

        return DatasetEvaluators(evaluator_list)



def train():


    DatasetCatalog.register("carplate_train", lambda x='train':  get_carplate_dicts(x))
    DatasetCatalog.register("carplate_val", lambda x='val':  get_carplate_dicts(x))
    MetadataCatalog.get("carplate_val").set(thing_classes=["box"])
    # carplate_metadata = MetadataCatalog.get("carplate_val")

    MetadataCatalog.get("carplate_val").set(evaluator_type='coco')




    cfg = get_cfg()
    cfg.merge_from_file(os.path.join(ROOT, CONFIG, "mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
    # cfg.merge_from_file(os.path.join(ROOT, CONFIG, "mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("carplate_train",)
    cfg.DATASETS.TEST = ("")
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.DEVICE = DEVICE
    cfg.MODEL.WEIGHTS = os.path.join(ROOT,WEIGHTS,"model_final_2d9806.pkl")  # Let training initialize from model zoo
    # cfg.MODEL.WEIGHTS = os.path.join(ROOT,WEIGHTS,"R-50.pkl")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.CHECKPOINT_PERIOD = 1000
    # cfg.TEST.EVAL_PERIOD = 50
    cfg.SOLVER.GAMMA = 0.05
    cfg.SOLVER.STEPS = (600, 1200, 1800, 2400, 3000, 3600, 4200, 4800)
    cfg.SOLVER.BASE_LR = 0.00005  # pick a good LR
    cfg.SOLVER.MAX_ITER = 5000    # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon)



    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = Trainer(cfg) 
    trainer.build_evaluator(cfg,'carplate_val',output_folder="./output/")
    trainer.resume_or_load(resume=False)
    trainer.train()

train()