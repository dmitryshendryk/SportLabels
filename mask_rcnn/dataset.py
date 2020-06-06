import os 
import json 
import cv2 
import numpy as np 
from detectron2.structures import BoxMode




def get_carplate_dicts(mode):
    ROOT = os.path.abspath('../')
    DATA_FOLDER = 'mask_rcnn/data'
    path = os.path.join(ROOT, DATA_FOLDER)
    json_file = os.path.join(path, "via_project_3Jun2020_12h29m.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)
    
    dataset_dicts = []
    dataset_len = len(list(imgs_anns['_via_img_metadata'].values()))
    dataset = list(imgs_anns['_via_img_metadata'].values())
    if mode == 'train':
        dataset = dataset[:dataset_len - int(dataset_len*0.1)]
    elif mode == 'val':
        dataset = dataset[dataset_len - int(dataset_len*0.1):]
    classes = ["box","size"]
    for idx, v in enumerate(list(dataset)):
        record = {}
        
        filename = os.path.join(path, v["filename"])
        height, width = cv2.imread(filename).shape[:2]
        
        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
      
        annos = v["regions"]
        objs = []
        for anno in annos:
#             assert not anno["region_attributes"]
            class_id = anno["region_attributes"]["type"]
            anno = anno["shape_attributes"]
            if anno['name'] == 'polyline':
                px = anno["all_points_x"]
                py = anno["all_points_y"]
                poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
                poly = [p for x in poly for p in x]
                obj = {
                    "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": [poly],
                    "category_id": classes.index(class_id),
                    "iscrowd": 0
                }
                objs.append(obj)
            elif anno['name'] == 'rect':
                px = [anno["x"], anno["x"]+ anno["width"],anno["x"] + anno["width"], anno["x"] ]
                py = [ anno["y"] + anno["height"], anno["y"]+ anno["height"], anno["y"], anno["y"]]
                poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
                poly = [p for x in poly for p in x]
                obj = {
                    "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": [poly],
                    "category_id": classes.index(class_id),
                    "iscrowd": 0
                }
                objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts
