import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

class DataSetLoader():
   
    def __init__(self):
        self.classes = ['box','size']

    def data_loader():

        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        data_dir = 'imagenet/data'
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                                data_transforms[x])
                        for x in ['train', 'val']}
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                    shuffle=True, num_workers=4)
                    for x in ['train', 'val']}
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
        class_names = image_datasets['train'].classes


        return dataloaders, class_names, dataset_sizes


  
