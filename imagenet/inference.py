
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
from PIL import Image
from torch.autograd import Variable
import shutil
from pathlib import Path


from imagenet.dataset import DataSetLoader




class ImageClassification():
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.__normalize = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])



    def __predict_image(self, image_path, model, device):
        image = Image.open(image_path)
        image_tensor = self.__normalize(image).float()
        image_tensor = image_tensor.unsqueeze_(0)
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)
        output_c = output.cpu()
        index = output_c.data.numpy().argmax()   
        return index


    def __load_model(self, device, weights):
        
        model = models.resnet18(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
        model = model.to(device)
        model.load_state_dict(torch.load(weights, map_location=torch.device(device)))

        was_training = model.training
        model.eval()

        return model 


    def start(self, device, weights):

        model = self.__load_model(device, weights)
        ds = DataSetLoader()
        for cl in ds.classes:
            Path(os.path.join(self.folder_path, cl)).mkdir(parents=True, exist_ok=True)

        for image in os.listdir(self.folder_path):
            if not os.path.isdir(os.path.join(self.folder_path, image)) and not image.startswith('.'): 
                index_cls = self.__predict_image(os.path.join(self.folder_path, image), model, device)
                
                src_path = os.path.join(self.folder_path, image)
                dist_path = os.path.join(self.folder_path, ds.classes[index_cls], image)
                shutil.move(src_path,dist_path)
                print(ds.classes[index_cls] , os.path.join(self.folder_path, image))

