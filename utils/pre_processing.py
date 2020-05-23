import numpy as np
from scipy import signal
from PIL import Image
import os 

ROOT = os.path.abspath('./')

def load_image(path):
    return np.asarray(Image.open(os.path.join(ROOT,path)))/255.0

def save(path, img):
    tmp = np.asarray(img*255.0, dtype=np.uint8)
    Image.fromarray(tmp).save(path)

def denoise_image(inp):
    # estimate 'background' color by a median filter
    bg = signal.medfilt2d(inp, 11)
    mask = inp < bg - 0.1
    return np.where(mask, inp, 1.0)

