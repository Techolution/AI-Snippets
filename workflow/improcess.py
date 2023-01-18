#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import torch
from torchvision import transforms
from skimage import io
import cv2

from libraries.data_loader_cache import normalize, im_preprocess 



class GOSNormalize(object):
    '''
    Normalize the Image using torch.transforms
    '''
    def __init__(self, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]):
        self.mean = mean
        self.std = std

    def __call__(self,image):
        image = normalize(image,self.mean,self.std)
        return image

#Static Variable
transform =  transforms.Compose([GOSNormalize([0.5,0.5,0.5],[1.0,1.0,1.0])])

def load_image(im_path, hypar):
    orig_im = io.imread(im_path)
    im, im_shp = im_preprocess(orig_im, hypar["cache_size"])
    im = torch.divide(im,255.0)
    shape = torch.from_numpy(np.array(im_shp))
    return orig_im ,transform(im).unsqueeze(0), shape.unsqueeze(0) # make a batch of image, shape

def extract_image(image, mask):
   segmented_image = cv2.bitwise_and(image,image, mask = mask)
   return segmented_image


