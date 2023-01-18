# -*- coding: utf-8 -*-n

import os
import torch
import argparse
from skimage import io

from workflow.build_model import build_param, load_model, predict
from workflow.improcess import load_image, extract_image


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_path", help='Image Path')
parser.add_argument("-s", "--save_path", help='Save Path')
args = parser.parse_args()


#Linux Devices with CUDA GPU Support
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Torch Backend Device is using {}'.format(device))

# This ensures that the current MacOS version is at least 12.3+
print(torch.backends.mps.is_available(), 'GPU MPS M1 Acceleration Availaible')
# This ensures that the current current PyTorch installation was built with MPS activated.
print(torch.backends.mps.is_built(), 'GPU MPS M1 Acceleration Built Succesfully')


"""# Load Model """

model_param = build_param()
net = load_model(model_param, device)

"""# Predict"""
original_image, image_tensor, orig_size = load_image(args.input_path, model_param) 
mask = predict(net,image_tensor, orig_size, model_param, device)
segment = extract_image(original_image, mask) 

io.imsave(os.path.join(args.save_path+"mask.png"), mask)
io.imsave(os.path.join(args.save_path+"segment.png"), segment)


