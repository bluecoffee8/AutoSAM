import pickle
import numpy as np
from PIL import Image
import os

import torch
import torch.nn as nn
from torchvision import transforms, utils
from torch.utils.data.dataset import Dataset
from torchvision.transforms.functional import resize

from models import HarDNet
from models import sam_seg_model_registry2

# from dataset import generate_dataset

# model = HarDNet()
# x = torch.zeros(size=(1, 3, 1024, 1024))
# x = model(x)
# print(x.shape)

# model = sam_seg_model_registry2['vit_b'](num_classes=4, checkpoint='medsam_vit_b.pth')
# state_dict = model.prompt_encoder.state_dict()
# print(state_dict)
# torch.cuda.set_device(0) # set torch device cuda
# model = model.cuda(0) # move model to gpu
# loc = 'cuda:{}'.format(0) # location of gpu
# # print(model.parameters())
# checkpoint = torch.load('model_best_autosam.pth.tar', map_location=loc)
# # model.prompt_encoder.load_state_dict(checkpoint['state_dict'])
# # print(model.prompt_encoder.base[0].named_parameters())
# # state_dict = checkpoint['state_dict']
# optimizer_state_dict = checkpoint['optimizer']
# print(optimizer_state_dict)
# # print(state_dict)
# x = torch.zeros(size=(1, 3, 224, 224))
# x = model(x)
# print(len(x))
# print(x[0].shape)
# print(x[1].shape)

# model = HarDNet()
# print(model.named_parameters())

# files = os.listdir('./Kvasir-SEG/images')
# img = Image.open('Kvasir-SEG/images/' + files[0])
# img = np.asarray(img).astype(np.float32).transpose([2, 0, 1])


# label = Image.open('Kvasir-SEG/masks/' + files[0])
# label = np.asarray(label)
# # print(img.shape)
# # print(label.shape)
# label = (label[:,:,0] > 128).astype(int)
# print(label.shape)
# print(np.sum(label))

# files = os.listdir('ACDC/imgs/patient_085_frame_01/')
# img = Image.open('ACDC/imgs/patient_085_frame_01/' + files[0])
# img = np.asarray(img).astype(np.float32).transpose([2, 0, 1])
# label = Image.open('ACDC/annotations/patient_085_frame_01/' + files[0])
# label = np.asarray(label)
# print(img.shape)
# print(label.shape)

split_dir = os.path.join('ACDC', "splits.pkl")
with open(split_dir, "rb") as f:
    splits = pickle.load(f)
tr_keys = splits[0]['train']
val_keys = splits[0]['val']
test_keys = splits[0]['test']

print(len(tr_keys))
print(len(val_keys))
print(len(test_keys))

# split_dir = os.path.join('Kvasir-SEG', "splits.pkl")
# with open(split_dir, "rb") as f:
#     splits = pickle.load(f)
# tr_keys = splits[0]['train']
# val_keys = splits[0]['val']
# test_keys = splits[0]['test']

# print(len(tr_keys))
# print(len(val_keys))
# print(len(test_keys))

# split_dir = 'ACDC/splits.pkl'
# with open(split_dir, "rb") as f:
#     splits = pickle.load(f)
# print(splits[0]['train'])