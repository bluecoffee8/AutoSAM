import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
import pickle
import numpy as np
from datetime import datetime
import nibabel as nib

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.models as models


from loss_functions.dice_loss import SoftDiceLoss
from loss_functions.metrics import dice_pytorch, SegmentationMetric

from models import sam_feat_seg_model_registry, sam_seg_model_registry
from dataset import generate_dataset, generate_test_loader
from evaluate import test_synapse, test_acdc


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument("--save_dir", type=str, default="eval/vit120_5")
parser.add_argument('--data_dir', type=str, default="ACDC/imgs/")
parser.add_argument('--src_dir', type=str, default="ACDC/")
parser.add_argument("--dataset", type=str, default="ACDC")
parser.add_argument('-b', '--batch-size', default=4, type=int,
                    metavar='N')
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N')
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument("--img_size", type=int, default=224)

args = parser.parse_args()

def test_cnn(model, args):
    print('Test')
    join = os.path.join
    if not os.path.exists(join(args.save_dir, "infer")):
        os.mkdir(join(args.save_dir, "infer"))
    if not os.path.exists(join(args.save_dir, "label")):
        os.mkdir(join(args.save_dir, "label"))

    split_dir = os.path.join(args.src_dir, "splits.pkl")
    with open(split_dir, "rb") as f:
        splits = pickle.load(f)
    for i in range(1):
        test_keys = splits[i]['test']

        model.eval()

        for key in test_keys:
            preds = []
            labels = []
            data_loader = generate_test_loader(key, args)
            with torch.no_grad():
                for i, tup in enumerate(data_loader):
                    if args.gpu is not None:
                        img = tup[0].float().cuda(args.gpu, non_blocking=True)
                        label = tup[1].long().cuda(args.gpu, non_blocking=True)
                    else:
                        img = tup[0]
                        label = tup[1]

                    mask = model(img)
                    mask_softmax = F.softmax(mask, dim=1)
                    mask = torch.argmax(mask_softmax, dim=1)

                    preds.append(mask.cpu().numpy())
                    labels.append(label.cpu().numpy())
                preds = np.concatenate(preds, axis=0)
                labels = np.concatenate(labels, axis=0).squeeze()
                print(preds.shape, labels.shape)
                if "." in key:
                    key = key.split(".")[0]
                ni_pred = nib.Nifti1Image(preds.astype(np.int8), affine=np.eye(4))
                ni_lb = nib.Nifti1Image(labels.astype(np.int8), affine=np.eye(4))
                nib.save(ni_pred, join(args.save_dir, 'infer', key + '.nii'))
                nib.save(ni_lb, join(args.save_dir, 'label', key + '.nii'))
            print("finish saving file:", key)

def test_vit(model, args):
    print('Test')
    join = os.path.join
    if not os.path.exists(join(args.save_dir, "infer")):
        os.mkdir(join(args.save_dir, "infer"))
    if not os.path.exists(join(args.save_dir, "label")):
        os.mkdir(join(args.save_dir, "label"))

    split_dir = os.path.join(args.src_dir, "splits.pkl")
    with open(split_dir, "rb") as f:
        splits = pickle.load(f)
    for i in range(1):
        test_keys = splits[i]['test']

        model.eval()

        for key in test_keys:
            preds = []
            labels = []
            data_loader = generate_test_loader(key, args)
            with torch.no_grad():
                for i, tup in enumerate(data_loader):
                    if args.gpu is not None:
                        img = tup[0].float().cuda(args.gpu, non_blocking=True)
                        label = tup[1].long().cuda(args.gpu, non_blocking=True)
                    else:
                        img = tup[0]
                        label = tup[1]

                    b, c, h, w = img.shape

                    mask, iou_pred = model(img)
                    mask = mask.view(b, -1, h, w)
                    mask_softmax = F.softmax(mask, dim=1)
                    mask = torch.argmax(mask_softmax, dim=1)

                    preds.append(mask.cpu().numpy())
                    labels.append(label.cpu().numpy())

                preds = np.concatenate(preds, axis=0)
                labels = np.concatenate(labels, axis=0).squeeze()
                print(preds.shape, labels.shape)
                if "." in key:
                    key = key.split(".")[0]
                ni_pred = nib.Nifti1Image(preds.astype(np.int8), affine=np.eye(4))
                ni_lb = nib.Nifti1Image(labels.astype(np.int8), affine=np.eye(4))
                nib.save(ni_pred, join(args.save_dir, 'infer', key + '.nii'))
                nib.save(ni_lb, join(args.save_dir, 'label', key + '.nii'))
            print("finish saving file:", key)

# model = sam_feat_seg_model_registry['vit_b'](num_classes=4, checkpoint='medsam_vit_b.pth') # for CNN/MLP model
model = sam_seg_model_registry['vit_b'](num_classes=4, checkpoint='medsam_vit_b.pth') # for VIT model

torch.cuda.set_device(args.gpu) # set torch device cuda
model = model.cuda(args.gpu) # move model to gpu
loc = 'cuda:{}'.format(args.gpu) # location of gpu

# checkpoint = torch.load('model_best_cnn_5.pth.tar', map_location=loc)
checkpoint = torch.load('model_best_vit_5.pth.tar', map_location=loc)

model.mask_decoder.load_state_dict(checkpoint['state_dict'], strict=False)

# test_cnn(model, args) 
test_vit(model, args)

test_acdc(args)

# split_dir = os.path.join(args.src_dir, "splits.pkl")
# with open(split_dir, "rb") as f:
#     splits = pickle.load(f)
# for i in range(5):
#     print(splits[i]['train'])
#     print()
#     print()