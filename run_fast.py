"""
This script runs the fast algorithm on single frame.

input: image path
output: polygon boxes

author: @lucky-verma
"""

import warnings
warnings.filterwarnings('ignore')

import argparse
import os
import sys
import logging
import time
import json
import torch
import dataset
import cv2
import mmcv

import numpy as np
import torchvision.transforms as transforms

from mmcv import Config
from PIL import Image
from pprint import pprint
from models import build_model
from dataset import build_data_loader
from models.utils import fuse_module, rep_model_convert

from dataset.utils import shrink
from dataset.utils import get_img
from dataset.utils import get_tt_ann as get_ann
from dataset.utils import random_scale, random_horizontal_flip, random_rotate
from dataset.utils import random_crop_padding_v2 as random_crop_padding
from dataset.utils import update_word_mask, get_vocabulary
from dataset.utils import scale_aligned_short

def get_img(img_path, read_type='pil'):
    try:
        if read_type == 'cv2':
            img = cv2.imread(img_path)
            img = img[:, :, [2, 1, 0]]
        elif read_type == 'pil':
            img = np.array(Image.open(img_path))
    except Exception as e:
        print(img_path)
        raise
    return img

def prepare_test_data(img_path, short_size=512, read_type='pil'):
    filename = img_path.split('/')[-1][:-4]
    img = get_img(img_path, read_type)
    if filename == 'img651' and img.shape[0] > img.shape[1]:
        img = cv2.transpose(img)
        img = cv2.flip(img, 0)
    img_meta = dict(
        org_img_size=np.array([img.shape[:2]])
    )

    img = scale_aligned_short(img, short_size)
    img_meta.update(dict(
        img_size=np.array([img.shape[:2]]),
        filename=filename
    ))

    img = Image.fromarray(img)
    img = img.convert('RGB')
    img = transforms.ToTensor()(img)
    img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)

    img = img.unsqueeze(0).cuda()

    data = dict(
        imgs=img,
        img_metas=img_meta
    )

    return data


def get_detections(img_path, ema=True):
    """
    This function takes an image path and returns the detections
    """
    # checkpoint path
    ckt_path = "fast_tiny_tt_512_finetune_ic17mlt.pth"

    # load config
    cfg = Config.fromfile("./config/fast/tt/fast_tiny_tt_512_finetune_ic17mlt.py")

    # model
    model = build_model(cfg.model)
    model = model.cuda()

    if os.path.isfile(ckt_path):
        start1 = time.time()

        print("Loading model and optimizer from checkpoint '{}'".format(ckt_path))
        logging.info("Loading model and optimizer from checkpoint '{}'".format(ckt_path))
        sys.stdout.flush()
        checkpoint = torch.load(ckt_path)

        if not ema:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint['ema']

        d = dict()
        for key, value in state_dict.items():
            tmp = key.replace("module.", "")
            d[tmp] = value
        model.load_state_dict(d)

        end1 = time.time()
        print("Loaded model in " + str(end1 - start1))
    else:
        print("No checkpoint found at '{}'".format(ckt_path))
        raise

    model = rep_model_convert(model)

    # fuse conv and bn
    model = fuse_module(model)

    model.eval()

    # prepare input
    data = prepare_test_data(img_path)
    # prepare input
    data['imgs'] = data['imgs'].cuda(non_blocking=True)
    data.update(dict(cfg=cfg))
    # forward
    with torch.no_grad():
        start2 = time.time()
        outputs = model(**data)
        end2 = time.time()
        print("Forward pass in " + str(end2 - start2))

    return outputs['results']


# Driver code
if __name__ == "__main__":
    # read image from file
    img_path = "./data/total_text/Images/Test/5f44dfcf85600a7fd4566462.jpg"
    print(get_detections(img_path))
