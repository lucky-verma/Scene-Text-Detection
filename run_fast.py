"""
This script runs the fast algorithm on single frame.

input: image path
output: polygon boxes

author: @lucky-verma
"""

import glob
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
import easyocr

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


### Initiate the FAST model
ckt_path = "fast_tiny_tt_512_finetune_ic17mlt.pth"

# load config
cfg = Config.fromfile(
    "./config/fast/tt/fast_tiny_tt_512_finetune_ic17mlt.py")

# model
model = build_model(cfg.model)
model = model.cuda()

ema=True

if os.path.isfile(ckt_path):
    start1 = time.time()

    print("Loading model and optimizer from checkpoint '{}'".format(
        ckt_path))
    logging.info("Loading model and optimizer from checkpoint '{}'".format(
        ckt_path))
    sys.stdout.flush()
    print("" * 2)
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

### Initialize the Easy OCR model
reader = easyocr.Reader(['ru'])


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
    img_meta = dict(org_img_size=np.array([img.shape[:2]]))

    img = scale_aligned_short(img, short_size)
    img_meta.update(dict(img_size=np.array([img.shape[:2]]),
                         filename=filename))

    img = Image.fromarray(img)
    img = img.convert('RGB')
    img = transforms.ToTensor()(img)
    img = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])(img)

    img = img.unsqueeze(0).cuda()

    data = dict(imgs=img, img_metas=img_meta)

    return data


def apply_ocr(img, boxes, single=False):
    """
    This function takes boxes from FAST algo which is converted cropped images. 
    These cropped images are then passed to the Easy OCR russian model to get the text. 
    """

    start = time.time()

    ### If single is true, then we will only perform OCR on the mask cropped image
    if single:
        for pts in boxes:
            mask = np.full((img.shape[0], img.shape[1]), 0, dtype=np.uint8)

            for pts in boxes:
                mask  = cv2.fillPoly(mask, pts =[np.array(pts).reshape((-1, 1, 2)).astype(np.int32)], color=(255,255,255))

            # get first masked value (foreground)
            masked = cv2.bitwise_and(image, image, mask=mask)

            results = reader.readtext(masked, detail=0)

            end = time.time()

            print("Single OCR time: " + str(end - start))

            # return the results
            return results, str(end - start)

    ### perform OCR on the cropped images
    results = []

    for pts in boxes:
        pts = np.array(pts).reshape((-1, 1, 2)).astype(np.int32)
        ## (1) Crop the bounding rect
        rect = cv2.boundingRect(pts)
        x,y,w,h = rect
        croped = img[y:y+h, x:x+w].copy()

        ## (2) make mask
        pts = pts - pts.min(axis=0)

        mask = np.zeros(croped.shape[:2], np.uint8)
        cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)

        ## (3) do bit-op
        dst = cv2.bitwise_and(croped, croped, mask=mask)

        results.append(reader.readtext(dst, detail=0))

    end = time.time()

    print("OCR time: " + str(end - start))

    # return the results
    return results, str(end - start)


def get_detections(img_path):
    """
    This function takes an image path and returns the detections
    """

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

    return outputs['results'], str(end2 - start2)


# Driver code
if __name__ == "__main__":
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, default='images')
    args = parser.parse_args()

    final_results = []

    # run on folder of images
    for img_path in glob.glob(args.img_dir + '/*.jpg'):
        # get detections
        detections, time1 = get_detections(img_path)

        image = cv2.imread(img_path)
        results, time2 = apply_ocr(image, detections[0]['bboxes'], single=True)

        temp_dict = {
            "filename": img_path.split('/')[-1],
            "FastTime": time1,
            "OCRTime": time2,
            "FastResults": detections,
            "OCRResults": results
        }

        # append to final results
        final_results.append(temp_dict)

    # total time
    total_time = 0
    for i in final_results:
        total_time += float(i['FastTime']) + float(i['OCRTime'])
    final_results.append({"TotalTime": total_time})

    # save final results
    with open('final_results.json', 'w') as fp:
        json.dump(final_results, fp, ensure_ascii=False)


    # # get detections
    # detections = get_detections(args.img_path)

    # print("" * 2)
    # print("Fast bboxes : ", detections)
    # print("" * 2)

    # # apply ocr
    # image = cv2.imread(args.img_path)
    # results = apply_ocr(image, detections[0]['bboxes'])
    # print("" * 2)
    # print("Easy OCR results : ", results)
