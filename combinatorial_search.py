### All utility functions for combinatorial search over perturbation mask

import  numpy as np
import itertools
import random
import math

from utils import *

import os
import time
import scipy.io as scio
import datetime
import re
import matplotlib.pyplot as plt
import pylab
import os
import csv
from skimage import transform, filters
from textwrap import wrap
import cv2
import sys
from PIL import Image


# return all nCr combinations of mask for top n pixels
def topCombinations(mask, limit_n, size_r):
    indices = np.argsort(mask.ravel())[: limit_n]
    all_masks = []
    for indices_subset in itertools.combinations(indices, size_r):
        ii = np.unravel_index(indices_subset, mask.shape)
        outMask = np.ones_like(mask)
        outMask[ii] = 0
        all_masks.append(outMask)
    return all_masks

# get probability of masked image
def get_mask_insertion_prob(mask, img, blurred_img, model, category, view=0, use_cuda=use_cuda):
    resize_wh = (img.data.shape[2], img.data.shape[3])
    if use_cuda:
        upsample = torch.nn.UpsamplingBilinear2d(size=resize_wh).cuda()
    else:
        upsample = torch.nn.UpsamplingBilinear2d(size=resize_wh)
    maskdata = mask.copy()
    # convert to insertion mask
    maskdata = np.subtract(1, maskdata).astype(np.float32)
    if use_cuda:
        Masktop = torch.from_numpy(maskdata).cuda()
    else:
        Masktop = torch.from_numpy(maskdata)
    Masktop = Variable(Masktop, requires_grad=False)
    MasktopLS = upsample(Masktop)
    Img_topLS = img.mul(MasktopLS) + \
                blurred_img.mul(1 - MasktopLS)
    outputstopLS = torch.nn.Softmax(dim=1)(model(Img_topLS))
    insertion_probability = outputstopLS[0, category].data.cpu().numpy().copy()
    if view:
        print('insertion_probability: ', insertion_probability)
        image = Img_topLS.data.cpu().numpy()
        image = image.squeeze()
        image = image.transpose(1,2,0)
        plt.imshow(image)
        plt.show()
    return insertion_probability

# check for duplicate mask
def duplicate(newmask, all_mp):
    for mask, p in all_mp:
        if (mask == newmask).all():
            return True
    return False

# remove redundant patches - necessary to get monotonic DNF
def remove_extra_patches(mask, pob, pth, img, blurred_img, model, category, use_cuda=use_cuda):
    no_patches_removed_flag = False
    while not no_patches_removed_flag:
        z0, z1, rows, cols = np.where(mask == 0)
        no_patches_removed_flag = True
        for i in range(len(rows)):
            mask2 = mask.copy()
            index = (z0[i], z1[i], rows[i], cols[i])
            mask2[index] = 1
            ptmp = get_mask_insertion_prob(mask2, img, blurred_img, model, category, 0, use_cuda)
            if ptmp > pth:
                mask = mask2
                pob = ptmp
                no_patches_removed_flag = False
    return mask, pob


# get probabilities for all masks when imposed on image
def get_all_mp(all_masks, img_ori, blurred_img_ori, model, category, prob_thresh, use_cuda=use_cuda):
    all_mp = []
    img = preprocess_image(img_ori, use_cuda, require_grad=False)
    blurred_img = preprocess_image(blurred_img_ori, use_cuda, require_grad=False)
    resize_wh = (img.data.shape[2], img.data.shape[3])
    if use_cuda:
        upsample = torch.nn.UpsamplingBilinear2d(size=resize_wh).cuda()
    else:
        upsample = torch.nn.UpsamplingBilinear2d(size=resize_wh)
    total = len(all_masks)
    count = 0
    for mask in all_masks:
        count += 1
        maskdata = mask.copy()
        # convert to insertion mask
        maskdata = np.subtract(1, maskdata).astype(np.float32)
        if use_cuda:
            Masktop = torch.from_numpy(maskdata).cuda()
        else:
            Masktop = torch.from_numpy(maskdata)
        Masktop = Variable(Masktop, requires_grad=False)
        MasktopLS = upsample(Masktop)
        Img_topLS = img.mul(MasktopLS) + \
                    blurred_img.mul(1 - MasktopLS)
        outputstopLS = torch.nn.Softmax(dim=1)(model(Img_topLS))
        insertion_probability = outputstopLS[0, category].data.cpu().numpy().copy()
        if insertion_probability > prob_thresh:
            mask, pob = remove_extra_patches(mask, insertion_probability, prob_thresh, img, blurred_img, model, category, use_cuda)
            if not duplicate(mask, all_mp):
                all_mp.append((mask, insertion_probability))
    return all_mp
