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
from copy import deepcopy


# beam search - returns all roots found
def beamSearch_topKSuccessors_roots(ref_mask, beam_width, numSuccessors, img, blurred_img, model, category, prob_thresh, full_image_probability, max_num_roots, root_size, use_cuda=use_cuda):
    roots_mp = []
    #preprocess image - needed for get_mask_insertion_prob()
    img = preprocess_image(img, use_cuda, require_grad=False)
    blurred_img = preprocess_image(blurred_img, use_cuda, require_grad=False)

    # init
    init_mask = np.ones((1,1,7,7))
    beam_masks = [init_mask]
    # beam_masks = beamSearch_Init(ref_mask, beam_width, numSuccessors)
    # num_patches_inserted += 1
    for i in range(root_size):
        # generate all successors
        all_successors_mp = beamSearch_get_all_successors_mp(ref_mask, beam_masks, numSuccessors, full_image_probability, img, blurred_img, model, category, use_cuda=use_cuda)
        if all_successors_mp == []: # no more successors left
            break
        # select top beam masks and add distict roots if found
        beam_masks, roots_mp = beamSearch_get_topk_masks_roots(roots_mp, all_successors_mp, beam_width, prob_thresh, full_image_probability, img, blurred_img, model, category)
        #print('roots_found: ', len(roots_mp))
        if len(roots_mp) > max_num_roots: # max roots limit reached
            break
    return roots_mp


# beam search status function
def beamSearch_topKSuccessors_status(ref_mask, beam_width, numSuccessors, img, blurred_img, model, category, prob_thresh, full_image_probability, use_cuda=use_cuda):
    status = False
    num_patches_inserted = 0
    #preprocess image - needed for get_mask_insertion_prob()
    img = preprocess_image(img, use_cuda, require_grad=False)
    blurred_img = preprocess_image(blurred_img, use_cuda, require_grad=False)

    # init
    init_mask = np.ones((1,1,7,7))
    beam_masks = [init_mask]
    # beam_masks = beamSearch_Init(ref_mask, beam_width, numSuccessors)
    # num_patches_inserted += 1
    while status == False:
        # generate all successors
        all_successors_mp = beamSearch_get_all_successors_mp(ref_mask, beam_masks, numSuccessors, full_image_probability, img, blurred_img, model, category, use_cuda=use_cuda)
        if all_successors_mp == []: # no more successors left
            break
        # select top beam masks
        beam_masks, status = beamSearch_get_topk_masks(all_successors_mp, beam_width, prob_thresh, full_image_probability, img, blurred_img, model, category)
        num_patches_inserted += 1
        # print('num_patches_inserted: ', num_patches_inserted)
    return status, num_patches_inserted


# init for beam search - select k initialisations from prunned mask
def beamSearch_Init(ref_mask, beam_width, numSuccessors):
    mask_list = []
    init_mask = np.ones((1,1,7,7))
    invalid_indices = np.where(init_mask == 0)
    # get successor indices to select successors from
    successor_indices, num_successors = beamSearch_get_successor_indices(invalid_indices, ref_mask, numSuccessors)
    # select initial points - draw samples equal to beam_width
    sampled_indices = np.random.choice(np.arange(num_successors), size=beam_width, replace=False)
    for index in sampled_indices:
        # mask_index = [0, 0, int(index/ncols), index%ncols]
        # print('mask_index: ', mask_index)
        new_mask = np.ones((1,1,7,7))
        new_mask[successor_indices[0][index]][successor_indices[1][index]][successor_indices[2][index]][successor_indices[3][index]] = 0
        mask_list.append(new_mask)
    return mask_list


# function to get successors - locally optimized using initial perturbation map as heuristic - since purely random generation of all successors is computationally expensive (inefficient)
def beamSearch_get_successor_indices(invalid_indices, ref_mask, numSuccessors):
    successor_indices = []
    # remove invalid indices from ref mask
    ref_mask_copy = deepcopy(ref_mask) # ref mask has continuous values in interval [0,1]
    ref_mask_copy[invalid_indices] = 2 # value=2 implies invalid patch
    # get successor indices from valid indices
    num_total_indices = ref_mask_copy.shape[2]*ref_mask_copy.shape[3]
    num_valid_indices = num_total_indices - len(invalid_indices[0])
    if numSuccessors > num_valid_indices:
        numSuccessors = num_valid_indices
    successor_indices = np.unravel_index(np.argsort(ref_mask_copy.ravel())[: numSuccessors], ref_mask_copy.shape)
    # print('numSuccessors_after: ', numSuccessors)
    return successor_indices, numSuccessors


# beam search - function to generate all successor (mask, prob) pairs
def beamSearch_get_all_successors_mp(ref_mask, beam_masks, numSuccessors, full_image_probability, img, blurred_img, model, category, use_cuda=use_cuda):
    mp_list = []
    for mask in beam_masks:
        invalid_indices = np.where(mask == 0)
        # get successor indices
        successor_indices, num_successors = beamSearch_get_successor_indices(invalid_indices, ref_mask, numSuccessors)
        for index in range(num_successors):
            new_mask = deepcopy(mask)
            new_mask[successor_indices[0][index]][successor_indices[1][index]][successor_indices[2][index]][successor_indices[3][index]] = 0 # add new patch to generate successor mask
            insertion_prob = get_mask_insertion_prob(new_mask, img, blurred_img, model, category, view=0, use_cuda=use_cuda)
            rel_prob = insertion_prob/full_image_probability
            mp_list.append((new_mask, insertion_prob, rel_prob))
    # print('len(mp_list): ', len(mp_list))
    return mp_list

# beam search - function to filter top k successors (k = beam_width)
def beamSearch_get_topk_masks_roots(roots_mp, all_successors_mp, beam_width, prob_thresh, full_image_probability, img, blurred_img, model, category, use_cuda=use_cuda):
    mask_list = []
    # sort successors masks in descending order of insertion prob
    all_successors_mp_sorted = sorted(all_successors_mp, key=lambda x: x[1], reverse=True)
    # select top k successors
    for mask, prob, rel_prob in all_successors_mp_sorted:
        if prob > prob_thresh * full_image_probability:
            # remove extra patches in root
            minimal_mask, minimal_ins_prob = remove_extra_patches(mask, prob, prob_thresh, full_image_probability, img, blurred_img, model, category, use_cuda)
            # add root if not duplicate
            if not duplicate(minimal_mask, roots_mp):
                minimal_rel_prob = minimal_ins_prob / full_image_probability
                item_mp = [minimal_mask, minimal_ins_prob, minimal_rel_prob]
                roots_mp.append(item_mp)
        else:
            mask_list.append(mask)
            if len(mask_list) > beam_width:
                break
    return mask_list, roots_mp


# beam search - function to filter top k successors (k = beam_width)
def beamSearch_get_topk_masks(all_successors_mp, beam_width, prob_thresh, full_image_probability, img, blurred_img, model, category, use_cuda=use_cuda):
    mask_list = []
    status = False
    # sort successors masks in descending order of insertion prob
    all_successors_mp_sorted = sorted(all_successors_mp, key=lambda x: x[1], reverse=True)
    # select top k successors
    for i in range(beam_width):
        mask_list.append(all_successors_mp_sorted[i][0])
    # status - True if a mask with insertion_prob > 0.9 * full_prob is found
    top_insertion_prob = all_successors_mp_sorted[0][1]
    if top_insertion_prob > prob_thresh * full_image_probability:
        status = True
    return mask_list, status


# this function returns success if one mask with insertion probability > 0.9 is found
def comb_search_status(mask, total_mask_pixels, max_patches, img, blurred_img, model, category, prob_thresh, probability, use_cuda=use_cuda):
    success = False

    #preprocess image - needed for get_mask_insertion_prob()
    img = preprocess_image(img, use_cuda, require_grad=False)
    blurred_img = preprocess_image(blurred_img, use_cuda, require_grad=False)

    indices = np.argsort(mask.ravel())[: total_mask_pixels]
    for indices_subset in itertools.combinations(indices, max_patches):
        ii = np.unravel_index(indices_subset, mask.shape)
        outMask = np.ones_like(mask)
        outMask[ii] = 0
        insertion_prob = get_mask_insertion_prob(outMask, img, blurred_img, model, category, view=0, use_cuda=use_cuda)
        if insertion_prob > (prob_thresh * probability):
            # print('insertion_prob:{} \t total_prob:{}'.format(insertion_prob, probability))
            success = True
            break
    return success


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
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imshow('window', image)
        cv2.waitKey(0)
    return insertion_probability
    

# check for duplicate mask
def duplicate(newmask, all_mp):
    for mask, p, rel_p in all_mp:
        if (mask == newmask).all():
            return True
    return False

# remove redundant patches - necessary to get monotonic DNF
def remove_extra_patches(mask, pob, pth, full_prob, img, blurred_img, model, category, use_cuda=use_cuda):
    no_patches_removed_flag = False
    while not no_patches_removed_flag:
        z0, z1, rows, cols = np.where(mask == 0)
        no_patches_removed_flag = True
        for i in range(len(rows)):
            mask2 = mask.copy()
            index = (z0[i], z1[i], rows[i], cols[i])
            mask2[index] = 1
            ptmp = get_mask_insertion_prob(mask2, img, blurred_img, model, category, 0, use_cuda)
            if ptmp > pth * full_prob:
                mask = mask2
                pob = ptmp
                no_patches_removed_flag = False
    return mask, pob


# get probabilities for all masks when imposed on image
def get_all_mp(all_masks, img_ori, blurred_img_ori, model, category, prob_thresh, full_prob, use_cuda=use_cuda):
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
        if insertion_probability > (prob_thresh * full_prob):
            mask, pob = remove_extra_patches(mask, insertion_probability, prob_thresh, full_prob, img, blurred_img, model, category, use_cuda)
            if not duplicate(mask, all_mp):
                all_mp.append((mask, insertion_probability))
    return all_mp
