### All utility functions for diverse subset selection of the set of candidate masks

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



def create_patch_weight_matrix(all_mp, coeff):
    pwm = np.zeros_like(all_mp[0][0])
    for k in range(len(all_mp)):
        mask, p = all_mp[k]
        pwm += (1-mask) * math.exp(coeff * p)
    return pwm

def union_bmasks(m1,m2):
    return m1*m2

def intersection_bmasks(m1,m2):
    m = m1+m2
    m = np.clip(m, 0, 1)
    return m

def intersection_score(m1,m2):
    intrmask = intersection_bmasks(m1,m2)
    return np.sum(1-intrmask)

def coverage(pwm, mask):
    pwm_intersection = pwm * (1-mask)
    return np.sum(pwm_intersection)

def max_coverage_set(pwm, all_mp):
    all_mc = []
    for mask, p in all_mp:
        cov = coverage(pwm, mask)
        all_mc.append((mask, p, cov))
    all_mc_sorted = list(sorted(all_mc, key=lambda x: x[2], reverse=True))
    #print('all_mc_sorted:\n', all_mc_sorted)
    maxcov = all_mc_sorted[0][2]
    mask_set = []
    for mask, p, cov in all_mc_sorted:
        if cov == maxcov:
            mask_set.append((mask,p))
        else:
            break
    return mask_set


def min_intersection_set(all_mp):
    mask_set = []
    l = len(all_mp)
    intr_mat = np.zeros((l,l))
    for i in range(l):
        for j in range(i+1,l):
            intr_mat[i][j] = intersection_score(all_mp[i][0], all_mp[j][0])
            intr_mat[j][i] = intr_mat[i][j]
    intr_array = np.sum(intr_mat, axis=0)
    minintr = np.min(intr_array)
    min_indices = np.where(intr_array == minintr)[0]
    size = np.size(min_indices)
    for k in range(size):
        mask_set.append(all_mp[min_indices[k]])
    return mask_set


def create_intersection_matrix(all_mp):
    l = len(all_mp)
    intr_mat = np.zeros((l,l))
    for i in range(l):
        for j in range(i+1,l):
            intr_mat[i][j] = intersection_score(all_mp[i][0], all_mp[j][0])
            intr_mat[j][i] = intr_mat[i][j]
        intr_mat[i][i] = intersection_score(all_mp[i][0], all_mp[i][0])
    return intr_mat


def tie_breaker_prob(indices, all_mp):
    maxprob = -1
    ans = -1
    for i in indices:
        if all_mp[i][1] > maxprob:
            maxprob = all_mp[i][1]
            ans = i
    return [ans]


def disparity_min_indices(all_mp, im, refset):
    minintersection = float('inf')
    minintersection_indices = []

    if len(refset) == 0:
        minintersection = im.min()
        for i in range(len(all_mp)):
            if im[i].min() == minintersection:
                minintersection_indices.append(i)
    else:
        maxintersection_pairs = []
        for i in range(len(all_mp)):
            if i not in refset:
                maxintersection = 0
                for j in refset:
                    if maxintersection < im[i][j]:
                        maxintersection = im[i][j]
                maxintersection_pairs.append((i,maxintersection))
                if minintersection > maxintersection:
                    minintersection = maxintersection
        for index, maxintersection in maxintersection_pairs:
            if maxintersection == minintersection:
                minintersection_indices.append(index)
    return minintersection_indices



def get_best_mask_index(all_mp, im, refset):
    filtered_mp_indices = disparity_min_indices(all_mp, im, refset)
    if len(filtered_mp_indices) > 1:
        filtered_mp_indices = tie_breaker_prob(filtered_mp_indices, all_mp)
    return filtered_mp_indices


def update_pwm(best_mp, pwm):
    best_mask, p = best_mp
    cov = coverage(pwm, best_mask)
    pwm = pwm * best_mask
    return pwm, cov


def set_filtering(all_mp):
    diverse_mp_indices = []
    disparity_min_set = []
    num_candidates = len(all_mp)
    im = create_intersection_matrix(all_mp)
    while len(diverse_mp_indices) < num_candidates:
        best_mp_index = get_best_mask_index(all_mp, im, diverse_mp_indices)
        assert len(best_mp_index) == 1
        best_mp_index = best_mp_index[0]
        diverse_mp_indices.append(best_mp_index)
        best_mp = all_mp[best_mp_index]
        disparity_min_set.append((best_mp[0], best_mp[1]))
    return disparity_min_set


def overlapThresh_dfs(n, im, refset, overlap_thresh):
    noOverlap_indices = []
    maximal_refset = []
    maximal_refset_length = 0
    index = refset[-1]
    for j in range(index+1,n): # consider only upper triangular elements as symmetric matrix
            # check for no overlap with refset
            valid = True
            for k in refset:
                if (im[j][k] > overlap_thresh):
                    valid = False
                    break
            if valid:
                noOverlap_indices.append(j)
    # base case
    if noOverlap_indices == []:
        return refset

    # trigger dfs for new indices
    for j in noOverlap_indices:
        candidate_refset = overlapThresh_dfs(n, im, refset + [j], overlap_thresh)
        # save maximal refset
        candidate_refset_length = len(candidate_refset)
        if maximal_refset_length < candidate_refset_length:
            maximal_refset_length = candidate_refset_length
            maximal_refset = candidate_refset
    return maximal_refset


# function to return maximal set of masks, all of which have zero overlap
def maximal_overlapThresh_set(all_mp, overlap_thresh):
    ans_mp = []
    max_set = []
    max_set_length = 0
    n = len(all_mp)
    im = create_intersection_matrix(all_mp)
    # trigger no overlap dfs
    for i in range(n):
        refset = [i]
        candidate_set = overlapThresh_dfs(n, im, refset, overlap_thresh)
        candidate_set_length = len(candidate_set)
        if max_set_length < candidate_set_length:
            max_set_length = candidate_set_length
            max_set = candidate_set
    for index in max_set:
        ans_mp.append(all_mp[index])
    return ans_mp
