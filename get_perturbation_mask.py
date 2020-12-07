### All utility function to obtain perturbation mask

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


def Get_blurred_img(input_img, img_label, model, resize_shape=(224, 224), Gaussian_param = [51, 50], Median_param = 11, blur_type= 'Gaussian', use_cuda = 1):
    ########################
    # Generate blurred images as the baseline

    # Parameters:
    # -------------
    # input_img: the original input image
    # img_label: the classification target that you want to visualize (img_label=-1 means the top 1 classification label)
    # model: the model that you want to visualize
    # resize_shape: the input size for the given model
    # Gaussian_param: parameters for Gaussian blur
    # Median_param: parameters for median blur
    # blur_type: Gaussian blur or median blur or mixed blur
    # use_cuda: use gpu (1) or not (0)
    ####################################################

    original_img = cv2.imread(input_img, 1)
    original_img = cv2.resize(original_img, resize_shape)
    img = np.float32(original_img) / 255

    if blur_type =='Gaussian':   # Gaussian blur
        Kernelsize = Gaussian_param[0]
        SigmaX = Gaussian_param[1]
        blurred_img = cv2.GaussianBlur(img, (Kernelsize, Kernelsize), SigmaX)

    elif blur_type == 'Black':
        blurred_img = img * 0

    elif blur_type == 'Median': # Median blur
        Kernelsize_M = Median_param
        blurred_img = np.float32(cv2.medianBlur(original_img, Kernelsize_M)) / 255

    elif blur_type == 'Mixed': # Mixed blur
        Kernelsize = Gaussian_param[0]
        SigmaX = Gaussian_param[1]
        blurred_img1 = cv2.GaussianBlur(img, (Kernelsize, Kernelsize), SigmaX)
        Kernelsize_M = Median_param
        blurred_img2 = np.float32(cv2.medianBlur(original_img, Kernelsize_M)) / 255
        blurred_img = (blurred_img1 + blurred_img2) / 2

    return img, blurred_img



def Integrated_Mask(ups, img, blurred_img, model, category, max_iterations = 15, integ_iter = 20,
                    tv_beta=2, l1_coeff = 0.01*300, tv_coeff = 0.2*300, size_init = 112, use_cuda =1):
    ########################
    # Obtaining perturbation mask using integrated gradient descent to find the smallest and smoothest area that maximally decrease the
    # output of a deep model

    # Parameters:
    # -------------
    # ups: upsampling factor
    # img: the original input image
    # blurred_img: the baseline for the input image
    # model: the model that you want to visualize
    # category: the classification target that you want to visualize (category=-1 means the top 1 classification label)
    # max_iterations: the max iterations for the integrated gradient descent
    # integ_iter: how many points you want to use when computing the integrated gradients
    # tv_beta: which norm you want to use for the total variation term
    # l1_coeff: parameter for the L1 norm
    # tv_coeff: parameter for the total variation term
    # size_init: the resolution of the mask that you want to generate
    # use_cuda: use gpu (1) or not (0)
    ####################################################

    # preprocess the input image and the baseline (low probability) image
    img = preprocess_image(img, use_cuda, require_grad=False)
    blurred_img = preprocess_image(blurred_img, use_cuda, require_grad=False)
    resize_size = img.data.shape
    resize_wh = (img.data.shape[2], img.data.shape[3])

    # initialize the mask
    mask_init = np.ones((int(resize_wh[0]/ups), int(resize_wh[1]/ups)), dtype=np.float32)
    mask = numpy_to_torch(mask_init, use_cuda, requires_grad=True)

    # upsampler
    if use_cuda:
        upsample = torch.nn.UpsamplingBilinear2d(size=resize_wh).cuda()
    else:
        upsample = torch.nn.UpsamplingBilinear2d(size=resize_wh)

    # You can choose any optimizer
    # The optimizer doesn't matter, because we don't need optimizer.step(), we just use it to compute the gradient
    optimizer = torch.optim.Adam([mask], lr=0.1)

    # containers for curve metrics
    curve1 = np.array([])
    curve2 = np.array([])
    curvetop = np.array([])
    curve_total = np.array([])

    # Integrated gradient descent

    # hyperparams
    alpha = 0.0001
    beta = 0.2

    for i in range(max_iterations):

        upsampled_mask = upsample(mask)
        upsampled_mask = upsampled_mask.expand(1, 3, upsampled_mask.size(2), upsampled_mask.size(3))

        # the l1 term and the total variation term
        loss1 = l1_coeff * torch.mean(torch.abs(1 - mask)) + tv_coeff * tv_norm(mask, tv_beta)
        loss_all = loss1.clone()

        # compute the perturbed image
        perturbated_input_base = img.mul(upsampled_mask) + blurred_img.mul(1 - upsampled_mask)

        loss2_ori = torch.nn.Softmax(dim=1)(model(perturbated_input_base))[0, category] # masking loss (no integrated)

        loss_ori = loss1 + loss2_ori
        if i==0:
            if use_cuda:
                curve1 = np.append(curve1, loss1.data.cpu().numpy())
                curve2 = np.append(curve2, loss2_ori.data.cpu().numpy())
                curvetop = np.append(curvetop, loss2_ori.data.cpu().numpy())
                curve_total = np.append(curve_total, loss_ori.data.cpu().numpy())
            else:
                curve1 = np.append(curve1, loss1.data.numpy())
                curve2 = np.append(curve2, loss2_ori.data.numpy())
                curvetop = np.append(curvetop, loss2_ori.data.numpy())
                curve_total = np.append(curve_total, loss_ori.data.numpy())
        if use_cuda:
            loss_oridata = loss_ori.data.cpu().numpy()
        else:
            loss_oridata = loss_ori.data.numpy()

        # calculate integrated gradient for next descent step
        for inte_i in range(integ_iter):

            # Use the mask to perturbated the input image.
            integ_mask = 0.0 + ((inte_i + 1.0) / integ_iter) * upsampled_mask
            perturbated_input_integ = img.mul(integ_mask) + blurred_img.mul(1 - integ_mask)

            # add noise
            noise = np.zeros((resize_wh[0], resize_wh[1], 3), dtype=np.float32)
            noise = noise + cv2.randn(noise, 0, 0.2)
            noise = numpy_to_torch(noise, use_cuda, requires_grad=False)
            perturbated_input = perturbated_input_integ + noise

            outputs = torch.nn.Softmax(dim=1)(model(perturbated_input))
            loss2 = outputs[0, category]
            loss_all = loss_all + loss2/20.0

        # compute the integrated gradients for the given target,
        # and compute the gradient for the l1 term and the total variation term
        optimizer.zero_grad()
        loss_all.backward()
        whole_grad = mask.grad.data.clone() # integrated gradient

        # LINE SEARCH with revised Armijo condition

        step = 200.0 # upper limit of step size
        MaskClone = mask.data.clone()
        MaskClone -= step * whole_grad
        MaskClone = Variable(MaskClone, requires_grad=False)
        MaskClone.data.clamp_(0, 1) # clamp the value of mask in [0,1]
        mask_LS = upsample(MaskClone)   # Here the direction is the whole_grad
        Img_LS = img.mul(mask_LS) + blurred_img.mul(1 - mask_LS)
        outputsLS = torch.nn.Softmax(dim=1)(model(Img_LS))
        loss_LS = l1_coeff * torch.mean(torch.abs(1 - MaskClone)) + tv_coeff * tv_norm(MaskClone, tv_beta) + outputsLS[0, category]

        if use_cuda:
            loss_LSdata = loss_LS.data.cpu().numpy()
        else:
            loss_LSdata = loss_LS.data.numpy()
        new_condition = whole_grad ** 2  # Here the direction is the whole_grad
        new_condition = new_condition.sum()
        new_condition = alpha * step * new_condition

        # finding best step size using backtracking line search
        while loss_LSdata > loss_oridata - new_condition.cpu().numpy():
            step *= beta
            MaskClone = mask.data.clone()
            MaskClone -= step * whole_grad
            MaskClone = Variable(MaskClone, requires_grad=False)
            MaskClone.data.clamp_(0, 1)
            mask_LS = upsample(MaskClone)
            Img_LS = img.mul(mask_LS) + blurred_img.mul(1 - mask_LS)
            outputsLS = torch.nn.Softmax(dim=1)(model(Img_LS))
            loss_LS = l1_coeff * torch.mean(torch.abs(1 - MaskClone)) + tv_coeff * tv_norm(MaskClone, tv_beta) + outputsLS[0, category]
            if use_cuda:
                loss_LSdata = loss_LS.data.cpu().numpy()
            else:
                loss_LSdata = loss_LS.data.numpy()

            new_condition = whole_grad ** 2  # Here the direction is the whole_grad
            new_condition = new_condition.sum()
            new_condition = alpha * step * new_condition

            if step<0.00001:
                break

        mask.data -= step * whole_grad # integrated gradient descent step - we have the updated mask at this point

        if use_cuda:
            curve1 = np.append(curve1, loss1.data.cpu().numpy())
            curve2 = np.append(curve2, loss2_ori.data.cpu().numpy()) # only masking loss
            curve_total = np.append(curve_total, loss_ori.data.cpu().numpy())
        else:
            curve1 = np.append(curve1, loss1.data.numpy())
            curve2 = np.append(curve2, loss2_ori.data.numpy())
            curve_total = np.append(curve_total, loss_ori.data.numpy())
        mask.data.clamp_(0, 1)
        if use_cuda:
            maskdata = mask.data.cpu().numpy()
        else:
            maskdata = mask.data.numpy()
        maskdata = np.squeeze(maskdata)
        maskdata, imgratio = topmaxPixel(maskdata, 40)
        maskdata = np.expand_dims(maskdata, axis=0)
        maskdata = np.expand_dims(maskdata, axis=0)
        if use_cuda:
            Masktop = torch.from_numpy(maskdata).cuda()
        else:
            Masktop = torch.from_numpy(maskdata)

        # Use the mask to perturb the input image.
        Masktop = Variable(Masktop, requires_grad=False)
        MasktopLS = upsample(Masktop)
        Img_topLS = img.mul(MasktopLS) + blurred_img.mul(1 - MasktopLS)
        outputstopLS = torch.nn.Softmax(dim=1)(model(Img_topLS))
        loss_top1 = l1_coeff * torch.mean(torch.abs(1 - Masktop)) + tv_coeff * tv_norm(Masktop, tv_beta)
        loss_top2 = outputstopLS[0, category]
        if use_cuda:
            curvetop = np.append(curvetop, loss_top2.data.cpu().numpy())
        else:
            curvetop = np.append(curvetop, loss_top2.data.numpy())


        if max_iterations > 3:

            if i == int(max_iterations / 2):
                if np.abs(curve2[0] - curve2[i]) <= 0.001:
                    l1_coeff = l1_coeff / 10

            elif i == int(max_iterations / 1.25):
                if np.abs(curve2[0] - curve2[i]) <= 0.01:
                    l1_coeff = l1_coeff / 5

    #######################################################################################

    upsampled_mask = upsample(mask)
    if use_cuda:
        mask = mask.data.cpu().numpy().copy()
    else:
        mask = mask.data.numpy().copy()

    return mask, upsampled_mask




def Deletion_Insertion_Comb_withOverlay(max_patches, mask, model, output_path, img_ori, blurred_img_ori, category, use_cuda=1, blur_mask=0, outputfig = 1):
    ########################
    # Compute the deletion and insertion scores
    #
    # parameters:
    # max_patches: number of literals in a root conjunction
    # mask: the generated mask
    # model: the model that you want to visualize
    # output_path: where to save the results
    # img_ori: the original image
    # blurred_img_ori: the baseline image
    # category: the classification target that you want to visualize (category=-1 means the top 1 classification label)
    # use_cuda: use gpu (1) or not (0)
    # blur_mask: blur the mask or not
    # outputfig: save figure or not
    ####################################################

    if blur_mask: # invert mask, blur and re-invert
        mask = (mask - np.min(mask)) / np.max(mask)
        mask = 1 - mask
        mask = cv2.GaussianBlur(mask, (51, 51), 50)
        mask = 1-mask

    blurred_insert = blurred_img_ori.copy()
    blurred_insert = preprocess_image(blurred_insert, use_cuda, require_grad=False)
    img = preprocess_image(img_ori, use_cuda, require_grad=False)
    blurred_img = preprocess_image(blurred_img_ori, use_cuda, require_grad=False)
    resize_wh = (img.data.shape[2], img.data.shape[3])
    if use_cuda:
        upsample = torch.nn.UpsamplingBilinear2d(size=resize_wh).cuda()
    else:
        upsample = torch.nn.UpsamplingBilinear2d(size=resize_wh)

    # containers to store curve metrics
    del_curve = np.array([])
    insert_curve = np.array([])
    xtick = np.arange(0, max_patches, 1)
    xnum = xtick.shape[0]
    xtick = xtick.shape[0]+ 10

    # get the ground truth label for the given category
    f_groundtruth = open('./GroundTruth1000.txt')
    line_i = f_groundtruth.readlines()[category]
    f_groundtruth.close()

    # initialize insertion and deletion masks
    insertion_maskdata = np.zeros(mask.shape)
    deletion_maskdata = np.ones(mask.shape)

    showimg_buffer = [] # buffer to store figures - we save them only if target_insertion_prob is achieved

    maskdata = mask.copy()
    maskdata = maskdata.astype(np.float32)
    if use_cuda:
        Masktop = torch.from_numpy(maskdata).cuda()
    else:
        Masktop = torch.from_numpy(maskdata)

    # Use the mask to perturb the input image - deletion mask.
    Masktop = Variable(Masktop, requires_grad=False)
    MasktopLS = upsample(Masktop)
    Img_topLS = img.mul(MasktopLS) + blurred_img.mul(1 - MasktopLS) # perturbed image

    outputstopLS = torch.nn.Softmax(dim=1)(model(Img_topLS)) # all probabilities
    deletion_loss = outputstopLS[0, category].data.cpu().numpy().copy() # probability of class under consideration
    del_mask = MasktopLS.clone()
    del_curve = np.append(del_curve, deletion_loss)

    # insertion mask
    maskdata = mask.copy()
    maskdata = np.subtract(1, maskdata).astype(np.float32)
    if use_cuda:
        Masktop = torch.from_numpy(maskdata).cuda()
    else:
        Masktop = torch.from_numpy(maskdata)
    Masktop = Variable(Masktop, requires_grad=False)
    MasktopLS = upsample(Masktop)
    Img_topLS = img.mul(MasktopLS) + \
                blurred_insert.mul(1 - MasktopLS)
    outputstopLS = torch.nn.Softmax(dim=1)(model(Img_topLS))
    insertion_loss = outputstopLS[0, category].data.cpu().numpy().copy()
    ins_mask = MasktopLS.clone()
    insert_curve = np.append(insert_curve, insertion_loss)

    # store result images
    if outputfig == 1:
        deletion_img = save_new(del_mask, img_ori * 255, blurred_img_ori)
        insertion_img = save_new(ins_mask, img_ori * 255, blurred_img_ori)
        showimg_buffer.append((deletion_img, insertion_img, del_curve, insert_curve, output_path, xtick, line_i))

    # round decimals
    deletion_loss = np.around(deletion_loss, decimals=3)
    insertion_loss = np.around(insertion_loss, decimals=3)

    return deletion_loss, insertion_loss, del_mask, showimg_buffer



def save_new(mask, img, blurred):
    ########################
    # generate the perturbed image for saving as result
    #
    # parameters:
    # mask: the generated mask
    # img: the original image
    # blurred: the baseline image
    ####################################################
    mask = mask.cpu().data.numpy()[0]
    mask = np.transpose(mask, (1, 2, 0))
    img = np.float32(img) / 255
    perturbated = np.multiply(mask, img) + np.multiply(1-mask, blurred)
    perturbated = cv2.cvtColor(perturbated, cv2.COLOR_BGR2RGB)
    return perturbated



def showimage(del_img, insert_img, del_curve, insert_curve, target_path, xtick, title):
    ########################
    # generate the result frame used for videos
    #
    # parameters:
    # del_img: the deletion image
    # insert_img: the insertion image
    # del_curve: the deletion curve
    # insert_curve: the insertion curve
    # target_path: where to save the results
    # xtick: xtick
    # title: title
    ####################################################
    pylab.rcParams['figure.figsize'] = (10, 10)
    f, ax = plt.subplots(2,2)
    f.suptitle('Category ' + title, y=0.04, fontsize=13)
    f.tight_layout()
    plt.subplots_adjust(left=0.005, bottom=0.1, right=0.98, top=0.93,
                        wspace=0.05, hspace=0.25)
    ax[0,0].imshow(del_img)
    ax[0,0].set_xticks([])
    ax[0,0].set_yticks([])
    ax[0,0].set_title("Deletion", fontsize=13)
    ax[1,0].imshow(insert_img)
    ax[1,0].set_xticks([])
    ax[1,0].set_yticks([])
    ax[1,0].set_title("Insertion", fontsize=13)
    ax[0,1].plot(del_curve,'r*-')
    ax[0,1].set_xlabel('number of blocks')
    ax[0,1].set_ylabel('classification confidence')
    ax[0,1].legend(['Deletion'])
    ax[0,1].set_xticks(range(0, xtick, 10))
    ax[0, 1].set_yticks(np.arange(0, 1.1, 0.1))
    ax[1,1].plot(insert_curve, 'b*-')
    ax[1, 1].set_xlabel('number of blocks')
    ax[1,1].set_ylabel('classification confidence')
    ax[1,1].legend(['Insertion'])
    ax[1, 1].set_xticks(range(0, xtick, 10))
    ax[1, 1].set_yticks(np.arange(0, 1.1, 0.1))

    plt.savefig(target_path + 'video'+ str(insert_curve.shape[0])+ '.jpg')
    plt.close()
