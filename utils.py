### Utility functions

import torch
from torch.autograd import Variable
from torchvision import models
import cv2
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from skimage import filters
from math import exp
import itertools
import math
import imageio
import os

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor


def preprocess_image(img, use_cuda=1, require_grad = False):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    preprocessed_img = img.copy()[:, :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    if use_cuda:
        preprocessed_img_tensor = torch.from_numpy(preprocessed_img).cuda()
    else:
        preprocessed_img_tensor = torch.from_numpy(preprocessed_img)
    preprocessed_img_tensor.unsqueeze_(0)
    return Variable(preprocessed_img_tensor, requires_grad=require_grad)


def numpy_to_torch(img, use_cuda=1, requires_grad=False):
    if len(img.shape) < 3:
        output = np.float32([img])
    else:
        output = np.transpose(img, (2, 0, 1))
    output = torch.from_numpy(output)
    if use_cuda:
        output = output.cuda()
    output.unsqueeze_(0)
    v = Variable(output, requires_grad=requires_grad)
    return v

def tv_norm(input, tv_beta):
    img = input[0, 0, :]
    row_grad = torch.mean(torch.abs((img[:-1, :] - img[1:, :])).pow(tv_beta))
    col_grad = torch.mean(torch.abs((img[:, :-1] - img[:, 1:])).pow(tv_beta))
    return row_grad + col_grad


def load_model_new(use_cuda = 1, model_name = 'resnet50'):
    if model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
    elif model_name == 'vgg19':
        model = models.vgg19(pretrained=True)
    model.eval()
    if use_cuda:
        model.cuda()
    for p in model.parameters():
        p.requires_grad = False
    return model


def get_topn_categories_probabilities_pairs(img, model, n, use_cuda=use_cuda):
    img = preprocess_image(img, use_cuda, require_grad=False)
    target = torch.nn.Softmax(dim=1)(model(img))
    target = target.squeeze()
    if use_cuda:
        target = target.data.cpu().numpy()
    else:
        target = target.data.numpy()
    topn_categories = np.argsort(-target)[:n]
    #print('top3_categories: ', top3_categories)
    topn_probabilities = [target[x] for x in topn_categories]
    #print('top3_probabilities: ', top3_probabilities)
    top3_cp = zip(topn_categories, topn_probabilities)
    return top3_cp

def topmaxPixel(HattMap, thre_num):
    ii = np.unravel_index(np.argsort(HattMap.ravel())[: thre_num], HattMap.shape)
    OutHattMap = HattMap*0
    OutHattMap[ii] = 1
    img_ratio = np.sum(OutHattMap) / OutHattMap.size
    OutHattMap = 1 - OutHattMap
    return OutHattMap, img_ratio

def add_topMaskPixel(current_mask, original_mask):
    ii = np.unravel_index(np.argsort(original_mask.ravel())[0], original_mask.shape)
    current_mask[ii] = 0
    original_mask[ii] = 1
    img_ratio = np.sum(current_mask) / current_mask.size
    return current_mask, original_mask, img_ratio

def filter_topmaxPixel(HattMap, thre_num):
    ii = np.unravel_index(np.argsort(HattMap.ravel())[: thre_num], HattMap.shape)
    OutHattMap = np.ones(HattMap.shape)
    OutHattMap[ii] = HattMap[ii]
    img_ratio = np.sum(OutHattMap) / OutHattMap.size
    return OutHattMap, img_ratio

def write_video(inputpath, outputname, img_num, fps = 10):
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    videoWriter = cv2.VideoWriter(outputname, fourcc, fps, (1000, 1000))
    for i in range(img_num):

        img_no = i+1
        #print(inputpath+'video'+str(img_no) +'.jpg')
        img12 = cv2.imread(inputpath+'video'+str(img_no) +'.jpg',1)
        videoWriter.write(img12)
    videoWriter.release()


def save_perturbation_heatmap(output_path, mask, img, blurred, blur_mask=0):
    mask = mask.cpu().data.numpy()[0]
    mask = np.transpose(mask, (1, 2, 0))
    mask = (mask - np.min(mask))
    if not (np.max(mask) == np.min(mask)):
        mask = mask / (np.max(mask)-np.min(mask))
    mask = 1 - mask

    if blur_mask:
        mask = cv2.GaussianBlur(mask, (11, 11), 10)
        mask = np.expand_dims(mask, axis=2)

    heatmap = np.uint8(255 * mask)
    heatmap = np.float32(heatmap) / 255
    img = np.float32(img) / 255
    perturbated = np.multiply(1 - mask, img) + np.multiply(mask, blurred)
    perturb = 1 * (1 - mask ** 0.8) * img + (mask ** 0.8)* heatmap
    cv2.imwrite(output_path + "original.png", np.uint8(255 * img))
    cv2.imwrite(output_path + "heatmap.png", np.uint8(255 * heatmap))
    cv2.imwrite(output_path + "imposed_heatmap.png", np.uint8(255 * perturb))
    cv2.imwrite(output_path + "blurred.png", np.uint8(255 * blurred))


def create_minsufexp_gif(imagefolder):
    images = []
    dirs = os.listdir(imagefolder)
    for dir in dirs:
        dirpath = imagefolder + '/' + dir
        if os.path.isdir(dirpath):
            #print('d1: ', dirpath)
            files = os.listdir(dirpath)
            for f in files:
                if 'InsertionImg' in f:
                    imagepath = dirpath + '/' + f
                    im = imageio.imread(imagepath)
                    images.append(im)
                    break
    if len(images) > 0:
        imageio.mimsave(imagefolder + 'dnf_gif.gif', images, duration=2.0)
