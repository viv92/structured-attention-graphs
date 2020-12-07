### author: Vivswan Shitole

import  numpy as np
import itertools
import random
import math
import os
import time
import scipy.io as scio
import datetime
import re
import matplotlib.pyplot as plt
import pylab
import csv
from skimage import transform, filters
from textwrap import wrap
import cv2
import sys
from PIL import Image

from utils import *
from get_perturbation_mask import *
from combinatorial_search import *
from diverse_subset_selection import *
from patch_deletion_tree import *




if __name__ == '__main__':

    # HYPERPARAMS
    ups = 30 # recommended values: [1 5 10 20]
    total_mask_pixels = 20 # recommended values: [30 20 15 10]. Higher value => more candidate mask pixels, but increase in compute cost
    max_patches = 4 # recommended values: [3-8]. Number of literals in a root conjunction
    prob_thresh = 0.9 # minimum probability threshold for root conjunctions
    numRoots = 3 # recommended values: [3-6]. Number of root conjunctions
    numCategories = 1 # generate SAG for top "N" predicted classes: useful for generating SAG explanations of wrong predictions.
    node_prob_thresh = 40 # lower bound on confidence score for expansion of node in SAG

    # traverse input image folder
    input_path = './Images/'
    dirs = os.listdir(input_path)
    for d in dirs:
        if not os.path.isdir(input_path + d):
            continue
        files = os.listdir(input_path + d)

        output_path = './Results/' + d + '/'
        if not os.path.isdir(output_path):
            os.makedirs(output_path)

        # enable cuda
        use_cuda = 0
        if torch.cuda.is_available():
            use_cuda = 1

        # load DNN model
        model = load_model_new(use_cuda=use_cuda, model_name='vgg19')

        # start
        for imgname in files:

            # current support for jpg and png image formats
            if imgname.endswith('JPEG') or imgname.endswith('jpg') or imgname.endswith('png'):
                input_img = input_path + d + '/' + imgname
                print('imgname:', imgname)
                imgprefix = imgname.split('.')[0]
                img_label = -1

                # start time stamp
                start_time = time.time()

                # get low probability blurred image
                img, blurred_img = Get_blurred_img(
                                    input_img,
                                    img_label,
                                    model,
                                    resize_shape=(224, 224),
                                    Gaussian_param=[51, 50],
                                    Median_param=11,
                                    blur_type='Black',
                                    use_cuda=use_cuda)

                # get top "numCategories" predicted categories with their probabilities
                top_cp = get_topn_categories_probabilities_pairs(img, model, numCategories, use_cuda=use_cuda)

                for category, probability in top_cp:

                    # get the ground truth label for the given category
                    f_groundtruth = open('./GroundTruth1000.txt')
                    category_name = f_groundtruth.readlines()[category]
                    category_name = category_name[:-2]
                    f_groundtruth.close()

                    # get perturbation mask
                    mask, upsampled_mask = Integrated_Mask(
                                             ups,
                                             img,
                                             blurred_img,
                                             model,
                                             category,
                                             max_iterations=15,
                                             integ_iter=20,
                                             tv_beta=2,
                                             l1_coeff=0.01 * 100,
                                             tv_coeff=0.2 * 100,
                                             size_init=28,
                                             use_cuda=use_cuda)

                    # filter top pixels of mask
                    mask, imgratio = filter_topmaxPixel(mask, total_mask_pixels) # note: the mask obtained here is not binary

                    # get all combinations of masks - all masks obtained here are binary masks
                    all_masks = topCombinations(mask, total_mask_pixels, max_patches)

                    # get all mask-probability pairs
                    all_mp = get_all_mp(all_masks, img, blurred_img, model, category, prob_thresh, use_cuda=use_cuda)
                    if len(all_mp) == 0:
                        continue

                    # sort masks for diversity
                    disparity_min_mp = set_filtering(all_mp)

                    # pick only the top "numRoots" diverse masks
                    # tmp_len = len(disparity_min_mp)
                    # if tmp_len > numRoots:
                    #     tmp_len = numRoots
                    # disparity_min_mp = disparity_min_mp[:tmp_len]

                    #ps = [x[1] for x in disparity_min_mp]

                    # end time stamp
                    end_time = time.time()
                    # time taken
                    time_taken = end_time - start_time
                    time_taken = np.around(time_taken, decimals=3)

                    # deletion insertion on filtered set of masks - just  to generate result figures
                    dnf = ""
                    for mask, ins_prob in disparity_min_mp:
                        output_file_videoimgs = imgprefix + '_'
                        delloss_top2, insloss_top2, minsufexpmask_upsampled, showimg_buffer = Deletion_Insertion_Comb_withOverlay(
                                                                                               max_patches,
                                                                                               mask,
                                                                                               model,
                                                                                               output_file_videoimgs,
                                                                                               img,
                                                                                               blurred_img,
                                                                                               category=category,
                                                                                               use_cuda=use_cuda,
                                                                                               blur_mask=0,
                                                                                               outputfig=1)


                        output_path_img = output_path + imgprefix + "_timetaken_" + str(time_taken) + "_category_" + str(category_name) + "_probthresh_" + str(prob_thresh) + "/"
                        output_file_perturbation_heatmaps = output_path_img + 'perturbation_'
                        output_path_count = output_path_img + '_insprob_' + str(ins_prob) + "/"
                        outvideo_path = output_path_count + 'VIDEO/'

                        # create MDNF expression
                        patch_boolean_list = get_patch_boolean(mask)
                        conjunction = ""
                        for b in patch_boolean_list:
                            conjunction += ' & P'+str(b)
                        conjunction = conjunction[3:]
                        dnf += ' | '+conjunction

                        # save obtained sample

                        if not os.path.isdir(outvideo_path):
                            os.makedirs(outvideo_path)

                        # save perturbation heatmaps
                        save_perturbation_heatmap(output_file_perturbation_heatmaps, upsampled_mask, img * 255, blurred_img, blur_mask=0)

                        # unpack result images
                        for item in showimg_buffer:
                            deletion_img, insertion_img, del_curve, insert_curve, out_pathx, xtick, line_i = item
                            out_pathx = outvideo_path + out_pathx
                            showimage(deletion_img, insertion_img, del_curve, insert_curve, out_pathx, xtick, line_i)

                        # save perturbation minsufexp heatmaps
                        output_file_perturbationminsufexp_heatmaps = output_path_count + imgprefix + '_perturbationminsufexp_'
                        save_perturbation_heatmap(output_file_perturbationminsufexp_heatmaps, minsufexpmask_upsampled, img * 255, blurred_img, blur_mask=0)

                        # write insertion image with probability score
                        insertion_img = cv2.cvtColor(insertion_img, cv2.COLOR_RGB2BGR)
                        # add score footer
                        prob_score = np.around(ins_prob, decimals=3)
                        footer = "p: " + str(prob_score)
                        # font
                        font = cv2.FONT_HERSHEY_DUPLEX
                        # org
                        org = (50, 200)
                        # fontScale
                        fontScale = 0.8
                        # Blue color in BGR
                        color = (255, 255, 255)
                        # Line thickness of 2 px
                        thickness = 2
                        insertion_img = cv2.putText(insertion_img, footer, org, font, fontScale, color, thickness, cv2.LINE_AA)
                        cv2.imwrite(output_path_count + imgprefix + 'InsertionImg.png', insertion_img * 255)

                        # write root conjunctions
                        conjunction_file = open(output_path_count + imgprefix + 'conjunction.txt', 'w+')
                        conjunction_file.write(conjunction)
                        conjunction_file.close()

                    # write MDNF expression
                    dnf_file = open(output_path_img + 'dnf.txt', 'w+')
                    dnf = dnf[3:]
                    dnf_file.write(dnf)
                    dnf_file.close()

                    # save SAG roots as a GIF
                    create_minsufexp_gif(output_path_img)

                    ## build patch deletion tree ##

                    # load original image
                    img_ori = cv2.imread(output_path_img + 'perturbation_original.png')
                    img_ori = cv2.cvtColor(img_ori, cv2.COLOR_RGB2BGR)

                    # create patchImages folder if not exists
                    current_patchImages_path = output_path_img + 'SAG_PatchImages_'+str(numRoots)+'roots'
                    if not os.path.isdir(current_patchImages_path):
                        os.makedirs(current_patchImages_path)

                    # create and save grid image
                    gridimage(img_ori, output_path_img + 'gridimage.png')

                    # get set of conjunctions from DNF expression
                    conjuncts = get_conjuncts_set(dnf)

                    # prune conjunctions to required number of roots in SAG
                    if len(conjuncts) > numRoots:
                        conjuncts = conjuncts[:numRoots]

                    # build tree
                    sag_tree = build_tree(conjuncts, ups, img_ori, blurred_img, model, category, current_patchImages_path, node_prob_thresh)

                    # book-keeping and save generated result files
                    f = output_path_img + 'SAG_'+str(numRoots)+'roots.dot'
                    sag_tree.write(f)
                    img = pgv.AGraph(f)
                    img.layout(prog='dot')
                    f2 = output_path_img + 'SAG_dag_'+str(numRoots)+'roots.png'
                    img.draw(f2)
                    img.close()
                    f3 = output_path_img + 'SAG_final_'+str(numRoots)+'roots.png'
                    img_tree = cv2.imread(f2)
                    h,w,c = img_tree.shape
                    hp = h
                    wp = hp
                    off = 50
                    h1 = hp - (off*2)
                    w1 = h1
                    dim = (h1,w1)
                    tmp_img_ori = deepcopy(img_ori)
                    tmp_img_ori = cv2.resize(tmp_img_ori, dim, interpolation=cv2.INTER_AREA)
                    tmp_img_ori = cv2.cvtColor(tmp_img_ori, cv2.COLOR_RGB2BGR)
                    img_ori_padded = np.ones((hp,wp,c)) * 0
                    img_ori_padded[off:off+h1, off:off+w1, :] = tmp_img_ori

                    # concatenate image and explanation tree
                    img_sag_final = np.concatenate((img_ori_padded, img_tree), axis=1)
                    # save generated sag image
                    cv2.imwrite(f3, img_sag_final)
