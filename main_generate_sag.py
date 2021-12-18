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
from search import *
from diverse_subset_selection import *
from patch_deletion_tree import *




if __name__ == '__main__':

    # HYPERPARAMS
    ups = 30
    prob_thresh = 0.9 # note that this is prob factor. So we are considering 0.9 * full_image_probability
    numCategories = 1
    node_prob_thresh = 40 # minimum score threshold to expand a node in the sag

    beam_width = 3 # suggested values [3,5,10,15]
    max_num_roots = 10 # upper limit on number of roots obtained via search - suggested values [10,20,30]
    overlap_thresh = 1 # number of patches allowed to overlap in roots - suggested values [0,1,2]
    numSuccessors = 15 # should be greater or equal to beam_width - 'q' hyperparam in the paper
    num_roots_sag = 3 # max number of roots to be displayed in the sag

    input_folder = 'Images'

    maxRootSize = 49 # max number of patches allowed for a root

    # enable cuda
    use_cuda = 0
    if torch.cuda.is_available():
        use_cuda = 1
    # load DNN model
    model = load_model_new(use_cuda=use_cuda, model_name='vgg19')

    images_no_roots_found = 0

    # traverse input image folder
    input_path = './'+input_folder+'/'
    dirs = os.listdir(input_path)
    # dirs = [folder]
    for d in dirs:
        if not os.path.isdir(input_path + d):
            continue
        files = os.listdir(input_path + d)

        total_files = len(files)
        file_counter = 0

        output_path = './Results/' + d + '/'
        if not os.path.isdir(output_path):
            os.makedirs(output_path)

        pre_existing_result_dirs = os.listdir(output_path)

        # start search
        success_counter = 0
        explained_images = []
        img_label = -1

        for imgname in files:

            # current support for jpg and png image formats
            if imgname.endswith('JPEG') or imgname.endswith('jpg') or imgname.endswith('png'):
                input_img = input_path + d + '/' + imgname
                print('imgname:', imgname)
                imgprefix = imgname.split('.')[0]
                img_label = -1

                # check if this file is already processes (results exist)
                for existing_dir in pre_existing_result_dirs:
                    if imgprefix in existing_dir:
                        print('skipping')
                        continue


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
                                             max_iterations=2,
                                             integ_iter=20,
                                             tv_beta=2,
                                             l1_coeff=0.01 * 100,
                                             tv_coeff=0.2 * 100,
                                             size_init=28,
                                             use_cuda=use_cuda)

                    # get all DISTINCT roots found via beam search
                    roots_mp = beamSearch_topKSuccessors_roots(mask, beam_width, numSuccessors, img, blurred_img, model, category, prob_thresh, probability, max_num_roots, maxRootSize, use_cuda=use_cuda)

                    numRoots = len(roots_mp)
                    print('numRoots_all = ', numRoots)
                    # get maximal set of non-overlapping roots
                    maximal_Overlap_mp = []
                    if numRoots > 0:
                        maximal_Overlap_mp = maximal_overlapThresh_set(roots_mp, overlap_thresh)
                    else:
                        images_no_roots_found += 1
                    numRoots_Overlap = len(maximal_Overlap_mp)
                    print('numRoots_Overlap = ', numRoots_Overlap)

                    # prune number of roots to be shown in the sag
                    if numRoots_Overlap > num_roots_sag:
                        maximal_Overlap_mp = maximal_Overlap_mp[:num_roots_sag]
                        numRoots_Overlap = num_roots_sag

                    # end time stamp
                    end_time = time.time()
                    # time taken
                    time_taken = end_time - start_time
                    time_taken = np.around(time_taken, decimals=3)

                    # deletion insertion on filtered set of masks - just  to generate result figures
                    dnf = ""
                    for mask, ins_prob, rel_prob in maximal_Overlap_mp:
                        output_file_videoimgs = imgprefix + '_'
                        delloss_top2, insloss_top2, minsufexpmask_upsampled, showimg_buffer = Deletion_Insertion_Comb_withOverlay(
                                                                                               maxRootSize,
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
                        output_path_count = output_path_img + '_insprob_' + str(ins_prob) + '_relprob_' + str(rel_prob) + "/"
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
                        insertion_img = cv2.cvtColor(insertion_img, cv2.COLOR_RGB2BGR)
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
                    current_patchImages_path = output_path_img + 'SAG_PatchImages_'+str(numRoots_Overlap)+'roots'
                    if not os.path.isdir(current_patchImages_path):
                        os.makedirs(current_patchImages_path)

                    # create and save grid image
                    gridimage(img_ori, output_path_img + 'gridimage.png')

                    # get set of conjunctions from DNF expression
                    conjuncts = get_conjuncts_set(dnf)

                    # prune conjunctions to required number of roots in SAG
                    if len(conjuncts) > numRoots_Overlap:
                        conjuncts = conjuncts[:numRoots_Overlap]

                    # build tree
                    sag_tree = build_tree(conjuncts, ups, img_ori, blurred_img, model, category, current_patchImages_path, node_prob_thresh, probability)

                    # book-keeping and save generated result files
                    f = output_path_img + 'SAG_'+str(numRoots_Overlap)+'roots.dot'
                    sag_tree.write(f)
                    img = pgv.AGraph(f)
                    img.layout(prog='dot')
                    f2 = output_path_img + 'SAG_dag_'+str(numRoots_Overlap)+'roots.png'
                    img.draw(f2)
                    img.close()
                    f3 = output_path_img + 'SAG_final_'+str(numRoots_Overlap)+'roots.png'
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

                file_counter += 1
                print('files processed: {}/{}'.format(file_counter, total_files))
                print('images with no roots found: ', images_no_roots_found)
