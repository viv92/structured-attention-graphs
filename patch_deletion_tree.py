### All utility functions to build the patch deletion tree

import os
import sys
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import cv2
import numpy as np
from copy import deepcopy
import pygraphviz as pgv
import torch

from utils import *


def get_patch_boolean(mask):
    boolean = []
    z0, z1, h, w = mask.shape
    z0, z1, rows, cols = np.where(mask == 0)
    for i in range(len(rows)):
        patchname = rows[i]*h + cols[i]
        boolean.append(patchname)
    return boolean


def get_edge_mask_red(mask, canny_param, intensity, kernel_size):
    upsampled_mask_newPatch_edge = deepcopy(mask)
    upsampled_mask_newPatch_edge = np.uint8(upsampled_mask_newPatch_edge * 255)
    upsampled_mask_newPatch_edge = cv2.Canny(upsampled_mask_newPatch_edge, canny_param, canny_param)
    morphkernel = np.ones((25, 25), np.uint8)
    upsampled_mask_newPatch_edge = cv2.morphologyEx(upsampled_mask_newPatch_edge, cv2.MORPH_CLOSE, morphkernel)
    upsampled_mask_newPatch_edge = cv2.Canny(upsampled_mask_newPatch_edge, 500, 500)
    upsampled_mask_newPatch_edge = cv2.GaussianBlur(upsampled_mask_newPatch_edge, (kernel_size, kernel_size), kernel_size-1)
    upsampled_mask_newPatch_edge *= intensity
    upsampled_mask_newPatch_edge = np.expand_dims(upsampled_mask_newPatch_edge, axis=2)
    return upsampled_mask_newPatch_edge


def create_node_image(parent_chain, index, edgepatch, parent_prob, ups, img_ori, blurred_img_ori, model, category, current_patchImages_path):
    width, height, channels = img_ori.shape
    resize_wh = (width, height)

    use_cuda = 0
    if torch.cuda.is_available():
        use_cuda = 1
        upsample = torch.nn.UpsamplingBilinear2d(size=resize_wh).cuda()
    else:
        upsample = torch.nn.UpsamplingBilinear2d(size=resize_wh)

    mask_w = int(width/ups)
    mask_h = int(height/ups)
    mask_insertion = np.zeros((mask_w, mask_h))
    mask_edgePatch = np.zeros((mask_w, mask_h))
    wh_mask_oldPatches = np.zeros((mask_w, mask_h))
    wh_mask_newPatch = np.zeros((mask_w, mask_h))
    wh_mask_combined = np.zeros((mask_w, mask_h))

    w = int(mask_w/7)
    h = int(mask_h/7)

    # create edgepatch mask
    edgepatch_flag = edgepatch != ""
    if edgepatch_flag:
        patchnum = int(edgepatch[1:])
        row = int(patchnum/7)
        col = int(patchnum%7)
        mask_edgePatch[row][col] = 1.1

    # create image
    for i in range(0,index+1):
        patchnum = parent_chain[i][0]
        patchnum = int(patchnum[1:])
        row = int(patchnum/7)
        col = int(patchnum%7)
        # y = h*row
        # x = w*col
        if i == index:
            wh_mask_newPatch[row][col] = 1.5 # 2.5
        else:
            wh_mask_oldPatches[row][col] = 1.5 # 2.5
        wh_mask_combined[row][col] = 1.5 # 2
        mask_insertion[row][col] = 1

    # get mask insertion probability
    mask_insertion = np.expand_dims(mask_insertion, axis=0)
    mask_insertion = np.expand_dims(mask_insertion, axis=0)
    mask_insertion = mask_insertion.astype(np.float32)
    if use_cuda:
        mask_insertion = torch.from_numpy(mask_insertion).cuda()
    else:
        mask_insertion = torch.from_numpy(mask_insertion)
    mask_insertion = Variable(mask_insertion, requires_grad=False)
    upsampled_mask_insertion = upsample(mask_insertion)
    img_ori_copy = deepcopy(img_ori)
    img_ori_copy = cv2.cvtColor(img_ori_copy, cv2.COLOR_BGR2RGB)
    img_ori_copy = np.float32(img_ori_copy) / 255
    blurred_img_copy = deepcopy(blurred_img_ori)
    img_ori_copy = preprocess_image(img_ori_copy, use_cuda=use_cuda, require_grad=False)
    blurred_img_copy = preprocess_image(blurred_img_copy, use_cuda=use_cuda, require_grad=False)
    insertion_img = img_ori_copy.mul(upsampled_mask_insertion) + blurred_img_copy.mul(1-upsampled_mask_insertion)
    prob_vector = torch.nn.Softmax(dim=1)(model(insertion_img))
    if use_cuda:
        ins_prob = prob_vector[0, category].data.cpu().numpy()
    else:
        ins_prob = prob_vector[0, category].data.numpy()

    #create edge patch image
    if edgepatch_flag:
        mask_edgePatch = np.expand_dims(mask_edgePatch, axis=0)
        mask_edgePatch = np.expand_dims(mask_edgePatch, axis=0)
        # if use_cuda:
        #     mask_edgePatch = torch.from_numpy(mask_edgePatch).cuda()
        # else:
        mask_edgePatch = torch.from_numpy(mask_edgePatch)
        upsampled_mask_edgePatch = upsample(mask_edgePatch)
        upsampled_mask_edgePatch = upsampled_mask_edgePatch.data.numpy()
        upsampled_mask_edgePatch = upsampled_mask_edgePatch.squeeze(0)
        upsampled_mask_edgePatch = np.transpose(upsampled_mask_edgePatch, (1,2,0))
        prob_drop = parent_prob - (ins_prob*100)
        if prob_drop < 20:
            ksize = 3
            intensity = 2
        elif prob_drop < 60:
            ksize = 3
            intensity = 10
        else:
            ksize = 7
            intensity = 50
        upsampled_mask_edgePatch_edge = get_edge_mask_red(upsampled_mask_edgePatch, 75, intensity, ksize)

    ######################### NOW DO THE WHITE MASK VERSION #######################
    wh_mask_oldPatches = np.expand_dims(wh_mask_oldPatches, axis=0)
    wh_mask_oldPatches = np.expand_dims(wh_mask_oldPatches, axis=0)
    # if use_cuda:
    #     wh_mask_oldPatches = torch.from_numpy(wh_mask_oldPatches).cuda()
    # else:
    wh_mask_oldPatches = torch.from_numpy(wh_mask_oldPatches)
    wh_upsampled_mask_oldPatches = upsample(wh_mask_oldPatches)
    wh_upsampled_mask_oldPatches = wh_upsampled_mask_oldPatches.data.numpy()
    wh_upsampled_mask_oldPatches = wh_upsampled_mask_oldPatches.squeeze(0)
    wh_upsampled_mask_oldPatches = np.transpose(wh_upsampled_mask_oldPatches, (1,2,0))

    wh_mask_newPatch = np.expand_dims(wh_mask_newPatch, axis=0)
    wh_mask_newPatch = np.expand_dims(wh_mask_newPatch, axis=0)
    # if use_cuda:
    #     wh_mask_newPatch = torch.from_numpy(wh_mask_newPatch).cuda()
    # else:
    wh_mask_newPatch = torch.from_numpy(wh_mask_newPatch)
    wh_upsampled_mask_newPatch = upsample(wh_mask_newPatch)
    wh_upsampled_mask_newPatch = wh_upsampled_mask_newPatch.data.numpy()
    wh_upsampled_mask_newPatch = wh_upsampled_mask_newPatch.squeeze(0)
    wh_upsampled_mask_newPatch = np.transpose(wh_upsampled_mask_newPatch, (1,2,0))

    wh_mask_combined = np.expand_dims(wh_mask_combined, axis=0)
    wh_mask_combined = np.expand_dims(wh_mask_combined, axis=0)
    # if use_cuda:
    #     wh_mask_combined = torch.from_numpy(wh_mask_combined).cuda()
    # else:
    wh_mask_combined = torch.from_numpy(wh_mask_combined)
    wh_upsampled_mask_combined = upsample(wh_mask_combined)
    wh_upsampled_mask_combined = wh_upsampled_mask_combined.data.numpy()
    wh_upsampled_mask_combined = wh_upsampled_mask_combined.squeeze(0)
    wh_upsampled_mask_combined = np.transpose(wh_upsampled_mask_combined, (1,2,0))

    # img_mean = np.ones_like(img_white)
    # img_mean[0] *= int(0.485 * 255)
    # img_mean[1] *= int(0.456 * 255)
    # img_mean[2] *= int(0.406 * 255)

    # different image colors used
    img_black = np.ones_like(img_ori.shape) * 0
    img_white = np.ones_like(img_ori.shape) * 255
    img_red = np.zeros_like(img_ori)
    img_red[:,:,0] = 255
    img_transparent = img_ori * 0.1 + img_black * 0.9

    # create node image
    patch_image = (img_ori * wh_upsampled_mask_oldPatches) + (img_ori * wh_upsampled_mask_newPatch) + (img_transparent * (1-wh_upsampled_mask_combined))
    if edgepatch_flag:
        patch_image += img_red * upsampled_mask_edgePatch_edge

    #save image by uid and return path
    patch_img = np.zeros_like(patch_image)
    for j in range(0,3):
        patch_img[:,:,j] = patch_image[:,:,2-j]
    uid = parent_chain[index][1]
    patch_img_path = current_patchImages_path + '/' + str(uid) + '.png'
    cv2.imwrite(patch_img_path, patch_img)

    # if leaf image, save in leaf images folder - to be used for user study setup
    # if index == len(parent_chain)-1:
    #     patch_img_leaf_path = current_patchImages_path_usrstudy + '/' + str(uid) + '.png'
    #     cv2.imwrite(patch_img_leaf_path, patch_img)

    return patch_img_path, ins_prob



# creates and saves grid image
def gridimage(image, savepath):
    my_dpi=100.
    fig=plt.figure(figsize=(float(image.shape[0]/my_dpi), float(image.shape[1]/my_dpi)), dpi=my_dpi)
    ax=fig.add_subplot(111)
    # Remove whitespace from around the image
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    # Set the gridding interval: here we use the major tick interval
    myInterval=float(image.shape[0]/7)
    loc = plticker.MultipleLocator(base=myInterval)
    ax.xaxis.set_major_locator(loc)
    ax.yaxis.set_major_locator(loc)
    # Add the grid
    ax.grid(which='major', axis='both', linestyle='-')
    # Add the image
    ax.imshow(image)
    # Find number of gridsquares in x and y direction
    nx=abs(int(float(ax.get_xlim()[1]-ax.get_xlim()[0])/float(myInterval)))
    ny=abs(int(float(ax.get_ylim()[1]-ax.get_ylim()[0])/float(myInterval)))
    # Add some labels to the gridsquares
    for j in range(ny):
        y=myInterval/2+j*myInterval
        for i in range(nx):
            x=myInterval/2.+float(i)*myInterval
            ax.text(x,y,'{:d}'.format(i+j*nx),color='w',ha='center',va='center')
    # Save the figure
    fig.savefig(savepath, dpi=my_dpi)


# returns a list of conjuncts from dnf string
def get_conjuncts_set(dnfstring):
    conjunct_sets = []
    conjuncts = dnfstring.split(' | ')
    for index, conj in enumerate(conjuncts):
        conj = conj.replace('(','') #filter for brackets
        conj = conj.replace(')','')
        literals = conj.split(' & ')
        conjunct_sets.append(literals)
    return conjunct_sets


def get_subconjuncts(conj):
    n = len(conj)
    if n == 1:
        return [] # base case
    subconjlist = []
    if n > 3:
        n = 3
    for i in range(n):
        subconj = deepcopy(conj)
        subconj.remove(conj[i])
        subconjlist.append((subconj, conj[i]))
    return subconjlist


def get_node_bg_color(ins_prob, basecolor):
    if ins_prob > 0.9:
        color = basecolor + '1'
    elif ins_prob > 0.5:
        color = basecolor + '3'
    else:
        color = basecolor + '4'
    return color


# global uid counter
incremental_uid = 0

# function to create node and add it to the tree
def add_lattice_node(conj, g, edgepatch, parent_prob, ups, img_ori, blurred_img_ori, model, category, current_patchImages_path, isleaf):
    global incremental_uid
    # using global incremental uid
    incremental_uid += 1
    uid = incremental_uid
    # create parent chain
    parent_chain = []
    dummy_uid = uid
    for lit in conj:
        parent_chain.append((lit,dummy_uid))
    dummy_index = len(parent_chain) - 1
    # create node
    shape = 'box'
    style = 'filled'
    penwidth=2
    imagepath, ins_prob = create_node_image(parent_chain, dummy_index, edgepatch, parent_prob, ups, img_ori, blurred_img_ori, model, category, current_patchImages_path)
    # get node background color
    node_bg_color = get_node_bg_color(ins_prob, 'snow')
    fillcolor = node_bg_color
    color = node_bg_color
    # add labels to node
    label = '< <table border="0"> <tr> <td width="2" height="2" border="0"><img src="' + imagepath + '"/></td> </tr> <tr> <td>'
    ins_prob = np.around(ins_prob, decimals=3)
    ins_prob = int(ins_prob * 100)
    xlabel = str(ins_prob)+'%'
    # different annotation style for leaves
    if isleaf:
        penwidth=3
        group=str(float('-inf'))
        rank=str(float('-inf'))
        xlabel = '<font color="black">' + xlabel + '</font>'
    else:
        group = ''
        rank = ''
        xlabel = '<font color="black">' + xlabel + '</font>'
    xlabel = '<b>' + xlabel + '</b>'
    label += xlabel + '</td> </tr> </table> >'
    # add the node
    g.add_node(uid, shape=shape, style=style, fillcolor=fillcolor, label=label, labelloc='b', fontsize=25, fontcolor='white', color=color, penwidth=penwidth, group=group, rank=rank, width=2.4, height=2.9, fixedsize='true')
    return uid, imagepath, ins_prob


def recursive_lattice(conjuncts, parent_uid, g, depth, parent_prob, ups, img_ori, blurred_img_ori, model, category, current_patchImages_path, node_prob_thresh):
    # iterate over leaves
    for conj, edgepatch in conjuncts:
        # add node for current leaf
        uid, imagepath, node_prob = add_lattice_node(conj, g, edgepatch, parent_prob, ups, img_ori, blurred_img_ori, model, category, current_patchImages_path, isleaf=False)
        g.add_edge(parent_uid, uid, color='white')
        # build rest of the lattice originating from current leaf recursively
        if node_prob > node_prob_thresh:
            subconjlist = get_subconjuncts(conj)
            if len(subconjlist) > 0:
                recursive_lattice(subconjlist, uid, g, depth+1, node_prob, ups, img_ori, blurred_img_ori, model, category, current_patchImages_path, node_prob_thresh)


def build_tree(conjuncts, ups, img_ori, blurred_img_ori, model, category, current_patchImages_path, node_prob_thresh):
    # initialize graph
    g = pgv.AGraph(directed=True, strict=True, overlap='false', bgcolor='black')
    edgepatch = ""
    depth = 1
    # iterate over leaves
    for conj in conjuncts:
        # add node for current leaf
        uid, _, node_prob = add_lattice_node(conj, g, edgepatch, 100, ups, img_ori, blurred_img_ori, model, category, current_patchImages_path, isleaf=True)
        # build rest of the lattice originating from current leaf recursively
        if node_prob > node_prob_thresh:
            subconjlist = get_subconjuncts(conj)
            if len(subconjlist) > 0:
                recursive_lattice(subconjlist, uid, g, depth+1, node_prob, ups, img_ori, blurred_img_ori, model, category, current_patchImages_path, node_prob_thresh)
    return g
