# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 18:12:15 2017

@author: or
"""

import numpy as np
import cv2
import os
import Image
import argparse
import ipdb
import matplotlib.pyplot as plt
import bgs_train_test

gt_ext = "_silhuette"
trimap_ext = "triMap_"
adverserial_ext = "_adv"
results_gt_prefix = "Annotation-gt"
results_trimap_prefix = "Annotation-trimap"
#gt_ext = getattr(bgs_train_test.bgs_test_train,'gt_ext')
#adverserial_ext = getattr(bgs_train_test.bgs_test_train,'adverserial_ext')
def show_gt_image_overlay(images_dir,trimap_dir = None):
    if os.path.isdir(images_dir):
           images_list = [os.path.join(images_dir,x)
                                     for x in os.listdir(images_dir)
                                     if x.endswith(".png") and x.find(gt_ext) == -1]
    elif os.path.isfile(images_dir):
        images = open(images_dir).readlines()
        images = [x[0:-1] for x in images if x.endswith('\n')]
        images_list = [x for x in images
                       if x.endswith(".png") and x.find(gt_ext) == -1
                       and x.find(adverserial_ext) == -1]
    else:
        raise Exception("invalid input format")

    if os.path.exists(results_gt_prefix) == False:
        os.mkdir(results_gt_prefix)
    if os.path.exists(results_trimap_prefix) == False:
        os.mkdir(results_trimap_prefix)
    for image_path in images_list:

        path = os.path.splitext(image_path)
        gt_path = path[0] + gt_ext + path[1]

        if os.path.exists(image_path) ==False or os.path.exists(gt_path) ==False:
            print "NOT_EXIST", image_path, " ", gt_path
            continue

        mask = Image.open(gt_path)
        mask = mask.convert('RGB')
        np_mask = np.array(mask)
        np_mask = np.multiply(np_mask,(1,0,0)).astype(np.uint8)
        mask = Image.fromarray(np_mask)
        img = Image.open(image_path)
        try:
            overlay = Image.blend(mask,img,0.7)
        except:
            ipdb.set_trace()
            print "BLEND_BUG", image_path, " ", gt_path
            continue
        split = image_path.split(os.sep)
        split = os.sep.join(split).replace(os.sep,"_")
        split = os.path.splitext(split)[0]

        fig_path =split +"_{}.jpg".format(results_gt_prefix)

        fig_path = os.path.join(results_gt_prefix,fig_path)
        overlay.save(fig_path)


        if trimap_dir != None:
            trimap_path = os.path.join(trimap_dir,
                                       split[ind1],split[ind1+1],
                                       trimap_ext+frame_num+".png")
            if os.path.isfile(trimap_path) == False:
                continue

            trimap = Image.open(trimap_path)
            trimap = trimap.convert('RGB')
            np_trimap = np.array(trimap)
            np_trimap[np.any(np_trimap == [0,0,0],axis = -1)] = (255,0,0)
            trimap = Image.fromarray(np_trimap)
            overlay = Image.blend(trimap,img,0.8)
            fig_path = split[ind1]+"_"+split[ind1 +1]+ "_"+split[ind1+2].split('.')[0] +"_{}.jpg".format(results_trimap_prefix)
            fig_path = os.path.join(results_trimap_prefix,fig_path)
            overlay.save(fig_path)






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', type=str, required=True, help= "path ot images list or directory")
    parser.add_argument('--trimap_dir', type=str, required=False, default = None, help="path to trimap or ant addtional output dir")
    args = parser.parse_args()
    show_gt_image_overlay(args.images_dir,args.trimap_dir)

