# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 18:17:05 2017

@author: or
"""

import caffe
caffe.set_mode_gpu()
caffe.set_device(0)
import numpy as np
import os
import cv2
import random
import argparse
import ipdb
import matplotlib.pyplot as plt

gt_ext = "_silhuette"
trimap_ext = "triMap_"
img_width = 224
img_height = 224

weights_regularization_ratios = {}


def regularization_weights_inspection(proto, model, inputs_data, trimap_data = None):

    trimap_r = None

    net = caffe.Net(proto,model,caffe.TRAIN)

    if os.path.isdir(inputs_data):
           images_list = [os.path.join(inputs_data,x)
                                for x in os.listdir(inputs_data)
                                if x.endswith(".png") and x.find(gt_ext) == -1]
    elif os.path.isfile(inputs_data):
        images = open(inputs_data).readlines()
        images = [x[0:-1] for x in images if x.endswith('\n')]
        images_list = [x for x in images
                             if x.endswith(".png") and x.find(gt_ext) == -1]
    else:
        raise Exception("invalid inputs format")

    for k,v in net.params.items():
            weights_regularization_ratios[k] = [0.0,0.0]

    for image_path in images_list:
        img_orig = cv2.imread(image_path)
        img = cv2.cvtColor(img_orig,cv2.COLOR_BGR2RGB).astype('float32')
        #subtract mean
        img -= np.array([104,117,123],dtype=np.float32)
        img_r1 = cv2.resize(img, (img_width,img_height))

        path = os.path.splitext(image_path)
        gt_path = path[0] + gt_ext + path[1]
        if not os.path.isfile(gt_path):
           raise Exception("missing ground truth per image {}".format(image_path))

        mask = cv2.imread(gt_path,cv2.CV_LOAD_IMAGE_GRAYSCALE)
        mask_r = cv2.resize(mask, (img_width,img_height),interpolation = cv2.INTER_NEAREST)

        split = image_path.split(os.sep)
        ind1 = split.index([x for x in split if x.startswith("w")][0])

        frame_num = split[ind1+2].split('_')[0]
        if trimap_data is not None:
            trimap_path = os.path.join(trimap_data,
                                       split[ind1],split[ind1+1],
                                       trimap_ext+frame_num+".png")

            if trimap_data is not None and not os.path.isfile(trimap_path):
                continue

            trimap = cv2.imread(trimap_path,cv2.CV_LOAD_IMAGE_GRAYSCALE)
            trimap_r = cv2.resize(trimap, (img_width,img_height),interpolation = cv2.INTER_NEAREST)

        img_r = img_r1.reshape([1,3,img_height,img_width])

        mask_r = mask_r.reshape([1,1,img_height,img_width])

        if trimap_r is not None:
            trimap_r = trimap_r.reshape([1,1,img_height,img_width])
            img_r = np.concatenate((img_r,trimap_r),axis =1)

        net.blobs[net.inputs[0]].reshape(*img_r.shape)
        net.blobs[net.inputs[1]].reshape(*mask_r.shape)
        net.blobs[net.inputs[0]].data[...] = img_r
        net.blobs[net.inputs[1]].data[...] = mask_r

        net.forward()
        net.backward()

        for k,v in net.params.items():
            weights_regularization_ratios[k][0] += np.mean(np.abs(v[0].diff))
            weights_regularization_ratios[k][1] += np.linalg.norm(v[0].data)

    for k,v in weights_regularization_ratios.items():
        v[0] /= len(images_list)
        v[1] /= len(images_list)
        value = v[0]/v[1]
        print k, " ", value



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--images_dir', type=str, required=True)
    parser.add_argument('--proto', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--trimap', type=str, required=False,default = None)

    args = parser.parse_args()

    regularization_weights_inspection(args.proto, args.model,args.images_dir,
                                      args.trimap)







