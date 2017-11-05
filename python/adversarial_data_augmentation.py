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
erosion_dilataion_kernel = np.ones((25,25),np.uint8)

def get_order(num):
    n = num
    i = 0
    while np.abs(n) < 0.01:
        n *= 10
        i += 1
    return i
    

def advesarial_data_augmentation(proto, model, inputs_data,  output_folder, 
                                 trimap_data = None,
                                 laying_rate = 1, precntage = 0.1):
                                
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
    
    random.shuffle(images_list)
    max_ind = int(precntage*len(images_list))
    images_list = images_list[0:max_ind]
        
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
        
        if np.random.uniform() > 0.5:
            mask_r = cv2.erode(mask_r, erosion_dilataion_kernel, iterations = 1)
        else:
            mask_r = cv2.dilate(mask_r, erosion_dilataion_kernel, iterations = 1)
        
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
        net.backward(diffs=[net.inputs[0]])
        
        grad_wrt_input = net.blobs[net.inputs[0]].diff[0,:3,:]
        
        if np.any(grad_wrt_input) == False:
            continue
        
        grad_wrt_input = grad_wrt_input.reshape(*img_r1.shape)
        order = get_order(np.mean(grad_wrt_input))
        laying_rate = 2*(10**order)
        img_r1 += laying_rate*grad_wrt_input
        img_r1 += np.array([104,117,123],dtype=np.float32)
        img_r1 = np.clip(img_r1, 0, 255)
     
        img_r1 = cv2.cvtColor(img_r1,cv2.COLOR_RGB2BGR).astype('uint8')
        img_r1 = cv2.resize(img_r1, (img.shape[1],img.shape[0])) 
        #need to add save image
        
        if np.any(img_r1 != img_orig):
                 
            path_to_save = '_'.join([split[ind1],split[ind1+1],frame_num])+"_adv.png"
            path_to_save_orig = '_'.join([split[ind1],split[ind1+1],frame_num])+"_orig.png"
            path_to_save = os.path.join(output_folder,path_to_save)
            path_to_save_orig = os.path.join(output_folder,path_to_save_orig)
            cv2.imwrite(path[0]+ "_adv" +path[1],img_r1)
            cv2.imwrite(path_to_save, img_r1)
            cv2.imwrite(path_to_save_orig, img_orig)
          
         
        




if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--images_dir', type=str, required=True)
    parser.add_argument('--proto', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--trimap', type=str, required=False,default = None)    
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--laying_rate', type=float, required=False, default=0.007)
    parser.add_argument('--percentage', type=float, required=False, default=0.1)
    args = parser.parse_args()
    
    advesarial_data_augmentation(args.proto, args.model,args.images_dir,
                                 args.output,args.trimap, args.laying_rate, args.percentage)
    
    
        
        
        
                
            