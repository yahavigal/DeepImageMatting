# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 05:51:25 2016

@author: or
"""

import  caffe
import numpy as np
import matplotlib.pyplot as plt
import argparse
import cv2
import math
import os
import shutil
import ipdb

def find_square_partition(number):
    sqrt = int(math.sqrt(number))
    width = number/sqrt


def show_mean_channels(proto, model,is_save_kernels = False):

    if is_save_kernels == True:
        if os.path.exists('kernels'):
            shutil.rmtree('kernels')
        os.mkdir('kernels')

    net = caffe.Net(proto,model,caffe.TEST)
    for layer,weights in net.params.items():

        if 'conv' not in layer:
            continue
        #height,width = find_square_partition(weights[0].shape[0])
        for i,channel in enumerate(weights[0].data):
            for j,kernel in enumerate(channel):

                name = '{} featuremap {} channel {} mean {} variance {}'.format(layer,i,j,np.mean(kernel),np.var(kernel))
                print name
                if is_save_kernels == True:
                    fig = plt.figure(figsize = (4,4))
                    plt.hist(kernel.flatten())
                    plt.savefig(os.path.join('kernels',name.replace(' ','_')+'.kernel.jpg'))
                    plt.close(fig)


if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('--proto', type=str, required=True, help= 'path to net definition file')
   parser.add_argument('--model', type=str, required=True, help= 'path to weights file')
   parser.add_argument('--save', action='store_true', help= 'optional: save kernels histograms')
   args = parser.parse_args()
   show_mean_channels(args.proto,args.model,args.save)


