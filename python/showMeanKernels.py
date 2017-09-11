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

parser = argparse.ArgumentParser()
parser.add_argument('--proto', type=str, required=True)
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--output',type=str, required=False,default='kernels')
parser.add_argument('--image',type=str, required=False)
args = parser.parse_args()

if args.image == None and args.output =='tops':
    raise "must to set input image inorder to get top blobs"
    

fig_index=1
j=1
list_fig=[]
list_fig.append(plt.figure(fig_index))

caffeNet = caffe.Net(args.proto,args.model,caffe.TEST)

if args.output == 'kernels':
    output = caffeNet.params.items()
else:
    
    output = caffeNet.blobs.items()
    img = cv2.imread(args.image)
    img = img.astype('float32')
    img = cv2.resize(img,(224,224))
    img -= np.array([104,117,123],dtype='float32')
    caffeNet.blobs[caffeNet.inputs[0]].data[0][:] = np.transpose(img,[2,0,1])
    caffeNet.forward()
    #ipdb.set_trace()
    

for layer,weights in output:
    if (j==11 or list_fig==[]): 
        global j
        j=1
        fig_index+=1
        f = plt.figure(fig_index)
        list_fig.append(f)
    ax = plt.subplot(5,2,j)
    if args.output =='kernels':
        data = weights[0].data
    else:
        data = weights.data
        
    plt.hist(data.flatten())
    plt.title(layer)
    plt.text(0.05,0.9,"Average: %.4g" % np.average(data),ha='center', va='center',transform=ax.transAxes)
    plt.text(0.05,0.8,"variance: %.4g" % np.var(data),ha='center', va='center',transform=ax.transAxes)
    plt.text(0.05,0.7,"Min: %.4g" % np.min(data),ha='center', va='center',transform=ax.transAxes)
    plt.text(0.05,0.6,"Max: %.4g" % np.max(data),ha='center', va='center',transform=ax.transAxes)      
    j+=1
plt.show()
    
