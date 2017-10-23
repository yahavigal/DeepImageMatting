# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 14:33:43 2017

@author: or
"""

import caffe 
import numpy as np 
import argparse
import os

def save_sparse_weights_mask(net, model,percent):

    net = caffe.Net(net, model, caffe.TEST)
    masks = {}
    for layer,blob in net.params.items():
        weights = blob[0].data
        pruning_percent = np.percentile(np.abs(weights),percent)
        masks[layer] = np.abs(weights) < pruning_percent

    path_to_save  = os.path.splitext(model)
    path_to_save = path_to_save[0]+"_SDS_mask"+path_to_save[1]    
    np.save(path_to_save,masks)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--net', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--percent', type=float, required=False, default = 30)
    args = parser.parse_args()

    save_sparse_weights_mask(args.net,args.model,args.percent)