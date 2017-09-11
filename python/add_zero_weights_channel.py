# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 00:39:20 2017

@author: or
"""

import caffe
import numpy as np 
import argparse
import os
import ipdb

def add_zeros_to_layer(proto, model, layer_name):
    net = caffe.Net(proto,model,caffe.TRAIN)
    if not layer_name in net.params:
        return
    
    layer_params = net.params[layer_name]
    shape = layer_params[0].data.shape[2:]
    z = np.zeros(shape)
    new_data = np.insert(layer_params[0].data,1,z, axis =1)
    layer_params[0].reshape(*new_data.shape)
    layer_params[0].data[...] = new_data
    save_path = os.path.join(os.path.split(model)[0],
                             os.path.split(model)[1].split('.')[0]+
                             "_zeroes_added.caffemodel")
    net.save(save_path)
    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--proto', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--layer', type=str, required=True)
    args = parser.parse_args()
    
    add_zeros_to_layer(args.proto,args.model,args.layer)
    