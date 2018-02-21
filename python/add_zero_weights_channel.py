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
import re
supproted_types = ['Convolution', 'BatchNorm']

def find_num_rep(layer, num_rep):
    if num_rep is None:
        return 1
    if layer not in num_rep:
        return 1
    return int(num_rep[num_rep.index(layer) +1])


def find_layer_type(net,layer):
    for i,name in enumerate(net._layer_names):
        if name == layer:
            return net.layers[i].type
    return None

def add_zeros_to_layer(proto, model, layer_name, num_rep):
    net = caffe.Net(proto,model,caffe.TRAIN)
    if 'all' in layer_name:
        layers = [x for x in net.params.keys() if find_layer_type(net, x) in supproted_types]
    else:
        layers = []
        for regex in layer_name:
            r = re.compile(regex)
            layers += [x for x in net.params.keys() if r.match(x) and find_layer_type(net,x) in supproted_types]
        if num_rep is not None:
            for regex in num_rep:
                r = re.compile(regex)
                layers += [x for x in net.params.keys() if r.match(x) and find_layer_type(net,x) in supproted_types]

    for layer in layers:
        if not layer in net.params:
            continue
        layer_type = find_layer_type(net, layer)
        repetitions = find_num_rep(layer,num_rep)
        if layer_type == 'BatchNorm':
            for rep in range(repetitions):
                ind = net.params[layer][0].data.shape[0]
                mean_blob = np.insert(net.params[layer][0].data,ind,0)
                var_blob = np.insert(net.params[layer][1].data,ind,0)
                net.params[layer][0].reshape(*mean_blob.shape)
                net.params[layer][0].data[...] = mean_blob
                net.params[layer][1].reshape(*var_blob.shape)
                net.params[layer][1].data[...] = var_blob
        else:
            for rep in range(repetitions):
                layer_params = net.params[layer]
                shape = layer_params[0].data.shape[2:]
                z = np.random.normal(loc=0,scale=0.02,size=shape)
                new_data = np.insert(layer_params[0].data,1,z, axis =1)
                layer_params[0].reshape(*new_data.shape)
                layer_params[0].data[...] = new_data

    save_path = os.path.join(os.path.split(model)[0],
                             os.path.split(model)[1].split('.')[0]+
                             "_zeroes_added.caffemodel")
    net.save(save_path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--proto', type=str, required=True, help='model definition in prototxt file')
    parser.add_argument('--model', type=str, required=True, help='path to weights file')
    parser.add_argument('--layer', type=str, required=True, nargs ='+',help='layer names (at least one) or \'all\' for all layers in tne net')
    parser.add_argument('--num_rep', type=str, required=False, nargs ='*',help='specific number of repetitions for specific layers default is 1')
    args = parser.parse_args()

    add_zeros_to_layer(args.proto,args.model,args.layer, args.num_rep)

