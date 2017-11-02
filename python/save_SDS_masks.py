# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 14:33:43 2017

@author: or
"""

import caffe
import numpy as np
import argparse
import os
import ipdb
types_to_prune = ["Convolution","InnerProduct"]

def save_sparse_weights_mask(net, model, percent, random, channel):

    net = caffe.Net(net, model, caffe.TEST)
    masks = {}
    for layer,blob in net.params.items():
        if net.layer_dict[layer].type not in types_to_prune:
            continue

        weights = blob[0].data
        if random == True:
            ipdb.set_trace()
            if channel ==True:
                masks[layer] = np.ones(weights.shape, dtype=bool)
                channels_to_prune =  np.random.choice([True, False], size=(weights.shape[0],), p=[1 - percent / 100.0, percent / 100.0])
                for i, single_channel in enumerate(weights):
                    if channels_to_prune[i] == False:
                        masks[layer][i][...] = False

            else:
                ipdb.set_trace()
                masks[layer] = np.random.choice([True, False], size=weights.shape, p=[1-percent/100.0, percent/100.0])
        else:
            if channel == True:
                ipdb.set_trace()
                pruning_percent = np.percentile(np.mean(np.abs(weights),axis = (1,2,3)),percent)
                masks[layer] = np.ones(weights.shape, dtype=bool)
                for i,single_channel in enumerate(weights):
                    if np.mean(np.abs(single_channel)) < pruning_percent:
                        masks[layer][i][...] = False

            else:
                pruning_percent = np.percentile(np.abs(weights),percent)
                masks[layer] = np.abs(weights) < pruning_percent

    path_to_save  = os.path.splitext(model)
    path_to_save = path_to_save[0]+"_DSD_mask"+path_to_save[1]
    np.save(path_to_save,masks)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--net', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--percent', type=float, required=False, default = 30)
    parser.add_argument('--random', action='store_true')
    parser.add_argument('--channel', action='store_true')
    args = parser.parse_args()

    save_sparse_weights_mask(args.net,args.model,args.percent,args.random,args.channel)
