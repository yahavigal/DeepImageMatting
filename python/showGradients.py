# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 16:20:20 2016
this script assumes 
@author: or
"""

import caffe 
import numpy as np 
import argparse
import os
import re
import matplotlib.pyplot as plt
from collections import defaultdict




parser = argparse.ArgumentParser()
parser.add_argument('--proto', type=str, required=True)
parser.add_argument('--folder', type=str, required=True)
parser.add_argument('--direction',action='store_true')
args = parser.parse_args()


net = caffe.Net(args.proto,caffe.TRAIN)
suffix = ".caffemodel"
    
norms = defaultdict(list)
iters=[]
for model in sorted([os.path.join(args.folder,x) for x in os.listdir(args.folder)],key=os.path.getctime):
    #ipdb.set_trace()
    if not model.endswith(suffix):
        continue
    iters.append(int(re.findall('\d+', model)[0]))
    net.copy_from(model)
    for k,v in net.params.items():
        if False == args.direction:
            norms[k].append(np.linalg.norm(v[0].diff))
        else:
            if len(norms[k]) == 0:
                continue
            prod = np.sum(norms[k-1]*v[0].diff)
            cos_theta = prod/(np.linalg.norm(norms[k-1])*np.linalg.norm(v[0].diff))
            norms[k].append(cos_theta)
            
                
       

        
fig_index=1
j=1
list_fig=[]
list_fig.append(plt.figure(fig_index))
      
for layer,weights in norms.items():
    if not np.all(weights):
        continue
    if (j==11 or list_fig==[]): 
        global j
        j=1
        fig_index+=1
        f = plt.figure(fig_index)
        list_fig.append(f)
    ax = plt.subplot(5,2,j)
    plt.title(layer)
    if False == args.direction:
        plt.plot(iters,weights)
    else:
        plt.scatter(iters,weights)
    plt.xlabel('# iter')
    plt.ylabel('gradient\'s norm')
    plt.subplots_adjust(hspace=0.5)
    j+=1

plt.show()
