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
from collections import defaultdict
import ipdb



parser = argparse.ArgumentParser()
parser.add_argument('--proto', type=str, required=True)
parser.add_argument('--folder', type=str, required=True)
args = parser.parse_args()


net = caffe.Net(args.proto,caffe.TEST)
suffix = ".caffemodel"
    
norms = defaultdict(list)
models = sorted([os.path.join(args.folder,x) for x in os.listdir(args.folder) 
                if x.endswith(suffix)],key=os.path.getctime)
                   
for model in models:
    #ipdb.set_trace()
    net.copy_from(model)
    for  k,v in net.params.items():
        if len(norms[k]) == 0  or np.any(norms[k][-1] != v[0].data):
            norms[k].append(v[0].data.copy())     
for k,v in norms.items():
    paths = []
    for i in range(len(v)):
        if i == 0:
            continue
        paths.append(np.linalg.norm(norms[k][i]-norms[k][i-1]))
    length = np.linalg.norm(norms[k][-1]-norms[k][0])
    print k, " ", length/np.sum(paths)
    
