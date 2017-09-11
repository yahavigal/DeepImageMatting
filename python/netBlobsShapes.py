# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 21:16:33 2016

@author: or
"""


import caffe
import argparse




parser = argparse.ArgumentParser()
parser.add_argument('--proto', type=str, required=True)
args = parser.parse_args()
net = caffe.Net(args.proto,caffe.TEST)

print "zozozozozozozozozozozozozozozozozozozoz"

for k,v in net.blobs.items():
    print k ,v.data.shape

print "zozozozozozozozozozozozozozozozozozozoz"
