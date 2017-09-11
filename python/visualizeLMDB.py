# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 16:31:01 2016

@author: or
"""

import lmdb
import numpy as np
import matplotlib.pyplot as plt
from caffe.proto import caffe_pb2
from caffe.io import datum_to_array
import cv2




def read_images_from_lmdb(db_name, visualize,fig_index=1):
    
    env = lmdb.open(db_name)
    txn = env.begin()
    cursor = txn.cursor()
    X = []
    y = []
    idxs = []
    for idx, (key, value) in enumerate(cursor):
        datum = caffe_pb2.Datum()
        datum.ParseFromString(value)
        X.append(np.array(datum_to_array(datum)))
        y.append(datum.label)
        idxs.append(idx)
    if visualize:
        fig = plt.figure(fig_index)
        print "Visualizing a few images..."
        for i in range(100):
            img = X[i]
            #convert the shape from (X,224,224) to (224,224,X)
            img = np.transpose(img,[1,2,0])
            
            if img.shape[2]==3:
                #rgb to bgr for visualization only
                img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            else:
                #replicate the mask for visualization puroposes
                img = np.repeat(img,3,axis=2)
                
            plt.subplot(10,10,i+1)
            plt.imshow(img,cmap='Greys')
            
            # the title is the label (1,-1)
            plt.title(y[i])
            plt.axis('off')          
        fig.show()
    print " ".join(["Reading from", db_name, "done!"])
    return X, y, idxs
 
 
read_images_from_lmdb("/home/or/caffe/or/deepMask/lmdb_100/train_rgb_lmdb",True)

read_images_from_lmdb("/home/or/caffe/or/deepMask/lmdb_100/train_mask_lmdb",True,2) 
raw_input() 