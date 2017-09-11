# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 13:27:41 2016

@author: or
"""
import caffe
import lmdb
import cv2
import ipdb
import numpy as np
import matplotlib.pyplot as plt

caffe.set_mode_gpu()

 
net = caffe.Net('../or/deepMask/proto/deploy_sharpMaskVGG.prototxt','../or/deepMask/snapshots/_iter_10000.caffemodel',caffe.TEST)
lmdb_env = lmdb.open('../or/deepMask/lmdb/test_rgb_lmdb')
lmdb_txn = lmdb_env.begin()
lmdb_cursor = lmdb_txn.cursor()
datum = caffe.proto.caffe_pb2.Datum()

for i, (key, value) in enumerate(lmdb_cursor):
    if i == 100:
        break
    datum.ParseFromString(value)
    label = datum.label
    data = caffe.io.datum_to_array(datum)
    data = np.transpose(data,[1,2,0])
    data-=np.array([104,117,123],dtype='uint8')
    data = np.transpose(data,[2,1,0])
    net.blobs[net.inputs[0]].data[:] = data    
    net.forward()
    predictionBlobName = net.outputs[1]
    prediction = net.blobs[predictionBlobName].data

    if prediction.shape[1] == 3:
        #rgb to bgr for visualization only
        prediction = cv2.cvtColor(prediction,cv2.COLOR_BGR2RGB)
    elif prediction.shape[1] == 1:
        #replicate the mask for visualization puroposes
        prediction = np.repeat(prediction,3,axis=1)
                
    plt.subplot(10,10,i+1)
    plt.imshow(np.transpose(prediction[0],[2,1,0]))
    plt.axis('off')
plt.show()           