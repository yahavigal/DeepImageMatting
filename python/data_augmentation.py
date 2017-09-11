# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 23:43:16 2017

@author: or
"""
import numpy as np
import cv2
from random import randint
import ipdb
import matplotlib.pyplot as plt
#this script has various data augmentation techniqus


rotaion_angels = [-45,-22,22,45]
gamma_coeff = [0.5,0.8,1.2,1.5]

def color_jitter(image):
    noise = np.eye(3) +  np.random.normal(0,0.01,9).reshape((3,3))
    return np.dot(image,noise)

#add random crops centerd at unknown regions (in case of trimap != None)
def random_cropping(image, crop_size, gt, target_size, ncrops = 1,  trimap = None):
    pass

#as proposed in Deep Image Matting
def horizontal_flipping(image, gt, trimap = None):
    if trimap is None:
        return [cv2.flip(image,flipCode = 1),cv2.flip(gt,flipCode = 1)]
    else:
        return [cv2.flip(image, flipCode = 1),
                cv2.flip(gt, flipCode = 1),
                cv2.flip(trimap,flipCode = 1)]

#need to figure out
def random_dilate_tri_map(trimap):
    pass

#as propose in alex net paper
def PCA_noise(image):
    
    A = image.reshape([image.shape[0]*image.shape[1],3])
    cov_mat = A.T.dot(A)
    eigen_values, eigen_vectors = np.linalg.eig(cov_mat)
    eigen_values /= (image.shape[0]*image.shape[1])
    noise = np.random.normal(0,0.0002,3).reshape((1,3))
    value_to_add = eigen_vectors.dot((noise*eigen_values).T)
    return np.add(image.astype('float32'),value_to_add.T)
    
#as proposed in Deep Automatic Protrait Matting
def rotate(image, gt, trimap = None):
    angel = rotaion_angels[randint(0,len(rotaion_angels)-1)]
    rows,cols = image.shape[0:2]

    M = cv2.getRotationMatrix2D((cols/2, rows/2), angel, 1)
    rotated_image = cv2.warpAffine(image,M,(cols,rows))
    rotated_gt = cv2.warpAffine(gt, M, (cols,rows))
    if trimap is not None:
        rotated_trimap = cv2.warpAffine(trimap, M, (cols,rows))
        return [rotated_image,rotated_gt,rotated_trimap]
    else:
        return [rotated_image,rotated_gt]

#as proposed in Deep Automatic Protrait Matting
def gamma_correction(image):
    
    gamma = gamma_coeff[randint(0,len(gamma_coeff)-1)]
    # build a lookup table mapping the pixel values [0, 255]
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
 
    # apply gamma correction using the lookup table
    return cv2.LUT(image.astype(np.uint8), table).astype(np.float32)