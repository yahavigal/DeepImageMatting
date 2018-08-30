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

#as proposed in Deep Image Matting
def horizontal_flipping(image, gt):
    return [cv2.flip(image,flipCode = 1),cv2.flip(gt,flipCode = 1)]


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
def rotate(image, gt):
    angel = rotaion_angels[randint(0,len(rotaion_angels)-1)]
    rows,cols = image.shape[0:2]

    M = cv2.getRotationMatrix2D((cols/2, rows/2), angel, 1)
    rotated_image = cv2.warpAffine(image,M,(cols,rows))
    rotated_gt = cv2.warpAffine(gt, M, (cols,rows))
    return [rotated_image,rotated_gt]

def translate(image, gt):
    rows,cols = image.shape[0:2]
    translation_x =  randint(int(0.1*cols),int(0.2*cols))
    coin = randint(0,1)
    if coin == 0:
        translation_x = -translation_x
    M = np.float32([[1,0,translation_x],[0,1,0]])
    translated_image = cv2.warpAffine(image,M,(cols,rows))
    if coin == 0:
        translated_image[:,translation_x:cols-1] = image[:,translation_x:cols-1]
    else:
        translated_image[:,0:translation_x] = image[:,0:translation_x]
    translated_gt = cv2.warpAffine(gt,M,(cols,rows))
    return [translated_image,translated_gt]

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


def mixup(image1, image2, gt1, gt2, trimap1 = None, trimap2 = None):
    cols = image1.shape[1]
    pixel_border = randint(int(0.33*cols), int(0.66*cols))
    image1[ : , : , pixel_border : ] = image2[ : , : , pixel_border : ]
    gt1[ : , : , pixel_border : ] = gt2[ : , : , pixel_border : ]
    if trimap1 is not None:
        trimap1[ : , : , pixel_border : ] = trimap2[ : , : , pixel_border : ]
    return image1, gt1, trimap1


def mixup_stitch_old(image1, image2, gt1, gt2, trimap1 = None, trimap2 = None):
    dim, rows, cols = image1.shpae
    while(1):
        x1 = randint(int(0.33*cols), int(0.66*cols))
        x2 = randint(int(0.33*cols), int(0.66*cols))
        y1 = randint(int(0.33*rows), int(0.66*rows))
        y2 = randint(int(0.33*rows), int(0.66*rows))
        if len(set([x1, x2, y1, y2])) == 4:
            a = float(y1-y2)/(x1-x2)
            b = y1 - a*x1
            if a!=1.0:
                break
    if abs(a)<1.0:
        for x in range(cols):
            image1[:, :int(a*x+b), x] = image2[:, :int(a*x+b), x]
            gt1[:, :int(a*x+b), x] = gt2[:, int(a*x+b), x]
            if trimap1 is not None:
                trimap1[:, :int(a*x+b), x] = trimap2[:, int(a*x+b), x]
    else:
        for y in range(rows):
            image1[:, y, :int((y-b)/a)] = image2[:, y, :int((y-b)/a)]
            gt1[:, y, :int((y-b)/a)] = gt2[:, y, :int((y-b)/a)]
            if trimap1 is not None:
                trimap1[:, y, :int((y-b)/a)]  = trimap2[:, y, :int((y-b)/a)]

    return image1, gt1, trimap1


def mixup_stitch(image1, image2, gt1, gt2, trimap1 = None, trimap2 = None):
    dim, rows, cols = image1.shape
    #ipdb.set_trace()
    #print '1'
    while(1):
        nonzeros = cv2.findNonZero(gt1[0].astype(np.uint8))
        pnt1 = np.random.randint(0, len(nonzeros))
        pnt2 = np.random.randint(0, len(nonzeros))
        x1, y1 = nonzeros[pnt1][0]
        x2, y2 = nonzeros[pnt2][0]
        if len(set([x1, x2, y1, y2])) == 4:
            a = float(y1-y2)/(x1-x2)
            b = y1 - a*x1
            if a!=1.0:
                break
    #ipdb.set_trace()
    #print '2'
    if abs(a)<1.0:
        for x in range(cols):
            image1[:, :int(a*x+b), x] = image2[:, :int(a*x+b), x]
            gt1[:, :int(a*x+b), x] = gt2[:, int(a*x+b), x]
            #if trimap1 is not None:
                #trimap1[:, :int(a*x+b), x] = trimap2[:, int(a*x+b), x]
    #ipdb.set_trace()
    else:
        for y in range(rows):
            image1[:, y, :int((y-b)/a)] = image2[:, y, :int((y-b)/a)]
            gt1[:, y, :int((y-b)/a)] = gt2[:, y, :int((y-b)/a)]
            #if trimap1 is not None:
                #trimap1[:, y, :int((y-b)/a)]  = trimap2[:, y, :int((y-b)/a)]
    #print image1.shape
    #print gt1.shape
    return image1, gt1#, trimap1
