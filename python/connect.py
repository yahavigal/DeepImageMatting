import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math as m
import ipdb

def calc_connectivity_metric(gt_mask, mask):
    resu = np.subtract(calc_connectivity_mask(gt_mask),calc_connectivity_mask(mask))
    print np.abs(resu)
   

def calc_connectivity_mask(mask):
    #top CC in corresopnding normalized DT
    toShow = False

    mask_thresh = mask >= 0.9
    if toShow == True:
        plt.imshow(mask_thresh)
        plt.show()
    cc_res = cv2.connectedComponentsWithStats((mask_thresh*255).astype(np.uint8),4, cv2.CV_32S)
    # find index of the biggest component
    max_cc_area_main = 0
    max_ind_main = 0;
    for ind in range (1, cc_res[0]):
        if max_cc_area_main < cc_res[2][ind,cv2.CC_STAT_AREA]:
	    max_cc_area_main = cc_res[2][ind,cv2.CC_STAT_AREA]
            max_ind_main = ind
    dt_res = cv2.distanceTransform((cc_res[1]!=max_ind_main).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    normFactor = m.sqrt( mask.shape[0]*mask.shape[0] + mask.shape[1]*mask.shape[1])
    # dt_res /= np.max(dt_res)
    dt_res /= normFactor
    if toShow == True:
        plt.imshow(dt_res)
        plt.show()
    theta = 0.15
    quantization = 0.1
    d = []
    #iterate over all quantized thresholds    
    for thresh in np.arange(0.1,0.9,quantization):
        current_thresh =  mask>=thresh
        if toShow == True:
            plt.imshow(current_thresh)
            plt.show()
        current_cc_res = cv2.connectedComponentsWithStats((current_thresh*255).astype(np.uint8),4, cv2.CV_32S)
        #if we get same numbers of CC's so we dont need to compute the rest
        if current_cc_res[0] == cc_res[0]:
            continue
        current_d = mask - (thresh + quantization*0.5)
        if toShow == True:
            plt.imshow(current_d)
            plt.show()

        #dont pay for biggest cc
	# find index of the biggest component
	max_cc_area = 0
        max_ind = 0;
        for ind in range (1, current_cc_res[0]):
	    if max_cc_area < current_cc_res[2][ind,cv2.CC_STAT_AREA]:
		max_cc_area = current_cc_res[2][ind,cv2.CC_STAT_AREA]
                max_ind = ind;

        current_d[current_cc_res[1] == max_ind] = 0
        if toShow == True:
            plt.imshow(current_d)
            plt.show()
        #dont pay for BG
        current_d[current_cc_res[1] == 0] = 0
        if toShow == True:
	    plt.imshow(current_d)
            plt.show()
        #dont pay for negligible errors
        current_d[current_d < theta] = 0   
        if toShow == True:
	    plt.imshow(current_d)
            plt.show()
        #pay according to the distance
        # current_connect = 1 - np.multiply(current_d,dt_res)        
        current_connect = np.multiply(current_d,dt_res)
	if toShow == True:
            plt.imshow(current_connect)
            plt.show()
        d.append(current_connect)

    #perfect connectivity
    if len(d) == 0:
        return 0
    
    #take the maximum accross thresholds
    max_penalty =  np.max(d,axis=0)
    if toShow == True:
        plt.imshow(max_penalty)
        plt.show()

    #connectivity_measure = np.average(np.abs(max_penalty))

    #blobs = max_penalty > 0;    
    #numPixsInBlobs = np.sum(blobs)    
    #connectivity_measure_1 = np.sum(np.abs(max_penalty))/numPixsInBlobs

    # normalization to free space
    # connectivity_measure = np.sum(np.abs(max_penalty))/(mask.shape[0]*mask.shape[1] - max_cc_area_main)

    # normalization to main blob area
    connectivity_measure = np.sum(np.abs(max_penalty))/(max_cc_area_main)

    return connectivity_measure  # np.array([connectivity_measure,connectivity_measure_1,connectivity_measure_2] )



#example of 2 images were on one of my machines
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--res', type=str, required=True)
    parser.add_argument('--gt', type=str, required=True)
    args = parser.parse_args()

    mask = cv2.imread(args.res,0)
    mask = np.divide(mask,255.0)
    gt_mask = cv2.imread(args.gt,0)
    gt_mask = np.divide(gt_mask,255.0)
    calc_connectivity_metric(gt_mask, mask)
   

