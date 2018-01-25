import cv2
import matplotlib.pyplot as plt
import numpy as np
import ipdb

def calc_connectivity_metric(mask):
    #top CC in corresopnding normalized DT
    mask_thresh = mask >= 0.9
    cc_res = cv2.connectedComponentsWithStats((mask_thresh*255).astype(np.uint8),4, cv2.CV_32S)
    dt_res = cv2.distanceTransform((cc_res[1]!=1).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    dt_res /= np.max(dt_res)
    theta = 0.15
    quantization = 0.1
    d = []
    #iterate over all quantized thresholds
    for thresh in np.arange(0.1,0.9,quantization):
        current_thresh =  mask>=thresh
        current_cc_res = cv2.connectedComponentsWithStats((current_thresh*255).astype(np.uint8),4, cv2.CV_32S)
        #if we get same numbers of CC's so we dont need to compute the rest
        if current_cc_res[0] == cc_res[0]:
            continue
        current_d = mask - (thresh - quantization)
        #dont pay for biggest cc
        current_d[current_cc_res[1] == 1] = 0
        #dont pay for BG
        current_d[current_cc_res[1] == 0] = 0
        #dont pay for negligible errors
        current_d[current_d < theta] = 0
        #pay according to the distance
        current_connect = 1 - np.multiply(current_d,dt_res)
        d.append(current_connect)

    #perfect connectivity
    if len(d) == 0:
        return 1

    #take the maximum accross thresholds
    max_penalty =  np.max(d,axis=0)
    connectivity_measure = np.average(np.abs(max_penalty))
    return connectivity_measure



#example of 2 images were on one of my machines
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--res', type=str, required=True)
    parser.add_argument('--gt', type=str, required=True)
    args = parser.parse_args()

    mask = cv2.imread(args.res,0)
    mask = np.divide(mask,255.0)
    gt = cv2.imread(args.gt,0)
    gt = np.divide(gt,255.0)
    print np.abs(calc_connectivity_metric(gt) - calc_connectivity_metric(mask))

