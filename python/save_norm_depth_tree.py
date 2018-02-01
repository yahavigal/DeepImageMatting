
import argparse
import os
import ipdb
import re
import shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg

#python save_norm_depth_tree.py --root /media/or/Data/deepImageMatting/ --images_list /media/or/Data/dataLists/dataSet_1/test_list_orig.txt --caseDirName Set1_07_2017 --outputSubDir set1_07_2017_depth_norm_v2

# --algoType 0

gt_ext = "_silhuette"
adverserial_ext = "_adv"

def save_depth_tree(root, images_input, caseDirName, outputSubDir, target_ext, algoType):

    output = os.path.join(root, outputSubDir)
    if not os.path.exists(output):
        os.mkdir(output)
    if os.path.isdir(images_input):
        images_list = [os.path.join(images_input, x)
                                  for x in os.listdir(images_input)
                                  if x.endswith(".png") and x.find(gt_ext) == -1]
    elif os.path.isfile(images_input):
        images = open(images_input).readlines()
        images = [x[0:-1] for x in images if x.endswith('\n')]
        images_list = [x for x in images
                                  if x.endswith(".png") and x.find(gt_ext) == -1 and x.find(adverserial_ext) == -1]

    #main loop
    for image_path in images_list:

        depth_path = image_path.replace("color",target_ext)
        depth_norm_path = depth_path.replace(caseDirName, outputSubDir)

        print('image_path = ', image_path)        
        print('depth_path = ', depth_path)
        print('depth_norm_path = ', depth_norm_path)

        if depth_norm_path == depth_path:
            print('src path and dst path are same, which is not allowed, continue')
            continue

        if not os.path.exists(depth_path):
            continue

        dst_dir = os.path.dirname(os.path.abspath(depth_norm_path))
        print('dst_dir = ', dst_dir)

        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)

	    # normalize depth
        d_im = cv2.imread(depth_path, 2)
        #plt.imshow(d_im)
        #plt.show()
        minVal, maxVal,_, _ = cv2.minMaxLoc(d_im);
        print 'minVal = ', minVal
        print 'maxVal = ', maxVal          

        if algoType == 0:
            thresh0 = 0
            thresh1 = 1800
            d_im[ d_im > thresh1] = 0
            d_im[ d_im < thresh0] = 0
            minVal, maxVal,_, _ = cv2.minMaxLoc(d_im);
            d_im_n = (255*(d_im - minVal)/(maxVal - minVal))
        elif algoType == 1:
            thresh0 = 0
            thresh1 = 1800
            d_im[ d_im > thresh1] = 0
            d_im[ d_im < thresh0] = 0
            d_im_f = np.float32(d_im )
            d_im_n = (255.*(d_im_f/1800.))
            d_im_n = np.uint8(d_im_n)
        elif algoType == 2:
            thresh0 = 0
            thresh1 = 1800
            d_im[ d_im > thresh1] = thresh1
            d_im[ d_im < thresh0] = thresh0
            d_im_f = np.float32(d_im )
            d_im_n = (255.*(d_im_f/1800.))
            d_im_n = np.uint8(d_im_n)
	elif algoType == 3: # normalize and fill depth
	    ipdb.set_trace()
            thresh0 = 0
            thresh1 = 1800
            d_im[ d_im > thresh1] = 0
            d_im[ d_im < thresh0] = 0
	    d_im_4 = cv2.resize(d_im, (d_im.shape[1]/4,d_im.shape[0]/4))
	    plt.imshow(d_im_4)
            plt.show()
            d_im_4_m = cv2.medianBlur(d_im_4, 5)
	    plt.imshow(d_im_4_m)
            plt.show()
            kernel = np.ones((5,5),np.uint8)
            d_im_4_d = cv2.morphologyEx(d_im_4, cv2.MORPH_CLOSE, kernel)
            plt.imshow(d_im_4_d)
            plt.show()
       
        cv2.imwrite(depth_norm_path,d_im_n)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--images_list', type=str, required=True)
    parser.add_argument('--caseDirName', type=str, required=True)
    parser.add_argument('--outputSubDir', type=str, required=True)
    parser.add_argument('--target_ext', type=str, required=False, default="depth")
    parser.add_argument('--algoType', type=int, required=False, default=0)
    args = parser.parse_args()

    save_depth_tree(args.root, args.images_list, args.caseDirName, args.outputSubDir,args.target_ext, args.algoType)
