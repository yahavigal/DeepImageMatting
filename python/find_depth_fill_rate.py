# coding: utf-8
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import ipdb
import argparse
def find_depth_fill_rate(root):
    total_fill_rate = 0
    num_of_files = 0
    fill_rates =[]
    for root,dirs,files in os.walk(root):
        for file in files:
            if 'depth' not in file:
                continue
            depth = cv2.imread(os.path.join(root,file),-1)
            gt_path = os.path.join(root,file).replace('depth','color_silhuette')
            gt = cv2.imread(gt_path,0)
            if depth is None or gt is None:
                continue
            indices = gt > 0
            relevant_pixels = depth[indices]
            fill_rate = np.sum(relevant_pixels == 0)/float(relevant_pixels.size)
            fill_rates.append(fill_rate)
            total_fill_rate += fill_rate
            num_of_files += 1
    print "average fill rate {} for {} files".format(total_fill_rate/num_of_files,num_of_files)
    plt.title("histogram of depth fill rates")
    weights = np.ones_like(fill_rates)/len(fill_rates)
    plt.hist(fill_rates,weights = weights,bins=100)
    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True, help="root to the depth tree")
    args = parser.parse_args()

    find_depth_fill_rate(args.root)
