# coding: utf-8
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import ipdb
import argparse
def find_depth_fill_rate(root):
    total_fill_rate =0
    num_of_files = 0
    fill_rates =[]
    for root,dirs,files in os.walk(root):
        for file in files:
            depth = cv2.imread(os.path.join(root,file),0)
            if depth is None:
                continue
            fill_rate = np.sum(depth ==0)/float(depth.size)
            fill_rates.append(fill_rate)
            total_fill_rate += fill_rate
            num_of_files += 1
    print "average fill rate {} for {} files".format(total_fill_rate/num_of_files,num_of_files)
    plt.title("histogram of depth fill rates")
    plt.hist(fill_rates,bins=100)
    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True, help="root to the depth tree")
    args = parser.parse_args()

    find_depth_fill_rate(args.root)
