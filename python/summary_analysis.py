import numpy as np
import matplotlib.pyplot as plt
import ipdb

path_no_mixup_60 = "/home/or/caffe-BGS-yahav/or/deepImageMatting/results_1/summary.txt"
path_no_mixup_150 = "/home/or/caffe-BGS-yahav/or/deepImageMatting/results_2/summary.txt"
path_mixup_stitch_60 = "/home/or/caffe-BGS-yahav/or/deepImageMatting/results_mixup_stitch_1/summary.txt"
path_mixup_stitch_150 = "/home/or/caffe-BGS-yahav/or/deepImageMatting/results_mixup_stitch_2/summary.txt"
paths = [path_no_mixup_60, path_no_mixup_150, path_mixup_stitch_60, path_mixup_stitch_150]
losses, accuracies = [], []
for path in paths:
    with open(path) as f:
        loss, accuracy = [], []
        for line in f.readlines():
            if " loss" in line:
                print line
                words = line.split()
                loss.append(words[6])
            if " mask_accuracy" in line:
                print line
                words = line.split()
                accuracy.append(words[6])
        losses.append(loss)
        accuracies.append(accuracy)
print "losses: {}, accuracies: {}".format(losses, accuracies)
