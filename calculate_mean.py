import os
from os import listdir
from PIL import Image
import ipdb

# Mean values of validation set:
[114.14099728830178, 102.61145065667115, 92.81196657998234]

mean_rgb = [0] * 3
counter = 0
with open("/media/or/Data/COCO_person_data/bgs_train2017/train.txt", 'r') as f:
    for line in f.readlines():
        im = Image.open(line.rstrip())
        counter += 1
        inner_rgb = [0] * 3
        size = im.size[0] * im.size[1]
        for pixel in list(im.getdata()):
            for channel in range(len(inner_rgb)):
                inner_rgb[channel] += pixel[channel]
        for channel in range(len(mean_rgb)):
            mean_rgb[channel] += inner_rgb[channel] / size
        print("{}: ({},{},{})".format(counter, inner_rgb[0]/size, inner_rgb[1]/size, inner_rgb[2]/size))

for channel in range(len(mean_rgb)):
    mean_rgb[channel] = mean_rgb[channel] / counter

print(mean_rgb)
#with open("train_mean.txt",'w+') as g:
#    g.write(str(mean_rgb))

