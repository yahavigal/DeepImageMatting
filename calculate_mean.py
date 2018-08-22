import os
from os import listdir
from PIL import Image

# Enter your folder path here:
FOLDER = "C:\\Users\\Assaf Avital\\PycharmProjects\\Mask_RCNN\\globerson_dataset"

# Mean values of validation set:
[114.14099728830178, 102.61145065667115, 92.81196657998234]

mean_rgb = [0] * 3
counter = 0
for f in os.listdir(FOLDER):
    if f.endswith("_color.png"):
        im = Image.open(FOLDER + "\\" + f)
        counter += 1
        inner_rgb = [0] * 3
        size = im.size[0] * im.size[1]
        for pixel in list(im.getdata()):
            for channel in range(len(inner_rgb)):
                inner_rgb[channel] += pixel[channel]
        for channel in range(len(mean_rgb)):
            mean_rgb[channel] += inner_rgb[channel] / size

for channel in range(len(mean_rgb)):
    mean_rgb[channel] = mean_rgb[channel] / counter

print(mean_rgb)