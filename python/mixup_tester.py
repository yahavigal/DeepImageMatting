#import data_augmentation as da
import matplotlib.pyplot as plt
import cv2
import ipdb
import numpy as np

def mixup_classic(image1, image2, gt1, gt2):
    alpha = 0.3
    lam = np.random.beta(alpha,alpha)
    x = lam * image1 + (1-lam) * image2
    y = lam * gt1 + (1-lam) * gt2
    return x, y

def mixup_stitch(image1, image2, gt1, gt2, trimap1 = None, trimap2 = None):
    rows, cols, dim = image1.shape
    while(1):
        ipdb.set_trace()
        nonzeros = cv2.findNonZero(gt1.astype(np.uint8))
        pnt1 = np.random.randint(0, len(nonzeros))
        pnt2 = np.random.randint(0, len(nonzeros))
        x1, y1 = nonzeros[pnt1][0]
        x2, y2 = nonzeros[pnt2][0]
        if len(set([x1, x2, y1, y2])) == 4:
            a = float(y1-y2)/(x1-x2)
            b = y1 - a*x1
            if a!=1.0:
                break
    if abs(a)<1.0:
        for x in range(cols):
            image1[:int(a*x+b), x, :] = image2[:int(a*x+b), x, :]
            gt1[:int(a*x+b), x] = gt2[:int(a*x+b), x]
    else:
        for y in range(rows):
            image1[y, :int((y-b)/a),:] = image2[y, :int((y-b)/a), :]
            gt1[y, :int((y-b)/a)] = gt2[y, :int((y-b)/a)]
    return image1, gt1

pref = '/home/or/share/DataSet_1_new/images/w10720472/standing/'
image1 = cv2.imread(pref + '231_color.png')
image2 = cv2.imread(pref + '381_color.png')
img1_copy = image1.copy()
gt1 = cv2.imread(pref + '231_color_silhuette.png',0)
gt2 = cv2.imread(pref + '381_color_silhuette.png',0)
gt1_copy = gt1.copy()
image, gt = mixup_stitch(image1, image2, gt1, gt2)
#ipdb.set_trace()
plt.subplot(231)
plt.imshow(img1_copy)
plt.subplot(234)
plt.imshow(gt1_copy)
plt.subplot(232)
plt.imshow(image2)
plt.subplot(235)
plt.imshow(gt2)
plt.subplot(233)
plt.imshow(image)
plt.subplot(236)
plt.imshow(gt)
plt.show()
ipdb.set_trace()
