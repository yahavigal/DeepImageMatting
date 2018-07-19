import numpy as np
import cv2
import ipdb

def fs_mask_accuracy(gt, pred, threshold = 0.5):
    #ipdb.set_trace()
    intersection = np.sum(np.bitwise_and(gt > threshold,pred > threshold))
    union = np.sum(np.bitwise_or(gt > threshold,pred > threshold))
    if union == 0:
        return 1.0
    return intersection/float(union)

def fs_mask_gradient(gt, pred):
    epsilon = 1e-2
    batch_gradient = 0

    for i in xrange(len(gt)):
        gt_i = gt[i]
        pred_i = pred[i]
        gt_gauss = cv2.GaussianBlur(gt_i, (5,5), 1.4, 1.4)
        pred_gauss = cv2.GaussianBlur(pred_i, (5,5), 1.4,1.4)
        gt_grad_x = cv2.Sobel(gt_gauss, -1, 1, 0)
        gt_grad_y = cv2.Sobel(gt_gauss, -1, 0, 1)
        pred_grad_x = cv2.Sobel(pred_gauss, -1, 1, 0)
        pred_grad_y = cv2.Sobel(pred_gauss, -1, 1, 0)
        gt_mag = cv2.magnitude(gt_grad_x, gt_grad_y)
        pred_mag = cv2.magnitude(pred_grad_x, pred_grad_y)

        cv2.divide(gt_grad_x,gt_mag,gt_grad_x)
        cv2.divide(gt_grad_y,gt_mag,gt_grad_y)
        pred_grad_x[pred_grad_x < epsilon] = 0
        pred_grad_y[pred_grad_y < epsilon] = 0
        gt_grad_x[gt_grad_x < epsilon] = 0
        gt_grad_y[gt_grad_y < epsilon] = 0
        cv2.divide(pred_grad_x, pred_mag, pred_grad_x)
        cv2.divide(pred_grad_y, pred_mag, pred_grad_y)

        normalized_magnitude = cv2.magnitude(gt_grad_x - pred_grad_x, gt_grad_y - pred_grad_y)
        batch_gradient += np.average(normalized_magnitude)
    return batch_gradient/len(gt)

def fs_mask_absdiff(gt, pred):
    return np.average(np.abs(gt - pred))
