import matplotlib.pyplot as plt
import cv2
import os
from PIL import Image
import numpy as np
import re
import ipdb


def sigmoid(arr):
    return 1.0/(1+np.exp(-arr))

def plot_test_images(data_provider,net, ind_in_batch, dump_bin, view_all,
                     infer_only_trimap, results_path,iou,input_bin):

    image_path = data_provider.images_path_in_batch[ind_in_batch]
    image_orig = data_provider.img_orig[ind_in_batch]
    gt_mask = data_provider.mask_orig[ind_in_batch]
    if 'alpha_pred_s' in net.blobs.keys():
        mask = net.blobs['alpha_pred_s'].data[ind_in_batch]
    else:
        mask = net.blobs['alpha_pred'].data[ind_in_batch]
    mask = mask.reshape((data_provider.img_height,data_provider.img_width,1))

    if (np.min(mask) < 0 or np.max(mask) > 1):
        mask = sigmoid(mask)

    if infer_only_trimap == True:
        trimap = data_provider.trimap_resized[ind_in_batch]
        mask[trimap == 255] = 1
        mask[trimap== 128 ] = 0

    mask_r = cv2.resize(mask, (image_orig.shape[1],image_orig.shape[0]))
    bg = cv2.imread('bg.jpg')
    bg = cv2.cvtColor(bg,cv2.COLOR_BGR2RGB)
    bg = cv2.resize(bg,(image_orig.shape[1],image_orig.shape[0]))
    bg  = np.multiply(bg/255.0,1 - mask_r[:,:,np.newaxis])
    overlay = np.multiply(image_orig/np.max(image_orig),mask_r[:,:,np.newaxis])
    mattImage = overlay + bg

    last_sep = [m.start() for m in re.finditer(r'{}'.format(os.sep),image_path)][data_provider.root_data_ind-1]
    split = os.path.splitext(image_path[last_sep:].replace(os.sep,"_"))[0]
    fig_path = split+"_iou_{}.jpg".format(iou)
    fig_path = os.path.join(results_path,fig_path)
    Image.fromarray((mattImage*255).astype(np.uint8)).save(fig_path)

    mask_path = split +"_iou_{}.mask.png".format(iou)
    mask_path = os.path.join(results_path,mask_path)
    cv2.imwrite(mask_path,255*mask_r)

    if dump_bin ==True:
        bin_path = fig_path.replace(".fig.jpg",".input.bin")
        output_bin_path = fig_path.replace(".fig.jpg",".output.bin")
        dump = open(bin_path,'w')
        out_dump = open(output_bin_path,'w')
        for item in input_bin:
            dump.writelines(str(int(item))+'\n')
        dump.close()
        for item in mask.flatten().tolist():
            out_dump.write(str(int(item*255))+'\n')
        out_dump.close()

    if view_all == True:
        fig = plt.figure(figsize = (8,8))
        plt.subplot(2,2,1)
        plt.axis('off')
        plt.title("trimap input")
        trimap = data_provider.trimap_orig[ind_in_batch]
        trimap = np.repeat(np.expand_dims(trimap,axis=2),3,axis=2)
        trimap[np.any(trimap == [0,0,0],axis = -1)] = (255,0,0)
        image_trimap = Image.fromarray(trimap)
        image_trimap = image_trimap.resize((image_orig.shape[1],image_orig.shape[0]))
        image_Image = Image.fromarray(image_orig.astype(np.uint8))

        trimap_blend = Image.blend(image_trimap,image_Image,0.8)
        plt.imshow(trimap_blend)

        plt.subplot(2,2,2)
        plt.axis('off')
        plt.title("GT input")
        gt_mask = np.repeat(np.expand_dims((gt_mask*255).astype(np.uint8),axis=2),3,axis=2)
        gt_mask[:,:,1:] = 0
        gt_input = Image.fromarray(gt_mask)
        gt_blend = Image.blend(gt_input,image_Image,0.8)
        plt.imshow(gt_blend)

        plt.subplot(2,2,3)
        plt.axis('off')
        plt.title("algo results")
        mask_r = np.repeat(np.expand_dims(mask_r,axis=2),3,axis=2)
        mask_r[:,:,1:] = 0
        algo_res = Image.fromarray((mask_r*255).astype(np.uint8))
        algo_res = Image.blend(algo_res,image_Image,0.8)
        plt.imshow(algo_res)
        fig_path = split +"_iou_{}.all.fig.jpg".format(iou)
        fig.canvas.set_window_title(fig_path)
        fig_path = os.path.join(results_path,fig_path)

        plt.savefig(fig_path)
        plt.close(fig)
