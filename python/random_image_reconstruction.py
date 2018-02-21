# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 18:17:05 2017

@author: or
"""

import caffe
caffe.set_mode_gpu()
caffe.set_device(0)
import numpy as np
import data_provider
import os
import argparse
import ipdb
import matplotlib.pyplot as plt
import shutil
import re
import math

def find_square(num):
    height = int(math.sqrt(num))
    width = num / height
    if width*height < num:
        height += 1
    return width,height


def random_image_reconstruction(proto, model, inputs_data, output_folder,
                                 layer_name, epsilon=1e-6, lr=1e-4, trimap_data = None):

    net = caffe.Net(proto,model,caffe.TRAIN)
    img_width = net.blobs[net.inputs[0]].shape[3]
    img_height = net.blobs[net.inputs[0]].shape[2]
    img_channels = net.blobs[net.inputs[0]].shape[1]
    layer_name_1 = net._layer_names[list(net._layer_names).index(layer_name)]
    end_layer = net._layer_names[0]

    provider = data_provider.DataProvider(inputs_data,'',trimap_data,img_width=img_width,
                                          img_height=img_height,use_data_aug=False,shuffle_data=False)

    if  os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)


    for _ in xrange(len(provider.images_list_test)):
        img,mask  = provider.get_test_data()
        net.blobs[net.inputs[0]].reshape(*img.shape)
        net.blobs[net.inputs[1]].reshape(*mask.shape)
        net.blobs[net.inputs[0]].data[...] = img
        net.blobs[net.inputs[1]].data[...] = mask
        net.forward()

        image_path = provider.images_path_in_batch[0]
        last_sep = [m.start() for m in re.finditer(r'{}'.format(os.sep),image_path)][provider.root_data_ind-1]
        split = os.path.splitext(image_path[last_sep:].replace(os.sep,"_"))[0]

        target_features = net.blobs[layer_name].data.copy()
        plot_width, plot_height = find_square(target_features.shape[1])
        fig = plt.figure(figsize=(plot_width*3,plot_height*3))
        plt.axis('off')
        for ind,feature in enumerate(target_features[0]):
            plt.subplot(plot_height,plot_width,ind+1)
            plt.imshow(np.squeeze(feature))
        plt.savefig(os.path.join(output_folder,split+'_target_features.fig.jpg'))
        plt.close(fig)
        random_noise = img.copy()
        for i in xrange(random_noise.shape[1]):
            random_noise[:,i,:] = np.random.uniform(0,255,size=(img_height,img_width))
            l2_loss = np.inf
            iters = 0
            last_loss = 0
            low_loss_counter = 0
            last_grad =0
            base_lr = lr
            while l2_loss > epsilon:

                #if base_lr < 1e-3:
                #    break

                net.blobs[net.inputs[0]].data[...] = random_noise
                net.blobs[net.inputs[1]].data[...] = mask
                net.forward()
                l2_loss = np.linalg.norm(net.blobs[layer_name].data - target_features)
                if iters % 1000 == 0:
                    print 'iter {} l2 loss with layer {} channel {}: {} lr is {} counter for lr is {} norm with orig image {}'.format(
                            iters, layer_name,i, l2_loss, base_lr, low_loss_counter,np.linalg.norm(img - random_noise))
                l2_grad = 2.0*(net.blobs[layer_name].data - target_features)
                net.blobs[layer_name].diff[...] = l2_grad
                net.backward(start=layer_name,end=end_layer)
                grad_wrt_input = net.blobs[net.inputs[0]].diff.copy()
                if np.any(grad_wrt_input) == False:
                    continue

                random_noise[:,i,:] -= base_lr*(0.1*grad_wrt_input + 0.9*last_grad)[:,i,:]
                #random_noise -= lr*grad_wrt_input
                last_grad = (0.1*grad_wrt_input + 0.9*last_grad).copy()
                iters += 1
                if  np.abs(last_loss - l2_loss)  < base_lr**2*1e-3:
                    low_loss_counter +=1
                if low_loss_counter == 500:
                    break
                    #base_lr*=0.0001
                last_loss = l2_loss
        fig = plt.figure(figsize=(9,9))
        plt.subplot(3,3,1)
        plt.title('R')
        plt.imshow(np.squeeze(random_noise[:,0,:]+104)/255.0)
        plt.subplot(3,3,2)
        plt.title('G')
        plt.imshow(np.squeeze(random_noise[:,1,:]+117)/255.0)
        plt.subplot(3,3,3)
        plt.title('B')
        plt.imshow(np.squeeze(random_noise[:,2,:]+123)/255.0)
        plt.subplot(3,3,4)
        plt.title('T or D')
        plt.imshow(np.squeeze(random_noise[:,3,:])/255.0)
        plt.subplot(3,3,5)
        plt.title('RGB')
        rgb = np.squeeze(random_noise[:,0:-1,:]).transpose([1,2,0]) + np.array([104,117,123])
        plt.imshow(rgb/255.0)
        plt.subplot(3,3,6)
        plt.title('RGB orig')
        rgb_orig = np.squeeze(img[:,0:-1,:]).transpose([1,2,0]) + np.array([104,117,123])
        plt.imshow(rgb_orig/255.0)
        plt.subplot(3,3,7)
        plt.title('RGB diff')
        plt.imshow(np.abs(rgb-rgb_orig))
        plt.subplot(3,3,8)
        plt.title('T or D orig')
        plt.imshow(np.squeeze(img[:,3,:])/255.0)
        plt.subplot(3,3,9)
        plt.title('T or D diff')
        plt.imshow(np.squeeze(np.abs(img[:,3,:]-random_noise[:,3,:])))

        plt.savefig(os.path.join(output_folder,split+'_reconstructed.fig.jpg'))
        plt.close(fig)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--images_dir', type=str, required=True)
    parser.add_argument('--proto', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--layer', type=str, required=True, help='layer name to reconstruct')
    parser.add_argument('--trimap', type=str, required=False,default = None)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--lr', type=float, required=False, default=1e-6, help='learning rate how to change the random noise')
    parser.add_argument('--epsilon', type=float, required=False, default=0.001, help= 'epsilon value when stop the iterations')
    args = parser.parse_args()
    #with keyboard.Listener(on_press=on_press,on_release=on_release) as listener:
    #    listener.join()
    random_image_reconstruction(args.proto, args.model, args.images_dir,
                                 args.output, args.layer,args.epsilon,args.lr ,args.trimap)







