# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 18:10:32 2017

@author: or
"""

import caffe
caffe.set_mode_gpu()
import numpy as np
import os
import cv2
import random
import argparse
import ipdb
import matplotlib.pyplot as plt
import data_augmentation
import time
import platform
from PIL import Image
import sys
import re

def find_data_root_ind(image,trimap_root):
    ind = 0
    image_split = image.split(os.sep)
    trimap_split = trimap_root.split(os.sep)
    while image_split[ind] == trimap_split[ind]:
        ind +=1
    return ind +1


current_milli_time = lambda: int(round(time.time() * 1000))


class bgs_test_train:
    def __init__(self, images_dir_test, images_dir_train, solver_path,weights_path,
                 snapshot_path, batch_size=32, snapshot = 100, snapshot_diff = False,
                 trimap_dir = None, DSD_flag = False, save_loss_per_image = False, shuffle_data = True):

        self.gt_ext = "_silhuette"
        self.trimap_ext = None
        if trimap_dir is not None:
            if "triMap" in trimap_dir:
                self.trimap_ext = "triMap_"
            else:
                self.trimap_ext = "_depth"
        self.adverserial_ext = "_adv"
        self.use_adv_data_train = False
        self.trimap_dir = trimap_dir
        self.batch_size = batch_size
        self.exp_name = platform.node() + " " + solver_path.split(os.sep)[-3]
        if trimap_dir is not None:
            self.exp_name += self.trimap_ext


        if os.path.isdir(images_dir_train):
           self.images_list_train = [os.path.join(images_dir_train,x)
                                     for x in os.listdir(images_dir_train)
                                     if x.endswith(".png") and x.find(self.gt_ext) == -1]
        elif os.path.isfile(images_dir_train):
            images = open(images_dir_train).readlines()
            images = [x[0:-1] for x in images if x.endswith('\n')]
            self.images_list_train = [x for x in images
                                      if x.endswith(".png") and x.find(self.gt_ext) == -1
                                      and (self.use_adv_data_train == False or x.find(self.adverserial_ext) != -1)]
        else:
            self.images_list_train = None

        if os.path.isdir(images_dir_test):
            self.images_list_test = [os.path.join(images_dir_test,x)
                                     for x in os.listdir(images_dir_test)
                                     if x.endswith(".png") and x.find(self.gt_ext) == -1]
        elif os.path.isfile(images_dir_test):
            images = open(images_dir_test).readlines()
            images = [x[0:-1] for x in images if x.endswith('\n')]
            self.images_list_test = [x for x in images
                                      if x.endswith(".png") and x.find(self.gt_ext) == -1]
        else:
            self.images_list_test = None

        self.trimap_dir =  trimap_dir

        if self.images_list_train is not None and shuffle_data == True:
            random.shuffle(self.images_list_train)

        if solver_path.find('solver') != -1:
            self.solver = caffe.get_solver(solver_path)
        else:
            self.solver = None
            self.net = caffe.Net(solver_path, weights_path,caffe.TEST)

        if weights_path != None and weights_path != "" and os.path.isfile(weights_path):
            if self.solver is not None:
                self.solver.net.copy_from(weights_path)
            else:
                self.net.copy_from(weights_path)
        if snapshot_path != "":
            self.results_path = snapshot_path.replace(snapshot_path.split('/')[-1],"results")
        else:
            self.results_path = os.path.join(os.sep.join(solver_path.split('/')[0:-2]),"results")

        self.DSD_masks = None
        self.DSD_flag = DSD_flag
        if weights_path is not None and self.DSD_flag == True:
            path_DSD_masks = os.path.splitext(weights_path)
            path_DSD_masks = path_DSD_masks[0]+"_DSD_mask"+path_DSD_masks[1]+".npy"
            self.DSD_masks = np.load(path_DSD_masks).item()


        self.img_width =128
        self.img_height = 128
        self.list_ind = 0
        self.epoch_ind = 0
        self.iter_ind = 0
        self.snapshot = snapshot
        self.snapshot_diff = snapshot_diff
        self.shuffle = shuffle_data
        self.use_data_aug = True
        self.infer_only_trimap = False
        self.dump_bin = False
        self.view_all = False
        self.save_test_by_loss = save_loss_per_image
        self.use_tf_inference = False
        self.snapshot_path = snapshot_path
        self.train_loss = []
        self.test_loss = []
        self.train_acc = []
        self.test_acc = []

        if self.use_tf_inference == True:
            sys.path.append(os.path.join(os.getcwd(),"..","or",solver_path.split(os.sep)[-3],"scripts"))
            import convert_to_tf
            convert_to_tf.load_caffe_weights(solver_path,weights_path)
            self.tf_trainer =  convert_to_tf.TF_trainer()



    def get_tuple_data_point(self, image_path):

        if not os.path.exists(image_path):
            if self.trimap_dir == None:
                return [None, None]
            else:
                return [None, None, None]
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB).astype('float32')

        if self.use_data_aug == True:

            #image based data augmentation
            coin = np.random.uniform(0,1,1)
            if coin <= 0.25:
                img = data_augmentation.color_jitter(img)
            elif coin <= 0.5:
                img = data_augmentation.PCA_noise(img)
            elif coin <= 0.75:
                img = data_augmentation.gamma_correction(img)

        #subtract mean
        img -= np.array([104,117,123],dtype=np.float32)
        img_r = cv2.resize(img, (self.img_width,self.img_height))
        path = os.path.splitext(image_path)
        gt_path = path[0] + self.gt_ext + path[1]
        if not os.path.isfile(gt_path):
            if self.trimap_dir == None:
                return [None, None]
            else:
                return [None, None, None]
        mask = cv2.imread(gt_path,0)
        mask_r = cv2.resize(mask, (self.img_width,self.img_height),interpolation = cv2.INTER_NEAREST)

        ind = find_data_root_ind(image_path,self.trimap_dir)
        split = image_path.split(os.sep)[ind:]
        split = os.sep.join(split)
        frame_num = re.findall(r'\d+', split)[-1]
        split = os.path.split(split)

        if self.trimap_dir != None:
            trimap_path = os.path.join(self.trimap_dir,
                                       split[0],frame_num+self.trimap_ext+".png")

            if not os.path.isfile(trimap_path):
                trimap_path = os.path.join(self.trimap_dir,split[0],self.trimap_ext +frame_num+".png")
                if not os.path.isfile(trimap_path):
                    return [None, None, None]

            trimap = cv2.imread(trimap_path,0)
            trimap_r = cv2.resize(trimap, (self.img_width,self.img_height),interpolation = cv2.INTER_NEAREST)

            if self.use_data_aug == True:
                #rotation / filipping data augmentation
                coin = np.random.uniform(0,1,1)
                if  coin <= 0.33:
                    img_r, mask_r, trimap_r = data_augmentation.horizontal_flipping(img_r, mask_r,trimap_r)
                elif coin <= 0.66:
                    img_r, mask_r, trimap_r = data_augmentation.rotate(img_r,mask_r,trimap_r)


            trimap_r = trimap_r.reshape([1,self.img_height,self.img_width])
            mask_r = mask_r.reshape([1,self.img_height,self.img_width])
            img_r = img_r.transpose([2,0,1])

            return img_r,mask_r,trimap_r
        else:
            mask_r = mask_r.reshape([1,self.img_height,self.img_width])
            img_r = img_r.transpose([2,0,1])
            return img_r,mask_r

    def get_batch_data(self,batch_size = 32):
        batch=[]
        masks = []
        while len(batch) < batch_size:
            if self.list_ind>= len(self.images_list_train):
                print "starting from beginning of the list"
                if self.shuffle == True:
                    random.shuffle(self.images_list_train)
                self.epoch_ind += 1
                self.list_ind = 0
            if self.trimap_dir == None:
                img_r,mask_r = self.get_tuple_data_point(self.images_list_train[self.list_ind])
            else:
                img_r,mask_r,trimap_r = self.get_tuple_data_point(self.images_list_train[self.list_ind])
            if img_r is None or mask_r is None:
                self.list_ind += 1
                continue

            if 'trimap_r' in locals():
                img_r = np.concatenate((img_r,trimap_r),axis = 0)

            batch.append(img_r)
            masks.append(mask_r)
            self.list_ind += 1


        return np.array(batch),np.array(masks)

    def train(self):
        images, masks = self.get_batch_data(self.batch_size)
        net = self.solver.net
        net.blobs[net.inputs[0]].reshape(*images.shape)
        net.blobs[net.inputs[1]].reshape(*masks.shape)
        net.blobs[net.inputs[0]].data[...]= images
        net.blobs[net.inputs[1]].data[...]= masks
        self.solver.step(1)

        # dense sparse dense (DSD)
        if self.DSD_flag == True and self.DSD_masks is not None:
            for layer,blob in net.params.items():
                if layer not in self.DSD_masks.keys():
                    continue
                blob[0].data[self.DSD_masks[layer]] = 0

        self.iter_ind += 1
        print self.iter_ind, " loss: " ,net.blobs['loss'].data
        self.train_loss.append(net.blobs['loss'].data.flatten()[0])
        if 'mask_accuracy' in net.blobs:
            print self.iter_ind, " mask accuray: " ,net.blobs['mask_accuracy'].data
            self.train_acc.append(net.blobs['mask_accuracy'].data.flatten()[0])

        if self.iter_ind % self.snapshot == 0:
            print "snapshot iter {}".format(self.iter_ind)
            snapshot_file = os.path.join(self.snapshot_path,"_iter_"+str(self.iter_ind)+".caffemodel")
            self.solver.net.save(snapshot_file, self.snapshot_diff)

        return net.blobs['loss'].data

    def test(self, is_save_fig = True):


        diff_caffe_tf = []
        times = []
        loss_per_image = {}
        avg_iou_tf =[]
        if self.use_tf_inference == True and self.images_list_train is not None:
            for i in xrange(5000):
                x,y = self.get_batch_data(1)
                _, loss, _  = self.tf_trainer.run_fine_tune_to_deconv(x,y)
                #print "loss for fine tune is: {} IOU is: {}".format(loss,iou)
            self.tf_trainer.save()

        #no data augmentation in test
        trimap_r = None
        self.use_data_aug = False
        if self.solver is not None:
            net = self.solver.net
        else:
            net = self.net

        for image in self.images_list_test:
            if self.trimap_dir != None:
                img_r,mask_r,trimap_r = self.get_tuple_data_point(image)
            else:
                img_r,mask_r = self.get_tuple_data_point(image)

            if img_r is None:
                continue

            if self.trimap_dir is not None and trimap_r is None:
                continue

            if self.trimap_dir is not None and 'trimap_r' in locals() and trimap_r is  None:
                continue

            if trimap_r is not None:
                img_r = np.concatenate((img_r,trimap_r),axis =0)
            ls = img_r.flatten().tolist()
            img_r = img_r.reshape(1,*img_r.shape)
            mask_r = mask_r.reshape(1,*mask_r.shape)
            net.blobs[net.inputs[0]].reshape(*img_r.shape)
            net.blobs[net.inputs[1]].reshape(*mask_r.shape)
            net.blobs[net.inputs[0]].data[...]= img_r
            net.blobs[net.inputs[1]].data[...]= mask_r

            start = current_milli_time()
            net.forward()
            times.append(current_milli_time() - start)

            if self.use_tf_inference ==True:
                tf_res,iou = self.tf_trainer.run_inference(img_r,mask_r)
                avg_iou_tf.append(iou)


            self.test_loss.append(net.blobs['loss'].data.flatten()[0])
            if self.save_test_by_loss == True:
                loss_per_image[self.test_loss[-1]] = image

            if 'mask_accuracy' in net.blobs:
                self.test_acc.append(net.blobs['mask_accuracy'].data.flatten()[0])
            if is_save_fig == True:
                fig,ax = plt.subplots(1)
                ax.axis('off')
                image_orig = cv2.imread(image)
                image_orig = cv2.cvtColor(image_orig,cv2.COLOR_BGR2RGB).astype('float32')
                gt_mask = mask_r.reshape((img_r.shape[2],img_r.shape[3]))
                gt_mask = cv2.resize(gt_mask, (image_orig.shape[1],image_orig.shape[0]),interpolation = cv2.INTER_NEAREST)
                mask = net.blobs['alpha_pred'].data
                if self.use_tf_inference ==True:
                    diff_caffe_tf.append(np.mean( np.bitwise_or(np.bitwise_and(mask >=0.5,tf_res<0.5),np.bitwise_and(mask <0.5,tf_res>=0.5))))
                mask = mask.reshape((img_r.shape[2],img_r.shape[3],1))

                if np.max(mask) == 255:
                    mask /= 255.0

                if self.infer_only_trimap == True:
                    trimap = trimap_r.reshape((trimap_r.shape[2],img_r.shape[3],1))
                    mask[trimap == 255] = 1
                    mask[trimap== 128 ] = 0
                    trimap = cv2.resize(trimap, (image_orig.shape[1],image_orig.shape[0]))
                mask_r = cv2.resize(mask, (image_orig.shape[1],image_orig.shape[0]))
                mask_r_thresh = mask_r.copy()
                if self.infer_only_trimap == True:
                    zv = np.bitwise_and(mask_r < 0.5, trimap == 0)
                    ov = np.bitwise_and(mask_r >= 0.5, trimap == 0)
                else:
                    zv = mask_r < 0.5
                    ov = mask_r >= 0.5
                mask_r_thresh[zv] = 0
                mask_r_thresh[ov] = 1
                bg = cv2.imread('bg.jpg')
                bg = cv2.cvtColor(bg,cv2.COLOR_BGR2RGB)
                bg = cv2.resize(bg,(image_orig.shape[1],image_orig.shape[0]))
                bg  = np.multiply(bg/255.0,1 - mask_r[:,:,np.newaxis])

                overlay = np.multiply(image_orig/np.max(image_orig),mask_r[:,:,np.newaxis])
                mattImage = overlay + bg
                overlay_thresh = np.multiply(image_orig/np.max(image_orig),
                                             mask_r_thresh[:,:,np.newaxis])
                split = os.path.splitext(image.replace(os.sep,"_"))[0]

                ax.imshow(mattImage)
                fig_path = split+"_iou_{}.fig.jpg".format(int(100*self.test_acc[-1]))
                fig_path = os.path.join(self.results_path,fig_path)
                plt.savefig(fig_path)

                ax.imshow(overlay_thresh)
                fig_path = split +"_iou_{}.threshold.fig.jpg".format(int(100*self.test_acc[-1]))
                fig_path = os.path.join(self.results_path,fig_path)
                plt.savefig(fig_path)
                plt.close(fig)

                if self.dump_bin ==True:
                    bin_path = fig_path.replace(".fig.jpg",".input.bin")
                    output_bin_path = fig_path.replace(".fig.jpg",".output.bin")
                    dump = open(bin_path,'w')
                    out_dump = open(output_bin_path,'w')
                    for item in ls:
                        dump.write(str(int(item))+'\n')
                    dump.close()
                    for item in mask.flatten().tolist():
                        out_dump.write(str(int(item*255))+'\n')
                    out_dump.close()

                if self.view_all == True:
                    fig = plt.figure(figsize = (8,8))
                    fig.canvas.set_window_title("bla bla")
                    plt.subplot(2,2,1)
                    plt.axis('off')
                    plt.title("trimap input")
                    trimap = trimap_r.reshape((trimap_r.shape[2],img_r.shape[3],1))
                    trimap = np.repeat(trimap,3,axis = 2)
                    trimap = cv2.resize(trimap,(image_orig.shape[1],image_orig.shape[0]))
                    trimap[np.any(trimap == [0,0,0],axis = -1)] = (255,0,0)
                    image_trimap = Image.fromarray(trimap)
                    image_Image = Image.fromarray(image_orig.astype(np.uint8))
                    trimap_blend = Image.blend(image_trimap,image_Image,0.8)
                    plt.imshow(trimap_blend)

                    plt.subplot(2,2,2)
                    plt.axis('off')
                    plt.title("GT input")
                    gt_input = np.array(Image.fromarray(gt_mask).convert('RGB'))
                    gt_input[np.any(gt_input == [255,255,255],axis = -1)] = (255,0,0)
                    gt_input = Image.fromarray(gt_input)
                    gt_blend = Image.blend(gt_input,image_Image,0.8)
                    plt.imshow(gt_blend)

                    plt.subplot(2,2,3)
                    plt.axis('off')
                    plt.title("algo results")
                    mask_r[mask_r > 0.5] = 1
                    algo_res = np.array(Image.fromarray(mask_r).convert('RGB'))
                    algo_res[np.any(algo_res == [1,1,1],axis = -1)] = (255,0,0)
                    algo_res = Image.fromarray(algo_res)
                    algo_res = Image.blend(algo_res,image_Image,0.8)
                    plt.imshow(algo_res)
                    fig_path = split +"_iou_{}.all.fig.jpg".format(int(100*self.test_acc[-1]))
                    fig_path = os.path.join(self.results_path,fig_path)
                    plt.savefig(fig_path)
                    plt.close(fig)





        print "{} average loss on test: {} average accuracy on test {}".format(self.exp_name,
                                                                        np.average(self.test_loss),
                                                                        np.average(self.test_acc))

        print "{} average time for inference: {}".format(self.exp_name,np.average(times))

        if self.use_tf_inference ==True:
            print "average accuracy on test in TF {}".format(np.average(avg_iou_tf))
            plt.hist(diff_caffe_tf, bins=100)
            plt.show()

        if self.save_test_by_loss == True:
            return loss_per_image
        return np.average(self.test_loss), np.average(self.test_acc)


    def plot_statistics(self):
        fig = plt.figure()
        fig.canvas.set_window_title(self.exp_name)

        plt.subplot(2,2,1)
        plt.title('train loss')
        plt.plot(xrange(len(self.train_loss)),self.train_loss)
        plt.xlabel('# iter')
        plt.ylabel('train loss')

        plt.subplot(2,2,2)
        plt.title('test loss')
        plt.plot(xrange(len(self.test_loss)),self.test_loss)
        plt.xlabel('# iter')
        plt.ylabel('test loss')

        plt.subplot(2,2,3)
        plt.title('train accuracy')
        plt.plot(xrange(len(self.train_acc)),self.train_acc)
        plt.xlabel('# iter')
        plt.ylabel('train accuracy')

        plt.subplot(2,2,4)
        plt.title('test accuracy')
        plt.plot(xrange(len(self.test_acc)),self.test_acc)
        plt.xlabel('# iter')
        plt.ylabel('test accuracy')

        plt.show()

def train_epochs(images_dir_test, images_dir_train, solver_path,weights_path,epochs_num, trimap_dir,DSD,shuffle):
    snapshot_path = solver_path.replace("proto","snapshots",1)
    snapshot_path = os.path.split(snapshot_path)[0]
    trainer = bgs_test_train(images_dir_test, images_dir_train, solver_path,weights_path,snapshot_path,
                             trimap_dir = trimap_dir,DSD_flag = DSD,shuffle_data=shuffle)

    while trainer.epoch_ind < epochs_num:
        trainer.train()

    trainer.test()
    trainer.plot_statistics()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, required=True, help="train directory path or list")
    parser.add_argument('--test_dir', type=str, required=True, help="test directory path or list")
    parser.add_argument('--trimap_dir', type=str, required=False, default = None,help="trimap or any addtional output")
    parser.add_argument('--solver', type=str, required=True,help="path to solver")
    parser.add_argument('--model', type=str, required=False, default = None, help="pre-trained weights path")
    parser.add_argument('--epochs', type=int, required=False, default = 60, help="number or epochs each epoch is equivalent to ")
    parser.add_argument('--DSD', action = 'store_true', help="use dense-sparse-dense mask and train with this restriction")
    parser.add_argument('--no_shuffle', action='store_false', help="training with no shuffle, shuufle the data by default")
    parser.add_argument('--gpu', type=int,required=False, default = 0, help= "GPU ID for multiple GPU machine")
    args = parser.parse_args()
    caffe.set_device(args.gpu)
    train_epochs(args.test_dir,args.train_dir,args.solver,args.model,args.epochs,args.trimap_dir,DSD=args.DSD,
                 shuffle=args.no_shuffle)





