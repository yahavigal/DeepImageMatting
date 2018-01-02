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
from collections import defaultdict
from data_provider import *
from google.protobuf import text_format
from caffe.proto import caffe_pb2
import publish_utils
import shutil

current_milli_time = lambda: int(round(time.time() * 1000))
def sigmoid(arr):
    return 1.0/(1+np.exp(-arr))

def check_threshold_param(net_file,threshold):
    if threshold == -1:
        return
    net = caffe_pb2.NetParameter()
    text_format.Merge(open(net_file).read(),net)
    for layer in net.layer:
        if layer.type == "MaskIOU":
            if threshold != layer.threshold_param.threshold:
                raise Exception("IOU threshold not consistent with the threshold supplied")


#python bgs_train_test.py --train_dir ../or/deepImageMatting/scripts/train_list.txt --test_dir ../or/deepImageMatting/scripts/test_list.txt --trimap_dir /media/or/Data/deepImageMatting/set1_07_2017_depth_norm/ --solver ../or/fastPortraitMatting/proto/solver.prototxt --model  ../or/fastPortraitMatting/snapshots/_iter_1792_MaxAccuracy9691.caffemodel

#or@ortrainubuntu5:~/caffe-BGS-win/python$ python bgs_train_test.py --train_dir ../or/deepImageMatting/scripts/composed/train_list.txt --test_dir ../or/deepImageMatting/scripts/composed/test_list.txt --trimap_dir /media/or/Data/composed/DataSet_2_composed_depth_norm/ --solver ../or/fastPortraitMatting/proto/solver.prototxt --model  ../or/fastPortraitMatting/snapshots/_iter_9180_MaxAccuracy8968.caffemodel


class bgs_test_train:
    def __init__(self, images_dir_test, images_dir_train, solver_path,weights_path,
                 snapshot_path, batch_size=128, snapshot = 100, snapshot_diff = False,
                 trimap_dir = None, DSD_flag = False, save_loss_per_image = False, shuffle_data = True,
                 threshold = -1,results_path=None):

        self.threhold_param = threshold
        if trimap_dir is not None:
            if "trimap" in trimap_dir.lower():
                self.trimap_ext = "_triMap"
            else:
                self.trimap_ext = "_depth"
        self.exp_name = platform.node() + " " + solver_path.split(os.sep)[-3]
        if trimap_dir is not None:
            self.exp_name += self.trimap_ext

        if solver_path.find('solver') != -1:
            self.solver = caffe.get_solver(solver_path)
            sp = caffe_pb2.SolverParameter()
            text_format.Merge(open(solver_path).read(),sp)
            check_threshold_param(sp.net,threshold)
            img_width = self.solver.net.blobs[self.solver.net.inputs[0]].shape[3]
            img_height = self.solver.net.blobs[self.solver.net.inputs[0]].shape[2]
        else:
            self.solver = None
            self.net = caffe.Net(solver_path, weights_path,caffe.TEST)
            check_threshold_param(solver_path,threshold)
            img_width = self.net.blobs[self.net.inputs[0]].shape[3]
            img_height = self.net.blobs[self.net.inputs[0]].shape[2]

        self.data_provider = DataProvider(images_dir_test,images_dir_train,trimap_dir,shuffle_data,
                                          batch_size=batch_size,use_data_aug=True,use_adv_data_train=False,
                                          threshold_param= self.threhold_param,img_width= img_width,img_height=img_height)
        self.exp_name += "_{}X{}".format(self.data_provider.img_width,self.data_provider.img_height)
        self.exp_name += "_threshold_{}".format(self.threhold_param)

        if weights_path != None and weights_path != "" and os.path.isfile(weights_path):
            if self.solver is not None:
                self.solver.net.copy_from(weights_path)
            else:
                self.net.copy_from(weights_path)
        if snapshot_path != "":
            self.results_path = snapshot_path.replace(snapshot_path.split('/')[-1],"results")
        else:
            if results_path is not None and len(results_path) > 0:
                self.results_path = results_path
            else:
                self.results_path = os.path.join(os.sep.join(solver_path.split('/')[0:-2]),"results")


        self.DSD_masks = None
        self.DSD_flag = DSD_flag
        if weights_path is not None and self.DSD_flag == True:
            path_DSD_masks = os.path.splitext(weights_path)
            path_DSD_masks = path_DSD_masks[0]+"_DSD_mask"+path_DSD_masks[1]+".npy"
            self.DSD_masks = np.load(path_DSD_masks).item()

        self.snapshot = snapshot
        self.snapshot_diff = snapshot_diff
        self.infer_only_trimap = False
        self.dump_bin = False
        self.view_all = True
        self.save_test_by_loss = save_loss_per_image
        self.use_tf_inference = False
        self.snapshot_path = snapshot_path
        self.train_measures = defaultdict(list)
        self.test_measures = defaultdict(list)
        self.maxAccuracy = 0.0;
        self.saveByMaxAccuracy = False

        if not os.path.exists(self.snapshot_path) and self.data_provider.images_list_train is not None:
            os.makedirs(self.snapshot_path)
        if not os.path.exists(self.results_path):
            os.makedirs(self.results_path)

        if self.use_tf_inference == True:
            sys.path.append(os.path.join(os.getcwd(),"..","or",solver_path.split(os.sep)[-3],"scripts"))
            import convert_to_tf
            convert_to_tf.load_caffe_weights(solver_path,weights_path)
            self.tf_trainer =  convert_to_tf.TF_trainer()

    def train(self):
        images, masks = self.data_provider.get_batch_data()
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
        self.data_provider.iter_ind += 1

        for output in net.outputs:
            if output == 'alpha_pred' or output == 'alpha_pred_s':
                continue
            self.train_measures[output].append(net.blobs[output].data.flatten()[0])
            print self.data_provider.iter_ind , " {}:  {} ".format(output,net.blobs[output].data)

        if self.saveByMaxAccuracy == True:
            if 'mask_accuracy' in self.train_measures.keys():
                isTosaveMaxAccuracyModel = False
                if self.maxAccuracy < self.train_measures['mask_accuracy'][-1] :
                    self.maxAccuracy = self.train_measures['mask_accuracy'][-1]
                    isTosaveMaxAccuracyModel = True

                if isTosaveMaxAccuracyModel == True:
                    isTosaveMaxAccuracyModel = False
                    print "snapshot iter {}".format(self.data_provider.iter_ind)
                    snapshot_file = os.path.join(self.snapshot_path,"_iter_"+str(self.data_provider.iter_ind)+"_MaxAccuracy" + str(int(10000*net.blobs['mask_accuracy'].data.flatten()[0]))+".caffemodel")
                    self.solver.net.save(snapshot_file, self.snapshot_diff)


        if self.data_provider.iter_ind % self.snapshot == 0:
            print "snapshot iter {}".format(self.data_provider.iter_ind)
            snapshot_file = os.path.join(self.snapshot_path,"_iter_"+str(self.data_provider.iter_ind)+".caffemodel")
            self.solver.net.save(snapshot_file, self.snapshot_diff)

        return net.blobs['loss'].data

    def test(self, is_save_fig = True):

        if self.solver is not None:
            net = self.solver.net
        else:
            net = self.net

        test_log_file = open(os.path.join(self.results_path,"test_log_file.txt"),"w")
        test_log_file.write('image_path ')
        for output in net.outputs:
            if output == 'alpha_pred' or output == 'alpha_pred_s':
                continue
            test_log_file.write(" {}".format(output))
        test_log_file.write('\n')
        diff_caffe_tf = []
        times = []
        loss_per_image = {}
        avg_iou_tf =[]
        if self.use_tf_inference == True and self.data_provider.images_list_train is not None:
            for i in xrange(5000):
                x,y = self.data_provider.get_batch_data(1)
                _, loss, _  = self.tf_trainer.run_fine_tune_to_deconv(x,y)
                #print "loss for fine tune is: {} IOU is: {}".format(loss,iou)
            self.tf_trainer.save()
            print "data saved"

        #no data augmentation in test
        trimap_r = None
        self.data_provider.use_data_aug = False

        for _ in xrange(len(self.data_provider.images_list_test)):
            img_r,mask_r = self.data_provider.get_test_data()
            if img_r is None or mask_r is None:
                continue
            image = self.data_provider.test_image_path
            net.blobs[net.inputs[0]].reshape(*img_r.shape)
            net.blobs[net.inputs[1]].reshape(*mask_r.shape)
            net.blobs[net.inputs[0]].data[...]= img_r
            net.blobs[net.inputs[1]].data[...]= mask_r
            ls = img_r.flatten().tolist()

            start = current_milli_time()
            net.forward()
            times.append(current_milli_time() - start)

            if self.use_tf_inference ==True:
                tf_res,iou = self.tf_trainer.run_inference(img_r,mask_r)
                avg_iou_tf.append(iou)

            test_log_file.write(image)
            for output in net.outputs:
                if output == 'alpha_pred' or output == 'alpha_pred_s':
                    continue
                self.test_measures[output].append(net.blobs[output].data.flatten()[0])
                test_log_file.write(" {}".format(self.test_measures[output][-1]))
            test_log_file.write('\n')


            if self.save_test_by_loss == True:
                loss_per_image[self.test_measures['loss'][-1]] = image

            if is_save_fig == True:
                fig,ax = plt.subplots(1)
                ax.axis('off')
                image_orig = self.data_provider.img_orig
                gt_mask = self.data_provider.mask_orig
                mask = net.blobs['alpha_pred_s'].data
                if self.use_tf_inference ==True:
                    diff_caffe_tf.append(np.mean( np.bitwise_or(np.bitwise_and(mask >=0.5,tf_res<0.5),np.bitwise_and(mask <0.5,tf_res>=0.5))))
                mask = mask.reshape((img_r.shape[2],img_r.shape[3],1))

                if (np.min(mask) < 0 or np.max(mask) > 1):
                    mask = sigmoid(mask)

                if self.infer_only_trimap == True:
                    trimap = self.data_provider.trimap_resized
                    mask[trimap == 255] = 1
                    mask[trimap== 128 ] = 0

                mask_r = cv2.resize(mask, (image_orig.shape[1],image_orig.shape[0]))
                bg = cv2.imread('bg.jpg')
                bg = cv2.cvtColor(bg,cv2.COLOR_BGR2RGB)
                bg = cv2.resize(bg,(image_orig.shape[1],image_orig.shape[0]))
                bg  = np.multiply(bg/255.0,1 - mask_r[:,:,np.newaxis])
                overlay = np.multiply(image_orig/np.max(image_orig),mask_r[:,:,np.newaxis])
                mattImage = overlay + bg

                last_sep = [m.start() for m in re.finditer(r'{}'.format(os.sep),image)][self.data_provider.root_data_ind-1]
                split = os.path.splitext(image[last_sep:].replace(os.sep,"_"))[0]
                iou = int(100*self.test_measures['mask_accuracy'][-1])
                ax.imshow(mattImage)
                fig_path = split+"_iou_{}.fig.jpg".format(iou)
                fig_path = os.path.join(self.results_path,fig_path)
                plt.savefig(fig_path)

                plt.close(fig)

                mask_path = split +"_iou_{}.mask.png".format(iou)
                mask_path = os.path.join(self.results_path,mask_path)
                cv2.imwrite(mask_path,255*mask_r)

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
                    plt.subplot(2,2,1)
                    plt.axis('off')
                    plt.title("trimap input")
                    trimap = self.data_provider.trimap_orig
                    trimap = np.repeat(np.expand_dims(trimap,axis=2),3,axis=2)
                    trimap[np.any(trimap == [0,0,0],axis = -1)] = (255,0,0)
                    image_trimap = Image.fromarray(trimap)
                    image_Image = Image.fromarray(image_orig.astype(np.uint8))
                    trimap_blend = Image.blend(image_trimap,image_Image,0.8)
                    plt.imshow(trimap_blend)

                    plt.subplot(2,2,2)
                    plt.axis('off')
                    plt.title("GT input")
                    gt_mask*=255
                    gt_mask = np.repeat(np.expand_dims(gt_mask,axis=2),3,axis=2)
                    gt_mask[:,:,1:] = 0
                    gt_input = Image.fromarray(gt_mask)
                    gt_blend = Image.blend(gt_input,image_Image,0.8)
                    plt.imshow(gt_blend)

                    plt.subplot(2,2,3)
                    plt.axis('off')
                    plt.title("algo results")
                    if self.threhold_param != -1:
                        mask_r[mask_r >= self.threhold_param] = 1
                        mask_r[mask_r < self.threhold_param] = 0
                    else:
                        mask_r/=255.0
                    mask_r = np.repeat(np.expand_dims(mask_r,axis=2),3,axis=2)
                    mask_r[:,:,1:] = 0
                    algo_res = Image.fromarray((mask_r*255).astype(np.uint8))
                    algo_res = Image.blend(algo_res,image_Image,0.8)
                    plt.imshow(algo_res)
                    fig_path = split +"_iou_{}.all.fig.jpg".format(iou)
                    fig.canvas.set_window_title(fig_path)
                    fig_path = os.path.join(self.results_path,fig_path)

                    #plt.subplot(2,2,4)
                    #plt.axis('off')
                    #plt.title("mask abs diff")
                    #plt.imshow(np.abs(mask_r - gt_mask))

                    plt.savefig(fig_path)
                    plt.close(fig)

        for output in net.outputs:
            if output == 'alpha_pred' or output == 'alpha_pred_s':
                continue
            print "{} average {} on test: {} ".format(self.exp_name,output,np.average(self.test_measures[output]))
            print "{} average {} on train: {} ".format(self.exp_name,output,np.average(self.train_measures[output]))

        print "{} average time for inference: {}".format(self.exp_name,np.average(times))

        if self.use_tf_inference ==True:
            print "average iou on test in TF {}".format(np.average(avg_iou_tf))
            plt.hist(diff_caffe_tf, bins=100)
            plt.show()

        test_log_file.close()

        if self.save_test_by_loss == True:
            return loss_per_image
        return np.average(self.test_measures['loss']), np.average(self.test_measures['mask_accuracy'])


    def plot_statistics(self):
        fig = plt.figure()
        fig.canvas.set_window_title(self.exp_name)

        if self.solver is not None:
            net = self.solver.net
        else:
            net = self.net

        for i,output in enumerate(net.outputs):
            if output == 'alpha_pred' or output == 'alpha_pred_s':
                continue
            plt.subplot(2,len(self.train_measures),i+1)
            plt.title('train {}'.format(output))
            plt.plot(xrange(len(self.train_measures[output])),self.train_measures[output])
            plt.xlabel('# iter')
            plt.ylabel('train {}'.format(output))

            plt.subplot(2,len(self.train_measures),i+1+len(self.train_measures))
            plt.title('test {}'.format(output))
            plt.plot(xrange(len(self.test_measures[output])),self.test_measures[output])
            plt.xlabel('# iter')
            plt.ylabel('test {}'.format(output))

        plt.show()

def train_epochs(images_dir_test, images_dir_train, solver_path,weights_path,epochs_num, trimap_dir,DSD,shuffle,threshold,publish):
    snapshot_path = solver_path.replace("proto","snapshots",1)
    snapshot_path = os.path.split(snapshot_path)[0]
    trainer = bgs_test_train(images_dir_test, images_dir_train, solver_path,weights_path,snapshot_path,
                             trimap_dir = trimap_dir,DSD_flag = DSD,shuffle_data=shuffle,threshold=threshold)

    while trainer.data_provider.epoch_ind < epochs_num:
        trainer.train()

    shutil.rmtree(trainer.results_path, ignore_errors=True)
    os.mkdir(trainer.results_path)

    trainer.test()
    if publish is not None:
        publish_utils.publish_results(publish,trainer)
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
    parser.add_argument('--threshold', type=float,required=False, default = -1, help= "threshold for mask if -1 no thresholding applied")
    parser.add_argument('--publish', type=str,required=False, default = None, help= "copy results folder into a share drive")
    args = parser.parse_args()
    caffe.set_device(args.gpu)
    train_epochs(args.test_dir,args.train_dir,args.solver,args.model,args.epochs,args.trimap_dir,DSD=args.DSD,
                 shuffle=args.no_shuffle,threshold=args.threshold,publish = args.publish)





