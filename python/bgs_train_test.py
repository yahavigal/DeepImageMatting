# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 18:10:32 2017

@author: or
"""

import caffe
caffe.set_mode_gpu()
import numpy as np
import os
import random
import argparse
import ipdb
import data_augmentation
import time
import platform
import sys
import re
import matplotlib.pyplot as plt
from collections import defaultdict
from data_provider import *
from temporal_data_provider import *
from data_provider_time_smoothing import *
from plot_test_images import *
from google.protobuf import text_format
from caffe.proto import caffe_pb2
import publish_utils
import shutil
from convert_train_to_deploy import *
import types
import fs_metrics

current_milli_time = lambda: int(round(time.time() * 1000))

def get_net_path(solver_path):
    sp = caffe_pb2.SolverParameter()
    text_format.Merge(open(solver_path).read(),sp)
    return sp.net.encode('ascii','ignore')


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


class bgs_test_train (object):
    def __init__(self, images_dir_test, images_dir_train, solver_path,weights_path,
                 snapshot_path, batch_size=32, snapshot = 100, snapshot_diff = False,
                 DSD_flag = False, save_loss_per_image = False, shuffle_data = True,
                 threshold = -1, results_path=None):

        self.threhold_param = threshold
        # trimap_dir can be or directory path ( standard till 15 May 2018) or subdirectory name that have to be changed in the image name in order to get path name
        self.exp_name = platform.node() + " " + solver_path.split(os.sep)[-3]

        if solver_path.find('solver') != -1:
            self.solver = caffe.get_solver(solver_path)
            sp = caffe_pb2.SolverParameter()
            text_format.Merge(open(solver_path).read(),sp)
            check_threshold_param(sp.net,threshold)
            img_width = self.solver.net.blobs[self.solver.net.inputs[0]].shape[3]
            img_height = self.solver.net.blobs[self.solver.net.inputs[0]].shape[2]
            self.solver_path = solver_path
        else:
            self.solver = None
            self.net = caffe.Net(solver_path, weights_path,caffe.TEST)
            check_threshold_param(solver_path,threshold)
            img_width = self.net.blobs[self.net.inputs[0]].shape[3]
            img_height = self.net.blobs[self.net.inputs[0]].shape[2]

        self.data_provider = DataProvider(images_dir_test,images_dir_train,shuffle_data, batch_size=batch_size, use_data_aug=True,
                                          threshold_param= self.threhold_param,img_width= img_width,img_height=img_height)

        self.data_provider.solver = self.solver

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
        self.dump_bin = False
        self.view_all = True
        self.save_test_by_loss = save_loss_per_image
        self.snapshot_path = snapshot_path
        self.train_measures = defaultdict(list)
        self.test_measures = defaultdict(list)

        if not os.path.exists(self.snapshot_path) and self.data_provider.images_list_train is not None:
            os.makedirs(self.snapshot_path)
        if not os.path.exists(self.results_path):
            os.makedirs(self.results_path)

        self.fs_metrics = [(x,getattr(fs_metrics,x))  for x in dir(fs_metrics) if isinstance(fs_metrics.__dict__.get(x), types.FunctionType)]


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

        if self.data_provider.iter_ind % self.snapshot == 0:
            print "snapshot iter {}".format(self.data_provider.iter_ind)
            snapshot_file = os.path.join(self.snapshot_path,"_iter_"+str(self.data_provider.iter_ind)+".caffemodel")
            self.solver.net.save(snapshot_file, self.snapshot_diff)


        return net.blobs['loss'].data

    def test_aux_std(self, net, times, test_log_file, is_save_fig):
	while self.data_provider.epoch_ind == 0:
            img_r,mask_r = self.data_provider.get_test_data()
            if img_r is None or mask_r is None or len(img_r) ==0 or len(mask_r) == 0:
                continue
            net.blobs[net.inputs[0]].reshape(*img_r.shape)
            net.blobs[net.inputs[1]].reshape(*mask_r.shape)
            net.blobs[net.inputs[0]].data[...]= img_r
            net.blobs[net.inputs[1]].data[...]= mask_r
            input_bin = img_r.flatten().tolist()
            start = current_milli_time()
            net.forward()
            times.append(current_milli_time() - start)

            for i in xrange(len(img_r)):

                single_image = img_r[i]
                single_mask = mask_r[i]
                image = self.data_provider.images_path_in_batch[i]

                test_log_file.write(image)
                for output in net.outputs:
                    if output == 'alpha_pred' or output == 'alpha_pred_s':
                        continue
                    if i==0:
                        self.test_measures[output].append(net.blobs[output].data.flatten()[0])
                    test_log_file.write(" {}".format(self.test_measures[output][-1]))
                for fs_metric in self.fs_metrics:
                    preds_blob = net.blobs['alpha_pred_s'].data.copy()
                    fs_w = self.data_provider.mask_orig[0].shape[1]
                    fs_h = self.data_provider.mask_orig[0].shape[0]
                    cv_preds = cv2.resize(np.squeeze(preds_blob.transpose([2,3,0,1])),(fs_w,fs_h))
                    #cv_preds = cv_preds.transpose([2,0,1])
                    cv_preds = np.expand_dims(cv_preds,axis=0)
                    self.test_measures[fs_metric[0]].append(fs_metric[1](np.array(self.data_provider.mask_orig),cv_preds))
                    test_log_file.write(" {}".format(fs_metric[0],self.test_measures[fs_metric[0]][-1]))

                test_log_file.write('\n')

                iou = int(100*self.test_measures['mask_accuracy'][-1])

                if self.save_test_by_loss == True:
                    loss_per_image[self.test_measures['loss'][-1]] = image

                if is_save_fig == True:
                    plot_test_images(self.data_provider,net,i,self.dump_bin,
                                     self.view_all, self.results_path, iou,input_bin)


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
        for fs_metric in self.fs_metrics:
            test_log_file.write(" {}".format(fs_metric[0]))

            self.test_measures[output] = []
        test_log_file.write('\n')
        times = []
        loss_per_image = {}

        self.test_aux_std(net, times, test_log_file, is_save_fig)

        with open(os.path.join(self.results_path,"summary.txt"),'w') as summary:
            for output in net.outputs:
                if output == 'alpha_pred' or output == 'alpha_pred_s':
                    continue
                str_test =  "{} average {} on test: {} variance {}".format(self.exp_name,output,
                                                                           np.average(self.test_measures[output]),
                                                                           np.var(self.test_measures[output]))
                str_train =  "{} average {} on train: {} variance {}".format(self.exp_name,output,
                                                                             np.average(self.train_measures[output]),
                                                                             np.var(self.train_measures[output]))
                print str_test
                print str_train
                summary.write(str_test + '\n')
                summary.write(str_train + '\n')

            for fs_metric in self.fs_metrics:
                str_test =  "{} average {} on test: {} variance {}".format(self.exp_name,fs_metric[0],
                                                                           np.average(self.test_measures[fs_metric[0]]),
                                                                           np.var(self.test_measures[fs_metric[0]]))
                print str_test
                summary.write(str_test + '\n')


            print "{} average time for inference: {}".format(self.exp_name,np.average(times))


        test_log_file.close()
        if hasattr(self,'solver_path'):
            self.deploy_file = convert_train_to_deploy(get_net_path(self.solver_path),self.results_path)
            shutil.copyfile(get_net_path(self.solver_path),os.path.join(self.results_path,'net.prototxt'))
            shutil.copyfile(self.solver_path,os.path.join(self.results_path,'slover.prototxt'))


        if self.save_test_by_loss == True:
            return loss_per_image
        return np.average(self.test_measures['loss']), np.average(self.test_measures['mask_accuracy'])


    def plot_statistics(self):
        fig = plt.figure(figsize=(20,10))
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

        plt.savefig(os.path.join(self.results_path,'stats.fig.jpg'))
        plt.show(block=False)

def train_epochs(images_dir_test, images_dir_train, solver_path,weights_path,epochs_num,DSD,shuffle,threshold):
    snapshot_path = solver_path.replace("proto","snapshots",1)
    snapshot_path = os.path.split(snapshot_path)[0]
    trainer = bgs_test_train(images_dir_test, images_dir_train, solver_path,weights_path,snapshot_path,
                             DSD_flag = DSD,shuffle_data=shuffle,threshold=threshold)

    while trainer.data_provider.epoch_ind < epochs_num:
        trainer.train()

    shutil.rmtree(trainer.results_path, ignore_errors=True)
    os.mkdir(trainer.results_path)

    trainer.data_provider.switch_to_test()
    trainer.test()
    trainer.plot_statistics()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
    epilog= '''Usage:
    python bgs_train_test.py --train_dir ../BGS_scripts/train_list.txt
                              --test_dir ../BGS_scripts/test_list.txt
                              --solver ../or/fastPortraitMatting/proto/solver.prototxt
                              --model  ../or/fastPortraitMatting/snapshots/_iter_100.caffemodel'''
                                    )
    parser.add_argument('--train_dir', type=str, required=True, help="train directory path or list")
    parser.add_argument('--test_dir', type=str, required=True, help="test directory path or list")
    parser.add_argument('--solver', type=str, required=True,help="path to solver")
    parser.add_argument('--model', type=str, required=False, default = None, help="pre-trained weights path")
    parser.add_argument('--epochs', type=int, required=False, default = 60, help="number or epochs each epoch is equivalent to ")
    parser.add_argument('--DSD', action = 'store_true', help="use dense-sparse-dense mask and train with this restriction")
    parser.add_argument('--no_shuffle', action='store_false', help="training with no shuffle, shuufle the data by default")
    parser.add_argument('--gpu', type=int,required=False, default = 0, help= "GPU ID for multiple GPU machine")
    parser.add_argument('--threshold', type=float,required=False, default = -1, help= "threshold for mask if -1 no thresholding applied")
    args = parser.parse_args()
    caffe.set_device(args.gpu)
    train_epochs(args.test_dir,args.train_dir,args.solver,args.model,args.epochs,DSD=args.DSD,shuffle=args.no_shuffle,threshold=args.threshold)
    raw_input()





