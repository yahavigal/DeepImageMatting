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
import TF_utils
from convert_train_to_deploy import *
from benchmark_utils import *
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

def copy_last_masks(net,data_provider):
    if 'alpha_pred_s' in net.blobs.keys():
        data_provider.last_masks = net.blobs['alpha_pred_s'].data.copy()*255
    else:
        data_provider.last_masks = net.blobs['alpha_pred'].data.copy()*255

#python bgs_train_test.py --train_dir ../or/deepImageMatting/scripts/train_list.txt --test_dir ../or/deepImageMatting/scripts/test_list.txt --trimap_dir /media/or/Data/deepImageMatting/set1_07_2017_depth_norm/ --solver ../or/fastPortraitMatting/proto/solver.prototxt --model  ../or/fastPortraitMatting/snapshots/_iter_1792_MaxAccuracy9691.caffemodel

#or@ortrainubuntu5:~/caffe-BGS-win/python$ python bgs_train_test.py --train_dir ../or/deepImageMatting/scripts/composed/train_list.txt --test_dir ../or/deepImageMatting/scripts/composed/test_list.txt --trimap_dir /media/or/Data/composed/DataSet_2_composed_depth_norm/ --solver ../or/fastPortraitMatting/proto/solver.prototxt --model  ../or/fastPortraitMatting/snapshots/_iter_9180_MaxAccuracy8968.caffemodel


class bgs_test_train (object):
    def __init__(self, images_dir_test, images_dir_train, solver_path,weights_path,
                 snapshot_path, batch_size=32, snapshot = 100, snapshot_diff = False,
                 trimap_dir = None, DSD_flag = False, save_loss_per_image = False, shuffle_data = True,
                 threshold = -1, temporal = None, results_path=None,comment=''):

        self.threhold_param = threshold
        self.comment = comment
        # trimap_dir can be or directory path ( standard till 15 May 2018) or subdirectory name that have to be changed in the image name in order to get path name
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
            self.solver_path = solver_path
        else:
            self.solver = None
            self.net = caffe.Net(solver_path, weights_path,caffe.TEST)
            check_threshold_param(solver_path,threshold)
            img_width = self.net.blobs[self.net.inputs[0]].shape[3]
            img_height = self.net.blobs[self.net.inputs[0]].shape[2]

        if temporal == None:
                self.data_provider = DataProvider(images_dir_test,images_dir_train,trimap_dir,shuffle_data,
                                                  batch_size=batch_size,use_data_aug=True,use_adv_data_train=False,
                                                  threshold_param= self.threhold_param,img_width= img_width,img_height=img_height)
        elif temporal == 'temporal':
            self.data_provider = TemporalDataProvider(images_dir_test,images_dir_train,trimap_dir,
                                              batch_size=batch_size,use_data_aug=False,use_adv_data_train=False,
                                              threshold_param= self.threhold_param,img_width= img_width,img_height=img_height)
        elif temporal == 'time_smooth':
            self.data_provider = TimeSmoothDataProvider(images_dir_test,images_dir_train,trimap_dir, shuffle_data,
                                              batch_size=batch_size,use_data_aug=False,use_adv_data_train=False,
                                              threshold_param= self.threhold_param,img_width= img_width,img_height=img_height)

        self.data_provider.solver = self.solver
        # current standard, change if standards changes
        self.data_provider.key_still = 'images'
        self.data_provider.key_video = 'videos'

        self.exp_name += "_{}X{}".format(self.data_provider.img_width,self.data_provider.img_height)
        self.exp_name += "_threshold_{}".format(self.threhold_param)
        if temporal == 'temporal' and self.data_provider.insert_prev_data == True:
            self.exp_name+='_temporal'

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
        self.temporal = temporal
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
            self.tf_trainer =  TF_utils.TF_trainer(solver_path.split(os.sep)[-3],self.data_provider.img_width,self.data_provider.img_height,
                                          batch_size =self.data_provider.batch_size)
        #if self.use_fs_metric == True:
        self.fs_metrics = [(x,getattr(fs_metrics,x))  for x in dir(fs_metrics) if isinstance(fs_metrics.__dict__.get(x), types.FunctionType)]


    def train(self):
        if self.temporal == 'time_smooth':
            images, masks, masks_prev, preds_prev = self.data_provider.get_batch_data()
        else:
            images, masks = self.data_provider.get_batch_data()

        net = self.solver.net
        net.blobs[net.inputs[0]].reshape(*images.shape)
        net.blobs[net.inputs[1]].reshape(*masks.shape)
        net.blobs[net.inputs[0]].data[...]= images
        net.blobs[net.inputs[1]].data[...]= masks

        if self.temporal == 'time_smooth':
            net.blobs[net.inputs[2]].reshape(*masks_prev.shape)
            net.blobs[net.inputs[3]].reshape(*preds_prev.shape)
            net.blobs[net.inputs[2]].data[...]= masks_prev
            net.blobs[net.inputs[3]].data[...]= preds_prev

        #part of ofir's method in comment for now
        #if self.temporal == 'temporal' and np.any(images[:,-1,:]) == False:
        #    self.solver.net.forward()
        #else:
        #    self.solver.step(1)
        
        self.solver.step(1)
	#ipdb.set_trace() # to see data in net including gradiets and data

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

        if self.temporal == 'temporal':
            copy_last_masks(net,self.data_provider)

        return net.blobs['loss'].data

    def test_aux_ts(self, net, times, test_log_file, is_save_fig):
         for self.data_provider.list_ind in xrange(0, len(self.data_provider.images_list_test) ):
            self.data_provider.get_current_clip_list( self.data_provider.images_list_test, self.data_provider.list_ind)
            pred_prev_r = None
            mask_prev_r = None
            for self.data_provider.current_clip_ind in xrange(0, len(self.data_provider.current_clip_list)):
                print self.data_provider.current_clip_ind
                image_path = self.data_provider.current_clip_list[self.data_provider.current_clip_ind];
		
                img_r, mask_r = self.data_provider.get_test_data( 1, image_path, pred_prev_r, mask_prev_r)
                if img_r is None or mask_r is None or len(img_r) ==0 or len(mask_r) == 0:
                    continue

                if pred_prev_r is None:
	            pred_prev_r = np.zeros_like(mask_r)

	        if mask_prev_r is None:
	            mask_prev_r = np.zeros_like(mask_r)               

                if img_r.ndim < 4:
		    img_r = np.expand_dims(img_r, axis = 0)
                if mask_r.ndim < 4:
                    mask_r = np.expand_dims(mask_r, axis = 0)
                if mask_prev_r.ndim < 4:
                    mask_prev_r = np.expand_dims( mask_prev_r, axis = 0)
                if pred_prev_r.ndim < 4:
                    pred_prev_r = np.expand_dims(pred_prev_r, axis = 0)

                
                net.blobs[net.inputs[0]].reshape(*img_r.shape)
                net.blobs[net.inputs[1]].reshape(*mask_r.shape)
                net.blobs[net.inputs[0]].data[...]= img_r
                net.blobs[net.inputs[1]].data[...]= mask_r
                net.blobs[net.inputs[2]].reshape(*mask_prev_r.shape)
                net.blobs[net.inputs[2]].data[...]= mask_prev_r
                net.blobs[net.inputs[3]].reshape(*pred_prev_r.shape)            
                net.blobs[net.inputs[3]].data[...]= pred_prev_r

                input_bin = img_r.flatten().tolist()

                start = current_milli_time()
                net.forward()
                times.append(current_milli_time() - start)
            
                mask_prev_r = mask_r.copy()
                if 'alpha_pred_s' in net.blobs.keys():
            	    pred_prev_r = net.blobs['alpha_pred_s'].data[0].copy()
        	else:
            	    pred_prev_r = net.blobs['alpha_pred'].data[0].copy()

                single_image = img_r
                single_mask = mask_r
                image = self.data_provider.current_clip_list[self.data_provider.current_clip_ind]
                
                test_log_file.write(image)
                for output in net.outputs:
                    if output == 'alpha_pred' or output == 'alpha_pred_s':
                        continue
                    self.test_measures[output].append(net.blobs[output].data.flatten()[0])
                    test_log_file.write(" {}".format(self.test_measures[output][-1]))
                test_log_file.write('\n')

                iou = int(100*self.test_measures['mask_accuracy'][-1])

                if self.save_test_by_loss == True:
                    loss_per_image[self.test_measures['loss'][-1]] = image

                if is_save_fig == True:
                    plot_test_images(self.data_provider,net,0,self.dump_bin,
                                     self.view_all,self.infer_only_trimap, self.results_path, iou, input_bin)



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
                if self.use_tf_inference ==True:
                    tf_res,iou = self.tf_trainer.run_inference(single_image,single_mask)
                    avg_iou_tf.append(iou)

                test_log_file.write(image)
                for output in net.outputs:
                    if output == 'alpha_pred' or output == 'alpha_pred_s':
                        continue
                    if i==0:
                        self.test_measures[output].append(net.blobs[output].data.flatten()[0])
                    test_log_file.write(" {}".format(self.test_measures[output][-1]))
                test_log_file.write('\n')

                iou = int(100*self.test_measures['mask_accuracy'][-1])

                if self.save_test_by_loss == True:
                    loss_per_image[self.test_measures['loss'][-1]] = image

                if is_save_fig == True:
                    plot_test_images(self.data_provider,net,i,self.dump_bin,
                                     self.view_all,self.infer_only_trimap, self.results_path, iou,input_bin)

                if self.temporal == 'temporal':
                    copy_last_masks(net,self.data_provider)
                    self.data_provider.iter_ind += 1

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
        diff_caffe_tf = []
        times = []
        loss_per_image = {}
        avg_iou_tf =[]
        if self.use_tf_inference == True and self.data_provider.images_list_train is not None:
            for i in xrange(5000):
                x,y = self.data_provider.get_batch_data(self.data_provider.batch_size)
                ipdb.set_trace()
                _, loss, _  = self.tf_trainer.step(x,y)
                print "loss for fine tune is: {} IOU is: {}".format(loss,iou)
            self.tf_trainer.save()
            print "data saved"

        #no data augmentation in test
        trimap_r = None

        if self.temporal == 'time_smooth':
            self.test_aux_ts(net, times, test_log_file, is_save_fig)
	else:
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

            print self.comment
            summary.write(self.comment + '\n')

            print "{} average time for inference: {}".format(self.exp_name,np.average(times))

        if self.use_tf_inference ==True:
            print "average iou on test in TF {}".format(np.average(avg_iou_tf))

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

def train_epochs(images_dir_test, images_dir_train, solver_path,weights_path,epochs_num, trimap_dir,DSD,shuffle,
                 threshold,publish,real,temporal,comment,benchmark):
    snapshot_path = solver_path.replace("proto","snapshots",1)
    snapshot_path = os.path.split(snapshot_path)[0]
    trainer = bgs_test_train(images_dir_test, images_dir_train, solver_path,weights_path,snapshot_path,
                             trimap_dir = trimap_dir,DSD_flag = DSD,shuffle_data=shuffle,threshold=threshold,
                             temporal=temporal,comment=comment)

    while trainer.data_provider.epoch_ind < epochs_num:
        trainer.train()

    shutil.rmtree(trainer.results_path, ignore_errors=True)
    os.mkdir(trainer.results_path)

    trainer.data_provider.switch_to_test()
    trainer.test()
    trainer.plot_statistics()
    if real is not None:
        trainer.data_provider.images_list_test = trainer.data_provider.create_list_from_file(real[0])
        trainer.data_provider.trimap_dir = real[1]
        trainer.data_provider.switch_to_test()
        trainer.results_path = os.path.join(trainer.results_path,"real_data")
        os.mkdir(trainer.results_path)
        trainer.test()
        trainer.plot_statistics()
    if publish is not None:
        trainer.results_path = trainer.results_path.replace("real_data","")
        if benchmark == True:
            trainer.solver.net.save(os.path.join(trainer.results_path,"final.caffemodel"), False)
            deploy_net = [os.path.join(trainer.results_path,x) for x in os.listdir(trainer.results_path) if 'deploy' in x and x.endswith('.prototxt')][0]
            trigger_benchmark(os.path.join(trainer.results_path,'final.caffemodel'),deploy_net)
        publish_utils.publish_results(publish,trainer)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
    epilog= '''Usage:
    python bgs_train_test.py --train_dir ../BGS_scripts/train_list.txt
                              --test_dir ../BGS_scripts/test_list.txt
                              --trimap_dir /media/or/Data/deepImageMatting/set1_07_2017_depth_norm
                              --solver ../or/fastPortraitMatting/proto/solver.prototxt
                              --model  ../or/fastPortraitMatting/snapshots/_iter_100.caffemodel'''
                                    )
    parser.add_argument('--train_dir', type=str, required=True, help="train directory path or list")
    parser.add_argument('--test_dir', type=str, required=True, help="test directory path or list")
    parser.add_argument('--trimap_dir', type=str, required=False, default = None,help="trimap or any addtional output, trimap_dir can be or directory path ( standard till 15 May 2018) or subdirectory name extension that have to be changed in the image name in order to get path name")
    parser.add_argument('--solver', type=str, required=True,help="path to solver")
    parser.add_argument('--model', type=str, required=False, default = None, help="pre-trained weights path")
    parser.add_argument('--epochs', type=int, required=False, default = 60, help="number or epochs each epoch is equivalent to ")
    parser.add_argument('--DSD', action = 'store_true', help="use dense-sparse-dense mask and train with this restriction")
    parser.add_argument('--no_shuffle', action='store_false', help="training with no shuffle, shuufle the data by default")
    parser.add_argument('--gpu', type=int,required=False, default = 0, help= "GPU ID for multiple GPU machine")
    parser.add_argument('--threshold', type=float,required=False, default = -1, help= "threshold for mask if -1 no thresholding applied")
    parser.add_argument('--publish', type=str,required=False, default = None, help= "copy results folder into a share drive")
    parser.add_argument('--real', type=str,required=False, default = None,nargs='+',
                        help= "additional test on other (real) data in case of use trimap or depth you should also add it")
    parser.add_argument('--temporal', choices = ['temporal', 'time_smooth'], required = False, default = None, 
                        help="train with temporal smoothness consistency: possible values are: temporal - Omer, time_smooth - Alexandra")
    parser.add_argument('--comment', type=str, required='--publish' in sys.argv ,default='', help="comment to explain your extra details of experiment mandatory for publish")
    parser.add_argument('--benchmark', action='store_true', help="trigger windows (and android) benchmark valid only in publish")
    args = parser.parse_args()
    caffe.set_device(args.gpu)
    train_epochs(args.test_dir,args.train_dir,args.solver,args.model,args.epochs,args.trimap_dir,DSD=args.DSD,
                 shuffle=args.no_shuffle,threshold=args.threshold,publish = args.publish,real =args.real, temporal=args.temporal,
                 comment = args.comment, benchmark=args.benchmark)
    raw_input()





