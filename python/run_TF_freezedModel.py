import os
import math
import argparse
import tensorflow as tf
from tensorflow import gfile as gfile
import numpy as np
import cv2
import matplotlib.pyplot as plt

import ipdb

# python run_TF_freezedModel.py --model_path ../or/fastPortraitMatting/results/Resu2save/toTF/95.91_128_toPlay/TF_95.905/fastPortraitMatting.pb --output_dir ../or/fastPortraitMatting/results/Resu2save/toTF/95.91_128_toPlay/TF_95.905/play/ --imgs_list_path /media/or/Data/dataLists/dataSet_1/train_list.txt --depth_trimap_root_dir set1_07_2017_depth_norm --images_root_dir Set1_07_2017 --target_ext depth


def mask_iou(pred,gt,thresh = 0.5):
    intersection = tf.logical_and(pred >= thresh,gt >= thresh)
    union = tf.logical_or(pred >= thresh,gt >= thresh)
    return tf.reduce_mean(tf.divide(tf.reduce_sum(tf.cast(intersection,dtype=np.int32)),tf.reduce_sum(tf.cast(union,dtype=np.int32))))

class run_TF_freezedModel:

    def __init__(self, args):
        self.img_width = 128 
        self.img_height = 128
        self.img_width_orig = 640 
        self.img_height_orig = 480
        self.gt_ext = "_silhuette" 
        self.clr_ext = "color"
        self.depth_trimap_ext = args.target_ext   
        self.depth_trimap_root_dir = args.depth_trimap_root_dir
        self.images_root_dir = args.images_root_dir
        self.get_images_list(args.imgs_list_path)
        self.dump_bin = False

    
    def get_images_list(self, imgs_list_path):
        # read images paths
        if os.path.isfile(imgs_list_path):
            images = open(imgs_list_path).readlines()
            images = [x[0:-1] for x in images if x.endswith('\n')]
            self.images_list = [x for x in images
                                if x.endswith(".png") and x.find(self.gt_ext) == -1]
        else:
            self.images_list = None
            print 'mgs_list_path must be a file'


    def get_data_for_image(self, image_path):
        if not os.path.exists(image_path):
            if self.depth_trimap_root_dir == None:
                return [None, None]
            else:
                return [None, None, None]
        
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB).astype('float32')

        #subtract mean
        img -= np.array([104,117,123],dtype=np.float32)
        img_r = cv2.resize(img, (self.img_width,self.img_height))

        gt_path = image_path.replace(self.clr_ext, self.clr_ext + self.gt_ext)
        if not os.path.isfile(gt_path):
            if self.depth_trimap_root_dir == None:
                return [None, None]
            else:
                return [None, None, None]
        
        mask = cv2.imread(gt_path,0)
        mask_r = cv2.resize(mask, (self.img_width,self.img_height),interpolation = cv2.INTER_NEAREST)     
        
        if self.depth_trimap_root_dir != None:
            d_trmp_path = image_path.replace(self.clr_ext, self.depth_trimap_ext)
            d_trmp_path = d_trmp_path.replace(self.images_root_dir, self.depth_trimap_root_dir)

            if not os.path.isfile(d_trmp_path):
                return [None, None, None]

            trimap = cv2.imread(d_trmp_path,0)
            trimap_r = cv2.resize(trimap, (self.img_width, self.img_height),interpolation = cv2.INTER_NEAREST)

            trimap_r = trimap_r.reshape([self.img_height,self.img_width,1])
            mask_r = mask_r.reshape([self.img_height,self.img_width,1])
            img_r = np.concatenate((img_r,trimap_r),axis = 2)     
            
        else:
            mask_r = mask_r.reshape([self.img_height,self.img_width,1])

        img_r = img_r.reshape(1,*img_r.shape)
        mask_r = mask_r.reshape(1,*mask_r.shape)

        return img_r,mask_r


    def load_graph_def( self, model_path, sess=None, use_moving_avarage=False):
        if os.path.isfile(model_path):
            with gfile.FastGFile(model_path, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name='')
                #print("garph imported ", graph_def)
        else:
            swss = sess if sess is not None else tf.get_default_session()
            saver = tf.train.import_meta_graph(model_path + '.meta')
            if use_moving_average:
                variable_averages = tf.train.ExponentialMovingAverage(0.9999)
                variables_to_restore = variable_averages.variables_to_restore()
                saver = tf.train.Saver(saver_def=saver.as_saver_def(), var_list=variables_to_restore)
            saver.restore(sess, model_path)
    
    def save_pred_as_img( self, pred_tf, iou, image_path, output_dir):

        pred = pred_tf.copy()
        sh = pred.shape
        pred = ( pred.reshape(sh[1], sh[2]) )*255
        pred_fs = cv2.resize(pred, (self.img_width_orig,self.img_height_orig))

        split = os.path.splitext(image_path.replace(os.sep,"_"))[0]
        img_path = split+"_iou_{}.mask.jpg".format(int(10000*iou))
        img_path = os.path.join( output_dir, img_path)

        cv2.imwrite(img_path, pred_fs)

    def dump_data_to_file(self, output_dir, image_path,input_data, pred_tf):
        if self.dump_bin ==True:
            output = os.path.join(output_dir, 'dumps')
            if not os.path.exists(output):
                os.makedirs(output)

            output_data = pred_tf.copy()
            sh = output_data.shape
            output_data = output_data.reshape(sh[1], sh[2])

            split = os.path.splitext(image_path.replace(os.sep,"_"))[0]
            bin_path_in  = os.path.join( output, split + "_input.bin")
            bin_path_out = os.path.join( output, split + "_output.bin")
            dump_in  = open(bin_path_in,'w')
            dump_out = open(bin_path_out,'w')
            ls_in  = input_data.flatten().tolist()
            ls_out = output_data.flatten().tolist()
            for item in ls_in:
                dump_in.write(str(int(item))+'\n')
            dump_in.close()
            for item in ls_out:
                dump_out.write(str(int(item*255))+'\n')
            dump_out.close()

    def run_tf_model(self, model_path, output_dir):

        preds_res = []
        ious = []
        c = 0
        imgs_to_check = []
        ious_to_check = []
       

        with tf.Graph().as_default():

            with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

                self.load_graph_def(model_path)
            
                images_placeholder = tf.get_default_graph().get_tensor_by_name("input_tensor_1:0")

                gt = tf.placeholder(np.float32, shape=(1,self.img_height, self.img_width,1))

                preds = tf.get_default_graph().get_tensor_by_name("alpha_pred:0")

                iou = mask_iou(preds[0], gt)                           

                for image_path in self.images_list:
                    img_r, mask_r = self.get_data_for_image(image_path)
                    feed_dict = {images_placeholder: img_r, gt: mask_r}
                    pred_res, iou = sess.run([preds, mask_iou(preds[0], gt,0.5)], feed_dict=feed_dict) 

                    ious.append(iou)
                    
                    print(len(self.images_list))
                    print(c)
                    c = c + 1
                    print(iou)
                    if iou < 0.9:
                        print( image_path)
                        imgs_to_check.append(image_path)
                        ious_to_check.append(iou)
      
 
                    self.save_pred_as_img( pred_res, iou, image_path, output_dir)
                    self.dump_data_to_file( output_dir, image_path, img_r, pred_res ) 

                                     
                   
        return ious, imgs_to_check, ious_to_check

#caffe [n,c,h,w]   TF [n,h,w,c ]

def main(args):
    
    tf_runer = run_TF_freezedModel(args)
    ious, imgs_to_check, ious_to_check = tf_runer.run_tf_model(args.model_path, args.output_dir)

    print('TF mean accuraccy is ', np.mean(ious))

    print (imgs_to_check)

    plt.title('set accuracy')
    plt.plot(xrange(len(ious)),ious)
    plt.xlabel('# frame')
    plt.ylabel('accuracy')
    fig_path = os.path.join(args.output_dir,'accuracy.png')
    plt.savefig(fig_path)
    plt.show()
    
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--imgs_list_path', type=str, required=True, help = 'txt file with pathes to images to be run')
    parser.add_argument('--images_root_dir',type=str,  required=True, help = 'name of the directory where is main data set in our tree is saved')
    parser.add_argument('--depth_trimap_root_dir',type=str, required=False, default = None, help = 'name of the directory where is depth or trimap data set in our tree is saved')
    parser.add_argument('--target_ext', type = str, required=False, default=None, help = 'file extension: trimap or depth')

    main(parser.parse_args())
