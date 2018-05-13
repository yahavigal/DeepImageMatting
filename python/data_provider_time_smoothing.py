import os
import random
import cv2
import numpy as np
from data_provider_original_ts import *
import ipdb
from random import randint
from collections import defaultdict

class TimeSmoothDataProvider(DataProvider) :

    def __init__(self, images_dir_test, images_dir_train, trimap_dir=None, shuffle_data=True,
                 batch_size = 32, use_data_aug = False, use_adv_data_train = False, threshold_param = -1,
                 img_width = 128, img_height = 128, 
                 use_gt_as_prev_prev_pred = False):

        super(TimeSmoothDataProvider,self).__init__(images_dir_test,images_dir_train,trimap_dir,shuffle_data,
                       batch_size=batch_size,use_data_aug=False,use_adv_data_train=False,
                       threshold_param= threshold_param,img_width= img_width,img_height=img_height)

        self.use_gt_as_prev_prev_pred = use_gt_as_prev_prev_pred


    def get_batch_data(self, batch_size = None):
        return self.get_train(self.images_list_train, batch_size)


    def get_test_data(self, batch_size = 1, image_path = None, pred_prev_r = None, mask_prev_r = None):
        return self.get_test(image_path, pred_prev_r, mask_prev_r)


    def switch_to_test(self):        
        super(TimeSmoothDataProvider, self).switch_to_test()
       
 
    def initialize_support_arrays(self):
        self.img_orig =[]
        self.img_resized=[]
        self.mask_orig =[]
        self.mask_resized =[]
        self.trimap_orig =[]
        self.mask_prev_orig =[]
        self.mask_prev_resized =[]
        self.pred_prev_orig =[]
        self.pred_prev_resized =[]     
        self.images_path_in_batch = []

    def replace_frameNum_in_image_path(self, image_path, frame_num_old, frame_num_new):        
        spl = os.path.split(image_path)
        new_name = spl[1].replace(frame_num_old, frame_num_new)
        new_path = os.path.join(spl[0], new_name)
       
	return new_path

    def get_current_clip_list(self,list_, ind):
        print 'starting {}'.format(list_[ind])
        current_dir = list_[ind]
        current_images = [os.path.join(current_dir,x) for x in os.listdir(current_dir) if self.gt_ext not in x and 'color'  in x]
        current_images = sorted(current_images, key=lambda x: int(x.split(os.sep)[-1].split('_')[0]))

        self.current_clip_list = current_images
        self.current_clip_ind = 0


    def get_gt_prev(self):
        if int(self.frame_num) > 1:
            gt_path_prev = self.replace_frameNum_in_image_path(self.gt_path, self.frame_num, str(int(self.frame_num) - 1 ) )
        else:
            return None
        #print gt_path
        #print gt_path_prev
        
        mask_prev = np.zeros_like(self.mask_orig[0])
        if os.path.exists(gt_path_prev):
            mask_prev = cv2.imread(gt_path_prev, 0)
            
        if self.threshold_param != -1:
            mask_prev[mask_prev < 256*self.threshold_param] = 0
            mask_prev[mask_prev >= 256*self.threshold_param] = 1
        else:
            mask_prev = np.divide(mask_prev,255.0)

        self.mask_prev_orig.append(mask_prev.copy())
        mask_prev_r = cv2.resize(mask_prev, (self.img_width, self.img_height), interpolation=cv2.INTER_NEAREST)
        self.mask_prev_resized.append(mask_prev_r)

        mask_prev_r = mask_prev_r.reshape([1, self.img_height, self.img_width])

        return mask_prev_r

    def run_net_forward( self, img_r, mask_r):
        batch = []
        masks = []
        masks_prev = []
        preds_prev = []

        batch.append(img_r)
        masks.append(mask_r)
        # following is fake date as it used for loss/back only and we do not need them at this stage
        masks_prev.append(mask_r)
        preds_prev.append(mask_r)

        batch_ar = np.array(batch)
        masks_ar = np.array(masks)
        masks_prev_ar = np.array(masks_prev)
        preds_prev_ar = np.array(preds_prev)
 
        # only forward propogation
        self.solver.net.blobs[self.solver.net.inputs[0]].reshape(*batch_ar.shape)
        self.solver.net.blobs[self.solver.net.inputs[0]].data[...]= batch_ar
        self.solver.net.blobs[self.solver.net.inputs[1]].reshape(*masks_ar.shape)
        self.solver.net.blobs[self.solver.net.inputs[1]].data[...]= masks_ar
        # following is fake date as it used for loss/back only and we do not need them at this stage
        self.solver.net.blobs[self.solver.net.inputs[2]].reshape(*masks_ar.shape)
        self.solver.net.blobs[self.solver.net.inputs[2]].data[...]= masks_ar
        self.solver.net.blobs[self.solver.net.inputs[3]].reshape(*masks_ar.shape)            
        self.solver.net.blobs[self.solver.net.inputs[3]].data[...]= masks_ar
     
        self.solver.net.forward()

    def get_prev_pred_from_sequence(self, image_path, num_frames_to_back = 8):
	num_first_frame = max(1, int(self.frame_num) - num_frames_to_back)
        prev_pred_curr = None
        for fn in range( num_first_frame, int(self.frame_num)):
	    image_path_prev = self.replace_frameNum_in_image_path( image_path, self.frame_num, str(fn))
	    
            if os.path.exists(image_path_prev):

	        if self.trimap_dir == None:
	            img_r, mask_r = self.get_tuple_data_point(image_path)
                else:
                    img_r, mask_r, trimap_r = self.get_tuple_data_point(image_path)

		if prev_pred_curr is None:
		    prev_pred_curr =  np.zeros_like(mask_r)

                prev_pred_curr_255 = prev_pred_curr*255
		if 'trimap_r' in locals():
                    img_r = np.concatenate((img_r, trimap_r, prev_pred_curr_255), axis=0)
                else:
                    img_r = np.concatenate((img_r, prev_pred_curr_255), axis=0)

                self.run_net_forward(img_r, mask_r)        

		if 'alpha_pred_s' in self.solver.net.blobs.keys():
            	    prev_pred_curr = self.solver.net.blobs['alpha_pred_s'].data[0].copy()
        	else:
            	    prev_pred_curr = self.solver.net.blobs['alpha_pred'].data[0].copy()

	    else:
                print "current image does not exist {}".format(image_path_prev)
                continue
 
 	return prev_pred_curr 


    def get_pred_for_previous_frame(self, image_path, mask_prev, pred_prev_prev_r = None):
        if hasattr(self, 'solver') == False:
           return None
   
        if int(self.frame_num) > 1:
	    image_path_prev = self.replace_frameNum_in_image_path(image_path, self.frame_num, str(int(self.frame_num) - 1 ))
        else:
            return None
   
        # print 'gets data for previous frame'
        isToAddToPathList = False
        if self.trimap_dir == None:
	    img_r, mask_r = self.get_tuple_data_point(image_path, isToAddToPathList)
        else:
            img_r, mask_r, trimap_r = self.get_tuple_data_point(image_path, isToAddToPathList)

        if pred_prev_prev_r is None: # if send no need to calculate
            pred_prev_prev_r = self.get_prev_pred_from_sequence(image_path)

        if pred_prev_prev_r is None: # not calculated in any way
	    pred_prev_prev_r = np.zeros_like(mask_r)
        
        pred_prev_prev_r_255 = pred_prev_prev_r*255
	if 'trimap_r' in locals():
            img_r = np.concatenate((img_r, trimap_r, pred_prev_prev_r_255), axis=0)
        else:
            img_r = np.concatenate((img_r, pred_prev_prev_r_255), axis=0)

        self.run_net_forward( img_r, mask_r)        

	if 'alpha_pred_s' in self.solver.net.blobs.keys():
            pred_prev = self.solver.net.blobs['alpha_pred_s'].data[0].copy()
        else:
            pred_prev = self.solver.net.blobs['alpha_pred'].data[0].copy()
    
        return pred_prev

    def get_tuple_data_point_train(self, image_path):        
        if self.trimap_dir == None:
	    img_r, mask_r = self.get_tuple_data_point(image_path)
        else:
            img_r, mask_r, trimap_r = self.get_tuple_data_point(image_path)

        if img_r is None or mask_r is None:
            if self.trimap_dir == None:
	        return None, None, None, None
            else:
                return None, None, None, None, None  
      
        mask_prev_r = self.get_gt_prev()

        if mask_prev_r is None:
            mask_prev_r = np.zeros_like(mask_r);   

        if self.use_gt_as_prev_prev_pred == True:
	    pred_prev = self.get_pred_for_previous_frame( image_path, mask_prev_r, mask_prev_r.copy())
        else:
            pred_prev = self.get_pred_for_previous_frame( image_path, mask_prev_r)

        if self.trimap_dir == None:
	    return img_r, mask_r, mask_prev_r, pred_prev
        else:
            return img_r, mask_r, mask_prev_r, pred_prev, trimap_r
   
    def get_train(self, list_, batch_size = None):
        if batch_size is None:
            batch_size = self.batch_size
        
        batch = []
        masks = []
        masks_prev = []
        preds_prev = []

        self.initialize_support_arrays()

        while len(batch) < batch_size:
            if self.list_ind >= len(list_):
                self.epoch_ind += 1
                self.list_ind = 0
                if self.is_test_phaze == True:
                    return batch, masks, masks_prev, preds_prev
                else:
                    print "starting from beginning of the list epoch {} finished".format(self.epoch_ind)
                    if self.shuffle == True:
                        random.shuffle(list_)

            if self.trimap_dir == None:
                img_r, mask_r, mask_prev_r, pred_prev_r = self.get_tuple_data_point_train(list_[self.list_ind])
            else:
                img_r, mask_r, mask_prev_r, pred_prev_r, trimap_r = self.get_tuple_data_point_train(list_[self.list_ind])

            if img_r is None or mask_r is None:
                del list_[self.list_ind]
                continue

            if mask_prev_r is None:
                mask_prev_r = np.zeros_like(mask_r);

            if pred_prev_r is None:
                pred_prev_r = np.zeros_like(mask_r);

            pred_prev_r_255 = pred_prev_r*255
            if 'trimap_r' in locals():
                img_r = np.concatenate((img_r, trimap_r, pred_prev_r_255), axis=0)
            else:
                img_r = np.concatenate((img_r, pred_prev_r_255), axis=0)

            batch.append(img_r)
            masks.append(mask_r)
            masks_prev.append(mask_prev_r)
            preds_prev.append(pred_prev_r)

            self.list_ind += 1

        return np.array(batch), np.array(masks), np.array(masks_prev), np.array(preds_prev)


    def get_test(self, image_path, pred_prev_r, mask_prev_r):
        self.initialize_support_arrays()
	if self.trimap_dir == None:
	    img_r, mask_r = self.get_tuple_data_point(image_path)
        else:
            img_r, mask_r, trimap_r = self.get_tuple_data_point(image_path)

	if pred_prev_r is None:
	    pred_prev_r = np.zeros_like(mask_r)

	if mask_prev_r is None:
	    mask_prev_r = np.zeros_like(mask_r)

        pred_prev_r_255 = pred_prev_r*255
	if 'trimap_r' in locals():
            img_r = np.concatenate((img_r, trimap_r, pred_prev_r_255), axis=0)
        else:
            img_r = np.concatenate((img_r, pred_prev_r_255), axis=0)

	return np.array(img_r), np.array(mask_r)




	

    
