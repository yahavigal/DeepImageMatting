import os
import cv2
import numpy as np
from data_provider import *

#import ipdb
#import matplotlib.pyplot as plt

class TimeSmoothDataProvider(DataProvider) :

    def __init__(self, images_dir_test, images_dir_train, trimap_dir=None, shuffle_data=True,
                 batch_size = 32, use_data_aug = False, use_adv_data_train = False, threshold_param = -1,
                 img_width = 128, img_height = 128, 
                 use_gt_as_prev_prev_pred = False, addDepthDiff = False, addColorDiff = False):

        super(TimeSmoothDataProvider,self).__init__(images_dir_test,images_dir_train,trimap_dir,shuffle_data,
                       batch_size=batch_size,use_data_aug=False,use_adv_data_train=False,
                       threshold_param= threshold_param,img_width= img_width,img_height=img_height)

        self.use_gt_as_prev_prev_pred = use_gt_as_prev_prev_pred
        # currently can be:
	# - all False, doesn't add any information from previous frame
	# - can't be all trues, only diffrenece or previous frame can be added
	# exapmples of possiblr variants: TFTF; TFFF; FFTF etc
	self.addDepthDiff = False
	self.addDepthPrev = False
	self.addColorDiff = False       
	self.addColorPrev = False


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

    def get_depth_path_from_image_path(self, image_path):
	if self.is_still:
	    curr_key = self.key_still
	else:
	    curr_key = self.key_video
	
	depth_path = image_path.replace(curr_key, curr_key + self.trimap_dir)
        depth_path = depth_path.replace(self.color_ext, self.trimap_ext)
	if not os.path.exists(depth_path):
	     print "TM depth file not found {} ".format(depth_path)
	       	
	return depth_path

    def get_current_clip_list(self,list_, ind):
        print 'starting {}'.format(list_[ind])
        current_dir = list_[ind]
        current_images = [os.path.join(current_dir,x) for x in os.listdir(current_dir) if self.gt_ext not in x and 'color'  in x]
        current_images = sorted(current_images, key=lambda x: int(x.split(os.sep)[-1].split('_')[0]))

        self.current_clip_list = current_images
        self.current_clip_ind = 0


    def get_gt_prev(self, frame_num):
        frame_num_prev = str(int(frame_num) - 1 )
        if int(frame_num) > 1:	    
            gt_path_prev = self.replace_frameNum_in_image_path(self.gt_path, frame_num,  frame_num_prev)
        else:
            return None, frame_num_prev
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

        return mask_prev_r, frame_num_prev

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

    def get_prev_pred_from_sequence(self, image_path, frame_num, num_frames_to_back = 8):

	num_first_frame = max(1, int(frame_num) - num_frames_to_back)
	if num_first_frame ==  frame_num:
	    return None
        prev_pred_curr = None
        for fn in range( num_first_frame, int(frame_num)):
	    image_path_prev = self.replace_frameNum_in_image_path( image_path, frame_num, str(fn))
	    
            if os.path.exists(image_path_prev):

	        if self.trimap_dir == None:
	            img_r, mask_r = self.get_tuple_data_point(image_path_prev)
                else:
                    img_r, mask_r, trimap_r = self.get_tuple_data_point(image_path_prev)

		if prev_pred_curr is None:
		    prev_pred_curr =  np.zeros_like(mask_r)

                prev_pred_curr_255 = prev_pred_curr*255
		if 'trimap_r' in locals():
                    img_r = np.concatenate((img_r, trimap_r, prev_pred_curr_255), axis=0)
                else:
                    img_r = np.concatenate((img_r, prev_pred_curr_255), axis=0)

		img_r = self.add_prev_frame_info(img_r, image_path_prev, str(fn))

                self.run_net_forward(img_r, mask_r)        

		if 'alpha_pred_s' in self.solver.net.blobs.keys():
            	    prev_pred_curr = self.solver.net.blobs['alpha_pred_s'].data[0].copy()
        	else:
            	    prev_pred_curr = self.solver.net.blobs['alpha_pred'].data[0].copy()

	    else:
                print "current image does not exist {}".format(image_path_prev)
                break
 
 	return prev_pred_curr 


    def get_pred_for_previous_frame(self, image_path, frame_num, mask_prev, pred_prev_prev_r = None):
        if hasattr(self, 'solver') == False:
           return None
   	frame_num_prev = str(int(frame_num) - 1 )
        if int(frame_num) > 1:	    
	    image_path_prev = self.replace_frameNum_in_image_path(image_path, frame_num, frame_num_prev)
        else:
            return None, frame_num_prev
   
        # print 'gets data for previous frame'
        isToAddToPathList = False
        if self.trimap_dir == None:
	    img_r, mask_r = self.get_tuple_data_point(image_path_prev, isToAddToPathList)
        else:
            img_r, mask_r, trimap_r = self.get_tuple_data_point(image_path_prev, isToAddToPathList)
	if img_r is None or mask_r is None or (self.trimap_dir is not None and trimap_r is None):
            return None, frame_num_prev
	
        frame_num_curr = self.frame_num
        if pred_prev_prev_r is None: # if send no need to calculate
            pred_prev_prev_r = self.get_prev_pred_from_sequence(image_path_prev, frame_num_prev)

        if pred_prev_prev_r is None: # not calculated in any way
	    pred_prev_prev_r = np.zeros_like(mask_r)
        
        pred_prev_prev_r_255 = pred_prev_prev_r*255
	if 'trimap_r' in locals():
            img_r = np.concatenate((img_r, trimap_r, pred_prev_prev_r_255), axis=0)
        else:
            img_r = np.concatenate((img_r, pred_prev_prev_r_255), axis=0)

	img_r = self.add_prev_frame_info(img_r, image_path_prev, frame_num_curr)

        self.run_net_forward( img_r, mask_r)        

	if 'alpha_pred_s' in self.solver.net.blobs.keys():
            pred_prev = self.solver.net.blobs['alpha_pred_s'].data[0].copy()
        else:
            pred_prev = self.solver.net.blobs['alpha_pred'].data[0].copy()
    
        return pred_prev, frame_num_prev

    def get_tuple_data_point_train(self, image_path):        
        if self.trimap_dir == None:
	    img_r, mask_r = self.get_tuple_data_point(image_path)
        else:
            img_r, mask_r, trimap_r = self.get_tuple_data_point(image_path)
 	frame_num_curr = self.frame_num
        if img_r is None or mask_r is None:
            if self.trimap_dir == None:
	        return None, None, None, None, None
            else:
                return None, None, None, None, None, None  
      
        mask_prev_r, frame_num_prev = self.get_gt_prev(frame_num_curr)

        if mask_prev_r is None:
            mask_prev_r = np.zeros_like(mask_r);   

        if self.use_gt_as_prev_prev_pred == True:
	    pred_prev, frame_num_prev = self.get_pred_for_previous_frame( image_path, frame_num_curr, mask_prev_r, mask_prev_r.copy())
        else:
            pred_prev, frame_num_prev = self.get_pred_for_previous_frame( image_path, frame_num_curr, mask_prev_r)

        if self.trimap_dir == None:
	    return img_r, mask_r, mask_prev_r, pred_prev, frame_num_prev
        else:
            return img_r, mask_r, mask_prev_r, pred_prev, trimap_r, frame_num_prev

    def get_diff_image_color(self, image_path, frame_num):
	if int(frame_num) > 1:
	    frame_num_prev = str(int(frame_num) - 1 )
	    image_path_prev = self.replace_frameNum_in_image_path(image_path, frame_num, frame_num_prev)
        else:
            return None

	if os.path.exists(image_path_prev):
            clr_im_prev = cv2.imread(image_path_prev)
	else:
	    return None

        clr_im = cv2.imread(image_path)
	clr_im_g = cv2.cvtColor(clr_im, cv2.COLOR_BGR2GRAY).astype('float32')
        clr_im_prev_g = cv2.cvtColor(clr_im_prev, cv2.COLOR_BGR2GRAY).astype('float32')
        d = clr_im_prev_g - clr_im_g;
	d_r = cv2.resize(d, (self.img_width, self.img_height))	

	return d_r

    def get_prev_gray_image(self, image_path, frame_num):
	if int(frame_num) > 1:
	    frame_num_prev = str(int(frame_num) - 1 )
	    image_path_prev = self.replace_frameNum_in_image_path(image_path, frame_num, frame_num_prev)
        else:
            return None

	if os.path.exists(image_path_prev):
            clr_im_prev = cv2.imread(image_path_prev)
	else:
	    return None

	g_im = cv2.cvtColor(clr_im_prev, cv2.COLOR_BGR2GRAY).astype('float32')        
	g_im_r = cv2.resize(g_im, (self.img_width, self.img_height))	

	return g_im_r

    def get_diff_image_depth(self, depth_path, frame_num):
	if int(frame_num) > 1:
	    frame_num_prev = str(int(frame_num) - 1 )
	    depth_path_prev = self.replace_frameNum_in_image_path(depth_path, frame_num, frame_num_prev)
        else:
            return None

	if os.path.exists(depth_path_prev):
            depth_im_prev = cv2.imread(depth_path_prev, 0).astype('float32')
	else:
	    return None

        depth_im = cv2.imread(depth_path,0).astype('float32')
        d = depth_im_prev - depth_im;
        d_r = cv2.resize(d, (self.img_width, self.img_height), interpolation=cv2.INTER_NEAREST)	

	return d_r

    def get_prev_depth(self, depth_path, frame_num):
	if int(frame_num) > 1:
	    frame_num_prev = str(int(frame_num) - 1 )
	    depth_path_prev = self.replace_frameNum_in_image_path(depth_path, frame_num, frame_num_prev)
        else:
            return None

	if os.path.exists(depth_path_prev):
            depth_im_prev = cv2.imread(depth_path_prev, 0).astype('float32')
	else:
	    return None

        d_r = cv2.resize(depth_im_prev, (self.img_width, self.img_height), interpolation=cv2.INTER_NEAREST)	

	return d_r

    def add_prev_frame_info(self, img_r, image_path, frame_num):
        #ipdb.set_trace()
	clr_d = None
	toAddColorInfo = False
	if self.addColorDiff:
	    toAddColorInfo = True
	    clr_d = self.get_diff_image_color(image_path, frame_num)
	elif self.addColorPrev:
	    toAddColorInfo = True
 	    clr_d = self.get_prev_gray_image(image_path, frame_num)
	if toAddColorInfo:
	    if clr_d is not None:
	        clr_d = clr_d.reshape([1, self.img_height, self.img_width])	# as far as it grey image
	    else:
	        clr_d = np.zeros([1, self.img_height, self.img_width])	
	    img_r = np.concatenate((img_r, clr_d), axis=0)

	depth_d = None
	toAddDepthInfo = False
        depth_path = self.get_depth_path_from_image_path(image_path)
	if self.addDepthDiff:
	    toAddDepthInfo = True
	    depth_d = self.get_diff_image_depth(depth_path, frame_num)
	elif self.addDepthPrev:
	    toAddDepthInfo = True
	    depth_d = self.get_prev_depth(depth_path, frame_num)
	if toAddDepthInfo: 
	    if depth_d is None:
	        depth_d = np.zeros([1, self.img_height, self.img_width])	
	    else:	
	        depth_d = depth_d.reshape([1, self.img_height, self.img_width])   
 	    img_r = np.concatenate((img_r, depth_d), axis=0)

	return img_r
   
    def get_train(self, list_, batch_size = None):
	#ipdb.set_trace()
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
                img_r, mask_r, mask_prev_r, pred_prev_r, frame_num_prev = self.get_tuple_data_point_train(list_[self.list_ind])
            else:
                img_r, mask_r, mask_prev_r, pred_prev_r, trimap_r, frame_num_prev = self.get_tuple_data_point_train(list_[self.list_ind])
	    #ipdb.set_trace()
            if img_r is None or mask_r is None:
                del list_[self.list_ind]
                continue

            if mask_prev_r is None:
                mask_prev_r = np.zeros_like(mask_r);

            if pred_prev_r is None:
                pred_prev_r = np.zeros_like(mask_r);

            pred_prev_r_255 = pred_prev_r*255
	    #ipdb.set_trace()
            if 'trimap_r' in locals():
                img_r = np.concatenate((img_r, trimap_r, pred_prev_r_255), axis=0)
            else:
                img_r = np.concatenate((img_r, pred_prev_r_255), axis=0)
	    if frame_num_prev is None:
		ipdb.set_trace()
		a=5

	    frame_num_curr = str(int(frame_num_prev) + 1 )
	    img_r = self.add_prev_frame_info(img_r, list_[self.list_ind], frame_num_curr)

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
	frame_num_curr = self.frame_num
	if pred_prev_r is None:
	    pred_prev_r = np.zeros_like(mask_r)

	if mask_prev_r is None:
	    mask_prev_r = np.zeros_like(mask_r)

        pred_prev_r_255 = pred_prev_r*255
	if 'trimap_r' in locals():
            img_r = np.concatenate((img_r, trimap_r, pred_prev_r_255), axis=0)
        else:
            img_r = np.concatenate((img_r, pred_prev_r_255), axis=0)

	img_r = self.add_prev_frame_info(img_r, image_path, frame_num_curr)

	return np.array(img_r), np.array(mask_r)




	

    
