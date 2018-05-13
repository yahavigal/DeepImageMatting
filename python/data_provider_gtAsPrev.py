import os
import random
import cv2
import data_augmentation
import numpy as np
import re
import ipdb

def find_data_root_ind(image,trimap_root):
    ind = 0
    image_split = image.split(os.sep)
    trimap_split = trimap_root.split(os.sep)
    while image_split[ind] == trimap_split[ind]:
        ind +=1
    return ind +1

class DataProvider(object) :

    def create_list_from_file(self,input_file):
        if os.path.isdir(input_file):
            list_images = [os.path.join(input_file, x)
                                      for x in os.listdir(input_file)
                                      if x.endswith(".png") and x.find(self.gt_ext) == -1]
        elif os.path.isfile(input_file):
            images = open(input_file).readlines()
            images = [x[0:-1] for x in images if x.endswith('\n')]
            list_images = [x for x in images
                                      if x.endswith(".png") and x.find(self.gt_ext) == -1
                                      and (self.use_adv_data_train == False or x.find(self.adverserial_ext) != -1)
                                      or os.path.isdir(x)]
        else:
            list_images = None
        return list_images

    def __init__(self, images_dir_test, images_dir_train,trimap_dir=None, shuffle_data=True,
                 batch_size = 32, use_data_aug = True, use_adv_data_train = False,threshold_param = -1,
                 img_width=128,img_height=128):

        self.gt_ext = "_silhuette"
        self.trimap_ext = None
        if trimap_dir is not None:
            if "trimap" in trimap_dir.lower():
                self.trimap_ext = "_triMap"
            else:
                self.trimap_ext = "_depth"
        self.adverserial_ext = "_adv"
        self.use_adv_data_train = use_adv_data_train
        self.trimap_dir = trimap_dir
        self.batch_size = batch_size
        self.shuffle = shuffle_data
        self.use_data_aug = use_data_aug
        self.threshold_param = threshold_param

        self.images_list_train = self.create_list_from_file(images_dir_train)
        self.images_list_test = self.create_list_from_file(images_dir_test)

        self.trimap_dir = trimap_dir

        if self.images_list_train is not None and shuffle_data == True:
            random.shuffle(self.images_list_train)


        self.img_width = img_width
        self.img_height = img_height
        self.list_ind = 0
        self.test_list_ind = 0
        self.root_data_ind = None
        self.epoch_ind = 0
        self.iter_ind = 0

        self.img_orig = []
        self.trimap_orig = []
        self.mask_orig = []
        self.mask_prev_orig = []
        self.pred_prev_orig = []

        self.img_resized = []
        self.trimap_resized = None
        self.mask_resized = []
        self.mask_prev_resized = []
        self.pred_prev_resized = []        

        self.images_path_in_batch = []

        self.img_transposed = None
        self.trimap_transposed = None
        self.mask_transposed = None

        self.is_test_phaze = False

    def get_tuple_data_point_aux(self, image_path, isToAdd = True):
	if not os.path.exists(image_path):
            if self.trimap_dir == None:
                return [None, None, None]
            else:
                return [None, None, None, None]
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype('float32')
        self.img_orig.append(img.copy())

        if self.use_data_aug == True:
	    ipdb.set_trace()
            # image based data augmentation
            coin = np.random.uniform(0, 1, 1)
            if coin <= 0.25:
                img = data_augmentation.color_jitter(img)
            elif coin <= 0.5:
                img = data_augmentation.PCA_noise(img)
            elif coin <= 0.75:
                img = data_augmentation.gamma_correction(img)

        # subtract mean
        img -= np.array([104, 117, 123], dtype=np.float32)
        img_r = cv2.resize(img, (self.img_width, self.img_height))
        self.img_resized.append(img_r)
        path = os.path.splitext(image_path)
        gt_path = path[0] + self.gt_ext + path[1]
        if not os.path.isfile(gt_path):
            del self.img_resized[-1]
            del self.img_orig[-1]
            if self.trimap_dir == None:
                return [None, None, None]
            else:
                return [None, None, None, None]
        mask = cv2.imread(gt_path, 0)
        if self.threshold_param != -1:
            mask[mask < 256*self.threshold_param] = 0
            mask[mask >= 256*self.threshold_param] = 1
        else:
            mask = np.divide(mask,255.0)
        self.mask_orig.append(mask.copy())
        mask_r = cv2.resize(mask, (self.img_width, self.img_height), interpolation=cv2.INTER_NEAREST)
        self.mask_resized.append(mask_r)
        if self.root_data_ind is None:
            self.root_data_ind = find_data_root_ind(image_path, self.trimap_dir)
        split = image_path.split(os.sep)[self.root_data_ind:]
        split = os.sep.join(split)
        frame_num = re.findall(r'\d+', split)[-1]
        split = os.path.split(split)
        self.frame_num = frame_num

#Alexandra: GT instead of previous mask
        #ipdb.set_trace()
        if int(frame_num) > 1:
            gt_path_prev = gt_path.replace(frame_num, str(int(frame_num) - 1 ))
        else:
            gt_path_prev = ''
        #print gt_path
        #print gt_path_prev
        
        mask_prev = np.zeros_like(mask)
        if gt_path_prev != '':
            if os.path.exists(gt_path_prev):
                mask_prev = cv2.imread(gt_path_prev, 0)
            #else:
                # print 'GT PREVIOUS DOES NOT EXIST'

        if self.threshold_param != -1:
            mask_prev[mask_prev < 256*self.threshold_param] = 0
            mask_prev[mask_prev >= 256*self.threshold_param] = 1
        else:
            mask_prev = np.divide(mask_prev,255.0)

        self.mask_prev_orig.append(mask_prev.copy())
        mask_prev_r = cv2.resize(mask_prev, (self.img_width, self.img_height), interpolation=cv2.INTER_NEAREST)
        self.mask_prev_resized.append(mask_prev_r)
#Alexandra: GT instead of previous mask end 

        if self.trimap_dir != None:
            trimap_path = os.path.join(self.trimap_dir,
                                       split[0], frame_num + self.trimap_ext + ".png")

            if not os.path.isfile(trimap_path):
                del self.img_resized[-1]
                del self.img_orig[-1]
                del self.mask_resized[-1]
                del self.mask_orig[-1]
                del self.mask_prev_resized[-1]
                del self.mask_prev_orig[-1]
                return [None, None, None, None]

            trimap = cv2.imread(trimap_path, 0)
            if trimap is None:
                del self.img_resized[-1]
                del self.img_orig[-1]
                del self.mask_resized[-1]
                del self.mask_orig[-1]
                del self.mask_prev_resized[-1]
                del self.mask_prev_orig[-1]
                return [None, None, None, None]
            self.trimap_orig.append(trimap.copy())
            trimap_r = cv2.resize(trimap, (self.img_width, self.img_height), interpolation=cv2.INTER_NEAREST)

            if self.use_data_aug == True:
                ipdb.set_trace()
                # rotation / filipping data augmentation
                coin = np.random.uniform(0, 1, 1)
                if '80cm' in image_path and coin <= 0.7:
                    img_r, mask_r, trimap_r = data_augmentation.translate(img_r, mask_r, trimap_r)
                else:
                    if coin <= 0.33:
                        img_r, mask_r, trimap_r = data_augmentation.horizontal_flipping(img_r, mask_r, trimap_r)
                    elif coin <= 0.66:
                        img_r, mask_r, trimap_r = data_augmentation.rotate(img_r, mask_r, trimap_r)
            if isToAdd:
                self.images_path_in_batch.append(image_path)
            trimap_r = trimap_r.reshape([1, self.img_height, self.img_width])
            mask_r = mask_r.reshape([1, self.img_height, self.img_width])
#Alexandra: GT instead of previous
            mask_prev_r = mask_prev_r.reshape([1, self.img_height, self.img_width])
            img_r = img_r.transpose([2, 0, 1])

            return img_r, mask_r, mask_prev_r, trimap_r
        else:
            mask_r = mask_r.reshape([1, self.img_height, self.img_width])
            mask_prev_r = mask_prev_r.reshape([1, self.img_height, self.img_width])
            img_r = img_r.transpose([2, 0, 1])
            return img_r, mask_r, mask_prev_r

    def get_tuple_data_point(self, image_path):        

        if self.trimap_dir == None:
	    img_r, mask_r, mask_prev_r = self.get_tuple_data_point_aux(image_path)
        else:
            img_r, mask_r, mask_prev_r, trimap_r = self.get_tuple_data_point_aux(image_path)
	
        pred_prev = self.get_data_for_previous_frame( image_path)

	if self.trimap_dir == None:
	    return img_r, mask_r, mask_prev_r, pred_prev
        else:
            return img_r, mask_r, mask_prev_r, pred_prev, trimap_r

    def get_data_for_previous_frame(self, image_path):
        #ipdb.set_trace()
        if hasattr(self, 'solver') == False:
           return None

        if int(self.frame_num) > 1:
            image_path_prev = image_path.replace(self.frame_num, str(int(self.frame_num) - 1 ))
        else:
            return None
   
        # print 'gets data for previous frame'
        img_r, mask_r, mask_prev_r, trimap_r = self.get_tuple_data_point_aux(image_path_prev, False)
        # for this step I'll take GT prev of prev to generate prev prediction, to see if it works
        if img_r is None or mask_r is None or len(img_r) ==0 or len(mask_r) == 0 or mask_prev_r is None:
	    return None

        mask_prev_r_255 = 255.0*mask_prev_r
	if 'trimap_r' in locals():     
            img_r = np.concatenate((img_r, trimap_r, mask_prev_r_255), axis=0)
        else:
            img_r = np.concatenate((img_r, mask_prev_r_255), axis=0)

 	batch = []
        masks = []
        masks_prev = []
        preds_prev = []
        batch.append(img_r)
        masks.append(mask_r)
        masks_prev.append(mask_prev_r)
        preds_prev.append(mask_prev_r)
        batch_ar = np.array(batch)
        masks_ar = np.array(masks)
        masks_prev_ar = np.array(masks_prev)
        preds_prev_ar = np.array(preds_prev)

        self.solver.net.blobs[self.solver.net.inputs[0]].reshape(*batch_ar.shape)
        self.solver.net.blobs[self.solver.net.inputs[1]].reshape(*masks_ar.shape)
        self.solver.net.blobs[self.solver.net.inputs[0]].data[...]= batch_ar
        self.solver.net.blobs[self.solver.net.inputs[1]].data[...]= masks_ar

        self.solver.net.blobs[self.solver.net.inputs[2]].reshape(*masks_prev_ar.shape)
        self.solver.net.blobs[self.solver.net.inputs[3]].reshape(*preds_prev_ar.shape)
        self.solver.net.blobs[self.solver.net.inputs[2]].data[...]= masks_prev_ar
        self.solver.net.blobs[self.solver.net.inputs[3]].data[...]= preds_prev_ar
        self.solver.net.forward()

	if 'alpha_pred_s' in self.solver.net.blobs.keys():
            pred_prev = self.solver.net.blobs['alpha_pred_s'].data[0].copy()
        else:
            pred_prev = self.solver.net.blobs['alpha_pred'].data[0].copy()
    
        return pred_prev

    def switch_to_test(self):
        self.list_ind = 0
        self.is_test_phaze = True
        self.epoch_ind = 0
        self.use_data_aug = False
        self.root_data_ind = None

    def get_batch(self,list_,batch_size = None):
        if batch_size is None:
            batch_size = self.batch_size

        batch = []
        masks = []
        masks_prev = []
        preds_prev = []
        self.img_orig =[]
        self.img_resized=[]
        self.mask_orig =[]
        self.mask_resized =[]
        self.mask_prev_orig =[]
        self.mask_prev_resized =[]
        self.pred_prev_orig =[]
        self.pred_prev_resized =[]
        self.trimap_orig =[]
        self.images_path_in_batch = []
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
                img_r, mask_r, mask_prev_r, pred_prev_r = self.get_tuple_data_point(list_[self.list_ind])
            else:
                img_r, mask_r, mask_prev_r, pred_prev_r, trimap_r = self.get_tuple_data_point(list_[self.list_ind])

            if img_r is None or mask_r is None:
                del list_[self.list_ind]
                continue

	    if pred_prev_r is None:
                pred_prev_r = np.zeros_like(mask_prev_r);
            
            pred_prev_r_255 = 255.0*pred_prev_r;
            if 'trimap_r' in locals():
                # img_r = np.concatenate((img_r, trimap_r, mask_prev_r), axis=0)		
                img_r = np.concatenate((img_r, trimap_r, pred_prev_r_255), axis=0)
            else:
                img_r = np.concatenate((img_r, pred_prev_r_255), axis=0)

            batch.append(img_r)
            masks.append(mask_r)
            masks_prev.append(mask_prev_r)
            preds_prev.append(pred_prev_r)
            self.list_ind += 1

        return np.array(batch), np.array(masks), np.array(masks_prev), np.array(preds_prev)

    def get_batch_data(self, batch_size=None):
        return self.get_batch(self.images_list_train)

    def get_test_data(self,batch_size = 1):
        return self.get_batch(self.images_list_test,batch_size)



