import os
import random
import cv2
import numpy as np
from data_provider import *
import ipdb
from random import randint
from collections import defaultdict

class TemporalDataProvider(DataProvider) :

    def __init__(self, images_dir_test, images_dir_train,trimap_dir=None,
                 batch_size = 32, use_data_aug = True, use_adv_data_train = False,threshold_param = -1,
                 img_width=128,img_height=128):

        super(TemporalDataProvider,self).__init__(images_dir_test,images_dir_train,trimap_dir,shuffle_data=True,
                       batch_size=batch_size,use_data_aug=True,use_adv_data_train=False,
                       threshold_param= threshold_param,img_width= img_width,img_height=img_height)

        self.train_last_clip_name = self.images_list_train[0].split(os.sep)[-3]
        self.test_last_clip_name = self.images_list_test[0].split(os.sep)[-3]
        self.last_masks = None
        self.clip_length = int(self.batch_size * 1.5)
        self.defects_list = defaultdict(int)
        self.get_current_clip_list(self.images_list_train,0)

    def get_current_clip_list(self,list_, ind):
        print 'starting {}'.format(list_[ind])
        current_dir = list_[ind]
        current_images = [os.path.join(current_dir,x) for x in os.listdir(current_dir) if self.gt_ext not in x
                                                                                       and 'color'  in x]
        current_images = sorted(current_images, key=lambda x: int(x.split(os.sep)[-1].split('_')[0]))
        if self.is_test_phaze == True:
            self.current_clip_list = current_images
        else:
            clip_start = randint(0,len(current_images) - self.clip_length + 5)
            self.current_clip_list = current_images[clip_start:clip_start+self.clip_length]
        self.current_clip_ind = 0

    def get_batch_data(self):
        return self.get_batch(self.images_list_train, self.batch_size)

    def get_test_data(self):
        return self.get_batch(self.images_list_test, self.batch_size)

    def switch_to_test(self):
        self.is_test_phaze = True
        self.list_ind = 0
        self.iter_ind = 0
        self.get_current_clip_list(self.images_list_test,self.list_ind)

    def get_batch(self, list_ , batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        batch = []
        masks = []
        self.img_orig =[]
        self.img_resized=[]
        self.mask_orig =[]
        self.mask_resized =[]
        self.trimap_orig =[]
        self.images_path_in_batch = []
        self.images_path_in_batch = []
        new_clip_started = False
        while len(batch) < batch_size:

            if self.current_clip_ind >= len(self.current_clip_list):
                masks = []
                batch = []
                self.list_ind += 1
                if self.list_ind >= len(list_):
                    print "starting from beginning of the list epoch {} finished".format(self.epoch_ind)
                    self.epoch_ind += 1
                    self.list_ind = 0
                else:
                    print 'in clip {} from {}'.format(self.list_ind,len(list_))
                self.get_current_clip_list(list_,self.list_ind)
                new_clip_started = True
                continue

            if self.trimap_dir == None:
                img_r, mask_r = self.get_tuple_data_point(self.current_clip_list[self.current_clip_ind])
            else:
                img_r, mask_r, trimap_r = self.get_tuple_data_point(self.current_clip_list[self.current_clip_ind])
            if img_r is None or mask_r is None:
                self.current_clip_ind = len(self.current_clip_list)
                self.defects_list[list_[self.list_ind]] += 1
                if self.defects_list[list_[self.list_ind]] > 5:
                    print 'delete {} after 5 defects'.format(list_[self.list_ind])
                    del list_[self.list_ind]
                continue

            if 'trimap_r' in locals():
                img_r = np.concatenate((img_r, trimap_r), axis=0)

            if self.iter_ind != 0 and new_clip_started == False:
                #print 'last masks has been injected'
                img_r = np.insert(img_r,img_r.shape[0],self.last_masks[len(batch)-1,0,:],axis=0)
            else:
                #print 'zeros has been injected'
                img_r = np.insert(img_r,img_r.shape[0],np.zeros((self.img_height,self.img_width),dtype=np.float32),axis=0)

            batch.append(img_r)
            masks.append(mask_r)
            self.current_clip_ind += 1
        self.current_clip_ind -= (self.batch_size -1)
        return np.array(batch), np.array(masks)
