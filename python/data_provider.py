import os
import random
import cv2
import data_augmentation
import numpy as np
import re
import multiprocessing
import ipdb

def worker(data_provider,images,batch,masks):
    try:
        for image_path in images:
            if data_provider.trimap_dir == None:
                img_r, mask_r = data_provider.get_tuple_data_point(image_path)
            else:
                img_r, mask_r, trimap_r = data_provider.get_tuple_data_point(image_path)
                img_r = np.concatenate((img_r, trimap_r), axis=0)
            batch.append(img_r)
            masks.append(mask_r)
    except:
        pass

manager = multiprocessing.Manager()
class DataProvider(object) :

    def find_data_root_ind(self, image,trimap_root):
        ind = 0
        image_split = image.split(os.sep)
        if self.trimap_is_dir:
    	    trimap_split = trimap_root.split(os.sep)
    	    while image_split[ind] == trimap_split[ind]:
               ind +=1
            ind += 1
        else:
            if self.is_still:
                ind = image_split.index(self.key_still)
            else:
		ind = image_split.index(self.key_video)
        return ind

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

        self.gt_ext = "silhouette"
        self.trimap_ext = None
        self.color_ext = "color"
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
        if trimap_dir is not None:
            if os.path.isdir(self.trimap_dir):
                self.trimap_is_dir = True
            else:
                self.trimap_is_dir = False

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
        self.images_path_in_batch = []

        self.is_test_phaze = False

    def get_tuple_data_point(self, image_path, isToAddToPathList = True):

        if not os.path.exists(image_path):
            print image_path
            if self.trimap_dir == None:
                return [None, None]
            else:
                return [None, None, None]

        img = cv2.imread(image_path)
        if self.is_test_phaze == True:
            self.img_orig.append(cv2.cvtColor(img.copy(),cv2.COLOR_BGR2RGB).astype('float32'))


        img_r = cv2.resize(img, (self.img_width, self.img_height))
        img_r = img_r[...,::-1].astype('float32')
        if self.use_data_aug == True:

            # image based data augmentation
            coin = np.random.uniform(0, 1, 1)
            if coin <= 0.25:
                img_r = data_augmentation.color_jitter(img_r)
            elif coin <= 0.5:
                img_r = data_augmentation.PCA_noise(img_r)
            elif coin <= 0.75:
                img_r = data_augmentation.gamma_correction(img_r)

        # subtract mean
        img_r -= np.array([104, 117, 123], dtype=np.float32)

        gt_path = image_path.replace(self.color_ext, self.gt_ext)
        self.gt_path = gt_path
        if not os.path.isfile(gt_path):
            print 'gt not found {}'.format(gt_path)
            del self.img_resized[-1]
            del self.img_orig[-1]
            if self.trimap_dir == None:
                return [None, None]
            else:
                return [None, None, None]
        mask = cv2.imread(gt_path, 0)
        mask_r = cv2.resize(mask, (self.img_width, self.img_height), interpolation=cv2.INTER_NEAREST)
        if self.threshold_param != -1:
            mask[mask_r < 256*self.threshold_param] = 0
            mask[mask_r >= 256*self.threshold_param] = 1
        else:
            mask_r = np.divide(mask_r,255.0)
        if self.is_test_phaze == True:
            self.mask_orig.append(mask.copy().astype('float32'))
        if self.trimap_dir is not None:
            frame_num = re.findall(r'\d+', image_path)[-2]

            self.frame_num = frame_num

        if isToAddToPathList and self.is_test_phaze == True:
            self.images_path_in_batch.append(image_path)

        if self.trimap_dir != None:
            trimap_path = image_path.replace(self.color_ext,self.trimap_dir).replace('640x480','160x120')
            if not os.path.isfile(trimap_path):
                print "depth file not found {} ".format(trimap_path)
                del self.img_orig[-1]
                del self.mask_orig[-1]
                return [None, None, None]

            trimap = cv2.imread(trimap_path, 0)
            if trimap is None:
                del self.img_orig[-1]
                del self.mask_orig[-1]
                return [None, None, None]

            if self.is_test_phaze == True:
                self.trimap_orig.append(trimap.copy())
            trimap_r = cv2.resize(trimap, (self.img_width, self.img_height), interpolation=cv2.INTER_NEAREST)

            if self.use_data_aug == True:
                # rotation / filipping data augmentation
                coin = np.random.uniform(0, 1, 1)
                if image_path and coin <= 0.7:
                    img_r, mask_r, trimap_r = data_augmentation.translate(img_r, mask_r, trimap_r)
                else:
                    if coin <= 0.33:
                        img_r, mask_r, trimap_r = data_augmentation.horizontal_flipping(img_r, mask_r, trimap_r)
                    elif coin <= 0.66:
                        img_r, mask_r, trimap_r = data_augmentation.rotate(img_r, mask_r, trimap_r)

            trimap_r = trimap_r.reshape([1, self.img_height, self.img_width])
            mask_r = mask_r.reshape([1, self.img_height, self.img_width])
            img_r = img_r.transpose([2, 0, 1])

            return img_r, mask_r, trimap_r
        else:
            mask_r = mask_r.reshape([1, self.img_height, self.img_width])
            img_r = img_r.transpose([2, 0, 1])
            return img_r, mask_r

    def switch_to_test(self):
        self.list_ind = 0
        self.is_test_phaze = True
        self.epoch_ind = 0
        self.use_data_aug = False
        self.root_data_ind = None

    def get_batch(self,list_,batch_size = None):
        if batch_size is None:
            batch_size = self.batch_size
        batch = manager.list()
        masks = manager.list()
        self.img_orig =[]
        self.mask_orig =[]
        self.trimap_orig =[]
        self.images_path_in_batch = []
        num_of_jobs = 8
        if self.is_test_phaze == True:
          num_of_jobs = 1
        samples_per_job = batch_size/num_of_jobs
        jobs= []
        for i in range(num_of_jobs):
            if self.list_ind >= len(list_):
                self.epoch_ind += 1
                self.list_ind = 0
                if self.is_test_phaze == True:
                    return np.array(batch), np.array(masks)
                else:
                    print "starting from beginning of the list epoch {} finished".format(self.epoch_ind)
                    if self.shuffle == True:
                        random.shuffle(list_)
            jobs_pers_p = []
            while len(jobs_pers_p) < samples_per_job:
                jobs_pers_p.append(list_[self.list_ind])
                self.list_ind += 1
                if self.list_ind >= len(list_):
                    self.epoch_ind += 1
                    self.list_ind = 0
                    if self.is_test_phaze == True:
                        return np.array(batch), np.array(masks)
                    else:
                        print "starting from beginning of the list epoch {} finished".format(self.epoch_ind)
                        if self.shuffle == True:
                            random.shuffle(list_)
            if self.is_test_phaze==False:
                p = multiprocessing.Process(target=worker, args=(self,jobs_pers_p,batch,masks))
                p.start()
                jobs.append(p)
            else:
                worker(self,jobs_pers_p,batch,masks)

        for job in jobs:
            job.join()


        return np.array(batch), np.array(masks)

    def get_batch_data(self, batch_size=None):
        return self.get_batch(self.images_list_train)

    def get_test_data(self,batch_size = 1):
        return self.get_batch(self.images_list_test,batch_size)

