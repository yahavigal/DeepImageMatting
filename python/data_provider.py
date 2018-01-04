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

class DataProvider :

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
                                      and (self.use_adv_data_train == False or x.find(self.adverserial_ext) != -1)]
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

        self.img_orig = None
        self.trimap_orig = None
        self.mask_orig = None

        self.img_resized = None
        self.trimap_resized = None
        self.mask_resized = None

        self.img_transposed = None
        self.trimap_transposed = None
        self.mask_transposed = None

    def get_tuple_data_point(self, image_path):

        if not os.path.exists(image_path):
            if self.trimap_dir == None:
                return [None, None]
            else:
                return [None, None, None]
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype('float32')
        self.img_orig = img.copy()

        if self.use_data_aug == True:

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
        self.img_resized = img_r
        path = os.path.splitext(image_path)
        gt_path = path[0] + self.gt_ext + path[1]
        if not os.path.isfile(gt_path):
            if self.trimap_dir == None:
                return [None, None]
            else:
                return [None, None, None]
        mask = cv2.imread(gt_path, 0)
        if self.threshold_param != -1:
            mask[mask < 256*self.threshold_param] = 0
            mask[mask >= 256*self.threshold_param] = 1
        else:
            mask/=255.0
        self.mask_orig = mask
        mask_r = cv2.resize(mask, (self.img_width, self.img_height), interpolation=cv2.INTER_NEAREST)
        self.mask_resized = mask_r
        if self.root_data_ind is None:
            self.root_data_ind = find_data_root_ind(image_path, self.trimap_dir)
        split = image_path.split(os.sep)[self.root_data_ind:]
        split = os.sep.join(split)
        frame_num = re.findall(r'\d+', split)[-1]
        split = os.path.split(split)

        if self.trimap_dir != None:
            trimap_path = os.path.join(self.trimap_dir,
                                       split[0], frame_num + self.trimap_ext + ".png")

            if not os.path.isfile(trimap_path):
                return [None, None, None]

            trimap = cv2.imread(trimap_path, 0)
            if trimap is None:
                return [None, None, None]
            self.trimap_orig = trimap.copy()
            trimap_r = cv2.resize(trimap, (self.img_width, self.img_height), interpolation=cv2.INTER_NEAREST)

            if self.use_data_aug == True:
                # rotation / filipping data augmentation
                coin = np.random.uniform(0, 1, 1)
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

    def get_batch_data(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        batch = []
        masks = []
        while len(batch) < batch_size:
            if self.list_ind >= len(self.images_list_train):
                print "starting from beginning of the list epoch {} finished".format(self.epoch_ind)
                if self.shuffle == True:
                    random.shuffle(self.images_list_train)
                self.epoch_ind += 1
                self.list_ind = 0
            if self.trimap_dir == None:
                img_r, mask_r = self.get_tuple_data_point(self.images_list_train[self.list_ind])
            else:
                img_r, mask_r, trimap_r = self.get_tuple_data_point(self.images_list_train[self.list_ind])
            if img_r is None or mask_r is None:
                del self.images_list_train[self.list_ind]
                continue

            if 'trimap_r' in locals():
                img_r = np.concatenate((img_r, trimap_r), axis=0)

            batch.append(img_r)
            masks.append(mask_r)
            self.list_ind += 1

        return np.array(batch), np.array(masks)


    def get_test_data(self,batch_size = 1):
        batch = []
        masks = []
        while len(batch) < batch_size:
            if self.test_list_ind >= len(self.images_list_test):
              return None,None
            if self.trimap_dir == None:
                img_r, mask_r = self.get_tuple_data_point(self.images_list_test[self.test_list_ind])
            else:
                img_r, mask_r, trimap_r = self.get_tuple_data_point(self.images_list_test[self.test_list_ind])
            if img_r is None or mask_r is None:
                del self.images_list_test[self.test_list_ind]
                continue

            if 'trimap_r' in locals():
                img_r = np.concatenate((img_r, trimap_r), axis=0)
            self.test_image_path = self.images_list_test[self.test_list_ind]
            batch.append(img_r)
            masks.append(mask_r)
            self.test_list_ind += 1

        return np.array(batch), np.array(masks)




