import os
from  PIL import Image
import argparse
import re
import numpy as np
import cv2

def convert_directory(dir_path):
    files = [os.path.join(dir_path,x) for x in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path,x)) and 'Color' in x]
    for image in files:
        image_buffer =open(image,'rb').read()
        image_obj = Image.frombytes('RGBA',(640,480) ,image_buffer)
        image_obj = image_obj.convert('RGB')

        frame_num = re.findall(r'\d+',image)[-1]
        path_tosave_color ='{}_color.png'.format(frame_num)
        path_tosave_color = os.path.join(dir_path, path_tosave_color)
        print(image)
        print(path_tosave_color)
        image_obj.save(path_tosave_color,'PNG')
        depth_image_path = image.replace('Color','Depth')
        depth_image_path = depth_image_path.replace('color','z16')
        depth_image_buffer = open(depth_image_path,'rb').read()
        np_depth = np.frombuffer(depth_image_buffer, dtype=np.uint16)
        np_depth = np.reshape(np_depth,(480,848))
        path_tosave_depth = '{}_depth.png'.format(frame_num)
        path_tosave_depth = os.path.join(dir_path, path_tosave_depth)
        print(path_tosave_depth)
        cv2.imwrite(path_tosave_depth,np_depth)
        mask_image_path = image.replace('Color','Mask')
        mask_image_path = mask_image_path.replace('color','mask')
        mask_image_buffer = open(mask_image_path,'rb').read()
        np_mask = np.frombuffer(mask_image_buffer, dtype=np.uint8)
        np_mask = np.reshape(np_mask,(480,640))
        path_tosave_mask = '{}_guess.png'.format(frame_num)
        path_tosave_mask = os.path.join(dir_path, path_tosave_mask)
        print(path_tosave_mask)
        cv2.imwrite(path_tosave_mask,np_mask)



def convert_tree(path):
    for root,subdirs,files in os.walk(path):
        if len(subdirs) == 0:
            convert_directory(root)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True, help='the path of root dir for conversion')
    args = parser.parse_args()
    convert_tree(args.root)
