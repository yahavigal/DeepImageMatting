# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 23:21:08 2017

@author: or
"""

# python test_net.py --finetune ../or/deepImageMatting/scripts/train_list.txt --test_dir ../or/deepImageMatting/scripts/test_list.txt --trimap_dir /media/or/Data/deepImageMatting/set1_07_2017_depth_norm/ --net ../or/fastPortraitMatting/results/Resu2save/toTF/95.91_128/train.prototxt  --model  ../or/fastPortraitMatting/results/Resu2save/toTF/95.91_128/_iter_1792_MaxAccuracy9691.caffemodel

from bgs_train_test import *
import argparse

def test_net(images_dir_test, net_path,weights_path, trimap_dir,save_loss_per_image =False,finetune=None,is_save_fig =True):
    if finetune is None:
        train_list =""
    else:
        train_list = finetune
    trainer = bgs_test_train(images_dir_test, train_list, net_path,weights_path,"",trimap_dir = trimap_dir,
                             save_loss_per_image=save_loss_per_image)
    
   
    var = trainer.test(is_save_fig=is_save_fig)
    trainer.plot_statistics()
    return var

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--test_dir', type=str, required=True)
    parser.add_argument('--trimap_dir', type=str, required=False, default = None)
    parser.add_argument('--net', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--finetune', type=str, required=False,default = None)
    args = parser.parse_args()

    test_net(args.test_dir,args.net,args.model,args.trimap_dir,save_loss_per_image= False,finetune=args.finetune)
        
            

    
