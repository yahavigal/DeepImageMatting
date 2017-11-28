# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 23:21:08 2017

@author: or
"""

from bgs_train_test import *
import argparse

def test_net(images_dir_test, net_path,weights_path, trimap_dir,save_loss_per_image =False,finetune=None):
    if finetune is None:
        train_list =""
    else:
        train_list = finetune
    trainer = bgs_test_train(images_dir_test, train_list, net_path,weights_path,"",trimap_dir = trimap_dir,
                             save_loss_per_image=save_loss_per_image)
    
   
    var = trainer.test()
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
        
            

    
