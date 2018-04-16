import caffe
import numpy as np
import argparse
from data_provider import *
from collections import defaultdict
import ipdb

def create_data_provider(train_dir,test_dir, trimap_dir, img_width, img_height, threshold):

    data_provider = DataProvider(test_dir,train_dir,trimap_dir,shuffle_data= False,
                                      batch_size=1,use_data_aug=False,use_adv_data_train=False,
                                      threshold_param=threshold,img_width= img_width,img_height=img_height)
    return data_provider


def calc_norms(net,dict_,images,masks):

        net.blobs[net.inputs[0]].reshape(*images.shape)
        net.blobs[net.inputs[1]].reshape(*masks.shape)
        net.blobs[net.inputs[0]].data[...]= images
        net.blobs[net.inputs[1]].data[...]= masks
        net.forward()
        net.backward()

        for k,v in net.params.items():
            dict_[k].append(np.average(np.abs(v[0].diff)))


def test_train_grad_diff(net_path, model_path,train_dir,test_dir,trimap_dir,threshold):

    net = caffe.Net(net_path,model_path,caffe.TRAIN)
    img_width = net.blobs[net.inputs[0]].shape[3]
    img_height = net.blobs[net.inputs[0]].shape[2]

    data_provider = create_data_provider(train_dir,test_dir,trimap_dir,img_width,img_height,threshold)

    train_grad_norms = defaultdict(list)
    test_grad_norms = defaultdict(list)

    #iterate over train data
    for _ in data_provider.images_list_train:
        images,masks = data_provider.get_batch_data()
        calc_norms(net,train_grad_norms,images,masks)

    #ipdb.set_trace()

    #iterate over test data
    for _ in data_provider.images_list_test:
        images,masks = data_provider.get_test_data()
        calc_norms(net,test_grad_norms,images,masks)

    for k,v in train_grad_norms.items():
        avg_train = np.average(v)
        avg_test = np.average(np.average(test_grad_norms[k]))
        print k, avg_train, avg_test, avg_test - avg_train,   (avg_test - avg_train)/(avg_test + avg_train)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='utility to find overfitting layer-wise',
    formatter_class=argparse.RawTextHelpFormatter,
    epilog= '''Usage:
    python test_train_grad_diff.py --train_dir ../BGS_scripts/train_list.txt
                              --test_dir ../BGS_scripts/test_list.txt
                              --trimap_dir /media/or/Data/deepImageMatting/set1_07_2017_depth_norm
                              --net ../or/fastPortraitMatting/proto/train.prototxt
                              --model  ../or/fastPortraitMatting/snapshots/_iter_100.caffemodel'''
                                    )
    parser.add_argument('--train_dir', type=str, required=True, help="train directory path or list")
    parser.add_argument('--test_dir', type=str, required=True, help="test directory path or list")
    parser.add_argument('--trimap_dir', type=str, required=False, default = None,help="trimap or any addtional output")
    parser.add_argument('--net', type=str, required=True,help="path to net definition file")
    parser.add_argument('--model', type=str, required=False, default = None, help="pre-trained weights path")
    parser.add_argument('--threshold', type=float,required=False, default = -1, help= "threshold for mask if -1 no thresholding applied")
    args = parser.parse_args()

    test_train_grad_diff(args.net,args.model,args.train_dir,args.test_dir,args.trimap_dir,args.threshold)


