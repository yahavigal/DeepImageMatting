import caffe
import tensorflow as tf
import argparse
import ipdb
import os
import numpy as np
import sys
sys.path.append('../../../python/')
from TF_utils import *


def deepImageMatting(input_tensor):

    #def conv_relu_pool_block(input,names,pool_name):
    #    for name in names:
    #        res = conv1_1 = conv(name, input_tensor, strides=[1, 1, 1, 1], padding="SAME", add_bias=True, apply_relu=True)
    #    pool, mask = max_pool_with_mask(res, name=pool_name)

    conv1_1 = conv('conv1_1', input_tensor, strides=[1, 1, 1, 1], padding="SAME", add_bias=True, apply_relu=True)
    conv1_2 = conv('conv1_2', conv1_1, strides=[1, 1, 1, 1], padding="SAME", add_bias=True, apply_relu=True)
    pool_1,mask_1 = max_pool_with_mask(conv1_2, name="pool_1")

    conv2_1 = conv('conv2_1', pool_1, strides=[1, 1, 1, 1], padding="SAME", add_bias=True, apply_relu=True)
    conv2_2 = conv('conv2_2', conv2_1, strides=[1, 1, 1, 1], padding="SAME", add_bias=True, apply_relu=True)
    pool_2,mask_2 = max_pool_with_mask(conv2_2, name="pool_2")

    conv3_1 = conv('conv3_1', pool_2, strides=[1, 1, 1, 1], padding="SAME", add_bias=True, apply_relu=True)
    conv3_2 = conv('conv3_2', conv3_1, strides=[1, 1, 1, 1], padding="SAME", add_bias=True, apply_relu=True)
    conv3_3 = conv('conv3_3', conv3_2, strides=[1, 1, 1, 1], padding="SAME", add_bias=True, apply_relu=True)
    pool_3,mask_3 = max_pool_with_mask(conv3_3, name="pool_3")

    conv4_1 = conv('conv4_1', pool_3, strides=[1, 1, 1, 1], padding="SAME", add_bias=True, apply_relu=True)
    conv4_2 = conv('conv4_2', conv4_1, strides=[1, 1, 1, 1], padding="SAME", add_bias=True, apply_relu=True)
    conv4_3 = conv('conv4_3', conv4_2, strides=[1, 1, 1, 1], padding="SAME", add_bias=True, apply_relu=True)
    pool_4, mask_4 = max_pool_with_mask(conv4_3, name="pool_4")

    conv5_1 = conv('conv5_1', pool_4, strides=[1, 1, 1, 1], padding="SAME", add_bias=True, apply_relu=True)
    conv5_2 = conv('conv5_2', conv5_1, strides=[1, 1, 1, 1], padding="SAME", add_bias=True, apply_relu=True)
    conv5_3 = conv('conv5_3', conv5_2, strides=[1, 1, 1, 1], padding="SAME", add_bias=True, apply_relu=True)
    pool_5, mask_5 = max_pool_with_mask(conv5_3, name="pool_5")

    deconv_6 = conv('deconv_6',pool_5,strides=[1, 1, 1, 1], padding="SAME", add_bias=True, apply_relu=True)
    unpool_5 = unpool(deconv_6,mask_5, 'unpool_5')
    deconv_5 = conv('deconv_5', unpool_5, strides=[1, 1, 1, 1], padding="SAME", add_bias=True, apply_relu=True)

    unpool_4 = unpool(deconv_5, mask_4, 'unpool_4')
    deconv_4 = conv('deconv_4', unpool_4, strides=[1, 1, 1, 1], padding="SAME", add_bias=True, apply_relu=True)

    unpool_3 = unpool(deconv_4, mask_3, 'unpool_3')
    deconv_3 = conv('deconv_3', unpool_3, strides=[1, 1, 1, 1], padding="SAME", add_bias=True, apply_relu=True)

    unpool_2 = unpool(deconv_3, mask_2, 'unpool_2')
    deconv_2 = conv('deconv_2', unpool_2, strides=[1, 1, 1, 1], padding="SAME", add_bias=True, apply_relu=True)

    unpool_1 = unpool(deconv_2, mask_1, 'unpool_1')
    deconv_1 = conv('deconv_1', unpool_1, strides=[1, 1, 1, 1], padding="SAME", add_bias=True, apply_relu=True)

    alpha_pred = conv('raw_alpha_pred', deconv_1, strides=[1, 1, 1, 1], padding="SAME", add_bias=True, apply_relu=False)

    alpha_pred = tf.nn.sigmoid(alpha_pred)

    return  alpha_pred




if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--net', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)

    args = parser.parse_args()
    sess = tf.InteractiveSession()

    load_caffe_weights(args.net,args.model)

    input_tensor = get_input_tensor(width =224,height =224)
    tf_model = deepImageMatting(input_tensor)
    tf.add_to_collection("alpha_pred",tf_model)

    init_all_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_all_op)
        saver = tf.train.Saver()
        saver.save(sess, "deepImageMatting")







