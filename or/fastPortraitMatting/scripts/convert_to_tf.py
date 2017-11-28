import caffe
import tensorflow as tf
import argparse
import ipdb
import os
import numpy as np
caffe_weights = {}


def conv(name, input, strides, padding, add_bias, apply_relu, atrous_rate=None, trainable = True):
    """
    Helper function for loading convolution weights from weight dictionary.
    """
    with tf.variable_scope(name):

        # Load kernel weights and apply convolution
        w_kernel = caffe_weights[name][0].data
        w_kernel = w_kernel.transpose([2,3,1,0])
        w_kernel = tf.Variable(initial_value=w_kernel, trainable=trainable,dtype=np.float32)

        if not atrous_rate:
            conv_out = tf.nn.conv2d(input, w_kernel, strides, padding)
        else:
            conv_out = tf.nn.atrous_conv2d(input, w_kernel, atrous_rate, padding)
        if add_bias:
            # Load bias values and add them to conv output
            w_bias = caffe_weights[name][1].data
            w_bias = tf.Variable(initial_value=w_bias, trainable=trainable,dtype=np.float32)
            conv_out = tf.nn.bias_add(conv_out, w_bias)

        if apply_relu:
            # Apply ReLu nonlinearity
            conv_out = tf.nn.relu(conv_out)


        return conv_out

def deconv(name, input, output_shape, padding, add_bias, apply_relu, atrous_rate=None):
    """
    Helper function for loading convolution weights from weight dictionary.
    """
    with tf.variable_scope(name):

        # Load kernel weights and apply convolution
        w_kernel = caffe_weights[name][0].data
        #ipdb.set_trace()
        w_kernel = w_kernel.transpose([2, 3, 1, 0])
        w_kernel = tf.Variable(initial_value=w_kernel, trainable=True,dtype=np.float32)

        #matmul
        #w_kernel = tf.reshape(w_kernel,shape=(2,32))
        #input_r= tf.reshape(input,shape=(2,4096))
        #deconv_out = tf.matmul(w_kernel,input_r, transpose_a=True)



        if not atrous_rate:
            #ipdb.set_trace()
            deconv_out = tf.nn.conv2d_transpose(input, w_kernel,strides=[1,2,2,1],output_shape= output_shape,
                                                padding=padding)
        else:
            deconv_out = tf.nn.atrous_conv2d_transpose(input, w_kernel, output_shape, atrous_rate, padding)
        if add_bias:
            # Load bias values and add them to conv output
            w_bias = caffe_weights[name][1].data
            w_bias = tf.Variable(initial_value=w_bias, trainable=False,dtype=np.float32)
            deconv_out = tf.nn.bias_add(deconv_out, w_bias)

        if apply_relu:
            # Apply ReLu nonlinearity
            deconv_out = tf.nn.relu(deconv_out)


        return deconv_out

def batch_normalization(name,input, trainable = True):

    with tf.variable_scope(name):
        # Load mean and variance
        scale_factor  = caffe_weights[name][2].data
        if scale_factor != 0:
            scale_factor = 1.0/scale_factor
        w_mean = caffe_weights[name][0].data*scale_factor
        w_variance = caffe_weights[name][1].data * scale_factor
        w_mean = tf.Variable(initial_value=w_mean, trainable=trainable,dtype=np.float32)
        w_variance = tf.Variable(initial_value=w_variance, trainable=trainable,dtype=np.float32)
        return tf.nn.batch_normalization(input,w_mean,w_variance,0,1,variance_epsilon=1e-5, name=name)



def fastPortraitMatting(input_tensor,batch_size =32):

    concat_axis = 3

    conv_1 = conv('conv_1',input_tensor,strides = [1,2,2,1],padding="SAME",add_bias=True,apply_relu=False)
    pool_1 = tf.nn.max_pool(input_tensor,ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME", name="pool_1")
    concat_1 = tf.concat([pool_1,conv_1], axis=3, name = "concat_1")
    conv_2 = conv('conv_2',concat_1,strides = [1,1,1,1],padding="SAME", atrous_rate=2, add_bias=True, apply_relu=True)
    concat_2 = tf.concat([conv_2,concat_1],axis=concat_axis, name = "concat_2")
    conv_3 = conv('conv_3',concat_2,strides=[1,1,1,1], padding="SAME", atrous_rate=4, add_bias=True, apply_relu=True)
    concat_3 = tf.concat([conv_3,concat_2], concat_axis, name="concat_3")

    conv_4 = conv('conv_4',concat_3, strides=[1, 1, 1, 1], padding="SAME", atrous_rate=6, add_bias=True, apply_relu=True)
    concat_4 = tf.concat([conv_4, concat_3], concat_axis, name="concat_4")
    conv_5 = conv('conv_5',concat_4, strides=[1, 1, 1, 1], padding="SAME", atrous_rate=8, add_bias=True, apply_relu=True)
    concat_5 = tf.concat([conv_2, conv_3, conv_4, conv_5], concat_axis, name="concat_5")
    conv_6 = conv('conv_6',concat_5, strides=[1, 1, 1, 1], padding="SAME", add_bias=True, apply_relu=False)

    upsampling = deconv("upsampling",conv_6,[batch_size,128,128,2],padding="SAME", add_bias=False, apply_relu=False)

    foreground, background = tf.split(upsampling,num_or_size_splits=2,axis=3,name="silcer_1")
    data_squared = tf.multiply(input_tensor,input_tensor,name="data_squared")

    concat_foreground = tf.concat([foreground,foreground,foreground,foreground],axis=concat_axis,name="concat_foreground")

    data_fg = tf.multiply(input_tensor,concat_foreground,name="data_fg")

    concat_feathering = tf.concat([input_tensor,background,foreground,data_squared,data_fg],concat_axis,name="concat_feathering")
    conv_feathering_1 = conv("conv_feathering_1",concat_feathering,[1,1,1,1],padding="SAME",
                             add_bias=True,apply_relu=False,trainable=True)

    bnorm_1 = batch_normalization("bnorm_1",conv_feathering_1,trainable=True)

    bnorm_1 = tf.nn.relu(bnorm_1,name="relu_feathering_1")

    conv_feathering_2 = conv("conv_feathering_2", bnorm_1,[1, 1, 1, 1], padding="SAME",
                             add_bias=True,apply_relu=False,trainable=True)

    a,b,c= tf.split(conv_feathering_2,num_or_size_splits=3,axis=concat_axis,name="slicer_2")
    a_mult_bg = tf.multiply(a,background,name="a_mult_bg")
    b_mult_fg = tf.multiply(b, foreground, name="b_mult_fg")
    alpha_pred =  tf.add_n([a_mult_bg, b_mult_fg, c], name="guided_filter")
    alpha_pred =  tf.nn.sigmoid(alpha_pred, name="alpha_pred")

    return  alpha_pred

def load_caffe_weights(net,model):
    net = caffe.Net(net,model,caffe.TEST)
    global caffe_weights
    caffe_weights = net.params

def init_graph():
    init_all_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_all_op)

def get_input_tensor(batch_size = 1):
    return tf.placeholder(dtype=np.float32,shape=(batch_size,128,128,4),name="input_tensor_"+str(batch_size))

def cross_entropy_loss(pred,gt):
    #ipdb.set_trace()
    return tf.reduce_mean(-gt*tf.log(pred + 1e-12) - (1- gt)*tf.log(1-pred +1e-12))

def alpha_prediction_loss(pred,gt):
    return tf.reduce_mean(tf.sqrt(tf.square(pred -gt) + 1e-12))


def mask_iou(pred,gt,thresh = 0.5):
    intersection = tf.logical_and(pred >= thresh,gt >= thresh)
    union = tf.logical_or(pred >= thresh,gt >= thresh)
    return tf.reduce_mean(tf.divide(tf.reduce_sum(tf.cast(intersection,dtype=np.int32)),tf.reduce_sum(tf.cast(union,dtype=np.int32))))


class TF_trainer:
    def __init__(self):
        self.batch_size = 1
        self.sess = tf.Session()
        self.input_tensor = get_input_tensor(self.batch_size)
        self.input_mask = tf.placeholder(np.float32,shape=[self.batch_size,128,128,1])
        self.graph = fastPortraitMatting(self.input_tensor,self.batch_size)
        self.cost = alpha_prediction_loss(self.graph, self.input_mask)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-5,beta1=0.99).minimize(self.cost)
        self.sess.run(tf.global_variables_initializer())

    def run_fine_tune_to_deconv(self, images, masks):
        data_tf_fromat = images.transpose([0, 2, 3, 1])
        mask_tf_fromat = masks.transpose([0, 2, 3, 1])

        iou = mask_iou(self.graph, self.input_mask)
        alpha_pred,loss,_ ,iou =  self.sess.run([self.graph,self.cost,self.optimizer,iou],
                                                feed_dict={self.input_tensor: data_tf_fromat, self.input_mask: mask_tf_fromat})
        return alpha_pred,loss,iou

    def run_inference(self, image,gt):
        data_tf_fromat = image.transpose([0, 2, 3, 1])
        mask_tf_fromat = gt.transpose([0, 2, 3, 1])
        pred,iou = self.sess.run([self.graph,mask_iou(self.graph,self.input_mask)],
                             feed_dict={self.input_tensor: data_tf_fromat, self.input_mask: mask_tf_fromat})
        return pred.transpose([0, 3, 1, 2]),iou

    def cross_entropy_loss(self,pred, gt):
        # ipdb.set_trace()
        return tf.reduce_mean(-gt * tf.log(pred) - (1 - gt) * tf.log(1 - pred))

    def alpha_prediction_loss(self,pred, gt):
        return tf.reduce_mean(tf.sqrt(tf.square(pred - gt) + 1e-12))

    def save(self):
        tf.add_to_collection("alpha_pred", self.graph)
        saver = tf.train.Saver()
        saver.save(self.sess, "fastPortraitMatting")


    def __del__(self):
        self.sess.close()



#def run_inference(graph,input_tensor,image):
#    data_tf_fromat = image.transpose([0, 2, 3, 1])
#    init_all_op = tf.global_variables_initializer()
#    with tf.Session() as sess:
#        sess.run(init_all_op)
#        pred = sess.run(graph,feed_dict={input_tensor:data_tf_fromat})
#        return pred.transpose([0,3,1,2])



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--net', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)

    args = parser.parse_args()

    load_caffe_weights(args.net,args.model)

    input_tensor = get_input_tensor()
    tf_model = fast_portrait_matting(input_tensor)
    tf.add_to_collection("alpha_pred",tf_model)

    init_all_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_all_op)
        saver = tf.train.Saver()
        saver.save(sess, "fastPortraitMatting")







