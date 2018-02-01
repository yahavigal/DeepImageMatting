import tensorflow as tf
import numpy as np
import caffe
import ipdb
import math
import sys
from google.protobuf import text_format
from caffe.proto import caffe_pb2
caffe_weights = {}

def load_caffe_weights(net,model):
    global caffe_weights
    if 'solver' in net:
        sp = caffe_pb2.SolverParameter()
        text_format.Merge(open(net).read(),sp)
        net = caffe.Net(sp.net.encode('ascii','ignore'),model,caffe.TEST)
    else:
        net = caffe.Net(net,model,caffe.TEST)

    caffe_weights = net.params

def init_graph():
    init_all_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_all_op)

def get_input_tensor(width, height, channels =4, batch_size = 1):
    arr = np.random.rand(batch_size,height,width,channels).astype(np.float32)
    return tf.Variable(initial_value=arr, name="input")
    #return tf.placeholder(dtype=np.float32,shape=(batch_size,height,width,channels),name="input_tensor_"+str(batch_size))

def cross_entropy_loss(pred,gt):
    #ipdb.aet_trace()
    return tf.reduce_mean(-gt*tf.log(pred + 1e-12) - (1- gt)*tf.log(1-pred +1e-12))

def alpha_prediction_loss(pred,gt):
    return tf.reduce_mean(tf.sqrt(tf.square(pred -gt) + 1e-12))


def mask_iou(pred,gt,thresh = 0.5):
    intersection = tf.logical_and(pred >= thresh,gt >= thresh)
    union = tf.logical_or(pred >= thresh,gt >= thresh)
    return tf.reduce_mean(tf.divide(tf.reduce_sum(tf.cast(intersection,dtype=np.int32)),tf.reduce_sum(tf.cast(union,dtype=np.int32))))


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
        w_kernel = w_kernel.transpose([2, 3, 1, 0])
        w_kernel = tf.Variable(initial_value=w_kernel, trainable=True,dtype=np.float32)

        if not atrous_rate:
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


def max_pool_with_mask(input,name):
    with tf.name_scope(name) as name:
        result, indices =  tf.nn.max_pool_with_argmax(input,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',
                                                       name=name)
        ipdb.set_trace()
        indices = tf.reshape(indices,[-1])

        #mask = tf.get_variable(name+'_mask',shape = input.get_shape(), initializer=tf.zeros_initializer())
        mask = tf.sparse_to_dense(indices,input.get_shape().as_list(),1,0)
        return result,mask

def unpool(input,mask, name):
    with tf.name_scope(name) as name:
        shape = input.get_shape().as_list()
        shape[1]= shape[1]/2.0
        shape[2] = shape[2]/2.0
        shape_11 = int(shape[1])
        shape_12 = int(math.ceil(shape[1]))
        shape_21 = int(shape[2])
        shape_22 = int(math.ceil(shape[2]))

        enlarged_input = tf.pad(input,
                                paddings=[[0,0],[shape_11,shape_12],[shape_21,shape_22],[0,0]],
                                mode='SYMMETRIC')
        result = tf.multiply(tf.cast(mask,tf.float32),enlarged_input)
    return result

class TF_trainer:
    def __init__(self,graph_name,image_width,image_height,image_channels = 4,batch_size =1):
        self.graph_name = graph_name
        self.batch_size = batch_size
        self.image_width = image_width
        self.image_height = image_height
        self.image_channels = image_channels
        self.sess = tf.InteractiveSession()
        self.input_tensor = tf.placeholder(dtype = np.float32,
                                           shape=(self.batch_size,self.image_height,self.image_width,self.image_channels),
                                           name = 'data')
        self.input_mask = tf.placeholder(np.float32,shape=[self.batch_size,self.image_height,self.image_width,1])

        sys.path.append('../or/{}/scripts/'.format(self.graph_name))
        import convert_to_tf
        self.graph = getattr(convert_to_tf,graph_name)(self.input_tensor)

        self.cost = alpha_prediction_loss(self.graph, self.input_mask)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-5,beta1=0.99).minimize(self.cost)

        self.sess.run(tf.global_variables_initializer())

    def step(self, images, masks):
        data_tf_fromat = images.transpose([0, 2, 3, 1])
        mask_tf_fromat = masks.transpose([0, 2, 3, 1])

        iou = mask_iou(self.graph, self.input_mask)
        alpha_pred,loss,_ ,iou =  self.sess.run([self.graph,self.cost,self.optimizer,iou],
                                                feed_dict={self.input_tensor: data_tf_fromat, self.input_mask: mask_tf_fromat})
        return alpha_pred,loss,iou

    def change_mode_to_inference():
        tf.reshape(self.input_tensor,shape=(1,self.image_height,self.image_width,self.image_channels))

    def run_inference(self, image,gt):
        data_tf_fromat = image.transpose([0, 2, 3, 1])
        mask_tf_fromat = gt.transpose([0, 2, 3, 1])
        pred,iou = self.sess.run([self.graph,mask_iou(self.graph,self.input_mask)],
                             feed_dict={self.input_tensor: data_tf_fromat, self.input_mask: mask_tf_fromat})
        return pred.transpose([0, 3, 1, 2]),iou

    def cross_entropy_loss(self,pred, gt):
        return tf.reduce_mean(-gt * tf.log(pred) - (1 - gt) * tf.log(1 - pred))

    def alpha_prediction_loss(self,pred, gt):
        return tf.reduce_mean(tf.sqrt(tf.square(pred - gt) + 1e-12))

    def save(self):
        tf.add_to_collection("alpha_pred", self.graph)
        saver = tf.train.Saver()
        saver.save(self.sess, self.graph_name)


    def __del__(self):
        self.sess.close()








