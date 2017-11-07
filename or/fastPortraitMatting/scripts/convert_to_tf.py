import caffe
import tensorflow as tf
import argparse
import ipdb

caffe_weights = {}


def conv(name, input, strides, padding, add_bias, apply_relu, atrous_rate=None):
    """
    Helper function for loading convolution weights from weight dictionary.
    """
    with tf.variable_scope(name):

        # Load kernel weights and apply convolution
        w_kernel = caffe_weights[name][0].data
        w_kernel = w_kernel.transpose([2,3,1,0])
        w_kernel = tf.Variable(initial_value=w_kernel, trainable=False)

        if not atrous_rate:
            conv_out = tf.nn.conv2d(input, w_kernel, strides, padding)
        else:
            conv_out = tf.nn.atrous_conv2d(input, w_kernel, atrous_rate, padding)
        if add_bias:
            # Load bias values and add them to conv output
            w_bias = caffe_weights[name][1].data
            w_bias = tf.Variable(initial_value=w_bias, trainable=False)
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
        w_kernel = tf.Variable(initial_value=w_kernel, trainable=False)

        if not atrous_rate:
            deconv_out = tf.nn.conv2d_transpose(input, w_kernel,strides=[1,2,2,1],output_shape= output_shape, padding=padding)
        else:
            deconv_out = tf.nn.atrous_conv2d_transpose(input, w_kernel, output_shape,atrous_rate, padding)
        if add_bias:
            # Load bias values and add them to conv output
            w_bias = caffe_weights[name][1].data
            w_bias = w_bias.transpose([2, 3, 1, 0])
            w_bias = tf.Variable(initial_value=w_bias, trainable=False)
            deconv_out = tf.nn.bias_add(deconv_out, w_bias)

        if apply_relu:
            # Apply ReLu nonlinearity
            deconv_out = tf.nn.relu(deconv_out)


        return deconv_out

def batch_normalization(name,input):

    with tf.variable_scope(name):
        # Load mean and variance
        scale_factor  = caffe_weights[name][2].data
        if scale_factor != 0:
            scale_factor = 1.0/scale_factor
        w_mean = caffe_weights[name][0].data*scale_factor
        w_variance = caffe_weights[name][1].data * scale_factor
        w_mean = tf.Variable(initial_value=w_mean, trainable=False)
        w_variance = tf.Variable(initial_value=w_variance, trainable=False)
        return tf.nn.batch_normalization(input,w_mean,w_variance,0,1,0, name=name)



def fast_portrait_matting(input_tensor):

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

    upsampling = deconv("upsampling",conv_6,[1,128,128,2],padding="SAME", add_bias=False, apply_relu=False)

    foreground, background = tf.split(upsampling,num_or_size_splits=2,axis=3,name="silcer_1")
    data_squared = tf.multiply(input_tensor,input_tensor,name="data_squared")

    concat_foreground = tf.concat([foreground,foreground,foreground,foreground],axis=concat_axis,name="concat_foreground")

    data_fg = tf.multiply(input_tensor,concat_foreground,name="data_fg")

    concat_feathering = tf.concat([input_tensor,background,foreground,data_squared,data_fg],concat_axis,name="concat_feathering")
    conv_feathering_1 = conv("conv_feathering_1",concat_feathering,[1,1,1,1],padding="SAME",
                             add_bias=True,apply_relu=False)

    bnorm_1 = batch_normalization("bnorm_1",conv_feathering_1)

    bnorm_1 = tf.nn.relu(bnorm_1,name="relu_feathering_1")

    conv_feathering_2 = conv("conv_feathering_2", bnorm_1, [1, 1, 1, 1], padding="SAME",
                             add_bias=True,apply_relu=False)

    a,b,c = tf.split(conv_feathering_2,num_or_size_splits=3,axis=concat_axis,name="slicer_2")
    a_mult_bg = tf.multiply(a,background,name="a_mult_bg")
    b_mult_fg = tf.multiply(b, foreground, name="b_mult_fg")
    alpha_pred =  tf.add_n([a_mult_bg, b_mult_fg, c], name="guided_filter")
    alpha_pred =  tf.nn.sigmoid(alpha_pred, name="sigmoid")

    return  alpha_pred

def load_caffe_weights(net,model):
    net = caffe.Net(net,model,caffe.TEST)
    global caffe_weights
    caffe_weights = net.params


def get_input_tensor():
    return tf.placeholder(dtype=tf.float32,shape=(1,128,128,4),name="input_tensor")




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
        saver.export_meta_graph("fastPortraitMatting.meta")







