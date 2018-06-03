from google.protobuf import text_format
from caffe.proto import caffe_pb2
import caffe
import os
import ipdb

def convert_train_to_deploy(net_file, prefix_out_path=None):
    old_net = caffe_pb2.NetParameter()
    new_net = caffe_pb2.NetParameter()
    text_format.Merge(open(net_file).read(),old_net)
    text_format.Merge(open(net_file).read(),new_net)
    while len(new_net.input) > 1:
        new_net.input.pop()
    del new_net.input_dim[4:]

    for i, layer in enumerate(old_net.layer):

        #remove all loss and accuracy
        if u'Loss' in layer.type:
            new_net.layer.remove(layer)
            continue
        if u'Dropout' == layer.type:
            new_net.layer.remove(layer)
            continue
        if u'Accuracy' in layer.type:
            new_net.layer.remove(layer)
            continue
        if u'accuracy' in layer.top[0]:
            new_net.layer.remove(layer)
            continue
        if u'alpha_pred_s' in layer.bottom[0]:
            new_net.layer.remove(layer)
            continue

    for i, layer in enumerate(new_net.layer):
        del layer.param[:]
        if layer.type == u'Convolution':
            layer.convolution_param.ClearField('bias_filler')
            layer.convolution_param.ClearField('weight_filler')
            pass
        #make sigmoid inplace
        if layer.type == u'Sigmoid':
            layer.top[0] = layer.bottom[0]

    net_name = new_net.name.replace(' ','_').encode('ascii','ignore')
    net_name = 'deploy_{}.prototxt'.format(net_name)
    full_path_file = os.path.join(prefix_out_path,net_name)
    with open(full_path_file,'w') as deploy:
        text_format.PrintMessage(new_net,deploy)
    return net_name




