import os
import argparse
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.framework import graph_util
import ipdb

#python freeze_graph.py /home/or/caffe-BGS-win/or/fastPortraitMatting/results/Resu2save/toTF/95.91_128/TF_95.905/fastPortraitMatting --output_layer 'alpha_pred'

def load_graph_def(model_path, sess=None):
    if os.path.isfile(model_path):
        with gfile.FastGFile(model_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
    else:
        saver = tf.train.import_meta_graph(model_path + '.meta')
        sess = sess if sess is not None else tf.get_default_session()
        saver.restore(sess, model_path)


def freeze_from_checkpoint(checkpoint_file, output_layer_name):

    print(tf.__version__)
    ipdb.set_trace()
    model_folder = os.path.basename(checkpoint_file)
    output_graph = os.path.join(model_folder, checkpoint_file + '.pb')

    with tf.Session() as sess:

        load_graph_def(checkpoint_file)

        graph = tf.get_default_graph()
        input_graph_def = graph.as_graph_def()

        # Exporting the graph
        print("Exporting graph...")
        output_graph_def = graph_util.convert_variables_to_constants(
            sess,
            input_graph_def,
            output_layer_name.split(","))

        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path')
    parser.add_argument('--output_layer', default='embeddings')
    args = parser.parse_args()
    freeze_from_checkpoint(checkpoint_file=args.model_path,
                           output_layer_name=args.output_layer)
