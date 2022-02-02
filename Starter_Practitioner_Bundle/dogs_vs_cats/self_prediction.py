import os
import numpy as np
import tensorflow as tf
import model_new
from PIL import Image
import matplotlib.pyplot as plt
import csv
import shutil
from tensorflow.python.platform import gfile


def get_one_image(img_dir):
    image = Image.open(img_dir)

    image = image.resize((128, 128))
    image = np.array(image)

    return image, img_dir


def test_model(model_path, img_path):
    image_array, img_dir = get_one_image(img_path)
    image = tf.cast(image_array, tf.float32)
    # image = tf.image.per_image_standardization(image)
    image = tf.reshape(image, [1, 128, 128, 3])

    with tf.Session() as sess:
        with gfile.FastGFile(model_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')
        sess.run(tf.global_variables_initializer())

        input_x = sess.graph.get_tensor_by_name('Cast_1:0')
        out = sess.graph.get_tensor_by_name('softmax_linear/softmax_linear:0')
        ret = sess.run(out, feed_dict={input_x: image.eval()})
        print(ret)


test_model(out_pb_path, img_path)