import inspect
import os

import numpy as np
import tensorflow as tf
import time
import helpers


VGG_MEAN = [103.939, 116.779, 123.68]
NUM_CLASSES = 2


class Vgg16:
    def __init__(self, vgg16_npy_path=None):
        if vgg16_npy_path is None:
            path = inspect.getfile(Vgg16)
            path = os.path.abspath(os.path.join(path, os.pardir))
            path = os.path.join(path, "vgg16.npy")
            vgg16_npy_path = path
            print path

        self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
        print("npy file loaded")

    def build(self, rgb):
        """
        load variable from npy to build the VGG

        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        """

        start_time = time.time()
        print("build model started")
        print(rgb)
        rgb_scaled = rgb * 255.0

        # Convert RGB to BGR
        red, green, blue = tf.split(3, 3, rgb_scaled)
        # assert red.get_shape().as_list()[1:] == [224, 224, 1]
        # assert green.get_shape().as_list()[1:] == [224, 224, 1]
        # assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        bgr = tf.concat(3, [
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])
        # assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

        # self.conv1_1 = self.conv_layer(bgr, "conv1_1")
        self.conv1_1 = self.conv2d(bgr, "conv1_1")
        print self.conv1_1
        # self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
        self.conv1_2 = self.conv2d(self.conv1_1, "conv1_2")
        print self.conv1_2
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')
        print self.pool1

        # self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.conv2_1 = self.conv2d(self.pool1, "conv2_1")
        print self.conv2_1
        # self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
        self.conv2_2 = self.conv2d(self.conv2_1, "conv2_2")
        print self.conv2_2
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')
        print self.pool2

        # self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.conv3_1 = self.conv2d(self.pool2, "conv3_1")
        print self.conv3_1
        # self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        self.conv3_2 = self.conv2d(self.conv3_1, "conv3_2")
        print self.conv3_2
        # self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
        self.conv3_3 = self.conv2d(self.conv3_2, "conv3_3")
        print self.conv3_3
        self.pool3 = self.max_pool(self.conv3_3, 'pool3')
        print self.pool3

        # self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.conv4_1 = self.conv2d(self.pool3, "conv4_1")
        print self.conv4_1
        # self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
        self.conv4_2 = self.conv2d(self.conv4_1, "conv4_2")
        print self.conv4_2
        # self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
        self.conv4_3 = self.conv2d(self.conv4_2, "conv4_3")
        print self.conv4_3
        self.pool4 = self.max_pool(self.conv4_3, 'pool4')
        print self.pool4

        # self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
        self.conv5_1 = self.conv2d(self.pool4, "conv5_1")
        print self.conv5_1
        # self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
        self.conv5_2 = self.conv2d(self.conv5_1, "conv5_2")
        print self.conv5_2
        # self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
        self.conv5_3 = self.conv2d(self.conv5_2, "conv5_3")
        print self.conv5_3
        self.pool5 = self.max_pool(self.conv5_3, 'pool5')
        print self.pool5
        # self.fc6 = self.conv_layer(self.pool5, 'fc6', shape=[7,7,512,4096], padding='VALID')
        self.fc6 = self.conv2d(self.pool5, "fc6", shape=[7,7,512,4096], padding='VALID')
        print self.fc6
        # self.fc7 = self.conv_layer(self.fc6, 'fc7', shape=[1,1,4096,4096], padding='VALID')
        self.fc7 = self.conv2d(self.fc6, 'fc7', shape=[1,1,4096,4096], padding='VALID')
        print self.fc7

        # self.fc8 = self.conv_layer(self.fc7, "fc8", shape=[1,1,4096,2], padding='VALID')
        self.fc8, nr_params = helpers.conv2d(self.fc7,name="fc8",kernel_width=1,transfer=tf.nn.relu, padding="VALID",num_filters=NUM_CLASSES)
        # print self.fc8
        # shape = self.fc8.get_shape().as_list()
        # self.logits = tf.reshape(self.fc8, [shape[0]*shape[1]*shape[2], 2])
        # self.prob = tf.nn.softmax(self.logits, name="prob")
        IMAGE_SIZE = rgb.get_shape().as_list()[1:3]
        self.resized = tf.image.resize_bilinear(self.fc8, IMAGE_SIZE, align_corners=None, name="upsample")
        print self.resized
        self.data_dict = None

        print (self.fc8)


        print("build model finished: %ds" % (time.time() - start_time))

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name, shape=None, padding='SAME'):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)
            if shape:
                filt = tf.reshape(filt, shape)
            # print filt

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding=padding)

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    def conv2d(self, bottom, name, shape=None, padding='SAME', decay_rate=None):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)
            if shape:
                filt = tf.reshape(filt, shape)
            # print filt
            if decay_rate:
                weight_decay = tf.mul(tf.nn.l2_loss(filt), decay_rate, name='weight_loss')
                tf.add_to_collection('losses', weight_decay)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding=padding)

            with tf.device('/cpu:0'):
                conv_biases = self.get_bias(name)
                # var = tf.get_variable(name, shape, initializer=initializer)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    def fc_layer(self, bottom, name):
        with tf.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])

            weights = self.get_fc_weight(name)
            biases = self.get_bias(name)

            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_filter(self, name):
        return tf.Variable(self.data_dict[name][0], name="filter")

    def get_bias(self, name):
        return tf.Variable(self.data_dict[name][1], name="biases")

    def get_fc_weight(self, name):
        return tf.Variable(self.data_dict[name][0], name="weights")
