from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import re
import sys
import tarfile
import math

from six.moves import urllib
import tensorflow as tf
TOWER_NAME = 'tower'
def conv2d(input, name, kernel_width, num_filters, transfer=tf.nn.elu, padding='SAME', decay_rate=0):
  c = input.get_shape()[3].value
  n = c * (kernel_width ** 2)
  with tf.variable_scope(name) as scope:
    kernel = _variable_with_weight_decay('weights', shape=[kernel_width, kernel_width,
                                                           c, num_filters],
                                         stddev=math.sqrt(2.0 / n), wd=decay_rate)
    conv = tf.nn.conv2d(input, kernel, [1, 1, 1, 1], padding=padding)
    biases = _variable_on_cpu('biases', [num_filters], tf.constant_initializer(0.1))
    bias = tf.nn.bias_add(conv, biases)
    x = transfer(bias, name=scope.name)
    _activation_summary(x)
    
  return x, kernel_width*kernel_width*c*num_filters+num_filters

def upscore_layer(bottom, shape,
                       num_classes, name, debug,
                       ksize=4, stride=2):
        strides = [1, stride, stride, 1]
        with tf.variable_scope(name):
            in_features = bottom.get_shape()[3].value

            # if shape is None:
            # Compute shape out of Bottom
            in_shape = tf.shape(bottom)

            h = ((in_shape[1] - 1) * stride) + 1
            w = ((in_shape[2] - 1) * stride) + 1
            new_shape = [in_shape[0], h, w, num_classes]
            # else:
            #     new_shape = [shape[0], shape[1], shape[2], num_classes]
            print ("new_shape")
            print (new_shape)
            output_shape = tf.pack(new_shape)
            print("output_shape")
            print (output_shape)
            # logging.debug("Layer: %s, Fan-in: %d" % (name, in_features))
            f_shape = [ksize, ksize, num_classes, in_features]
            print ("f_shape")
            print (f_shape)
            # create
            num_input = ksize * ksize * in_features / stride
            stddev = (2 / num_input)**0.5

            # weights = self.get_deconv_filter(f_shape)
            weights = _variable_with_weight_decay('weights',
                                         shape=f_shape,
                                         stddev=stddev,
                                         wd=0)
            print ("weights")
            print(weights)
            deconv = tf.nn.conv2d_transpose(bottom, weights, output_shape,
                                            strides=strides, padding='VALID')
        return deconv

def conv2d_transpose(input, name, kernel_width, num_filters,
                     transfer=tf.nn.elu, padding='VALID', 
                     decay_rate=0, output_shape=None, 
                     strides=[1, 1, 1, 1]):
  print("transpose")
  print(input)
  c = input.get_shape()[3].value
  print (c)
  n = c * (kernel_width ** 2)
  print(n)
  print(output_shape)
  with tf.variable_scope(name) as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[kernel_width,
                                                kernel_width,
                                                num_filters,
                                                c],
                                         stddev=math.sqrt(2.0 / n),
                                         wd=decay_rate)
    print (kernel)
    conv = tf.nn.conv2d_transpose(input,
                                  kernel,
                                  output_shape,
                                  strides=strides,
                                  padding=padding)
    print (conv)
    b = conv.get_shape().as_list()[-1]
    print (b)
    biases = _variable_on_cpu('biases', [b], tf.constant_initializer(0.1))

    print("conv: "+str(conv.get_shape()))
    print("biases: "+str(biases.get_shape()))
    bias = tf.nn.bias_add(conv, biases)
    x = transfer(bias, name=scope.name)
    _activation_summary(x)
  return x, kernel_width*kernel_width*c*num_filters + b

def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.
 
  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
 
  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer)
  return var
 
 
def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.
 
  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.
 
  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
 
  Returns:
    Variable Tensor
  """
  var = _variable_on_cpu(name, shape,
                         tf.truncated_normal_initializer(stddev=stddev))
  # print (var)
  if wd:
    weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var

def residual(input, name, num_conv):
  x = input
  c = input.get_shape()[3].value
  with tf.variable_scope(name) as scope:
    for i in range(num_conv):
      x = conv2d(x, "conv%d" % (i), 3, c)
      x = batch_norm_conv(x, "bn%d" % (i))
      if i != num_conv - 1:
        x = tf.nn.elu(x)
    return tf.nn.elu(x + input)
 
def batch_norm_conv(input, name):
  with tf.variable_scope(name) as scope:
    mean, variance = tf.nn.moments(input, [0, 1, 2])
    return tf.nn.batch_normalization(input, mean, variance, None, None, 1e-6)
 
def batch_norm_fc(input, name):
  with tf.variable_scope(name) as scope:
    mean, variance = tf.nn.moments(input, [0])
    return tf.nn.batch_normalization(input, mean, variance, None, None, 1e-6)


def _activation_summary(x):
  """Helper to create summaries for activations.
  Creates a summary that provides a histogram of activations.
  Creates a summary that measure the sparsity of activations.
  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.histogram_summary(tensor_name + '/activations', x)
  tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def pool(input, name, kernel_width, stride):
  with tf.variable_scope(name) as scope:
    kernel = [1, kernel_width, kernel_width, 1]
    strides = [1, stride, stride, 1]
    return tf.nn.max_pool(input, 
                          ksize=kernel, 
                          strides=strides, 
                          padding='SAME', 
                          name=scope.name)
