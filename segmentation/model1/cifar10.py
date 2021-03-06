# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Builds the CIFAR-10 network.

Summary of available functions:

 # Compute input images and labels for training. If you would like to run
 # evaluations, use inputs() instead.
 inputs, labels = distorted_inputs()

 # Compute inference on the model inputs to make a prediction.
 predictions = inference(inputs)

 # Compute the total loss of the prediction with respect to the labels.
 loss = loss(predictions, labels)

 # Create a graph to run one step of training with respect to the loss.
 train_op = train(loss, global_step)
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import re
import sys
import tarfile

from six.moves import urllib
import tensorflow as tf
import random

import cifar10_input
import helpers


FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('root_dir', '/home/magnus/thesis-project/segmentation',
                            """Root directory of the segmentation task""")
tf.app.flags.DEFINE_string('data_dir', os.path.join(FLAGS.root_dir,'data'),
                           """Path to the my data directory.""")
CLASSES = ["bkg", "person", "chair", "car"]

# Global constants describing the my data set.
IMAGE_SIZE = cifar10_input.IMAGE_SIZE
NUM_CLASSES = cifar10_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
DECAY_RATE = 5e-4

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.0005       # Initial learning rate.


# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'


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
  if wd is not None:
    weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var


def distorted_inputs():
  """Construct distorted input for COCO training using the Reader ops.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 3D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE] size.

  Raises:
    ValueError: If no data_dir
  """
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  # data_dir = os.path.join(FLAGS.data_dir, 'test_records.tfrecords')
  data_dir = FLAGS.data_dir
  return cifar10_input.distorted_inputs(data_dir=data_dir,
                                        batch_size=FLAGS.batch_size)


def inputs(eval_data):
  """Construct input for COCO evaluation using the Reader ops.

  Args:
    eval_data: bool, indicating if one should use the train or eval data set.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 3D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE] size.

  Raises:
    ValueError: If no data_dir
  """
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  # data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
  data_dir = FLAGS.data_dir
  return cifar10_input.inputs(eval_data=eval_data, data_dir=data_dir,
                              batch_size=FLAGS.batch_size)


def inference(images):
  """Build the CIFAR-10 model.

  Args:
    images: Images returned from distorted_inputs() or inputs().

  Returns:
    Logits.
  """
  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().
  #
  # conv1
  print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
  print(images.get_shape())
  nr_params = 0
  shape = images.get_shape().as_list()
  SIZE1 = tf.constant(shape[1:3])
  conv1, nrp = helpers.conv2d(images,
                         name="conv1",
                         kernel_width=5,
                         num_filters=64,
                         transfer=tf.nn.relu,
                         decay_rate=DECAY_RATE)
  print("conv1: "+str(conv1.get_shape()))
  nr_params += nrp
  # pool1
  pool1 = helpers.pool(conv1, name="pool1", kernel_width=2, stride=2)
  print("pool1: "+str(pool1.get_shape()))
  # norm1
  norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm1')

  shape = norm1.get_shape().as_list()
  SIZE2 = tf.constant(shape[1:3])
  # conv2
  conv2, nrp = helpers.conv2d(norm1,
                         name="conv2",
                         kernel_width=5,
                         num_filters=64,
                         transfer=tf.nn.relu,
                         decay_rate=DECAY_RATE)
  nr_params += nrp
  print("conv2: "+str(conv2.get_shape()))
  # norm2
  norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm2')
  pool2 = helpers.pool(norm2, name="pool2", kernel_width=3, stride=2)
  print("pool2: "+str(pool2.get_shape()))
  deconv_shape = pool2.get_shape()
  # conv3
  conv3, nrp = helpers.conv2d(pool2,
                        name="conv3",
                        kernel_width=6,
                        num_filters=128,
                        transfer=tf.nn.relu,
                        padding="VALID",
                        decay_rate=DECAY_RATE)
  print("conv3: "+str(conv3.get_shape()))
  nr_params += nrp
  # conv 4 1x1
  conv4, nrp = helpers.conv2d(conv3,
                        name="conv4-1x1",
                        kernel_width=1,
                        num_filters=128,
                        transfer=tf.nn.relu,
                        decay_rate=DECAY_RATE)
  print("conv4: "+str(conv4.get_shape()))
  nr_params += nrp
  # deconv
  deconv1, nrp = helpers.conv2d_transpose(conv4,
                          name="deconv1",
                          kernel_width=6,
                          num_filters=64,
                          transfer=tf.nn.relu,
                          padding="VALID",
                          output_shape=deconv_shape,
                          decay_rate=DECAY_RATE)
  print("deconv1: "+str(deconv1.get_shape()))
  nr_params += nrp
  unpool1 = tf.image.resize_nearest_neighbor(deconv1,
                                             SIZE2,
                                             align_corners=None,
                                             name="unpool1")
  print("unpool1: "+str(unpool1.get_shape()))
  # conv5
  conv5, nrp = helpers.conv2d(unpool1,
                        name="conv5",
                        kernel_width=5,
                        num_filters=64,
                        transfer=tf.nn.relu,
                        decay_rate=DECAY_RATE)
  print("conv5: "+str(conv5.get_shape()))
  nr_params += nrp
  unpool2 = tf.image.resize_nearest_neighbor(conv5,
                                              SIZE1,
                                              align_corners=None,
                                              name="unpool2")
  print("unpool2: "+str(unpool2.get_shape()))
  # conv6
  conv6, nrp = helpers.conv2d(unpool2,
                        name="conv6",
                        kernel_width=5,
                        num_filters=NUM_CLASSES,
                        transfer=tf.nn.relu,
                        decay_rate=DECAY_RATE)
  print("conv6  : "+str(conv6.get_shape()))
  nr_params += nrp
  
  
  return conv6, nr_params


def loss(logits, labels):
  """Add L2Loss to all the trainable variables.

  Add summary for "Loss" and "Loss/avg".
  Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 3-D tensor
            of shape [batch_size,IMAGE_SIZE,IMAGE_SIZE]

  Returns:
    Loss tensor of type float.
  """
  labels = tf.cast(labels, tf.int64)
  label_shape = labels.get_shape().as_list()
  reshaped_labels = tf.reshape(labels,
                              [label_shape[0]*label_shape[1]*label_shape[2]])
  print(reshaped_labels.get_shape())
  logits_shape =logits.get_shape().as_list()
  reshaped_logits = tf.reshape(logits,
                              [logits_shape[0]*logits_shape[1]*logits_shape[2],
                              logits_shape[3]]) 
  cross_entropy_per_pixel = tf.nn.sparse_softmax_cross_entropy_with_logits(
                                  reshaped_logits, reshaped_labels,
                                  name='cross_entropy_per_pixel')
  no_loss_mask = tf.not_equal(reshaped_labels, -1)

  filtered_cross_entropy = tf.boolean_mask(cross_entropy_per_pixel,
                                           no_loss_mask,
                                           name='no_loss_mask')
  cross_entropy_mean = tf.reduce_mean(filtered_cross_entropy, name='cross_entropy')
#  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)

  return tf.add_n(tf.get_collection('losses'), name='total_loss')

def accuracy(logits, labels):
  def tf_count(t, val):
    elements_equal_to_value = tf.equal(t, val)
    as_ints = tf.cast(elements_equal_to_value, tf.int32)
    count = tf.reduce_sum(as_ints)
    return count
  
  labels = tf.cast(labels, tf.int64)
  label_shape = labels.get_shape().as_list()
  reshaped_labels = tf.reshape(labels,
                              [label_shape[0]*label_shape[1]*label_shape[2]])

  logits_shape = logits.get_shape().as_list()
  reshaped_logits = tf.reshape(logits,
                              [logits_shape[0]*logits_shape[1]*logits_shape[2],
                              logits_shape[3]])

  predictions = tf.argmax(reshaped_logits, dimension=1)
  shaped_predictions = tf.argmax(logits, dimension=3)
  correct_predictions = tf.equal(predictions, reshaped_labels)
  accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name='accuracy')
  tf.add_to_collection('accuracy', accuracy)
  tf.histogram_summary('predictions_hist', predictions)
  imgs_to_summarize = tf.expand_dims(tf.cast(shaped_predictions, 'float32'), -1)
  tf.image_summary('predictions', imgs_to_summarize)

  cat_names = CLASSES
  precision = []
  cat_acc = []
  for cat_id,cat in enumerate(cat_names):
    cat_pred = tf.equal(predictions, cat_id, name=cat+"_pred")
    cat_truth = tf.equal(reshaped_labels, cat_id, name=cat+"_truth")
    non_cat_truth = tf.not_equal(reshaped_labels, cat_id, name=cat+"_non_truth")
      
    tp = tf.logical_and(cat_pred, cat_truth, name=cat+"_tp")
    tp_count = tf.reduce_sum(tf.cast(tp, "float"), name=cat+"_tp_count")
    fp = tf.logical_and(cat_pred, non_cat_truth, name=cat+"_fp")
    fp_count = tf.reduce_sum(tf.cast(fp, "float"), name=cat+"_fp_count")

    tf.scalar_summary('cat_precisions/'+cat+'_fp_count', fp_count)
    tf.scalar_summary('cat_precisions/'+cat+'_tp_count', tp_count)
  
    precision.append( tp_count / (tp_count + fp_count) )

    cat_correct = tf.logical_and(cat_truth, cat_pred, name=cat+"_correct")
    cat_acc.append(tf.reduce_mean(tf.cast(cat_correct, "float"), name=cat+"_accuracy"))
  
  precisions = tf.pack(precision)  
  accuracies = tf.pack(cat_acc)
  tf.add_to_collection('precisions',precisions)

  return accuracy, precisions, accuracies

def _add_accuracy_precision_summaries(accuracy, precision):
  accuracy_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  accuracies = tf.get_collection('accuracy')
  accuracy_average_op = accuracy_averages.apply(accuracies + [accuracy])

  for a in accuracies + [accuracy]:
    tf.scalar_summary(a.op.name +' (raw)', a)
    tf.scalar_summary(a.op.name, accuracy_averages.average(a))

  precision_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  precisions = tf.get_collection('precision')
  precision_average_op = accuracy_averages.apply(precisions + [precision])  

  for p in precisions + [precisions]:
    tf.scalar_summary("human_precision (raw)", p)
    tf.scalar_summary("human_precision", precision_averages.average(p))
  return accuracy_average_op, precision_average_op


def _add_loss_summaries(total_loss):
  """Add summaries for losses in CIFAR-10 model.

  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.

  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.scalar_summary(l.op.name +' (raw)', l)
    tf.scalar_summary(l.op.name, loss_averages.average(l))

  return loss_averages_op


def train(total_loss, global_step):
  """Train CIFAR-10 model.

  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.

  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """
  # Variables that affect learning rate.
  num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

  # Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
  tf.scalar_summary('learning_rate', lr)

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)

  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
    # opt = tf.train.GradientDescentOptimizer(lr)
    opt = tf.train.AdamOptimizer(learning_rate=0.0001,
                                       beta1=0.9,
                                       beta2=0.999,
                                       epsilon=1e-08,
                                       use_locking=False,
                                       name='Adam')#.minimize(loss,global_step=batch)
    grads = opt.compute_gradients(total_loss)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.histogram_summary(var.op.name, var)

  # Add histograms for gradients.
  for grad, var in grads:
    if grad is not None:
      tf.histogram_summary(var.op.name + '/gradients', grad)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')

  return train_op


def maybe_download_and_extract():
  """Download and extract the tarball from Alex's website."""
  dest_directory = FLAGS.data_dir
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
          float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)
