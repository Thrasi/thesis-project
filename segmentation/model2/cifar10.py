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


FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 1,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('root_dir', '/home/magnus/thesis-project/segmentation',
                            """Root directory of the segmentation task""")
tf.app.flags.DEFINE_string('data_dir', os.path.join(FLAGS.root_dir,'data'),
                           """Path to the my data directory.""")
tf.app.flags.DEFINE_string('histograms', False, """Set to true to save histogram data.""")
CLASSES = ["bkg", "person"]

# Global constants describing the my data set.
NUM_CLASSES = cifar10_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
DECAY_RATE = 5e-4

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.000001       # Initial learning rate.
MOMENT_DECAY_RATE1 = 0.999
MOMENT_DECAY_RATE2 = 0.999

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

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

def accuracy(logits, labels, num_classes):
  collection = "metrics"
  def tf_count(t, val):
    elements_equal_to_value = tf.equal(t, val)
    as_ints = tf.cast(elements_equal_to_value, tf.int32)
    count = tf.reduce_sum(as_ints)
    return count

  labels = tf.cast(labels, tf.int64)
  reshaped_labels = tf.reshape(labels, [-1])

  reshaped_logits = tf.reshape(logits, (-1, num_classes))
  predictions = tf.argmax(reshaped_logits, dimension=1)
  shaped_predictions = tf.argmax(logits, dimension=3)
  correct_predictions = tf.equal(predictions, reshaped_labels)
  accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name='accuracy')
  tf.add_to_collection(collection, accuracy)
  
  if FLAGS.histograms:
    tf.histogram_summary('predictions_hist', predictions)
    
  imgs_to_summarize = tf.expand_dims(tf.cast(shaped_predictions, 'float32'), -1)
  tf.image_summary('predictions', imgs_to_summarize)

  cat_names = CLASSES
  precision = []
  cat_acc = []
  i_o_u = []
  for cat_id,cat in enumerate(cat_names):
    cat_pred = tf.equal(predictions, cat_id, name=cat+"_pred")
    cat_truth = tf.equal(reshaped_labels, cat_id, name=cat+"_truth")
    non_cat_truth = tf.not_equal(reshaped_labels, cat_id, name=cat+"_non_truth")
      
    tp = tf.logical_and(cat_pred, cat_truth, name=cat+"_tp")
    tp_count = tf.reduce_sum(tf.cast(tp, "float"), name=cat+"_tp_count")
    fp = tf.logical_and(cat_pred, non_cat_truth, name=cat+"_fp")
    fp_count = tf.reduce_sum(tf.cast(fp, "float"), name=cat+"_fp_count")
    
    prec = tf.div(tp_count, tp_count+fp_count+tf.constant(1e-10), name=cat+"_precision")
    
    precision.append( prec )
    tf.add_to_collection(collection, prec)
    
    cat_correct = tf.logical_and(cat_truth, cat_pred, name=cat+"_correct")
    cat_acc.append(tf.reduce_mean(tf.cast(cat_correct, "float"), name=cat+"_accuracy"))
    cat_sum = tf.reduce_sum(tf.cast(cat_correct, "float"))
    truth_sum = tf.reduce_sum(tf.cast(cat_truth, "float"))
    cat_accuracy = tf.div(cat_sum, truth_sum, name=cat+"_accuracy")
    tf.add_to_collection(collection, cat_accuracy)

    
    intersection = tp_count
    union = tf.reduce_sum(tf.cast(cat_pred, "float")) + tf.reduce_sum(tf.cast(cat_truth, "float")) - tp_count
    iou = tf.div(intersection, union, name=cat+"_iou")
    i_o_u.append(iou)
    tf.add_to_collection(collection, iou)
  
  precisions = tf.pack(precision)  
  accuracies = tf.pack(cat_acc)
  ious = tf.pack(i_o_u)
  tf.add_to_collection('precisions',precisions)

  return accuracy, precisions, accuracies, ious

def _add_metric_summaries():
  metric_averages = tf.train.ExponentialMovingAverage(0.9, name='metric_avg')
  metrics = tf.get_collection("metrics")
  metric_averages_op = metric_averages.apply(metrics)

  for m in metrics:
    tf.scalar_summary("metrics/"+m.op.name + ' (raw)', m)
    tf.scalar_summary("metrics/"+m.op.name, metric_averages.average(m))

  return metric_averages_op


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

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)
  metric_averages_op = _add_metric_summaries()
  # Compute gradients.
  with tf.control_dependencies([loss_averages_op, metric_averages_op]):
    # opt = tf.train.GradientDescentOptimizer(lr)
    opt = tf.train.AdamOptimizer(learning_rate=INITIAL_LEARNING_RATE,
                                       beta1=MOMENT_DECAY_RATE1,
                                       beta2=MOMENT_DECAY_RATE2,
                                       epsilon=1e-08,
                                       use_locking=False,
                                       name='Adam')
    grads = opt.compute_gradients(total_loss)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  if FLAGS.histograms:
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
