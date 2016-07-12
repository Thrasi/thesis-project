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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time
import os

import numpy as np
import tensorflow as tf

import cifar10

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', os.path.join(FLAGS.root_dir,'model1/eval'),
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', os.path.join(FLAGS.root_dir,'model1/train'),
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60*10 ,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 10000,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                         """Whether to run eval only once.""")
CLASSES = ["bkg", "person", "chair", "car"]


def eval_once(saver, summary_writer, summary_op, accuracy, precision, accuracies):
  """Run Eval once.

  Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
    summary_op: Summary op.
  """
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/cifar10_train/model.ckpt-0,
      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
      print('No checkpoint file found')
      return
    print("passed checkpoint check")
    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))
      print("passed for loop")
      num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
      true_count = 0  # Counts the number of correct predictions.
      total_sample_count = num_iter * FLAGS.batch_size
      step = 0
      acc = 0
      prec = [0 for i in range(len(CLASSES))]
      cat_acc = [0 for i in range(len(CLASSES))]
      print ("while loop")
      while step < num_iter and not coord.should_stop():
        acc_val, prec_val, cat_acc_val = sess.run([accuracy, precision, accuracies])
        #true_count += np.sum(predictions)
        acc += acc_val
        for i in xrange(len(prec)):
          prec[i] += prec_val[i]
          cat_acc[i] += cat_acc_val[i]
#        prec += hp_val
        step += 1
      print("passed while loop")
      # Compute precision @ 1.
      prec = [ p / float(step) for p in prec ]
      cat_acc = [ a / float(step) for a in cat_acc ]
      acc = acc / float(step)
      #precision = true_count / total_sample_count
      print('%s: precision @ 1 = %.3f\n    accuracy = %.3f\n ' % (datetime.now(), 0, acc))
      print (prec)
      print(cat_acc)

      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
#      summary.value.add(tag='Precision @ 1', simple_value=precision)
      summary.value.add(tag='Accuracy', simple_value=acc)
      for i,s in enumerate(CLASSES):
        summary.value.add(tag="precision/"+s+" (raw)",simple_value=float(prec[i]))
        summary.value.add(tag="accuracies/"+s+" (raw)",simple_value=float(cat_acc[i]))
#      summary.value.add(tag='Human precision', simple_value=prec)

      summary_writer.add_summary(summary, global_step)
    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)
    print("done evaluating")
    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)


def evaluate():
  """Eval CIFAR-10 for a number of steps."""
  with tf.Graph().as_default() as g:
    # Get images and labels for CIFAR-10.
    eval_data = FLAGS.eval_data == 'test'
    print (eval_data)
    images, labels, ground_truth = cifar10.inputs(eval_data=eval_data)
    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits, _ = cifar10.inference(images)
    print(logits)
    print(logits.get_shape())
    print("after inference node creation")
    loss = cifar10.loss(logits, labels)
    accuracy, precision, accuracies = cifar10.accuracy(logits, ground_truth)
    labels = tf.cast(labels, tf.int64)

    label_shape = labels.get_shape().as_list()
    reshaped_labels = tf.reshape(labels,
                                [label_shape[0]*label_shape[1]*label_shape[2]])
    logits_shape =logits.get_shape().as_list()
    reshaped_logits = tf.reshape(logits,
                                [logits_shape[0]*logits_shape[1]*logits_shape[2],
                                logits_shape[3]]) 

    # Calculate predictions.
    # top_k_op = tf.nn.in_top_k(logits, labels, 1)
    #top_k_op = tf.nn.in_top_k(reshaped_logits, reshaped_labels, 1)

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        cifar10.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

    summary_writer = tf.train.SummaryWriter(FLAGS.eval_dir, g)

    while True:
      print("evaluate:")
      eval_once(saver, summary_writer, summary_op, accuracy, precision, accuracies)
      if FLAGS.run_once:
        break
      time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):  # pylint: disable=unused-argument
  if argv[-1] == "clean":
    FLAGS.eval_dir = os.path.join(FLAGS.eval_dir, argv[-2])
    FLAGS.checkpoint_dir = os.path.join(FLAGS.checkpoint_dir, argv[-2])
    if tf.gfile.Exists(FLAGS.eval_dir):
      print ("cleaning: "+ FLAGS.eval_dir)
      tf.gfile.DeleteRecursively(FLAGS.eval_dir)
    tf.gfile.MakeDirs(FLAGS.eval_dir)
  else:
    FLAGS.eval_dir = os.path.join(FLAGS.eval_dir, argv[-1])
    FLAGS.checkpoint_dir = os.path.join(FLAGS.checkpoint_dir, argv[-1])
    print("Attempt to continue from prexistion evaluation")
  evaluate()

if __name__ == '__main__':
  tf.app.run()
