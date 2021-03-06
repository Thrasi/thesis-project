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
import os.path
import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import cifar10
import matplotlib.pyplot as plt

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', 
                            os.path.join(FLAGS.root_dir,'model1/train'),
                            """Directory where to write event logs """
                            """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 
                            1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', 
                            False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_string('checkpoint_dir',
                            os.path.join(FLAGS.root_dir,'model1/train'),
                            """Directory where to read model checkpoints.""")
CLASSES = ["bkg", "person", "chair", "car"]


def train():
  """Train a model for a number of steps."""
  with tf.Graph().as_default():
    global_step = tf.Variable(0, trainable=False)

    # Get images and labels for a segmentation model.
    images, labels, ground_truth = cifar10.distorted_inputs()
    tf.histogram_summary('label_hist/with_ignore', labels)
    tf.histogram_summary('label_hist/ground_truth', ground_truth)
    
    # Build a Graph that computes the logits predictions from the
    # inference model.
    print("before inference")
    print(images.get_shape())
    logits, nr_params = cifar10.inference(images)
    print("nr_params: "+str(nr_params) )
    print("after inference")
    # Calculate loss.
    loss = cifar10.loss(logits, labels)
    accuracy, precision, cat_accs = cifar10.accuracy(logits, ground_truth)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = cifar10.train(loss, global_step)

    # Create a saver.
    saver = tf.train.Saver(tf.all_variables())
#    tf.image_summary('images2', images)
    print (logits)
#    tf.image_summary('predictions', logits)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

    # Build an initialization operation to run below.
    init = tf.initialize_all_variables()

    # Start running operations on the Graph.
    sess = tf.Session(config=tf.ConfigProto(
        log_device_placement=FLAGS.log_device_placement))
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/cifar10_train/model.ckpt-0,
      # extract global_step from it.
      global_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
    else:
      print('No checkpoint file found')
      print('Initializing new model')
      sess.run(init)
      global_step = 0


    # Start the queue runners.
    tf.train.start_queue_runners(sess=sess)

    summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)

    for step in xrange(global_step, FLAGS.max_steps):
      start_time = time.time()
      _, loss_value, accuracy_value, precision_value, cat_accs_val  = sess.run([train_op,
                                                                                loss,
                                                                                accuracy,
                                                                                precision,
                                                                                cat_accs])
                                                                  
      duration = time.time() - start_time

      print (precision_value)
      print (cat_accs_val)
      
      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      #precision_value = [0 if np.isnan(p) else p for p in precision_value]
      #print (precision_value)
      if step % 10 == 0:
        num_examples_per_step = FLAGS.batch_size
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = float(duration)

        format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                      'sec/batch)\n Accuracy = %.4f, mean average precision = %.4f')
        print (format_str % (datetime.now(), step, loss_value,
                             examples_per_sec, sec_per_batch,
                             accuracy_value, np.mean(precision_value)))

      if step % 100 == 0:
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, step)

        summary = tf.Summary()
        summary.value.add(tag='Accuracy (raw)', simple_value=float(accuracy_value))
        for i,s in enumerate(CLASSES):
          summary.value.add(tag="precision/"+s+" (raw)",simple_value=float(precision_value[i]))
          summary.value.add(tag="accs/"+s+" (raw)",simple_value=float(cat_accs_val[i]))
#        summary.value.add(tag='Human precision (raw)', simple_value=float(precision_value))
        summary_writer.add_summary(summary, step)
        print("hundred steps")
      # Save the model checkpoint periodically.
      if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
        print("thousand steps")
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)


def main(argv=None):  # pylint: disable=unused-argument
  if argv[-1] == "clean":
    FLAGS.train_dir = os.path.join(FLAGS.train_dir, argv[-2])
    if tf.gfile.Exists(FLAGS.train_dir):
      print("cleaning: " + FLAGS.train_dir)
      tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
  else:
    FLAGS.train_dir = os.path.join(FLAGS.train_dir, argv[-1])
    print("Attempt to continue from pre-existing model")
  

  train()

if __name__ == '__main__':
  tf.app.run()
