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

"""Routine for decoding my data file for the coco dataset binary file format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

# Global constants describing the CIFAR-10 data set.
NUM_CLASSES = 2
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 45041
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 21579
PAD_WIDTH = 0
TOTAL_NUM_CHANNELS = 4
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('training_data', 'cocoFullImagetrain.tfrecords',
                           """Name of training data file """)

def read_cifar10(filename_queue):
  """Reads and parses examples from COCO data files.

  Recommendation: if you want N-way read parallelism, call this function
  N times.  This will give you N independent Readers reading different
  files & positions within those files, which will give better mixing of
  examples.

  Args:
    filename_queue: A queue of strings with the filenames to read from.

  Returns:
    An object representing a single example, with the following fields:
      mask: an int32 Tensor with the label in the range 0..4.
      image_raw: a [height, width, depth] uint8 Tensor with the image data
  """
  class COCORecord(object):
    pass
  result = COCORecord()

  print("inni i read and decode")
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(
    serialized_example,
    features={
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'image_and_mask': tf.FixedLenFeature([], tf.string),
    })

  result.height = tf.cast(features['height'], tf.int32)
  result.width = tf.cast(features['width'], tf.int32)
  shape = tf.pack([result.width,result.height, TOTAL_NUM_CHANNELS])
  result.image_and_mask = tf.decode_raw(features['image_and_mask'], tf.int16)
  result.image_and_mask = tf.reshape(result.image_and_mask, shape)

  return result


def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
  """Construct a queued batch of images and labels.

  Args:
    image: 3-D Tensor of [height, width, 3] of type.float32.
    label: 2-D Tensor of type.int32
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.
    shuffle: boolean indicating whether to use a shuffling queue.

  Returns:
    images: Images. 4D tensor of [batch_size, height, width, 3] size.
    labels: Labels. 3D tensor of [batch_size, height, width] size.
  """

  # Display the training images in the visualizer.
  print("image in gen_batch")
  print(image)
  image = tf.expand_dims(image, 0)
  label = tf.expand_dims(label, 0)
  tf.image_summary('images', image)
  tf.image_summary('truth', label)
  # tf.image_summary('truth_with_ignore', label_with_ignore_batch)

  return image, tf.cast(label, dtype=tf.int32)


def distorted_inputs(data_dir, batch_size):
  """Construct distorted input for COCO training using the Reader ops.

  Args:
    data_dir: Path to the COCO data directory.
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 3D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE] size.
  """

  filenames = [os.path.join(data_dir, FLAGS.training_data)]

  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

  # Create a queue that produces the filenames to read.
  filename_queue = tf.train.string_input_producer(filenames)

  # Read examples from files in the filename queue.
  read_input = read_cifar10(filename_queue)
  reshaped_image = tf.cast(read_input.image_and_mask, tf.float32)

  height = read_input.height
  width = read_input.width
  print(width)
  print(height)
  # Image processing for training the network. Note the many random
  # distortions applied to the image.

  # Randomly crop a [height, width] section of the image.

  distorted_image = reshaped_image
  # distorted_image = tf.random_crop(distorted_image, [height, width, TOTAL_NUM_CHANNELS])


  # Randomly flip the image horizontally.
  distorted_image = tf.image.random_flip_left_right(distorted_image)

  # Separate the image and mask.
  label = distorted_image[0:,0:,3:4]
  distorted_image = tf.pad(distorted_image[0:,0:,0:3],
                           [[PAD_WIDTH,PAD_WIDTH],[PAD_WIDTH,PAD_WIDTH],[0,0]], 
                           mode="SYMMETRIC", 
                           name="mirror_pad")
  print(distorted_image)
  distorted_image = distorted_image[0:,0:,0:3]

  # Because these operations are not commutative, consider randomizing
  # the order their operation.
  distorted_image = tf.image.random_brightness(distorted_image,
                                               max_delta=63)
  distorted_image = tf.image.random_contrast(distorted_image,
                                             lower=0.2, upper=1.8)

  # Subtract off the mean and divide by the variance of the pixels.
  float_image = tf.image.per_image_whitening(distorted_image)

  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                           min_fraction_of_examples_in_queue)
  print ('Filling queue with %d COCO images before starting to train. '
         'This will take a few minutes.' % min_queue_examples)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(float_image, label, 
                                         min_queue_examples, batch_size,
                                         shuffle=False)


def inputs(eval_data, data_dir, batch_size):
  """Construct input for CIFAR evaluation using the Reader ops.

  Args:
    eval_data: bool, indicating if one should use the train or eval data set.
    data_dir: Path to the CIFAR-10 data directory.
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 3D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE] size.
  """
  if not eval_data:
    filenames = [os.path.join(data_dir, "cocoFullImageval.tfrecords")]
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
  else:
    print (data_dir)
    filenames = [os.path.join(data_dir, "cocoFullImageval.tfrecords")]
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

  # Create a queue that produces the filenames to read.
  filename_queue = tf.train.string_input_producer(filenames)

  # Read examples from files in the filename queue.
  read_input = read_cifar10(filename_queue)

  reshaped_image = tf.cast(read_input.image_and_mask, tf.float32)

  height = read_input.height
  width = read_input.width

  ### Here we don't crop the center because the image is 64x64
  ### But if the image is larger we need to think if how to do this.
  ### Or if All since we use an FCN
  resized_image = reshaped_image
  label = resized_image[0:,0:,3:4]

  resized_image = resized_image[0:,0:,0:3]

  # Subtract off the mean and divide by the variance of the pixels.
  float_image = tf.image.per_image_whitening(resized_image)
  print (float_image)
  print (label)

  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(num_examples_per_epoch *
                           min_fraction_of_examples_in_queue)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(float_image, label,
                                         min_queue_examples, batch_size,
                                         shuffle=True)
