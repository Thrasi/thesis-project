import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import colors

def read_and_decode(filename_queue):
  class COCORecord(object):
    pass
  result = COCORecord()

  print "inni i read and decode"
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(
    serialized_example,
    features={
        # 'image_raw': tf.FixedLenFeature([], tf.string),
        # 'mask': tf.FixedLenFeature([], tf.string),
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'image_and_mask': tf.FixedLenFeature([], tf.string),
    })
  result.height = tf.cast(features['height'], tf.int32)
  result.width = tf.cast(features['width'], tf.int32)
  # width = result.width

  # result.width  =64
  # result.height=64
  shape = tf.pack([result.width,result.height,5])
  # print shape  
  result.image_and_mask = tf.decode_raw(features['image_and_mask'], tf.int16)
  result.image_and_mask = tf.reshape(result.image_and_mask,
                                     shape)

  # a = result.image_and_mask[:,:,:3]
  # print a
  # print type(result.image_and_mask)

  # result.image = tf.decode_raw(features['image_raw'], tf.uint8)
  # result.image = tf.reshape(result.image, [64,64,3])

  # result.mask = tf.decode_raw(features['mask'], tf.uint8)
  # result.mask = tf.reshape(result.mask, [64,64])
  return result

filenames = ["/home/mb/Documents/kth/thesis-project/segmentation/coco64by64train.tfrecords"]
filename_queue = tf.train.string_input_producer(filenames)
result = read_and_decode(filename_queue)
init_op = tf.initialize_all_variables()

sess = tf.Session()

sess.run(init_op)
tf.train.start_queue_runners(sess=sess)
for i in range(200):
  # image_and_mask = sess.run([result.image_and_mask])
  # image = image_and_mask[0][:,:,:3]
  # mask = image_and_mask[0][:,:,3]
  # [image, mask, image_and_mask] = sess.run([result.image, result.mask, result.image_and_mask])
  [image_and_mask] = sess.run([result.image_and_mask])
  # plt.subplot(221)
  # plt.imshow(image)
  # plt.subplot(222)
  # plt.imshow(mask)
  plt.subplot(131)
  plt.imshow(image_and_mask[:,:,:3]/255.)
  plt.subplot(132)
  cmap = colors.ListedColormap(['black', 'blue', 'red', 'green', 'cyan', 'yellow'])
  bounds=[-2,-0.5,0.5,1.5,2.5,3.5,1000]
  norm = colors.BoundaryNorm(bounds, cmap.N)
        # plt.imshow(image_with_ignore, cmap=cmap, norm=norm)
  plt.imshow(image_and_mask[:,:,3], cmap=cmap, norm=norm)

  plt.subplot(133)
  plt.imshow(image_and_mask[:,:,4], cmap=cmap, norm=norm)

  plt.show()
