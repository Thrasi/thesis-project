import tensorflow as tf
import matplotlib.pyplot as plt

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
        'image_raw': tf.FixedLenFeature([], tf.string),
        'mask': tf.FixedLenFeature([], tf.string),
    })

  result.image = tf.decode_raw(features['image_raw'], tf.uint8)
  result.image = tf.reshape(result.image, [64,64,3])

  result.mask = tf.decode_raw(features['mask'], tf.uint8)
  result.mask = tf.reshape(result.mask, [64,64])
  return result

filenames = ["/home/mb/Documents/kth/thesis-project/segmentation/data/coco64by64.tfrecords"]
filename_queue = tf.train.string_input_producer(filenames)

result = read_and_decode(filename_queue)
init_op = tf.initialize_all_variables()

sess = tf.Session()

sess.run(init_op)
tf.train.start_queue_runners(sess=sess)
for i in range(200):
  image, mask = sess.run([result.image, result.mask])
  plt.subplot(121)
  plt.imshow(image)
  plt.subplot(122)
  plt.imshow(mask)
  plt.show()
