import sys
import os
import numpy as np
import scipy as sp
import scipy.ndimage
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from pycocotools.coco import mask, COCO
import skimage.io as io
from skimage.draw import polygon
import pylab
import random

FIELD = 64
L_SCALE = 0.5
U_SCALE = 2

def get_expanded_bbox(bbox, img):
  """Returns a bounding box expanded by a margin"""

  center_column = bbox[0] + bbox[2]/2
  center_row = bbox[1] + bbox[3]/2
  col_margin = int((1.2 * bbox[2]) / 2)
  row_margin = int((1.2 * bbox[2]) / 2)
  box = bbox[:]

  box[0] = max(0, bbox[0] - col_margin)
  box[1] = max(0, bbox[1] - row_margin)
  if bbox[0]+bbox[2]+col_margin > img['width']:
    box[2] = img['width'] - box[0]
  else:
    box[2] = bbox[0]+bbox[2]+col_margin - box[0]
  if bbox[1]+bbox[3]+row_margin > img['height']:
    box[3] = img['height'] - box[1]
  else:
    box[3] = bbox[1]+bbox[3]+row_margin - box[1]
  return box

def get_range(start, length):
  if length >= FIELD:
    lower = int(start)
    upper = int(start+length-FIELD)
  else:
    lower = int(start+length-FIELD)
    upper = int(start)
    lower = max(0, lower)
  return lower, upper

def get_ranges(bbox):
  """
  """
  col_min, col_max = get_range(bbox[0], bbox[2])
  row_min, row_max = get_range(bbox[1], bbox[3])
  return row_min, row_max, col_min,  col_max

def sample_coords(bbox, img):
  """Returns random coordinates of the upper left corner
   of crop of size FIELD within an expanded bounding box"""
#   # print "box: "+ str(box)
#   print "Old box: {}".format(bbox)
#   box = get_expanded_bbox(bbox, img)
#   print "New box: {}".format(box)
  box = bbox[:]
  box = get_expanded_bbox(bbox, img)
  row_min, row_max, col_min, col_max = get_ranges(box)
  row = random.randint(row_min, row_max)
  col = random.randint(col_min, col_max)
  return row, col


def get_num_of_crops(width, height, FIELD):
  """Returns a number based on how many times FIELD fits
  into the bounding box"""
  nrx = 1 + int(width)/FIELD
  nry = 1 + int(height)/FIELD
  return nrx*nry

# def annsToMasks(anns):
#   if len(anns) == 0:
#     return 0
#   polygons = []
#   masks = []
#   color =[]
#   for ann in anns:
#     c = ann['category_id']
#     if type(ann['segmentation']) == list:
#       for seg in ann['segmentation']:
#         poly = np.array(seg).reshape((len(seg)/2, 2))
#         polygons.append(Polygon(poly, True,alpha=0.4))
#         # t = self.imgs[ann['image_id']]
#         t = coco.loadImgs(ann['image_id'])
#         rle = mask.frPyObjects([seg], t['height'], t['width'])
#         m = mask.decode(rle)
#         masks.append(m*c)
#     else:
#       t = self.imgs[ann['image_id']]
#       if type(ann['segmentation']['counts']) == list:
#         rle = mask.frPyObjects([ann['segmentation']], t['height'], t['width'])
#       else:
#         rle = [ann['segmentation']]
#       m = mask.decode(rle)
#       masks.append(m*c)
#   p = PatchCollection(polygons, facecolors=color, edgecolors=(0,0,0,1), linewidths=3, alpha=0.4)
#   return masks

def get_crops(img, annotations, image, ground_truth):
  """Returns a list of tuples containing a crop from the 
  image and the ground truth of that crop"""
  count = 0
  crops = []
  # plt.subplot(5,5,1)
  # plt.imshow(image)
  # plt.subplot(5,5,2)
  # plt.imshow(ground_truth)
  index = 3
  for annotation in annotations:
    bbox = annotation['bbox']
    number_of_crops = get_num_of_crops(bbox[2], bbox[3], FIELD)
    LOWER = int(FIELD*L_SCALE)
    UPPER = FIELD*U_SCALE
    if LOWER < bbox[2] < UPPER and LOWER < bbox[3] < UPPER:
      # if index <= 25:
      #   plt.subplot(5,5,index)
      index+=1
      bbox=map(int,bbox)
      # if index <= 25:
      #   # print image.shape
      #   plt.imshow(image[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2],:])
      count+=1
      # print number_of_crops
      for c in xrange(number_of_crops):
        col, row = sample_coords(bbox, img)
        img_crop = image[col:col+FIELD, row:row+FIELD, :]
        mask_crop = ground_truth[col:col+FIELD, row:row+FIELD]
        masked_pixels = np.sum(mask_crop > 0)
        size = mask_crop.shape[0]*mask_crop.shape[1]
        if 0.2 < masked_pixels/float(size) < 0.8:
          crops.append([img_crop, mask_crop])
          

        # if index <= 24:
        #   plt.subplot(5,5, index)
        #   index+=1
        #   plt.imshow(img_crop)
        #   plt.subplot(5,5, index)
        #   index+=1
        #   plt.imshow(mask_crop)
  # plt.show()
  return count, crops

def label_mask_categories(masks, categoryIds):
  for i,c in enumerate(categoryIds):
    for mask in masks:
      if mask.max() == c:
        mask[mask==c] = i+1
  return masks

def get_ground_truth_mask(coco, annotations, categoryIds):
  annotations = [a for a in annotations if a['category_id'] in categoryIds]
  masks = coco.annsToMasks(annotations)
  masks = label_mask_categories(masks, categoryIds)
  concats = np.concatenate(masks, axis=2)
  mask = concats.max(axis=2)
  return mask

def show_crops(crops, image):
  plt.subplot(5,5,1)
  plt.imshow(image)
  index = 2
  for (crop, mask) in crops:
    if index > 25:
      continue
    plt.subplot(5,5,index)
    plt.imshow(crop)
    index+=1
    plt.subplot(5,5,index)
    plt.imshow(mask)
    index+=1
  plt.show()

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def test(coco, categories, path_to_files , output_file):
  writer = tf.python_io.TFRecordWriter(output_file)
  categoryIds = coco.getCatIds(catNms=categories)
  imgIds = []
  count = 0
  nr_crops = 0
  for i,cid in enumerate(categoryIds):
    ids = coco.getImgIds(catIds=cid)
    imgIds.extend(ids)
    print "category: {} has {} images".format(i,len(ids))
  imgIds = list(set(imgIds))
  print "number of imageIds: {}".format(len(imgIds))
  random.shuffle(imgIds)
  n=1
  number_of_drops = 0
  for imgId in imgIds:
    if n%1000==0:
      print "processed {} images".format(n)
    n+=1
    annotationIds = coco.getAnnIds(imgId)
    annotations = coco.loadAnns(annotationIds)
    annotations = [a for a in annotations if a['category_id'] in categoryIds]
    mask = get_ground_truth_mask(coco, annotations, categoryIds)

    img = coco.loadImgs(imgId)[0]
    image = io.imread(os.path.join(path_to_files, img['file_name']))
    # A few images don't fit the general (row,col,3)
    if len(image.shape) < 3:
      # print "2 D"
      continue
    c, crops = get_crops(img, annotations, image, mask)
    # show_bboxes(img, annotations, image, mask)
    nr_crops+=len(crops)
    count+=c
    for crop in crops:
      image_raw = crop[0].tostring() 
      mask = crop[1].tostring()
      example = tf.train.Example(features=tf.train.Features(feature={
          # 'height': _int64_feature(random.randint(0,100))
          'mask': _bytes_feature(mask),
          'image_raw': _bytes_feature(image_raw)
          }))
      # print example.mask
      if crop[0].shape[0]*crop[0].shape[1]*crop[0].shape[2] != 12288:
        number_of_drops+=1
      else:
        writer.write(example.SerializeToString())
      #   print "im shape: "+str(crop[0].shape)
      #   print "nr el: "+str(crop[0].shape[0]*crop[0].shape[1]*crop[0].shape[2])
      # if crop[1].shape[0]*crop[1].shape[1] != 4096:
      #   print "mask shape: "+str(crop[1].shape)
      #   print "nr el: "+str(crop[1].shape[0]*crop[1].shape[1])
      # m = _bytes_feature(mask)
      # im = _bytes_feature(image_raw)
      # decoded_im = tf.decode_raw(im, tf.uint8)
      # rez_im = tf.reshape(decoded_im, [64,64,3])
      # decoded_m = tf.decode_raw(m, tf.uint8)
      # rez_m = tf.reshape(decoded_m, [64,64])
    # plt.subplot(1,3,1)
    # plt.imshow(image)
    # coco.showAnns(annotations)
    # plt.subplot(1,3,2)
    # plt.imshow(mask)
    # plt.show()
    # show_crops(crops, image)
  writer.close()
  print "nr of annotations: {}".format(count)
  print "nr of crops: {}".format(nr_crops)
  print "nr of drops: {}".format(number_of_drops)


  return count

def show_bboxes(img, annotations, image, mask):
  plt.subplot(4,4,1)
  plt.imshow(image)
  i=2
  for ann in annotations:
    if i <= 16:
      plt.subplot(4,4,i)
      i+=1
      bbox = ann['bbox']
      crop = image[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2],:]
      plt.imshow(crop)
  plt.show()

def load_ann_file(ann_file_path):
  return COCO(ann_file_path)


if __name__=='__main__':
  ["person", "cat", "couch", "car"]
  FILE_PATH = "/home/mb/Documents/kth/thesis-project/segmentation/coco/images/val2014"
  ANN_FILE_PATH = "/home/mb/Documents/kth/thesis-project/segmentation/coco/annotations/instances_val2014.json"
  TF_RECORD_PATH ="/home/mb/Documents/kth/thesis-project/segmentation/coco64by64.tfrecords"
  CATEGORIES = ["person", "cat", "couch", "car"]
  # read_files(FILE_PATH)
  # ANNOTATION_FILE = "../annotations/instances_val2014.json"
  # prepare(ANNOTATION_FILE)
  coco = load_ann_file(ANN_FILE_PATH)

  count = test(coco, CATEGORIES, FILE_PATH, TF_RECORD_PATH)
  print "number of samples: {}".format(count)
  # find_images(coco, CATEGORIES, FILE_PATH)