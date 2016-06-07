import sys
import os
import numpy as np
import scipy as sp
import scipy.ndimage
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib import colors
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

def get_crops(img, annotations, image, ground_truth, categoryIds):
  """Returns a list of tuples containing a crop from the 
  image and the ground truth of that crop"""
  count = 0
  crops = []
  counts = {cat_id:0 for cat_id in categoryIds}
  unique_counts = {cat_id:0 for cat_id in categoryIds}
  cats = []
  index = 3
  for annotation in annotations:
    bbox = annotation['bbox']
    number_of_crops = get_num_of_crops(bbox[2], bbox[3], FIELD)
    LOWER = int(FIELD*L_SCALE)
    UPPER = FIELD*U_SCALE
    if LOWER < bbox[2] < UPPER and LOWER < bbox[3] < UPPER:
      index+=1
      bbox=map(int,bbox)
      count+=1
      # print number_of_crops
      ground = get_ground_truth_mask(coco, [annotation], categoryIds)
      cat = annotation['category_id']
      ignore_class = ground != ground_truth

      image_with_ignore = ground_truth.copy()

      image_with_ignore[ignore_class] = -1
      # print np.max(image_with_ignore)
      # print np.unique(ground_truth)
      # print np.unique(ground)
      # print np.unique(ignore_class)
      # print np.unique(image_with_ignore)
      # print image_with_ignore.dtype
      # plt.subplot(231)
      # plt.imshow(image)
      # plt.subplot(232)
      # plt.imshow(ground_truth)
      # plt.subplot(233)
      # plt.imshow(ground)
      # plt.subplot(234)
      # plt.imshow(ignore_class)
      # plt.subplot(235)
      # cmap = colors.ListedColormap(['black','blue', 'red'])
      # bounds=[-2,-0.5,0.5,1000]
      # norm = colors.BoundaryNorm(bounds, cmap.N)
      # plt.imshow(image_with_ignore, cmap=cmap, norm=norm)
      # plt.imshow(image_with_ignore,cmap="hot")

      # plt.show()
      cropped = False
      for c in xrange(number_of_crops):
        col, row = sample_coords(bbox, img)
        img_crop = image[col:col+FIELD, row:row+FIELD, :]

        # mask_crop = ground_truth[col:col+FIELD, row:row+FIELD]
        mask_crop_with_ignore = image_with_ignore[col:col+FIELD, row:row+FIELD]
        mask_crop_with_ignore = np.expand_dims(mask_crop_with_ignore, axis=2)

        mask_crop = ground_truth[col:col+FIELD, row:row+FIELD]
        mask_crop = np.expand_dims(mask_crop, axis=2)

        masked_pixels = np.sum(mask_crop_with_ignore > 0)
        size = mask_crop_with_ignore.shape[0]*mask_crop_with_ignore.shape[1]
        
        if 0.2 < masked_pixels/float(size) < 0.8:
          crops.append([img_crop, mask_crop_with_ignore, mask_crop])
          counts[cat] += 1
          cropped = True
      if cropped:
        unique_counts[cat] += 1
      # for cat in categoryIds:
      #   if counts[cat] > 0:
      #     unique_counts[cat] += 1
          
  return count, crops, counts, unique_counts

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
  return mask.astype('int16')

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def test(coco, categories, path_to_files , output_file):
  writer = tf.python_io.TFRecordWriter(output_file)
  categoryIds = coco.getCatIds(catNms=categories)
  total_counts = {cat_id:0 for cat_id in categoryIds}
  total_unique_counts = {cat_id:0 for cat_id in categoryIds}
  imgIds = []
  count = 0
  nr_crops = 0
  for i,cid in enumerate(categoryIds):
    ids = coco.getImgIds(catIds=cid)
    imgIds.extend(ids)
    print "category: {} has {} images, {}".format(cid,len(ids), categories[i])
  imgIds = list(set(imgIds))
  print "number of imageIds: {}".format(len(imgIds))
  random.shuffle(imgIds)
  n=1
  number_of_drops = 0
  total_crops_written = 0
  for imgId in imgIds:
    # print n
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
    c, crops, counts, unique_counts = get_crops(img, annotations, image, mask, categoryIds)
    # show_bboxes(img, annotations, image, mask)
    for cat in categoryIds:
      total_counts[cat] += counts[cat]
      total_unique_counts[cat] += unique_counts[cat]
    nr_crops+=len(crops)
    count+=c
    for crop in crops:
      # image_raw = crop[0].tostring() 
      # mask = crop[1].tostring()
      # print crop[0].shape
      # print crop[1].shape
      # print crop[2].shape
      image_and_mask = np.concatenate((crop[0], crop[1], crop[2]), axis=2)#,dtype='uint8')#.astype('uint8')
      image_and_mask = image_and_mask.astype('int16')
      # print np.min(crop[1])
      # print np.max(crop[1])
      # print "      "
      # image_and_mask = image_and_mask #/ 255.
      # image = image_and_mask[:,:,:3] /255.
      # plt.subplot(231)
      # plt.imshow(crop[0])
      # plt.subplot(232)
      # plt.imshow(image_and_mask[:,:,:3])
      # plt.subplot(233)
      # plt.imshow(image_and_mask[:,:,0])
      # plt.subplot(234)
      # plt.imshow(image_and_mask[:,:,1])
      # plt.subplot(235)
      # plt.imshow(image_and_mask[:,:,2])
      # plt.subplot(236)
      # plt.imshow(image_and_mask[:,:,3])
      # plt.show()
      example = tf.train.Example(features=tf.train.Features(feature={
          'height': _int64_feature(64),
          'width': _int64_feature(64),
          'image_and_mask': _bytes_feature(image_and_mask.tostring()),
          }))
      if crop[0].shape[0]*crop[0].shape[1]*crop[0].shape[2] != 12288:
        number_of_drops+=1
      else:
        total_crops_written+=1
        writer.write(example.SerializeToString())

  writer.close()
  print "nr of annotations: {}".format(count)
  print "nr of crops: {}".format(nr_crops)
  print "nr of drops: {}".format(number_of_drops)
  print "nr of written crops: {}".format(total_crops_written)
  print total_counts
  print total_unique_counts
  print categories
  print categoryIds


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
  dataset = "val"
  root_path = "/home/mb/Documents/kth/thesis-project/segmentation"
  FILE_PATH = os.path.join(root_path,"coco/images/"+dataset+"2014")
  ANN_FILE_PATH = os.path.join(root_path, "coco/annotations/instances_"+dataset+"2014.json")
  TF_RECORD_PATH =os.path.join(root_path, "coco64by64"+dataset+".tfrecords")
  CATEGORIES = ["person", "car", "chair"]
  # read_files(FILE_PATH)
  # ANNOTATION_FILE = "../annotations/instances_val2014.json"
  # prepare(ANNOTATION_FILE)
  coco = load_ann_file(ANN_FILE_PATH)
  # CATEGORIES = []
  # for i in range(1, len(coco.cats)+1):
  #   try:
  #     CATEGORIES.append(coco.cats[i]['name'])
  #   except:
  #     pass


  count = test(coco, CATEGORIES, FILE_PATH, TF_RECORD_PATH)
  print "number of samples: {}".format(count)
  # find_images(coco, CATEGORIES, FILE_PATH)