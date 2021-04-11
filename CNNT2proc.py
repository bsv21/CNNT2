# Code used for Radiology: Artificial Intelligence paper "Synthesizing Quantitative T2 Maps in 
# Right Lateral Knee Femoral Condyles from Multi-Contrast Anatomical Data with a Conditional GAN" by Sveinsson et al (cleaned-up and commented version).

# This code is heavily based on the Pix2Pix paper by Isola et al. (arXiv:1611.07004) and its tutorial 
# implementation by Google's Tensorflow team, retrieved from 
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/eager/python/examples/pix2pix/pix2pix_eager.ipynb in Oct 2018. 
# This website seems to now be defunct but an updated version is now at 
# https://www.tensorflow.org/tutorials/generative/pix2pix, which is cited in the Sveinsson paper. Many parts of that code are used as given,
# with some new parts added.


# Import TensorFlow - this implementation used Tensorflow 1.10.0.
import tensorflow as tf
# Eager execution was also used.
tf.enable_eager_execution()

import os
os.environ["CUDA_VISIBLE_DEVICES"]="2,3"
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import clear_output

# The 'fileNumber" value will be used for assigning the output of the network to distinct folders and files. 
# During the development of the network, this value was incremented for every new network version, 
# starting with 1, then 2, etc., with the network published having value 120.
fileNumber =              # Fill in with integer value
EPOCHS = 20
# The following numpy arrays will contain various error metrics and will be appended after each epoch. 
# Based on these arrays, the errors and losses can be plotted as a function on epoch number.
# 'sosArray' will contain the sum-of-squared-difference between the MESE and CNN T2 maps, 
# for the first image in a batch. Used for tuning the network. After program finishes or 
# CTRL+C is pressed, this array is written into the file 'sosFile'.
sosArray = np.array([1])
# 'dlossArray' will contain the discriminator loss (measuring failure to classify 
# patches as MESE and CNN from whole (unmasked) images). Used for tuning the network. 
# After program finishes or CTRL+C is pressed, this array is written into the file 'dlossFile'.
dlossArray = np.array([1])
# 'mdlossArray' will contain the discriminator loss (measuring failure to classify 
# patches as MESE and CNN from masked images, focusing on femoral cartilage). Used for tuning the network.
# After program finishes or CTRL+C is pressed, this array is written into the file 'mdlossFile'.
mdlossArray = np.array([1])
# 'glossArray' will contain the total generator loss, consisting of a weighted sum of the generator 
# loss for the whole image (glossGAN, see below), generator loss for a masked image (glossMGAN, see below), 
# the L1 error for the whole image (wholeL1, see below), and the L1 error for the masked image (maskL1, 
# see below). Used for tuning the network. After program finishes or CTRL+C is pressed, this 
# array is written into the file 'glossFile'.
glossArray = np.array([1])
# 'glossGANArray' will contain the (unweighted) generator loss for the whole image 
# - the performance in creating image patches that fool the discriminator for the whole image. 
# Used for tuning the network. After program finishes or CTRL+C is pressed, 
# this array is written into the file 'glossGANFile'.
glossGANArray = np.array([1])
# 'glossMGANArray' will contain the (unweighted) generator loss for the masked image - 
# the performance in creating image patches that fool the 'mask discriminator', 
# that only looks at the masked cartilage region. Used for tuning the network. 
# After program finishes or CTRL+C is pressed, this array is written into the file 'glossMGANFile'.
glossMGANArray = np.array([1])
# 'wholeL1Array' will contain the (unweighted) generator loss contribution from 
# the L1 difference between the whole MESE and CNN T2 maps. Used for tuning the network. 
# After program finishes or CTRL+C is pressed, this array is written into the file 'wholeL1file'.
wholeL1array = np.array([1])
# 'maskL1Array' will contain the (unweighted) generator loss contribution from 
# the L1 difference between the MESE and CNN T2 maps over a mask 
# focusing on the femoral cartilage. Used for tuning the network. 
# After program finishes or CTRL+C is pressed, this array is written into the file 'maskL1file'.
maskL1array = np.array([1])
# Choose directory for saving the arrays described above in the form of text files.
errorMetricFilePath =                                                # Fill in with string value in the form of '/path/to/metric/files/'
sosFile = errorMetricFilePath + 'sosFile_version' + str(fileNumber)
dlossFile = errorMetricFilePath + 'dlossFile_version' + str(fileNumber)
mdlossFile = errorMetricFilePath + 'mdlossFile_version' + str(fileNumber)
glossFile = errorMetricFilePath + 'glossFile_version' + str(fileNumber)
glossGANFile = errorMetricFilePath + 'glossGANFile_version' + str(fileNumber)
glossMGANFile = errorMetricFilePath + 'glossMGANFile_version' + str(fileNumber)
wholeL1file = errorMetricFilePath + 'wholeL1file_version' + str(fileNumber)
maskL1file = errorMetricFilePath + 'maskL1file_version' + str(fileNumber)

# If the user presses CTRL+C, we stop the program but write whatever had been stored in the metric arrays to their corresponding files. Normally this is done when the program finishes running.
import signal
import sys
def signal_handler(sig, frame):
        print('You pressed Ctrl+C!')
        np.save(sosFile, sosArray)
        np.save(dlossFile, dlossArray)
        np.save(mdlossFile, mdlossArray)
        np.save(glossFile, glossArray)
        np.save(wholeL1file, wholeL1array)
        np.save(maskL1file, maskL1array)
        sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)


print ('***\nSetting constants\n***\n')

# Define a path to the input data directory. This assumes the following structure:
# That PATH has the subdirectories 'train', 'val', and 'test'.
# Each of the subdirectories contains input data in the form of jpg image files.
# The image files do not have to be in any particular order.
# The image files are assumed to consist of a 3x6 matrix of images.
# The 3 rows represent 3 consecutive slices, although in the final formulation of the paper, only the middle row was used.
# The columns represent 240x288 images, in the order:
#     (1) T2 map
#     (2) DESS
#     (3) Sag TSE
#     (4) Cor FLASH (reformatted)
#     (5) Cor TSE (reformatted)
#     (6) Cartilage mask - produced with the network described in http://arxiv.org/abs/1902.01977
PATH = 									        # Fill in with string value of the form '/path/to/input/data/'

BUFFER_SIZE = 400
BATCH_SIZE = 16
# The image will be resampled to RESAMPLED_HEIGHT x RESAMPLED_WIDTH
RESAMPLED_WIDTH = 256
RESAMPLED_HEIGHT = 256
# Set weightings for the different components of the generator loss function, relative to the "GAN loss" (measuring
# the success of the generator in creating images that the "whole image" discriminator cannot distinguish as synthesized)
# These weights were chosen after a lot of trial and error. Note that the mask contribution is given weight 3x the contribution
# for the whole image, both for the L1 loss and the GAN loss.
LAMBDA_WL1 = 25
LAMBDA_ML1 = 75
LAMBDA_MGAN = 3

print ('***\nDefining load_image\n***\n')

# The function 'load_image' reads in the input jpg files, representing a 3x6 image matrix,
# and splits it up into its respective images. It then does data augmentation if desired
# (using the second argument to the function), resizes the immages to RESAMPLED_HEIGHTxRESAMPLED_WIDTH,
# normalizes the images, and saves all 18 images in a tensor of size (batch_size,RESAMPLED_HEIGHT,RESAMPLED_WIDTH,18)
def load_image(image_file, augment_data):
  image = tf.read_file(image_file)
  image = tf.image.decode_jpeg(image)

  # w_total and h_total will be the width and height of the 3x6 image matrix
  w_total = tf.shape(image)[1]
  h_total = tf.shape(image)[0]

  # Now we divide w_total and h_total by 6 and 3 to get the width and height of individual images.
  # This assumes that the images are all the same size.
  w = w_total // 6
  h = h_total // 3
  # Extract T2, DESS, Sag TSE, FLASH, cor TSE, and cartilage mask from top row (slice N-1)
  T2_image0 = image[:h, :w, :]
  DESS_image0 = image[:h, w:2*w, :]
  TSEsag_image0 = image[:h, 2*w:3*w, :]
  FLASH_image0 = image[:h, 3*w:4*w, :]
  TSEcor_image0 = image[:h, 4*w:5*w, :]
  mask_image0 = image[:h, 5*w:6*w, :]
  # Extract T2, DESS, Sag TSE, FLASH, cor TSE, and cartilage mask from top row (slice N)
  T2_image1 = image[h:2*h, :w, :]
  DESS_image1 = image[h:2*h, w:2*w, :]
  TSEsag_image1 = image[h:2*h, 2*w:3*w, :]
  FLASH_image1 = image[h:2*h, 3*w:4*w, :]
  TSEcor_image1 = image[h:2*h, 4*w:5*w, :]
  mask_image1 = image[h:2*h, 5*w:6*w, :]
  # Extract T2, DESS, Sag TSE, FLASH, cor TSE, and cartilage mask from top row (slice N+1)
  T2_image2 = image[2*h:3*h, :w, :]
  DESS_image2 = image[2*h:3*h, w:2*w, :]
  TSEsag_image2 = image[2*h:3*h, 2*w:3*w, :]
  FLASH_image2 = image[2*h:3*h, 3*w:4*w, :]
  TSEcor_image2 = image[2*h:3*h, 4*w:5*w, :]
  mask_image2 = image[2*h:3*h, 5*w:6*w, :]

  # Cast all 18 images to float32 tensors.
  T2_image0 = tf.cast(T2_image0, tf.float32)
  DESS_image0 = tf.cast(DESS_image0, tf.float32)
  TSEsag_image0 = tf.cast(TSEsag_image0, tf.float32)
  FLASH_image0 = tf.cast(FLASH_image0, tf.float32)
  TSEcor_image0 = tf.cast(TSEcor_image0, tf.float32)
  mask_image0 = tf.cast(mask_image0, tf.float32)
  T2_image1 = tf.cast(T2_image1, tf.float32)
  DESS_image1 = tf.cast(DESS_image1, tf.float32)
  TSEsag_image1 = tf.cast(TSEsag_image1, tf.float32)
  FLASH_image1 = tf.cast(FLASH_image1, tf.float32)
  TSEcor_image1 = tf.cast(TSEcor_image1, tf.float32)
  mask_image1 = tf.cast(mask_image1, tf.float32)
  T2_image2 = tf.cast(T2_image2, tf.float32)
  DESS_image2 = tf.cast(DESS_image2, tf.float32)
  TSEsag_image2 = tf.cast(TSEsag_image2, tf.float32)
  FLASH_image2 = tf.cast(FLASH_image2, tf.float32)
  TSEcor_image2 = tf.cast(TSEcor_image2, tf.float32)
  mask_image2 = tf.cast(mask_image2, tf.float32)

  # If the user has chosen data augmentation, we randomly "jitter" and flip the image.
  if augment_data:
    
    # Start by resizing the images to 286x286 (instead of the desired resampling of 256x256)
    imgH_jitter = RESAMPLED_HEIGHT + 30
    imgW_jitter = RESAMPLED_WIDTH + 30
    T2_image0 = tf.image.resize_images(T2_image0, [imgH_jitter, imgW_jitter], align_corners=True, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    DESS_image0 = tf.image.resize_images(DESS_image0, [imgH_jitter, imgW_jitter], align_corners=True, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    TSEsag_image0 = tf.image.resize_images(TSEsag_image0, [imgH_jitter, imgW_jitter], align_corners=True, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    FLASH_image0 = tf.image.resize_images(FLASH_image0, [imgH_jitter, imgW_jitter], align_corners=True, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    TSEcor_image0 = tf.image.resize_images(TSEcor_image0, [imgH_jitter, imgW_jitter], align_corners=True, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    mask_image0 = tf.image.resize_images(mask_image0, [imgH_jitter, imgW_jitter], align_corners=True, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    T2_image1 = tf.image.resize_images(T2_image1, [imgH_jitter, imgW_jitter], align_corners=True, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    DESS_image1 = tf.image.resize_images(DESS_image1, [imgH_jitter, imgW_jitter], align_corners=True, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    TSEsag_image1 = tf.image.resize_images(TSEsag_image1, [imgH_jitter, imgW_jitter], align_corners=True, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    FLASH_image1 = tf.image.resize_images(FLASH_image1, [imgH_jitter, imgW_jitter], align_corners=True, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    TSEcor_image1 = tf.image.resize_images(TSEcor_image1, [imgH_jitter, imgW_jitter], align_corners=True, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    mask_image1 = tf.image.resize_images(mask_image1, [imgH_jitter, imgW_jitter], align_corners=True, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    T2_image2 = tf.image.resize_images(T2_image2, [imgH_jitter, imgW_jitter], align_corners=True, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    DESS_image2 = tf.image.resize_images(DESS_image2, [imgH_jitter, imgW_jitter], align_corners=True, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    TSEsag_image2 = tf.image.resize_images(TSEsag_image2, [imgH_jitter, imgW_jitter], align_corners=True, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    FLASH_image2 = tf.image.resize_images(FLASH_image2, [imgH_jitter, imgW_jitter], align_corners=True, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    TSEcor_image2 = tf.image.resize_images(TSEcor_image2, [imgH_jitter, imgW_jitter], align_corners=True, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    mask_image2 = tf.image.resize_images(mask_image2, [imgH_jitter, imgW_jitter], align_corners=True, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    
    # Now randomly crop the images to 256x256. Effectively gives a slight zoom to a random image region.
    # Start by randomly cropping the top row (slice N-1)
    stacked_image0 = tf.stack([DESS_image0, TSEsag_image0, FLASH_image0, TSEcor_image0, mask_image0, T2_image0], axis=0)
    cropped_image0 = tf.random_crop(stacked_image0, size=[6, RESAMPLED_HEIGHT, RESAMPLED_WIDTH, 1])
    DESS_image0, TSEsag_image0, FLASH_image0, TSEcor_image0, mask_image0, T2_image0 = cropped_image0[0], cropped_image0[1], cropped_image0[2], cropped_image0[3], cropped_image0[4], cropped_image0[5]
    # Now randomly crop the middle row (slice N)
    stacked_image1 = tf.stack([DESS_image1, TSEsag_image1, FLASH_image1, TSEcor_image1, mask_image1, T2_image1], axis=0)
    cropped_image1 = tf.random_crop(stacked_image1, size=[6, RESAMPLED_HEIGHT, RESAMPLED_WIDTH, 1])
    DESS_image1, TSEsag_image1, FLASH_image1, TSEcor_image1, mask_image1, T2_image1 = cropped_image1[0], cropped_image1[1], cropped_image1[2], cropped_image1[3], cropped_image1[4], cropped_image1[5]
    # Finally randomly crop the bottom row (slice N+1)
    stacked_image2 = tf.stack([DESS_image2, TSEsag_image2, FLASH_image2, TSEcor_image2, mask_image2, T2_image2], axis=0)
    cropped_image2 = tf.random_crop(stacked_image2, size=[6, RESAMPLED_HEIGHT, RESAMPLED_WIDTH, 1])
    DESS_image2, TSEsag_image2, FLASH_image2, TSEcor_image2, mask_image2, T2_image2 = cropped_image2[0], cropped_image2[1], cropped_image2[2], cropped_image2[3], cropped_image2[4], cropped_image2[5]

    # Now flip every image horizontally with a 50% probability.
    if np.random.random() > 0.5:
      # Flip top row (slice N-1)
      T2_image0 = tf.image.flip_left_right(T2_image0)
      DESS_image0 = tf.image.flip_left_right(DESS_image0)
      TSEsag_image0 = tf.image.flip_left_right(TSEsag_image0)
      FLASH_image0 = tf.image.flip_left_right(FLASH_image0)
      TSEcor_image0 = tf.image.flip_left_right(TSEcor_image0)
      mask_image0 = tf.image.flip_left_right(mask_image0)
      # Flip middle row (slice N)
      T2_image1 = tf.image.flip_left_right(T2_image1)
      DESS_image1 = tf.image.flip_left_right(DESS_image1)
      TSEsag_image1 = tf.image.flip_left_right(TSEsag_image1)
      FLASH_image1 = tf.image.flip_left_right(FLASH_image1)
      TSEcor_image1 = tf.image.flip_left_right(TSEcor_image1)
      mask_image1 = tf.image.flip_left_right(mask_image1)
      # Flip bottom row (slice N+1)
      T2_image2 = tf.image.flip_left_right(T2_image2)
      DESS_image2 = tf.image.flip_left_right(DESS_image2)
      TSEsag_image2 = tf.image.flip_left_right(TSEsag_image2)
      FLASH_image2 = tf.image.flip_left_right(FLASH_image2)
      TSEcor_image2 = tf.image.flip_left_right(TSEcor_image2)
      mask_image2 = tf.image.flip_left_right(mask_image2)

  # Data augmentation has not been selected. Do simple resizing to 256x256.
  else:
    # Resize top row (slice N-1)
    T2_image0 = tf.image.resize_images(T2_image0, size=[RESAMPLED_HEIGHT, RESAMPLED_WIDTH], align_corners=True, method=2)
    DESS_image0 = tf.image.resize_images(DESS_image0, size=[RESAMPLED_HEIGHT, RESAMPLED_WIDTH], align_corners=True, method=2)
    TSEsag_image0 = tf.image.resize_images(TSEsag_image0, size=[RESAMPLED_HEIGHT, RESAMPLED_WIDTH], align_corners=True, method=2)
    FLASH_image0 = tf.image.resize_images(FLASH_image0, size=[RESAMPLED_HEIGHT, RESAMPLED_WIDTH], align_corners=True, method=2)
    TSEcor_image0 = tf.image.resize_images(TSEcor_image0, size=[RESAMPLED_HEIGHT, RESAMPLED_WIDTH], align_corners=True, method=2)
    mask_image0 = tf.image.resize_images(mask_image0, size=[RESAMPLED_HEIGHT, RESAMPLED_WIDTH], align_corners=True, method=2)
    # Resize middle row (slice N)
    T2_image1 = tf.image.resize_images(T2_image1, size=[RESAMPLED_HEIGHT, RESAMPLED_WIDTH], align_corners=True, method=2)
    DESS_image1 = tf.image.resize_images(DESS_image1, size=[RESAMPLED_HEIGHT, RESAMPLED_WIDTH], align_corners=True, method=2)
    TSEsag_image1 = tf.image.resize_images(TSEsag_image1, size=[RESAMPLED_HEIGHT, RESAMPLED_WIDTH], align_corners=True, method=2)
    FLASH_image1 = tf.image.resize_images(FLASH_image1, size=[RESAMPLED_HEIGHT, RESAMPLED_WIDTH], align_corners=True, method=2)
    TSEcor_image1 = tf.image.resize_images(TSEcor_image1, size=[RESAMPLED_HEIGHT, RESAMPLED_WIDTH], align_corners=True, method=2)
    mask_image1 = tf.image.resize_images(mask_image1, size=[RESAMPLED_HEIGHT, RESAMPLED_WIDTH], align_corners=True, method=2)
    # Resize bottom row (slice N+1)
    T2_image2 = tf.image.resize_images(T2_image2, size=[RESAMPLED_HEIGHT, RESAMPLED_WIDTH], align_corners=True, method=2)
    DESS_image2 = tf.image.resize_images(DESS_image2, size=[RESAMPLED_HEIGHT, RESAMPLED_WIDTH], align_corners=True, method=2)
    TSEsag_image2 = tf.image.resize_images(TSEsag_image2, size=[RESAMPLED_HEIGHT, RESAMPLED_WIDTH], align_corners=True, method=2)
    FLASH_image2 = tf.image.resize_images(FLASH_image2, size=[RESAMPLED_HEIGHT, RESAMPLED_WIDTH], align_corners=True, method=2)
    TSEcor_image2 = tf.image.resize_images(TSEcor_image2, size=[RESAMPLED_HEIGHT, RESAMPLED_WIDTH], align_corners=True, method=2)
    mask_image2 = tf.image.resize_images(mask_image2, size=[RESAMPLED_HEIGHT, RESAMPLED_WIDTH], align_corners=True, method=2)
  

  # Now normalize the images to [-1, 1], except the mask, which we normalize to [0, 1].
  # Normalize top row (slice N-1)
  T2_image0 = (T2_image0 / 127.5) - 1
  DESS_image0 = (DESS_image0 / 127.5) - 1
  TSEsag_image0 = (TSEsag_image0 / 127.5) - 1
  FLASH_image0 = (FLASH_image0 / 127.5) - 1
  TSEcor_image0 = (TSEcor_image0 / 127.5) - 1
  mask_image0 = (mask_image0 / 255.0)             # Set between 0 and 1
  # Normalize middle row (slice N)
  T2_image1 = (T2_image1 / 127.5) - 1
  DESS_image1 = (DESS_image1 / 127.5) - 1
  TSEsag_image1 = (TSEsag_image1 / 127.5) - 1
  FLASH_image1 = (FLASH_image1 / 127.5) - 1
  TSEcor_image1 = (TSEcor_image1 / 127.5) - 1
  mask_image1 = (mask_image1 / 255.0)             # Set between 0 and 1
  # Normalize bottom row (slice N+1)
  T2_image2 = (T2_image2 / 127.5) - 1
  DESS_image2 = (DESS_image2 / 127.5) - 1
  TSEsag_image2 = (TSEsag_image2 / 127.5) - 1
  FLASH_image2 = (FLASH_image2 / 127.5) - 1
  TSEcor_image2 = (TSEcor_image2 / 127.5) - 1
  mask_image2 = (mask_image2 / 255.0)             # Set between 0 and 1


  # Now we combine DESS, sag TSE, FLASH, and cor TSE into one four-channel image.
  # We include all three slices of each sequence before we add the next sequence.
  # In the final implementation, only the middle slice (image1) was used, 
  # and the FLASH images were discarded.
  combined_image = tf.concat([DESS_image0, DESS_image1, DESS_image2, 
                              TSEsag_image0, TSEsag_image1, TSEsag_image2, 
                              FLASH_image0, FLASH_image1, FLASH_image2, 
                              TSEcor_image0, TSEcor_image1, TSEcor_image2], axis=2)


  # Return the combined image (DESS, Sag TSE, FLASH, Cor TSE, 3 slices each) and the center slice of the MESE T2 map and the mask.
  return combined_image, T2_image1, mask_image1

print ('***\nReading input data\n***\n')

# Read the train data, contained as jpgs in PATH/train/
train_dataset = tf.data.Dataset.list_files(PATH+'train/*.jpg')
# The training data gets shuffled.
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
# The second argument to load_image augments the data if True but
# we don't use this. 
train_dataset = train_dataset.map(lambda x: load_image(x, False))
train_dataset = train_dataset.batch(BATCH_SIZE)

# Read the test data, contained as jpgs in PATH/test/
test_dataset = tf.data.Dataset.list_files(PATH+'test/*.jpg', shuffle=False)
test_dataset = test_dataset.map(lambda x: load_image(x, False))
test_dataset = test_dataset.batch(BATCH_SIZE)

# Read the validation data, contained as jpgs in PATH/val/
val_dataset = tf.data.Dataset.list_files(PATH+'val/*.jpg', shuffle=False)
val_dataset = val_dataset.map(lambda x: load_image(x, False))
val_dataset = val_dataset.batch(BATCH_SIZE)

# We will only use one output channel, for the T2 map.
OUTPUT_CHANNELS = 1




print ('***\nDefining Downsample class\n***\n')

# The 'Downsample' class will handle the downsampling step in the generator
# (the first half of the U-net)
class Downsample(tf.keras.Model):
    
  def __init__(self, filters, size, apply_batchnorm=True):
    super(Downsample, self).__init__()
    self.apply_batchnorm = apply_batchnorm
    initializer = tf.random_normal_initializer(0., 0.02)

    self.conv1 = tf.keras.layers.Conv2D(filters, (size, size), strides=2, padding='same', kernel_initializer=initializer, use_bias=False)
    if self.apply_batchnorm:
        self.batchnorm = tf.keras.layers.BatchNormalization()
  
  def call(self, x, training):
    x = self.conv1(x)
    if self.apply_batchnorm:
        x = self.batchnorm(x, training=training)
    x = tf.nn.leaky_relu(x)
    return x 


print ('***\nDefining Upsample class\n***\n')

# The 'Upsample' class will handle the upsampling step 
# (2D transposed convolution) in the generator (the second half of the U-net)
# and the concatenation with the corresponding step in the first U-net half.
class Upsample(tf.keras.Model):
    
  def __init__(self, filters, size, apply_dropout=False):
    super(Upsample, self).__init__()
    self.apply_dropout = apply_dropout
    initializer = tf.random_normal_initializer(0., 0.02)

    self.up_conv = tf.keras.layers.Conv2DTranspose(filters, (size, size), strides=2, padding='same', kernel_initializer=initializer, use_bias=False)
    self.batchnorm = tf.keras.layers.BatchNormalization()
    if self.apply_dropout:
        self.dropout = tf.keras.layers.Dropout(0.5)

  def call(self, x1, x2, training):
    x = self.up_conv(x1)
    x = self.batchnorm(x, training=training)
    if self.apply_dropout:
        x = self.dropout(x, training=training)
    x = tf.nn.relu(x)
    x = tf.concat([x, x2], axis=-1)
    return x


print ('***\nDefining Generator class\n***\n')
# The 'Generator' class will be a U-Net that takes in 256x256x3 data, representing
# 256x256 DESS, Sag TSE, and Cor TSE images.
class Generator(tf.keras.Model):
    
  def __init__(self):
    super(Generator, self).__init__()
    initializer = tf.random_normal_initializer(0., 0.02)
   
    # Here we create the downsampling steps of the U-Net generator, using
    # the 'Downsample' class. Number of filters increases, kernel
    # size stays constant at 4x4. 
    self.down1 = Downsample(64, 4, apply_batchnorm=False)
    self.down2 = Downsample(128, 4)
    self.down3 = Downsample(256, 4)
    self.down4 = Downsample(512, 4)
    self.down5 = Downsample(512, 4)
    self.down6 = Downsample(512, 4)
    self.down7 = Downsample(512, 4)
    self.down8 = Downsample(512, 4)

    # Similarly, apply upsampling steps of the U-Net generator, using
    # the 'Upsample' class. Number of filters decreases, kernel
    # size again stays constant at 4x4.
    self.up1 = Upsample(512, 4, apply_dropout=True)
    self.up2 = Upsample(512, 4, apply_dropout=True)
    self.up3 = Upsample(512, 4, apply_dropout=True)
    self.up4 = Upsample(512, 4)
    self.up5 = Upsample(256, 4)
    self.up6 = Upsample(128, 4)
    self.up7 = Upsample(64, 4)

    self.last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, (4, 4), strides=2, padding='same', kernel_initializer=initializer)

    # The 'merge' layer will be used to concatenate the input data with the final data.
    self.merge = tf.keras.layers.Concatenate()

    # The 'final2Dconv' layer will be used to do a 2D convolution on the merged data
    # - essentially, the idea is that the final output will be a weighted combination
    # of the unchanged input and the U-net output. This could be helpful since it can be
    # argued that the T2 map could be pretty well approximated with a relatively simple
    # linear combination of the input images.
    self.final2Dconv = tf.keras.layers.Conv2D(1, (4,4), strides=(1, 1), padding='same', data_format=None,
                                           dilation_rate=(1, 1), activation=None, use_bias=True,
                                           kernel_initializer='glorot_uniform', bias_initializer='zeros',
                                           kernel_regularizer=None, bias_regularizer=None,
                                           activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
  

  # Based on the Tensorflow tutorial of Pix2Pix, we use tf.contrib.eager.defun, which should give a performance speedup.
  @tf.contrib.eager.defun
  # Define what happens when you call the U-Net with input tensor 'x' and the option 'training'
  def call(self, x, training):
    # Start by splitting the 'x' input into its individual images. Has 3 slices (N-1, N, N+1) from 4 sequences (DESS, Sag TSE, FLASH, and Cor TSE)
    dess0, dess1, dess2, tsesag0, tsesag1, tsesag2, flash0, flash1, flash2, tsecor0, tsecor1, tsecor2 = tf.split(x, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 3)
    # We only use slice N from DESS, Sag TSE, and Cor TSE.
    x0 = self.merge([dess1, tsesag1, tsecor1])

    # Propagate the tensor down the first half of the U-Net
    x1 = self.down1(x0, training=training) # (bs, 128, 128, 64)
    x2 = self.down2(x1, training=training) # (bs, 64, 64, 128)
    x3 = self.down3(x2, training=training) # (bs, 32, 32, 256)
    x4 = self.down4(x3, training=training) # (bs, 16, 16, 512)
    x5 = self.down5(x4, training=training) # (bs, 8, 8, 512)
    x6 = self.down6(x5, training=training) # (bs, 4, 4, 512)
    x7 = self.down7(x6, training=training) # (bs, 2, 2, 512)
    x8 = self.down8(x7, training=training) # (bs, 1, 1, 512)

    # Now propage the tensor up the second half of the U-Net
    x9 = self.up1(x8, x7, training=training) # (bs, 2, 2, 1024)
    x10 = self.up2(x9, x6, training=training) # (bs, 4, 4, 1024)
    x11 = self.up3(x10, x5, training=training) # (bs, 8, 8, 1024)
    x12 = self.up4(x11, x4, training=training) # (bs, 16, 16, 1024)
    x13 = self.up5(x12, x3, training=training) # (bs, 32, 32, 512)
    x14 = self.up6(x13, x2, training=training) # (bs, 64, 64, 256)
    x15 = self.up7(x14, x1, training=training) # (bs, 128, 128, 128)
    
    x16 = self.last(x15) # (bs, 256, 256, 3)

    # Merge together the original input with the output of the U-Net 
    x0and16 = self.merge([dess1, tsesag1, tsecor1, x16])
    # Perform a convolution to get a weighted sum of the original inputs and the U-Net output.
    x17 = self.final2Dconv(x0and16)
    x17 = tf.nn.tanh(x17)

    return x17

print ('***\nDefining DiscDownsample class\n***\n')

# The class 'DiscDownsample' defines the downsampling steps for the
# discriminators. Both discriminators (working on the whole image and on
# the mask) use the same downsampling function.
class DiscDownsample(tf.keras.Model):
    
  def __init__(self, filters, size, apply_batchnorm=True):
    super(DiscDownsample, self).__init__()
    self.apply_batchnorm = apply_batchnorm
    initializer = tf.random_normal_initializer(0., 0.02)

    self.conv1 = tf.keras.layers.Conv2D(filters, (size, size), strides=2, padding='same', kernel_initializer=initializer, use_bias=False)

    if self.apply_batchnorm:
        self.batchnorm = tf.keras.layers.BatchNormalization()
 
  # Calling the DiscDownsample function performs a convolution on the input,
  # performs batch normalization unless otherwise desired, and then
  # passes through a leaky ReLU activation function.
  def call(self, x, training):
    x = self.conv1(x)
    if self.apply_batchnorm:
        x = self.batchnorm(x, training=training)
    x = tf.nn.leaky_relu(x)
    return x 



print ('***\nDefining Discriminator class\n***\n')
# The 'Discriminator' class is a patch discriminator that looks at patches
# from the whole image and decides whether they represent true (MESE) data
# or synthesized (CNN) data.
# The discriminator will output a 30x30 "image".
# Each pixel in this 30x30 output is sensitive to a 70x70 patch in the input.
# So we are checking whether the discriminator can tell if 70x70 patches in the input look real.
class Discriminator(tf.keras.Model):
    
  def __init__(self):
    super(Discriminator, self).__init__()
    initializer = tf.random_normal_initializer(0., 0.02)
   
    # Here we create the downsampling steps of the whole-image discriminator, using
    # the 'DiscDownsample' class. Number of filters increases, kernel size stays constant at 4x4. 
    self.down1 = DiscDownsample(64, 4, False)
    self.down2 = DiscDownsample(128, 4)
    self.down3 = DiscDownsample(256, 4)
    
    # We zero pad, resulting in our images going from 32x32 to 34x34.
    self.zero_pad1 = tf.keras.layers.ZeroPadding2D()
    # We then convolve with kernel size 4x4 and stride 1, resulting
    # in image size going from 34x34 to 31x31.
    self.conv = tf.keras.layers.Conv2D(512, (4, 4), strides=1, kernel_initializer=initializer, use_bias=False)
    self.batchnorm1 = tf.keras.layers.BatchNormalization()
    
    # Define another zero padding, making the image go from 31x31 to 33x33
    self.zero_pad2 = tf.keras.layers.ZeroPadding2D()
    # Final convolution makes images go from 33x33 to 30x30.
    self.last = tf.keras.layers.Conv2D(1, (4, 4), strides=1, kernel_initializer=initializer)
  
  # Based on the Tensorflow tutorial of Pix2Pix, we use tf.contrib.eager.defun, which should give a performance speedup.
  @tf.contrib.eager.defun
  def call(self, combAnat_im, mese_im, training):
    # concatenating the combined anatomical images and the MESE map
    x = tf.concat([combAnat_im, mese_im], axis=-1) # (bs, 256, 256, 1)
    x = self.down1(x, training=training) # (bs, 128, 128, 64)
    x = self.down2(x, training=training) # (bs, 64, 64, 128)
    x = self.down3(x, training=training) # (bs, 32, 32, 256)

    x = self.zero_pad1(x) # (bs, 34, 34, 256)
    x = self.conv(x)      # (bs, 31, 31, 512)
    x = self.batchnorm1(x, training=training)
    x = tf.nn.leaky_relu(x)
    
    x = self.zero_pad2(x) # (bs, 33, 33, 512)
    # The loss function will expect raw logits so we don't add an activation function.
    x = self.last(x)      # (bs, 30, 30, 1)

    return x

print ('***\nDefining MaskDiscriminator class\n***\n')

# The 'MaskDiscriminator' class works similarly to the 'Discriminator' class,
# but is designed to focus on an area covering the cartilage. It is therefore
# built to be sensitive to smaller patches. While the 'Discriminator' class,
# which looked at the whole image, gave a 30x30 output where each
# output was sensitive to a 70x70 patch in the input, the
# 'MaskDiscriminator' class produces a 128x128 output where
# each pixel is sensitive to 11x11 patches in the input..
class MaskDiscriminator(tf.keras.Model):
    
  def __init__(self):
    super(MaskDiscriminator, self).__init__()
    initializer = tf.random_normal_initializer(0., 0.02)
   
    # We only have one downsampling layer here. This will result in the
    # mask discriminator not examining as much "structure", which is
    # reasonable as we there should not be much structure within the cartile. 
    self.down1 = DiscDownsample(64, 3, False)
  
    # First zero padding will take image from 130x130 to 128x128 
    self.zero_pad1 = tf.keras.layers.ZeroPadding2D()
    # We use a smaller convolution kernel here than for the full 'Discriminator' class
    # (3x3), as the area we are interested in is small.
    self.conv1 = tf.keras.layers.Conv2D(128, (3, 3), strides=1, kernel_initializer=initializer, use_bias=False)
    self.batchnorm1 = tf.keras.layers.BatchNormalization()
    
    # Second zero padding will change the image from 130x130 to 128x128 again.
    self.zero_pad2 = tf.keras.layers.ZeroPadding2D()
    # Finally do 2D convolution, again with 3x3 kernel.
    self.last = tf.keras.layers.Conv2D(1, (3, 3), strides=1, kernel_initializer=initializer)
  
  # Based on the Tensorflow tutorial of Pix2Pix, we use tf.contrib.eager.defun, which should give a performance speedup.
  @tf.contrib.eager.defun
  def call(self, combAnat_im, mese_im, training):
    # We concatenate the combined anatomical images with the MESE map.
    x = tf.concat([combAnat_im, mese_im], axis=-1) # (bs, 256, 256, 1)
    x = self.down1(x, training=training) # (bs, 128, 128, 64)
    
    x = self.zero_pad1(x)  # (bs, 130, 130, 64)
    x = self.conv1(x)      # (bs, 128, 128, 128)
    x = self.batchnorm1(x, training=training)
    x = tf.nn.leaky_relu(x)
    
    x = self.zero_pad2(x) # (bs, 130, 130, 128)
    # The loss function will expect raw logits so we don't add an activation function.
    x = self.last(x)      # (bs, 128, 128, 1)      # Each of the 128x128 pixels should correspond to a 11x11 patch

    return x


print ('***\nCreating generator and discriminator objects\n***\n')

# Now create the U-Net generator, the discriminator (for the whole image), and the mask discriminator (for the
# cartilage region) objects, based on the 'Generator', 'Discriminator', and 'MaskDiscriminator' classes.
generator = Generator()
discriminator = Discriminator()
maskDiscriminator = MaskDiscriminator()



print ('***\nDefining discriminator_loss functions\n***\n')

# Create the loss function for the "whole image" discriminator.
def discriminator_loss(disc_real_output, disc_generated_output):
  # disc_real_output holds the output that the discriminator gave for real images. Discriminator should want this to be big (meaning likely real).
  real_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels = tf.ones_like(disc_real_output), logits = disc_real_output)

  # disc_generated_output holds the output that the discriminator gave for generated images. Discriminator should want this to be small (meaning likely generated).
  generated_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels = tf.zeros_like(disc_generated_output), logits = disc_generated_output)

  total_disc_loss = real_loss + generated_loss

  return total_disc_loss


# Create the loss function for the maskdiscriminator, that only looks at the cartilage region.
def maskdiscriminator_loss(mdisc_real_output, mdisc_generated_output, mask):
    
  # We resize the mask to be the same size as the discriminator output.
  mask_resized = tf.image.resize_images(mask, [tf.shape(mdisc_real_output)[1], tf.shape(mdisc_real_output)[2]], align_corners=True, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  
  # Create an image that has the real mdisc values within the mask, 1 otherwise
  mdisc_real_output_masked1 = tf.where(tf.greater_equal(mask_resized,0.5), mdisc_real_output, tf.ones_like(mdisc_real_output))

  # mdisc_real_output holds the output that the mask discriminator gave for real images. Maskdiscriminator should want this to be big (meaning likely real).
  # Compare this to a vector of ones, penalize if different from one.
  real_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels = tf.ones_like(mdisc_real_output), logits = mdisc_real_output_masked1)
  
  # Create an image that has the generated mdisc values within the mask, 0 otherwise
  mdisc_generated_output_masked0 = tf.where(tf.greater_equal(mask_resized,0.5), mdisc_generated_output, tf.zeros_like(mdisc_generated_output))

  # disc_generated_output holds the output that the discriminator gave for generated images. Maskdiscriminator should want this to be small (meaning likely generated).
  # Compare this to a vector of zeros, penalize if different from zero.
  generated_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels = tf.zeros_like(mdisc_generated_output), logits = mdisc_generated_output_masked0)

  total_mdisc_loss = real_loss + generated_loss

  return total_mdisc_loss


print ('***\nDefining generator_loss function\n***\n')

# Create the loss function for the U-Net generator.
def generator_loss(disc_generated_output, maskDisc_generated_output, gen_output, mese_im, mask):
  # disc_generated_output holds the output that the discriminator gave for generated images. Generator should want this to be big (meaning likely real).
  gan_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels = tf.ones_like(disc_generated_output),logits = disc_generated_output) 

  # We resize the mask to be the same size as the discriminator output.
  mask_resized = tf.image.resize_images(mask, [tf.shape(maskDisc_generated_output)[1], tf.shape(maskDisc_generated_output)[2]], align_corners=True, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  # Create an image that has the real mdisc values within the mask, 1 otherwise
  maskDisc_generated_output_masked1 = tf.where(tf.greater_equal(mask_resized,0.5), maskDisc_generated_output, tf.ones_like(maskDisc_generated_output))

  # maskDisc_generated_output holds the output that the mask discriminator gave for generated images. Generator should want this to be big (meaning likely real).
  maskGan_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels = tf.ones_like(maskDisc_generated_output_masked1),logits = maskDisc_generated_output_masked1)
 
  # Compute the L1 loss - the mean absolute difference between the MESE map and the synthesized CNN map
  l1_loss = tf.reduce_mean(tf.abs(mese_im - gen_output))

  # Compute the L1 loss - the mean absolute difference between the MESE map and the synthesized CNN map over the mask. The mask goes from 0 to 1.
  l1_loss_m = tf.divide(tf.reduce_sum(tf.abs(tf.multiply(mese_im - gen_output, mask))), tf.reduce_sum(mask))

  # Add up all the different losses with chosen weightings.
  total_gen_loss = gan_loss + (LAMBDA_MGAN * maskGan_loss) + (LAMBDA_WL1 * l1_loss) + (LAMBDA_ML1 * l1_loss_m)

  return total_gen_loss, gan_loss, maskGan_loss, l1_loss, l1_loss_m


# We use the Adam optimizer for the U-Net generator and both the discriminators.
generator_optimizer = tf.train.AdamOptimizer(2e-4, beta1=0.5)
discriminator_optimizer = tf.train.AdamOptimizer(2e-4, beta1=0.5)
maskDiscriminator_optimizer = tf.train.AdamOptimizer(2e-4, beta1=0.5)

# Set the directory that saves the network weights after each iteration
checkpoint_dir  = ''        # Insert checkpoint directory name here in the form of /path/to/directory
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
# Specify what is included in the checkpoint
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 maskDiscriminator_optimizer=maskDiscriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator,
                                 maskDiscriminator=maskDiscriminator)

# You can start training from an older checkpoint (useful for when you split up the training time into more than one time period).
# Specify the checkpoint directory for that older checkpoint - if same as current checkpoint directory, then a new training will
# begin from scratch.
checkpoint_dir_prev = checkpoint_dir



print ('***\nDefining generate_images function\n***\n')

def generate_images(model, input_imag, mese_imag, mask_imag, testTrainVal):
  global sosArray
  # From the Tensorflow Pix2Pix tutorial:
  # The training=True is intentional here since we want the batch statistics while running the model on the test dataset. 
  # If we use training=False, we will get the accumulated statistics learned from the training dataset (which we don't want)
  prediction = model(input_imag, training=True)

  # Define the figure which will contain the results. Will contain a 2x2 figure matrix.
  plt.figure(figsize=(2,2), dpi=256, frameon=False)

  # Split input into tensors with size 1 along dimension 3
  DESSimage0, DESSimage1, DESSimage2, TSEsagimage0, TSEsagimage1, TSEsagimage2, FLASHimage0, FLASHimage1, FLASHimage2, TSEcorimage0, TSEcorimage1, TSEcorimage2 = tf.split(input_imag, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 3)

  # Create the sum-of-square difference between the true image and the synthesized image.
  sumsquarediff = tf.reduce_sum(tf.math.squared_difference(tf.squeeze(mese_imag[0]), tf.squeeze(prediction[0])))

  # Write the sum-of-square difference into the 'sosArray'
  ssdvalue = tf.keras.backend.get_value(sumsquarediff)
  sosArray = np.append(sosArray, ssdvalue)

  # We default the number of images to print as the whole batch. If we are training, we override this as 1,
  # as it would otherwise be a lot of images.
  tgt_shape = mese_imag.get_shape()
  imsToPrint = tgt_shape[0]
 
  # Check cases for the variable 'testTrainVal', determining if we are running on the training, testing, or validation set. 
  if (testTrainVal == 1):    # This means that we are running on the training set
    resultDir =       # Fill in with a string containing the directory where the images from training are put
    # Write 1 image in training, but do whole batch for testing and validation
    imsToPrint = 1
  elif (testTrainVal == 0):  # This means that we are running on the testing set
    resultDir =       # Fill in with a string containing the directory where the images from testing are put
  else:                      # We are not doing training or testing so we assume we are doing validation
    resultDir =       # Fill in with a string containing the directory where the images from validation are put

  # We will enumarate the resulting images sequentially. For this, we need to know how many images have already been produced
  # and store in the training/testing/validation directory.
  path, dirs, files = next(os.walk(resultDir))
  file_count = len(files)

  # Specify what goes into the output image (for the training, this is just the first image in the batch, for the others
  # it is the whole batch).
  for indInBatch in range(imsToPrint): 
    # We want to display true T2, DESS, predicted T2, mask
    # The output image will be a 2x2 image matrix as follows:
    # (0,0): The MESE image
    # (0,1): The DESS image
    # (1,0): The synthesized CNN image
    # (1,1): The cartilage mask
    topImag = np.concatenate((tf.squeeze(mese_imag[indInBatch])*0.5+0.5, tf.squeeze(DESSimage1[indInBatch])*0.5+0.5), axis=1)
    bottomImag = np.concatenate((tf.squeeze(prediction[indInBatch])*0.5+0.5, tf.squeeze(mask_imag[indInBatch])), axis=1)
    display_imag = np.concatenate((topImag,bottomImag), axis=0)
    display_imag = display_imag * 255
    
    new_im = Image.fromarray(display_imag)
    new_im = new_im.convert("L")
    # The resulting images will be given names such as "resultfig00123.png" 
    resultImNameTest = resultDir + "/resultfig_" + str(format(int(float(file_count) + indInBatch + 1), '05d')) + ".png"
    new_im.save(resultImNameTest)
  


print ('***\nDefining train function\n***\n')

# Define how the network is trained, writing various performance metrics into respective arrays that
# later get written into text files.
def train(dataset, epochs):
  global dlossArray
  global mdlossArray
  global glossArray
  global glossGANArray
  global glossMGANArray
  global wholeL1array
  global maskL1array

  # Loop through the epochs. For each epoch, we measure the error metrics again so we start by resetting them.
  for epoch in range(epochs):
    start = time.time()
    disc_loss_sum = 0
    mdisc_loss_sum = 0
    gen_loss_sum = 0
    gen_lossGAN_sum = 0
    gen_lossMGAN_sum = 0
    gen_lossL1_whole_sum = 0
    gen_lossL1_mask_sum = 0
    epoch_cycles = 0

    # For each image, loop through the images in the data set, looking at individual batches.
    for combAnat_im, mese_im, mask in dataset:
      
      with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape, tf.GradientTape() as maskDisc_tape:

        # Generate an output CNN T2 map based on the input batch.
        gen_output = generator(combAnat_im, training=True)

        # Create an output for the "full image" discriminator for the true MESE map.
        disc_real_output = discriminator(combAnat_im, mese_im, training=True)

        # Create an output for the "full image" discriminator for the generated CNN map.
        disc_generated_output = discriminator(combAnat_im, gen_output, training=True)
       
        # Create masked version of the input by multiplying it with the cartilage mask. 
        masked_input_image = tf.multiply(combAnat_im, mask)
        # Create masked version of the true MESE map  by multiplying it with the cartilage mask. 
        masked_mese_im = tf.multiply(mese_im, mask)
        # Create masked version of the synthesized CNN map  by multiplying it with the cartilage mask. 
        masked_gen_output = tf.multiply(gen_output, mask)

        # Create an output for the mask discriminator for the masked MESE map.
        maskDisc_real_output = maskDiscriminator(masked_input_image, masked_mese_im, training=True)
        # Create an output for the mask discriminator for the masked CNN map.
        maskDisc_generated_output = maskDiscriminator(masked_input_image, masked_gen_output, training=True)

        # Compute the loss for the U-Net generator based on the generated CNN map and the outputs from the two discriminators.
        # Only the first term is used for updating the generator, the other parameters are used during tuning.
        gen_loss, gen_lossGAN, gen_lossMGAN, gen_lossL1_whole, gen_lossL1_mask  = generator_loss(disc_generated_output, maskDisc_generated_output, gen_output, mese_im, mask)
        # Compute the "whole image" discriminator loss based its performance for the true MESE map and the synthesized CNN map.
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)
        # Compute the mask discriminator loss based its performance for the masked true MESE map and the masked synthesized CNN map.
        maskDisc_loss = maskdiscriminator_loss(maskDisc_real_output, maskDisc_generated_output, mask)


      # Compute gradients for the three networks and apply them to their respective optimizers.
      generator_gradients = gen_tape.gradient(gen_loss, generator.variables)
      discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.variables)
      maskDiscriminator_gradients = maskDisc_tape.gradient(maskDisc_loss, maskDiscriminator.variables)

      generator_optimizer.apply_gradients(zip(generator_gradients, generator.variables))
      discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.variables))
      maskDiscriminator_optimizer.apply_gradients(zip(maskDiscriminator_gradients, maskDiscriminator.variables))

      # Add the performance metrics to the cumulative values from past batches.
      dloss = tf.keras.backend.get_value(disc_loss)
      disc_loss_sum += dloss
      mdloss = tf.keras.backend.get_value(maskDisc_loss)
      mdisc_loss_sum += mdloss
      gloss = tf.keras.backend.get_value(gen_loss)
      gen_loss_sum += gloss
      glossGAN = tf.keras.backend.get_value(gen_lossGAN)
      gen_lossGAN_sum += glossGAN
      glossMGAN = tf.keras.backend.get_value(gen_lossMGAN)
      gen_lossMGAN_sum += glossMGAN
      glossL1_whole = tf.keras.backend.get_value(gen_lossL1_whole)
      gen_lossL1_whole_sum += glossL1_whole
      glossL1_mask = tf.keras.backend.get_value(gen_lossL1_mask)
      gen_lossL1_mask_sum += glossL1_mask
      epoch_cycles += 1

    # We have finished looping over the batches for this epoch.
    # Save average performance metrics over this epoch in their respective data arrays. Will be written into text files when the program stops.
    if epoch % 1 == 0:
        clear_output(wait=True)
        dlossArray = np.append(dlossArray, disc_loss_sum/epoch_cycles)
        mdlossArray = np.append(mdlossArray, mdisc_loss_sum/epoch_cycles)
        glossArray = np.append(glossArray, gen_loss_sum/epoch_cycles)
        glossGANArray = np.append(glossGANArray, gen_lossGAN_sum/epoch_cycles)
        glossMGANArray = np.append(glossMGANArray, gen_lossMGAN_sum/epoch_cycles)
        wholeL1array = np.append(wholeL1array, gen_lossL1_whole_sum/epoch_cycles)
        maskL1array = np.append(maskL1array, gen_lossL1_mask_sum/epoch_cycles)
        # Take a random entry from the test dataset and generate a prediction with the network.
        for combIn, mese, mask in test_dataset.take(1):
          # The final argument "1" tells the program to write the image in the "train" output directory
          generate_images(generator, combIn, mese, mask, 1)

    # Save the weights after this epoch in the checkpoint.          
    checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1, time.time()-start))
    print ('Time at end of epoch {} is {}\n'.format(epoch + 1, time.asctime(time.localtime(time.time()))))


######################################################
print ('***\nRunning train function\n***\n')

# If an older checkpoint exists in the directory stored in 'checkpoint_dir_prev', we load it and train from there.
# Otherwise, we train from scratch.
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir_prev))
print ('Loaded previous checkpoint (if exists), time is {}\n'.format(time.asctime(time.localtime(time.time()))))

# Perform training on the training data set.
train(train_dataset, EPOCHS)
print ('Finished running training, time is {}\n'.format(time.asctime(time.localtime(time.time()))))

# Restore the latest checkpoint in checkpoint_dir
print ('***\nRunning checkpoint_restore\n***\n')
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
print ('Finished running checkpoint_restore, time is {}\n'.format(time.asctime(time.localtime(time.time()))))

# Run the trained model on the validation dataset
print ('***\nRunning the trained model on the val dataset.\n***\n')
for combAnat_imag, mese_imag, mask_imag in val_dataset:
  generate_images(generator, combAnat_imag, mese_imag, mask_imag, 2)
print ('Finished running the trained model on the val dataset, time is {}\n'.format(time.asctime(time.localtime(time.time()))))

# Run the trained model on the test dataset
print ('***\nRunning the trained model on the test dataset.\n***\n')
for combAnat_imag, mese_imag, mask_imag in test_dataset:
  generate_images(generator, combAnat_imag, mese_imag, mask_imag, 0)
print ('Finished running the trained model on the test dataset, time is {}\n'.format(time.asctime(time.localtime(time.time()))))

# We are done. Save the performance metrics in their respective text files.        
np.save(sosFile, sosArray)
np.save(dlossFile, dlossArray)
np.save(mdlossFile, mdlossArray)
np.save(glossFile, glossArray)
np.save(glossGANFile, glossGANArray)
np.save(glossMGANFile, glossMGANArray)
np.save(maskL1file, maskL1array)
np.save(wholeL1file, wholeL1array)
