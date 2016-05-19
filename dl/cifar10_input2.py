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

"""Routine for decoding the CIFAR-10 binary file format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

# Process images of this size. Note that this differs from the original CIFAR
# image size of 32 x 32. If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.
IMAGE_SIZE = 208
ORIGINAL_WIDTH = 640
ORIGINAL_HEIGHT = 480

# Global constants describing the CIFAR-10 data set.
# NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000
#NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 5876
#NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 5876
NUM_CLASSES = 7
#NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 5000
#NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 1000


# Games
LABEL_PATH = "/home/neo/projects/deepLearning/data/label/"
IMAGE_PATH = "/home/neo/projects/deepLearning/data/image/"
LABEL_SUFIX = "_emotion"
LABEL_FORMAT = "*.txt"
IMAGE_FORMAT = "*.png"
LABEL = []

def generate_image_dict(path):
  image = []
  for root, dirs, files in os.walk(path, True):
    for name in files:
      image.append(os.path.join(root, name))
  return image

def search_label(filename):
  for lab in LABEL:
    if lab[0][:8] == filename[:8]:
      return lab[1]

def rename_image_filename(image):
  with tf.Session() as ppro:
    # Initialize the variables define ("Read more about it !!")
    tf.initialize_all_variables().run()
    # Start to populate the label queue
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    # Execute the image section of the graph
    imageTensor = ppro.run([image])
    key = imageTensor[0]
    # Shutdown the queue coordinator.
    coord.request_stop()
    coord.join(threads)
    return key[len(IMAGE_PATH):-len(IMAGE_FORMAT)+1]
    print key[len(IMAGE_PATH):-len(IMAGE_FORMAT)+1]

def rename_label_filename(label):
  return label[:-(len(LABEL_SUFIX)+len(LABEL_FORMAT)-1)]

def generate_label_dict(path):
  for root, dirs, files in os.walk(path, True):
    for name in files:
      f = open(os.path.join(root, name), 'r')
      LABEL.append([rename_label_filename(name), int(float(f.read()))])
  return LABEL

def read_cifar10(filename_queue):
  # Read the images and generate the decode from PNG image
  imageReader = tf.WholeFileReader()
  image_key, image_value = imageReader.read(filename_queue)
  image_decode = tf.image.decode_png(image_value, channels=1)
  image_decode = tf.cast(image_decode, tf.float32)
  # Preprocess data
  image_key = rename_image_filename(image_key)    # rename image filename 
  #label = search_label(image_key)
  #label = 1
  #label = random.choice([1, 2, 3, 4, 5, 6, 7])
  label = random.choice([1, 2, 3, 4])
  # CREATE OBJECT
  class Record(object):
    pass
  record = Record()
  # Instantiate object
  record.key = image_key
  record.label = tf.cast(label, tf.int32)
  record.image = image_decode
  #with tf.Session() as ppro:
  #  result = ppro.run([record.label])
  #  print(result)
  return record

def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size):
  """Construct a queued batch of images and labels.

  Args:
    image: 3-D Tensor of [height, width, 3] of type.float32.
    label: 1-D Tensor of type.int32
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, height, width, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  # Create a queue that shuffles the examples, and then
  # read 'batch_size' images + labels from the example queue.
  num_preprocess_threads = 16
  images, label_batch = tf.train.shuffle_batch(
      [image, label],
      batch_size=batch_size,
      num_threads=num_preprocess_threads,
      capacity=min_queue_examples + 3 * batch_size,
      min_after_dequeue=min_queue_examples)

  # Display the training images in the visualizer.
  tf.image_summary('images', images, max_images=10)

  return images, tf.reshape(label_batch, [batch_size])


def distorted_inputs(data_dir, batch_size):
  """Construct distorted input for CIFAR training using the Reader ops.

  Args:
    data_dir: Path to the CIFAR-10 data directory.
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  # Create a queue that produces the filenames to read.
  # filename_queue = tf.train.string_input_producer(filenames)
  generate_label_dict(LABEL_PATH)
  # filename_matching = tf.train.match_filenames_once(os.path.join(IMAGE_PATH, IMAGE_FORMAT))
  filename_queue = tf.train.string_input_producer(generate_image_dict(IMAGE_PATH))

  # Read examples from files in the filename queue.
  read_input = read_cifar10(filename_queue)
  # reshaped_image = tf.cast(read_input.uint8image, tf.float32)
  reshaped_image = read_input.image
  height = IMAGE_SIZE
  width = IMAGE_SIZE
  # Image processing for training the network. Note the many random
  # distortions applied to the image.
  reshaped_image.set_shape([ORIGINAL_HEIGHT, ORIGINAL_WIDTH, 1])
  distorted_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, width, height)
  # Randomly crop a [height, width] section of the image.
  # Randomly flip the image horizontally.
  distorted_image = tf.image.random_flip_left_right(distorted_image)
  # Because these operations are not commutative, consider randomizing
  # randomize the order their operation.
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
  print ('Filling queue with %d CIFAR images before starting to train. '
         'This will take a few minutes.' % min_queue_examples)
  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(float_image, read_input.label,
                                         min_queue_examples, batch_size)

def inputs(eval_data, data_dir, batch_size):
  """Construct input for CIFAR evaluation using the Reader ops.
  Args:
    eval_data: bool, indicating if one should use the train or eval data set.
    data_dir: Path to the CIFAR-10 data directory.
    batch_size: Number of images per batch.
  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
  # Create a queue that produces the filenames to read.
  generate_label_dict(LABEL_PATH)
  filename_queue = tf.train.string_input_producer(generate_image_dict(IMAGE_PATH))
  # Read examples from files in the filename queue.
  read_input = read_cifar10(filename_queue)
  reshaped_image = read_input.image
  height = IMAGE_SIZE
  width = IMAGE_SIZE
  # Image processing for evaluation.
  # Crop the central [height, width] of the image.
  reshaped_image.set_shape([ORIGINAL_HEIGHT, ORIGINAL_WIDTH, 1])
  resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, width, height)
  # Subtract off the mean and divide by the variance of the pixels.
  float_image = tf.image.per_image_whitening(resized_image)
  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  #min_queue_examples = int(num_examples_per_epoch *
  min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                           min_fraction_of_examples_in_queue)
  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(float_image, read_input.label,
                                         min_queue_examples, batch_size)