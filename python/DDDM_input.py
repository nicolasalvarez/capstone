# coding=utf-8

"""
Routine for convert and decode the DDDM dataset.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import numpy as np
import pandas as pd
import sklearn.utils
import random
from six.moves import xrange
from PIL import Image

# Original image size
ORIG_WIDTH = 640
ORIG_HEIGHT = 480

# The input layer (that contains the image) should be divisible by 2 many times. Common numbers include 32, 64, 96, ...
# Possible (HEIGHT, WIDTH): (48, 64) or (72, 96)
HEIGHT = 48
WIDTH = 64
IMG_SIZE = WIDTH  # Final images must be squares.
DEPTH = 3

# Fraction of dataset for validation
VAL_SET_FRACTION = 0.2

# Global constants describing the DDDM data set.
NUM_CLASSES = 10
TOTAL_EXAMPLES = 22423
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = int(TOTAL_EXAMPLES * VAL_SET_FRACTION)
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = TOTAL_EXAMPLES - NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
NUM_EXAMPLES_FOR_TEST = 79726
NO_TRAIN_BATCH_FILES = 6
NO_VAL_BATCH_FILES = 2


def resize_img(img_path, width, height):
    """
    Resize image to new_width and new_height using Bicubic algorithm.
    :param img_path: path to the image to resize
    :param width: width
    :param height: height
    :raise InvalidImageSize if image size is not equal to (ORIG_WIDTH, ORIG_HEIGHT)
    :return: image resized
    """
    img = Image.open(img_path)
    # Resize original images. Discard images with wrong size.
    if img.size != (ORIG_WIDTH, ORIG_HEIGHT):
        raise ValueError('InvalidImageSize')
    return img.resize((width, height), resample=Image.BICUBIC)


def convert_dataset(data_dir, batch_dir):
    """
    Converts the dataset from original files to binary files called data_batch_1.bin, data_batch_2.bin, ...,
    as well as val_batch.bin. Binary files are written in "data_dir/batch_dir" directory.

    Each of these files is formatted as follows:
    <1 x label><height*width*depth x pixel>
    ...
    <1 x label><height*width*depth x pixel>

    The first byte is the label of the first image, which is a number in the range 0-9. The next height*width*depth
    bytes are the values of the pixels of the image. The values are stored in row-major order.

    The file test_batch.bin does not have label. Instead, it has a image number of 32 bits.

    :param data_dir: Path to the DDDM data directory.
    :param batch_dir: Batch data directory name.
    :return: None
    """

    # Maximum batch file size in MB.
    batch_file_size = 30

    batch_directory = os.path.join(data_dir, batch_dir)
    file_idx = 1
    base_name = 'data'
    batch_f = open(os.path.join(batch_directory, base_name + '_batch_%d.bin' % file_idx), "wb")

    # Convert original images to binary batch files
    print('Converting training and validation datasets to binary files.')

    # Shuffle dataset before convert it to binary file
    train_df = pd.read_csv(os.path.join(data_dir, 'driver_imgs_list.csv'))
    train_df = sklearn.utils.shuffle(train_df).reset_index()
    for img_idx, row in train_df.iterrows():
        # Resize image
        try:
            img = resize_img(os.path.join(data_dir, 'imgs/train/' + row['classname'] + '/' + row['img']), WIDTH, HEIGHT)
        except ValueError as x:
            print(x, row['classname'] + '/' + row['img'])
            continue

        if img_idx > len(train_df.index) * (1 - VAL_SET_FRACTION) and base_name != 'val':
            batch_f.close()
            file_idx = 1
            base_name = 'val'
            batch_f = open(os.path.join(batch_directory, base_name + '_batch_%d.bin' % file_idx), "wb")
        # Save images to batch file with format <1 x label><Height*Width*Depth x pixel>
        elif os.path.getsize(batch_f.name) > batch_file_size * (1024 ** 2):
            batch_f.close()
            file_idx += 1
            batch_f = open(os.path.join(batch_directory, base_name + '_batch_%d.bin' % file_idx), "wb")
        # Write image label and image
        batch_f.write(np.array(int(row['classname'][-1]), dtype=np.uint8).tobytes())
        batch_f.write(np.array(img).flatten().tobytes())
    batch_f.close()
    print('Done.')

    print('Converting testing datasets to binary files.')
    batch_f = open(os.path.join(batch_directory, 'test_batch.bin'), "wb")
    test_img_path = os.path.join(data_dir, 'imgs/test')

    for img_file in sorted(os.listdir(test_img_path)):
        # Resize image
        try:
            img = resize_img(os.path.join(os.path.join(test_img_path, img_file)), WIDTH, HEIGHT)
        except ValueError as x:
            print(x, img_file)
            continue
        # Write image number and image
        batch_f.write(np.array(int(img_file[4:-4]), dtype=np.int32).tobytes())
        batch_f.write(np.array(img).flatten().tobytes())
    batch_f.close()
    print('Done.')


def read_DDDM_img(filename_queue):
    """Reads and parses examples from DDDM data files.

    Args:
      filename_queue: A queue of strings with the filenames to read from.

    Returns:
      A DDDMRecord object representing a single example.
    """

    class DDDMRecord(object):
        """
        Class that represent a single example, with the following fields:
        height: number of rows in the result
        width: number of columns in the result
        depth: number of color channels in the result
        key: a scalar string Tensor describing the filename & record number
          for this example.
        label: an int32 Tensor with the label in the range 0..9.
        uint8image: a [height, width, depth] uint8 Tensor with the image data
        """
        pass
    result = DDDMRecord()

    # Dimensions of the images in the DDDM dataset.
    label_bytes = 1
    result.height = HEIGHT
    result.width = WIDTH
    result.depth = DEPTH

    image_bytes = result.height * result.width * result.depth
    # Every record consists of a label followed by the image, with a fixed number of bytes for each.
    record_bytes = label_bytes + image_bytes

    # Read a record, getting filenames from the filename_queue.
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key, value = reader.read(filename_queue)

    # Convert from a string to a vector of uint8 that is record_bytes long.
    record_bytes = tf.decode_raw(value, tf.uint8)

    # The first bytes represent the label, which we convert from uint8->int32.
    result.label = tf.cast(tf.slice(record_bytes, [0], [label_bytes]), tf.int32)

    # The remaining bytes after the label represent the image, which we reshape
    # from [height * width * depth] to [height, width, depth].
    result.uint8image = tf.reshape(tf.slice(record_bytes, [label_bytes], [image_bytes]),
                                   [result.height, result.width, result.depth])

    return result


def _generate_image_and_label_batch(image, label, min_queue_examples, batch_size, shuffle):
    """Construct a queued batch of images and labels.

    Args:
      image: 3-D Tensor of [height, width, 3] of type.float32.
      label: 1-D Tensor of type.int32
      min_queue_examples: int32, minimum number of samples to retain in the queue that provides of batches of examples.
      batch_size: Number of images per batch.
      shuffle: boolean indicating whether to use a shuffling queue.

    Returns:
      images: Images. 4D tensor of [batch_size, height, width, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.
    """
    # Create a queue that shuffles the examples, and then
    # read 'batch_size' images + labels from the example queue.
    num_preprocess_threads = 16
    if shuffle:
        images, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        images, label_batch = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size)

    # Display the training images in the visualizer.
    tf.image_summary('images', images)

    return images, tf.reshape(label_batch, [batch_size])


def distorted_inputs(data_dir, batch_size):
    """Construct distorted input for DDDM training using the Reader ops.

    Args:
      data_dir: Path to the DDDM data directory.
      batch_size: Number of images per batch.

    Returns:
      images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.
    """
    filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i) for i in xrange(1, NO_TRAIN_BATCH_FILES)]
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    # Create a queue that produces the filenames to read.
    filename_queue = tf.train.string_input_producer(filenames)

    # Read examples from files in the filename queue.
    read_input = read_DDDM_img(filename_queue)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)

    # Image processing for training the network. Note the many random distortions applied to the image.

    # Pad and randomly crop a width section of the image.
    reshaped_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, IMG_SIZE, int(IMG_SIZE * 1.2))
    distorted_image = tf.random_crop(reshaped_image, [IMG_SIZE, IMG_SIZE, 3])

    def rand_bright(img):
        """
        Randomly adjust the brightness of image.
        :param img: image
        :return:
        """
        return tf.image.random_brightness(img, max_delta=63)

    def rand_contr(img):
        """
        Randomly adjust the contrast of image.
        :param img: image
        :return:
        """
        return tf.image.random_contrast(img, lower=0.2, upper=1.8)

    def rand_sat(img):
        """
        Randomly adjust the saturation of image.
        :param img: image
        :return:
        """
        return tf.image.random_saturation(img, lower=0.2, upper=1.8)

    def rand_hue(img):
        """
        Randomly adjust the hue of image.
        :param img: image
        :return:
        """
        return tf.image.random_hue(img, max_delta=0.5)

    # Randomize the order of not commutative operations
    ops_list = [rand_bright, rand_contr, rand_sat, rand_hue]
    random.shuffle(ops_list)
    for op in ops_list:
        distorted_image = op(distorted_image)

    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_whitening(distorted_image)

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * min_fraction_of_examples_in_queue)
    print ('Filling queue with %d DDDM images before starting to train.' % min_queue_examples)

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(float_image, read_input.label, min_queue_examples, batch_size, shuffle=True)


def inputs(eval_data, data_dir, batch_size):
    """Construct input for DDDM evaluation using the Reader ops.

    Args:
      eval_data: bool, indicating if one should use the train (False) or eval (True) data set.
      data_dir: Path to the DDDM data directory.
      batch_size: Number of images per batch.

    Returns:
      images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.
    """
    if not eval_data:
        filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i) for i in xrange(1, NO_TRAIN_BATCH_FILES)]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    else:
        filenames = [os.path.join(data_dir, 'val_batch_%d.bin' % i) for i in xrange(1, NO_VAL_BATCH_FILES)]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    # Create a queue that produces the filenames to read.
    filename_queue = tf.train.string_input_producer(filenames)

    # Read examples from files in the filename queue.
    read_input = read_DDDM_img(filename_queue)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)

    # Image processing for evaluation.
    # Pad the height of the image to make it square.
    resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, IMG_SIZE, IMG_SIZE)

    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_whitening(resized_image)

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoch * min_fraction_of_examples_in_queue)

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(float_image, read_input.label, min_queue_examples, batch_size, shuffle=False)
