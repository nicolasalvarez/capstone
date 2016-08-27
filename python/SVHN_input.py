"""
Routine for decoding the SVHN binary file format.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from six.moves import xrange
import tensorflow as tf
from scipy.io import loadmat
import numpy as np
from random import shuffle

# Process images of this size. Note that this differs from the original SVHN
# image size of 32 x 32. If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.
IMAGE_SIZE = 24

# Global constants describing the SVHN data set.
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 73257
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 26032
NO_BATCH_FILES = 8

def convert_dataset(data_dir, batch_dir):
    """
    Converts the dataset from .mat files to binary files called data_batch_1.bin, data_batch_2.bin, ...,
    as well as test_batch.bin. Binary files are written in "data_dir/batch_dir" directory.

    Each of these files is formatted as follows:
    <1 x label><3072 x pixel>
    ...
    <1 x label><3072 x pixel>

    The first byte is the label of the first image, which is a number in the range 0-9. The next 3072 bytes are the
    values of the pixels of the image. The first 1024 bytes are the red channel values, the next 1024 the green,
    and the final 1024 the blue. The values are stored in row-major order, so the first 32 bytes are the red channel
    values of the first row of the image.

    :param data_dir: Path to the SVHN data directory.
    :param batch_dir: Batch data directory name.
    :return: None
    """

    # Maximum batch file size.
    batch_file_size = 30

    # Load train and test dataset from mat files
    train = loadmat(os.path.join(data_dir, 'train_32x32.mat'))
    test = loadmat(os.path.join(data_dir, 'test_32x32.mat'))

    print("Keys: Train =",train.keys(), "Test =", test.keys())
    print("X Shape: Train =",train['X'].shape, "Test =", test['X'].shape)
    print("Y Shape: Train =",train['y'].shape, "Test =", test['y'].shape)

    # Total examples in datasets
    no_train_examples = train['X'].shape[3]
    no_test_examples = test['X'].shape[3]

    X_train = train['X']
    y_train = train['y']
    X_test = test['X']
    y_test = test['y']

    batch_directory = os.path.join(data_dir, batch_dir)

    # Save train data to bin file
    idx = 1
    batch_f = open(os.path.join(batch_directory, 'data_batch_%d.bin' % idx), "wb")
    for sample in range(no_train_examples):
        if os.path.getsize(batch_f.name) > batch_file_size * (1024 ** 2):
            batch_f.close()
            idx += 1
            batch_f = open(os.path.join(batch_directory, 'data_batch_%d.bin' % idx), "wb")
        batch_f.write(np.array(y_train[sample] % 10).tobytes())
        batch_f.write(np.array(X_train[:, :, :, sample]).flatten().tobytes())
    batch_f.close()

    # Save test data to bin file
    batch_f = open(os.path.join(batch_directory, 'test_batch.bin'), "wb")
    for sample in range(no_test_examples):
        batch_f.write(np.array(y_test[sample] % 10).tobytes())
        batch_f.write(np.array(X_test[:, :, :, sample]).flatten().tobytes())
    batch_f.close()


def read_SVHN(filename_queue):
    """Reads and parses examples from SVHN data files.

    Recommendation: if you want N-way read parallelism, call this function
    N times.  This will give you N independent Readers reading different
    files & positions within those files, which will give better mixing of
    examples.

    Args:
      filename_queue: A queue of strings with the filenames to read from.

    Returns:
      A SVHNRecord object representing a single example.
    """

    class SVHNRecord(object):
        """
        Class that represent a single example, with the following fields:
        height: number of rows in the result (32)
        width: number of columns in the result (32)
        depth: number of color channels in the result (3)
        key: a scalar string Tensor describing the filename & record number
          for this example.
        label: an int32 Tensor with the label in the range 0..9.
        uint8image: a [height, width, depth] uint8 Tensor with the image data
        """
        pass
    result = SVHNRecord()

    # Dimensions of the images in the SVHN dataset.
    # See http://ufldl.stanford.edu/housenumbers/ for a description of the input format.
    label_bytes = 1
    result.height = 32
    result.width = 32
    result.depth = 3
    image_bytes = result.height * result.width * result.depth
    # Every record consists of a label followed by the image, with a
    # fixed number of bytes for each.
    record_bytes = label_bytes + image_bytes

    # Read a record, getting filenames from the filename_queue.  No
    # header or footer in the SVHN format, so we leave header_bytes
    # and footer_bytes at their default of 0.
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key, value = reader.read(filename_queue)

    # Convert from a string to a vector of uint8 that is record_bytes long.
    record_bytes = tf.decode_raw(value, tf.uint8)

    # The first bytes represent the label, which we convert from uint8->int32.
    result.label = tf.cast(tf.slice(record_bytes, [0], [label_bytes]), tf.int32)

    # The remaining bytes after the label represent the image, which we reshape
    # from [depth * height * width] to [height, width, depth].
    result.uint8image = tf.reshape(tf.slice(record_bytes, [label_bytes], [image_bytes]),
                                   [result.height, result.width, result.depth])

    return result


def _generate_image_and_label_batch(image, label, min_queue_examples, batch_size, shuffle):
    """Construct a queued batch of images and labels.

    Args:
      image: 3-D Tensor of [height, width, 3] of type.float32.
      label: 1-D Tensor of type.int32
      min_queue_examples: int32, minimum number of samples to retain
        in the queue that provides of batches of examples.
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
    """Construct distorted input for SVHN training using the Reader ops.

    Args:
      data_dir: Path to the SVHN data directory.
      batch_size: Number of images per batch.

    Returns:
      images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.
    """
    filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i) for i in xrange(1, NO_BATCH_FILES)]
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    # Create a queue that produces the filenames to read.
    filename_queue = tf.train.string_input_producer(filenames)

    # Read examples from files in the filename queue.
    read_input = read_SVHN(filename_queue)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    # Image processing for training the network. Note the many random distortions applied to the image.

    # Randomly crop a [height, width] section of the image.
    distorted_image = tf.random_crop(reshaped_image, [height, width, 3])

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
    shuffle(ops_list)
    for op in ops_list:
        distorted_image = op(distorted_image)

    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_whitening(distorted_image)

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * min_fraction_of_examples_in_queue)
    print ('Filling queue with %d SVHN images before starting to train. '
           'This will take a few minutes.' % min_queue_examples)

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(float_image, read_input.label, min_queue_examples, batch_size, shuffle=True)


def inputs(eval_data, data_dir, batch_size):
    """Construct input for SVHN evaluation using the Reader ops.

    Args:
      eval_data: bool, indicating if one should use the train (False) or eval (True) data set.
      data_dir: Path to the SVHN data directory.
      batch_size: Number of images per batch.

    Returns:
      images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.
    """
    if not eval_data:
        filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i) for i in xrange(1, NO_BATCH_FILES)]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    else:
        filenames = [os.path.join(data_dir, 'test_batch.bin')]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    # Create a queue that produces the filenames to read.
    filename_queue = tf.train.string_input_producer(filenames)

    # Read examples from files in the filename queue.
    read_input = read_SVHN(filename_queue)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    # Image processing for evaluation.
    # Crop the central [height, width] of the image.
    resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, width, height)

    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_whitening(resized_image)

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoch * min_fraction_of_examples_in_queue)

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(float_image, read_input.label, min_queue_examples, batch_size, shuffle=False)
