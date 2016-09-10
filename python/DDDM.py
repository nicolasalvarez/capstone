# coding=utf-8

"""
DDDM ConvNet model.

Summary of available functions:

 # Compute input images and labels for training. If you would like to run
 # evaluations, use inputs() instead.
 inputs, labels = distorted_inputs()

 # Compute inference on the model inputs to make a prediction.
 predictions = inference(inputs)

 # Compute the total loss of the prediction with respect to the labels.
 loss = loss(predictions, labels)

 # Create a graph to run one step of training with respect to the loss.
 train_op = train(loss, global_step)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from python import DDDM_input
from math import sqrt

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 128, """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', '/media/nalvarez/Data/State_Farm_Distracted_Driver_Detection', """Path to
the DDDM data directory.""")
tf.app.flags.DEFINE_string('batch_dir', 'data_batch_files', """Batch datasets directory name.""")

# Global constants describing the DDDM data set.
IMG_SIZE = DDDM_input.IMG_SIZE
HEIGHT = DDDM_input.HEIGHT
WIDTH = DDDM_input.WIDTH
DEPTH = DDDM_input.DEPTH
NUM_CLASSES = DDDM_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = DDDM_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = DDDM_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

# URL and file names for downloading dataset
DATA_URL = 'https://www.kaggle.com/c/state-farm-distracted-driver-detection/download'
DATASET_FILES = ['imgs.zip', 'driver_imgs_list.csv.zip']

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999       # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 10           # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.5    # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1         # Initial learning rate.


def _activation_summary(x):
    """
    Helper to create summaries for activations.

    Creates a summary that provides a histogram of activations.
    Creates a summary that measures the sparsity of activations.

    Args:
      x: Tensor
    Returns:
      nothing
    """
    tf.histogram_summary(x.op.name + '/activations', x)
    tf.scalar_summary(x.op.name + '/sparsity', tf.nn.zero_fraction(x))


def distorted_inputs():
    """
    Construct distorted input for DDDM training using the Reader ops.

    Returns:
      images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.

    Raises:
      ValueError: If no data_dir
    """
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')
    data_dir = os.path.join(FLAGS.data_dir, FLAGS.batch_dir)
    images, labels = DDDM_input.distorted_inputs(data_dir=data_dir, batch_size=FLAGS.batch_size)
    return images, labels


def inputs(eval_data):
    """
    Construct input for DDDM evaluation using the Reader ops.

    Args:
      eval_data: bool, indicating if one should use the train or eval data set.

    Returns:
      images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.

    Raises:
      ValueError: If no data_dir
    """
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')
    data_dir = os.path.join(FLAGS.data_dir, FLAGS.batch_dir)
    images, labels = DDDM_input.inputs(eval_data=eval_data, data_dir=data_dir, batch_size=FLAGS.batch_size)
    return images, labels


def inference(images, dropout_prob):
    """
    Build the DDDM model: INPUT -> [CONV -> RELU -> CONV -> RELU -> POOL]*2 -> [FC ->RELU]*2 -> FC

    Args:
      images: Images returned from distorted_inputs() or inputs().
      dropout_prob: dropout probability

    Returns:
      Logits.
    """
    # conv1
    conv1_filters = 32
    conv1_filter_size = 5
    with tf.variable_scope('conv1') as scope:
        kernel = tf.Variable(tf.truncated_normal([conv1_filter_size, conv1_filter_size, DEPTH, conv1_filters],
                                                 stddev=1.0/sqrt(float(conv1_filter_size ** 2 * DEPTH))),
                             name='weights')
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[conv1_filters]), name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope.name)
        _activation_summary(conv1)

    # conv2
    conv2_filters = 32
    conv2_filter_size = 5
    with tf.variable_scope('conv2') as scope:
        kernel = tf.Variable(tf.truncated_normal([conv2_filter_size, conv2_filter_size, conv1_filters, conv2_filters],
                                                 stddev=1.0/sqrt(float(conv2_filter_size ** 2 * conv1_filters))),
                             name='weights')
        conv = tf.nn.conv2d(conv1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[conv2_filters]), name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope.name)
        _activation_summary(conv2)

    # pool1 64x64 -> 32X32
    pool1 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

    # conv3
    conv3_filters = 32
    conv3_filter_size = 5
    with tf.variable_scope('conv3') as scope:
        kernel = tf.Variable(tf.truncated_normal([conv3_filter_size, conv3_filter_size, conv2_filters, conv3_filters],
                                                 stddev=1.0/sqrt(float(conv3_filter_size ** 2 * conv2_filters))),
                             name='weights')
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[conv3_filters]), name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias, name=scope.name)
        _activation_summary(conv3)

    # conv4
    conv4_filters = 32
    conv4_filter_size = 5
    with tf.variable_scope('conv4') as scope:
        kernel = tf.Variable(tf.truncated_normal([conv4_filter_size, conv4_filter_size, conv3_filters, conv4_filters],
                                                 stddev=1.0/sqrt(float(conv4_filter_size ** 2 * conv3_filters))),
                             name='weights')
        conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[conv4_filters]), name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(bias, name=scope.name)
        _activation_summary(conv4)

    # pool2 32X32 -> 16x16
    pool2 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    # FC1
    fc1_neurons = 192#384
    with tf.variable_scope('FC1') as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        reshape = tf.reshape(pool2, [FLAGS.batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = tf.Variable(tf.truncated_normal([dim, fc1_neurons], stddev=1.0/sqrt(float(dim))), name='weights')
        biases = tf.Variable(tf.constant(0.0, shape=[fc1_neurons]), name='biases')
        fc1 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
        fc1 = tf.nn.dropout(fc1, dropout_prob)
        _activation_summary(fc1)

    # FC2
    fc2_neurons = 96#192
    with tf.variable_scope('FC2') as scope:
        weights = tf.Variable(tf.truncated_normal([fc1_neurons, fc2_neurons], stddev=1.0/sqrt(float(fc1_neurons))),
                              name='weights')
        biases = tf.Variable(tf.constant(0.0, shape=[fc2_neurons]), name='biases')
        fc2 = tf.nn.relu(tf.matmul(fc1, weights) + biases, name=scope.name)
        fc2 = tf.nn.dropout(fc2, dropout_prob)
        _activation_summary(fc2)

    # softmax, i.e. softmax(WX + b)
    with tf.variable_scope('softmax_linear') as scope:
        weights = tf.Variable(tf.truncated_normal([fc2_neurons, NUM_CLASSES], stddev=1.0/sqrt(float(fc2_neurons))),
                              name='weights')
        biases = tf.Variable(tf.constant(0.0, shape=[NUM_CLASSES]), name='biases')
        softmax_linear = tf.add(tf.matmul(fc2, weights), biases, name=scope.name)
        _activation_summary(softmax_linear)

    return softmax_linear


def loss(logits, labels):
    """
    Add L2Loss to all the trainable variables.

    Add summary for "Loss" and "Loss/avg".
    Args:
      logits: Logits from inference().
      labels: Labels from distorted_inputs or inputs(). 1-D tensor
              of shape [batch_size]

    Returns:
      Loss tensor of type float.
    """
    # Calculate the average cross entropy loss across the batch.
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the weight decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
    """
    Add summaries for losses in DDDM model.

    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.

    Args:
      total_loss: Total loss from loss().
    Returns:
      loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summary to all individual losses and the total loss; do the same for the averaged version of
    # the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss as the original loss name.
        tf.scalar_summary(l.op.name + ' (raw)', l)
        tf.scalar_summary(l.op.name, loss_averages.average(l))

    return loss_averages_op


def train(total_loss, global_step):
    """
    Train DDDM model.

    Create an optimizer and apply to all trainable variables. Add moving
    average for all trainable variables.

    Args:
      total_loss: Total loss from loss().
      global_step: Integer Variable counting the number of training steps
        processed.
    Returns:
      train_op: op for training.
    """
    # Variables that affect learning rate.
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE, global_step, decay_steps, LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)
    tf.scalar_summary('learning_rate', lr)

    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(total_loss)

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.histogram_summary(var.op.name, var)

    # Add histograms for gradients.
    for grad, var in grads:
        if grad is not None:
            tf.histogram_summary(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op


def check_and_maybe_convert_dataset():
    """
    Check if dataset files are present at directory "data_dir" and convert original dataset to binary if directory
    "batch_dir" does not exist.
    """
    dest_directory = FLAGS.data_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)

    # Check if dataset files are present
    for ds_file in DATASET_FILES:
        if not os.path.exists(os.path.join(dest_directory, ds_file)[:-4]):
            print('Please download and unzip the file', DATA_URL+'/'+ds_file, 'to', dest_directory)
            raise ValueError('DatasetFilesNotFound')

    # Convert original dataset to binary if required
    batch_directory = os.path.join(dest_directory, FLAGS.batch_dir)
    if not os.path.exists(batch_directory):
        os.makedirs(batch_directory)
        DDDM_input.convert_dataset(dest_directory, FLAGS.batch_dir)
