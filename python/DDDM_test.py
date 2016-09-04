# coding=utf-8

"""
Routine for testing the Distracted Driver Detection model.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf
from python import DDDM
import os
import pandas as pd

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/DDDM_train', """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('num_examples', 79726, """Number of examples to run.""")
tf.app.flags.DEFINE_string('submission_file', 'nalvarez_submission.csv', """Submission file name.""")


def get_filename_queue():
    """
    Check if test_batch.bin file exists and returns a list with filename.
    :return: list with filename
    """
    batch_directory = os.path.join(FLAGS.data_dir, FLAGS.batch_dir)

    filenames = [os.path.join(batch_directory, 'test_batch.bin')]
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)
    return filenames


def evaluate():
    """
    Evaluate the images in file test_batch.bin and store probabilities estimated by DDDM in submission_file file.
    :return: None
    """
    # Dimensions of the images in the DDDM dataset.
    img_name_bytes = 4
    image_bytes = DDDM.HEIGHT * DDDM.WIDTH * DDDM.DEPTH
    record_bytes = img_name_bytes + image_bytes

    # Read a record, getting filenames from the filename_queue.
    filename_queue = tf.train.string_input_producer(get_filename_queue())
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    key, value = reader.read(filename_queue)

    # Get image name (just the number) as a int32
    img_name = tf.cast(tf.slice(tf.decode_raw(value, tf.int32), [0], [1]), tf.int32)

    # Get uint8 image and subtract off the mean and divide by the variance of the pixels.
    uint8image = tf.reshape(tf.slice(tf.decode_raw(value, tf.uint8), [img_name_bytes], [image_bytes]),
                            [DDDM.HEIGHT, DDDM.WIDTH, DDDM.DEPTH])
    reshaped_image = tf.cast(uint8image, tf.float32)

    resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, DDDM.IMG_SIZE, DDDM.IMG_SIZE)

    images = tf.image.per_image_whitening(resized_image)

    images, names = tf.train.batch([images, img_name], batch_size=FLAGS.batch_size, capacity=FLAGS.batch_size*10)

    names = tf.reshape(names, [FLAGS.batch_size])

    # Build a Graph that computes the logits predictions from the inference model.
    logits = DDDM.inference(images)
    img_prob = tf.nn.softmax(logits)

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(DDDM.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print('No checkpoint file found')
            return

        # Start populating the filename queue.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # Create submission file.
        submission_file = os.path.join(FLAGS.data_dir, FLAGS.submission_file)
        if os.path.exists(submission_file):
            os.remove(submission_file)
        f = open(submission_file, 'wb')
        f.write('img,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9\n')
        f.close()

        num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
        step = 0

        while step < num_iter and not coord.should_stop():
            print ('Running step', step+1, 'of', num_iter)

            names_batch, prob_batch = sess.run([names, img_prob])

            # Create dataframe with images probabilities
            df_val = pd.DataFrame(prob_batch, columns=['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'])

            # Make additive smoothing, also called Laplace smoothing
            df_val = (df_val + 1) / (1+DDDM.NUM_CLASSES)

            # Reassembly images file names
            names_batch = ['img_'+str(img_no)+'.jpg' for img_no in names_batch]

            # Add names column to dataframe removing padding of the last batch
            padding = (step + 1) * FLAGS.batch_size - FLAGS.num_examples
            if padding > 0:
                df_val = df_val[:-padding]
                df_val.insert(0, 'img', names_batch[:-padding])
            else:
                df_val.insert(0, 'img', names_batch)

            # Write data to file
            with open(submission_file, 'a') as f:
                df_val.to_csv(f, index=False, header=False)

            step += 1

        print('Testing ended.')

        coord.request_stop()
        coord.join(threads)


def main(argv=None):
    """"""
    DDDM.check_and_maybe_convert_dataset()
    evaluate()


if __name__ == '__main__':
    tf.app.run()
