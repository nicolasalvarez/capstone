# coding=utf-8

"""
Routine for validating the Distracted Driver Detection model.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import time
import os
from datetime import datetime
import numpy as np
import tensorflow as tf
import DDDM

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', os.path.join(FLAGS.data_dir, 'DDDM_eval'),
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test', """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', os.path.join(FLAGS.data_dir, 'DDDM_train'),
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('num_examples', DDDM.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL, """Number of examples to run.""")


def validate(saver, summary_writer, logits, labels, top_k_op, summary_op):
    """
    Run Eval once.

    Args:
      saver: Saver.
      summary_writer: Summary writer.
      top_k_op: Top K op.
      summary_op: Summary op.
    """
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
            # Assuming model_checkpoint_path looks something like: /my-favorite-path/DDDM_train/model.ckpt-0,
            # extract global_step from it.
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print('No checkpoint file found')
            return

        # Start the queue runners.
        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))

            num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
            true_count = 0  # Counts the number of correct predictions.
            total_sample_count = num_iter * FLAGS.batch_size
            step = 0

            log_loss = 0.0
            sm_log_loss = 0.0
            while step < num_iter and not coord.should_stop():
                probs, labels_, predictions = sess.run([tf.nn.softmax(logits), labels, top_k_op])

                # Compute  multi-class logarithmic loss with and without laplace smoothing.
                smooth_probs = (probs + 1) / (1 + DDDM.NUM_CLASSES)
                for i in range(FLAGS.batch_size):
                    log_loss -= math.log(probs[i][labels_[i]])
                    sm_log_loss -= math.log(smooth_probs[i][labels_[i]])
                true_count += np.sum(predictions)
                step += 1

            # Compute precision @ 1.
            precision = true_count / total_sample_count
            log_loss /= total_sample_count
            sm_log_loss /= total_sample_count
            print("log_loss", log_loss, "sm_log_loss", sm_log_loss)
            print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            summary.value.add(tag='Precision @ 1', simple_value=precision)
            summary_writer.add_summary(summary, global_step)
        except Exception as e:
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)


def evaluate():
    """Eval DDDM for a number of steps."""
    with tf.Graph().as_default() as g:
        # Get images and labels for DDDM.
        eval_data = FLAGS.eval_data == 'test'
        images, labels = DDDM.inputs(eval_data=eval_data)

        # Build a Graph that computes the logits predictions from the inference model.
        logits = DDDM.inference(images, dropout_prob=tf.constant(1.0))

        # Calculate predictions.
        top_k_op = tf.nn.in_top_k(logits, labels, 1)

        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(DDDM.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.merge_all_summaries()

        summary_writer = tf.train.SummaryWriter(FLAGS.eval_dir, g)

        validate(saver, summary_writer, logits, labels, top_k_op, summary_op)


def main(argv=None):
    DDDM.check_and_maybe_convert_dataset()
    if tf.gfile.Exists(FLAGS.eval_dir):
        tf.gfile.DeleteRecursively(FLAGS.eval_dir)
    tf.gfile.MakeDirs(FLAGS.eval_dir)
    evaluate()


if __name__ == '__main__':
    tf.app.run()
