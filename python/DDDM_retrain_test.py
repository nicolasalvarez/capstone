# coding=utf-8

"""
Routine for testing the retrained Inception V3 model. Please see:
http://arxiv.org/abs/1512.00567 (model description)
https://www.tensorflow.org/versions/r0.10/how_tos/image_retraining/index.html (model retrain tutorial)

Preparing model:
 - Install bazel ( check tensorflow's github for more info )
- For retraining, prepare folder structure as
    - root_folder_name
        - class 1
            - file1
            - file2
        - class 2
            - file1
            - file2
- Clone tensorflow
- Go to root of tensorflow
- bazel build tensorflow/examples/image_retraining:retrain
- bazel-bin/tensorflow/examples/image_retraining/retrain --image_dir /path/to/root_folder_name  --output_graph
/path/output_graph.pb -- output_labels /path/output_labels.txt -- bottleneck_dir /path/bottleneck
** Training done. **
For testing through bazel,
    bazel build tensorflow/examples/label_image:label_image && \
    bazel-bin/tensorflow/examples/label_image/label_image \
    --graph=/path/output_graph.pb --labels=/path/output_labels.txt \
    --output_layer=final_result \
    --image=/path/to/test/image
For testing through python, run the following python program.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import DDDM
import os
import numpy as np

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('num_examples', 79726, """Number of examples to run.""")
tf.app.flags.DEFINE_string('submission_file', 'nalvarez_submission_retrained_model.csv',
                           """Submission file name using the retrained model.""")
tf.app.flags.DEFINE_string('model_path', '../retrained_model/output_graph.pb', """Retrained model path""")
tf.app.flags.DEFINE_string('labels_path', '../retrained_model/output_labels.txt', """Labels file path""")


def create_graph():
    """
    Creates a graph from saved GraphDef file and returns a saver.
    """
    with tf.gfile.FastGFile(FLAGS.model_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def test_imgs_inference():
    """
    Predicts the test dataset and store the results in a submission file.
    :return: None
    """
    # Creates graph from saved GraphDef.
    create_graph()

    test_img_path = os.path.join(FLAGS.data_dir, 'imgs/test')

    # Get labels from file
    f = open(FLAGS.labels_path, 'rb')
    lines = f.readlines()
    labels = [str(w).replace("\n", "") for w in lines]
    f.close()

    # Create submission file.
    submission_file = os.path.join(FLAGS.data_dir, FLAGS.submission_file)
    if os.path.exists(submission_file):
        os.remove(submission_file)
    subm_f = open(submission_file, 'wb')
    subm_f.write('img')
    for label in labels:
        subm_f.write("," + label)
    subm_f.write('\n')

    with tf.Session() as sess:
        img_count = 0
        for image_file in os.listdir(test_img_path):

            image_data = tf.gfile.FastGFile(os.path.join(test_img_path, image_file), 'rb').read()

            softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
            predictions = sess.run(softmax_tensor,
                                   {'DecodeJpeg/contents:0': image_data})
            predictions = np.squeeze(predictions)
            # top_k = predictions.argsort()[::-1]  # Get predictions
            # print(image_file, '%s (score = %.5f)' % (labels[top_k[0]], predictions[top_k[0]]))

            # Print progress
            img_count += 1
            if img_count % 1000 == 0 or img_count == FLAGS.num_examples:
                print ("%d/%d images predicted.", img_count, FLAGS.num_examples)

            # Save predictions to file
            predictions.tofile(subm_f, sep=",", format="%.5f")
            subm_f.write('\n')
        subm_f.close()
        return

def main(argv=None):
    print("Predicting testing images.")
    test_imgs_inference()
    print("End")

if __name__ == '__main__':
    tf.app.run()
