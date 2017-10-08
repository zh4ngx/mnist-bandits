# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""A very simple MNIST classifier.

See extensive documentation at
https://www.tensorflow.org/get_started/mnist/beginners
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None

BATCH_SIZE = 100
NUM_ACTIONS = 10
NUM_HIDDEN = 16
BASE_LEARNING_RATE = 0.1
NUM_EPOCHS = 100

# Reproduce results
SEED = 12345
np.random.seed(SEED)


def main(_):
    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True, seed=SEED)

    iterations_per_epoch = len(mnist.train.labels) // BATCH_SIZE
    num_iterations = iterations_per_epoch * NUM_EPOCHS

    # Create the model
    x = tf.placeholder(tf.float32, [None, 784])

    weights_in = tf.Variable(tf.truncated_normal(shape=[784, NUM_HIDDEN], stddev=0.1))
    weights_out = tf.Variable(tf.truncated_normal(shape=[NUM_HIDDEN, NUM_ACTIONS], stddev=0.01))
    b = tf.Variable(tf.zeros([NUM_ACTIONS]))
    h = tf.matmul(x, weights_in)
    y = tf.matmul(h, weights_out) + b

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, NUM_ACTIONS])

    # Implement MTR L2 Loss by masking all rewards except logged action
    mask = tf.placeholder(tf.int32, [None])
    one_hot_mask = tf.one_hot(indices=mask, depth=NUM_ACTIONS)
    error_term = y_ - y
    masked_error = tf.multiply(error_term, one_hot_mask)
    loss = tf.nn.l2_loss(masked_error) / BATCH_SIZE
    global_step = tf.Variable(0, trainable=False)

    learning_rate = tf.train.exponential_decay(BASE_LEARNING_RATE, global_step,
                                               iterations_per_epoch, 0.96, staircase=True)

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # Train
    for iteration in range(num_iterations):
        batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
        # Uniform Random logging policy
        batch_mask = np.random.randint(NUM_ACTIONS, size=BATCH_SIZE)

        _, batch_loss, batch_error_term, batch_masked_error, batch_weights_in, batch_weights_out, batch_b, batch_h, batch_lr = sess.run(
            [
                train_step,
                loss,
                error_term,
                masked_error,
                weights_in,
                weights_out,
                b,
                h,
                learning_rate,
            ], feed_dict={x: batch_xs, y_: batch_ys, mask: batch_mask})

        if iteration % 1000 == 0:
            correct = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
            valid_acc = tf.reduce_mean(tf.cast(correct, tf.float32))
            iter_valid_acc = sess.run(valid_acc, feed_dict={x: mnist.validation.images, y_: mnist.validation.labels})
            print("Iteration {0}: Loss {1}, Valid Acc {2}, LR {3}".format(iteration, batch_loss, iter_valid_acc,
                                                                          batch_lr))

    # Test trained model
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                        y_: mnist.test.labels}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
