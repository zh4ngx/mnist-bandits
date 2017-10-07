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
LEARNING_RATE = 0.05

# Reproduce results
SEED = 12345
np.random.seed(SEED)


def main(_):
    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True, seed=SEED)

    # Create the model
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, NUM_ACTIONS]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.matmul(x, W) + b

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, NUM_ACTIONS])

    # Implement MTR L2 Loss by masking all rewards except logged action
    mask = tf.placeholder(tf.int32, [None])
    one_hot_mask = tf.one_hot(indices=mask, depth=NUM_ACTIONS)
    error_term = y_ - y
    masked_error = tf.multiply(error_term, one_hot_mask)
    loss = tf.nn.l2_loss(masked_error) / BATCH_SIZE

    train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # Only train on one epoch - never see an example twice
    num_iterations = len(mnist.train.labels) // BATCH_SIZE

    # Train
    for iteration in range(num_iterations):
        batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
        # Uniform Random logging policy
        batch_mask = np.random.randint(NUM_ACTIONS, size=BATCH_SIZE)

        _, batch_loss, batch_error_term, batch_masked_error = sess.run([
            train_step,
            loss,
            error_term,
            masked_error,
        ], feed_dict={x: batch_xs, y_: batch_ys, mask: batch_mask})

        if iteration % 50 == 0:
            print("Iteration {0}: Loss {1}".format(iteration, batch_loss))

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
