"""
title: TensorFlow > Develop > GET STARTED > Getting Started With TensorFlow
reference: https://www.tensorflow.org/get_started/get_started
"""
import tensorflow as tf

# Basic constant operations
# The value returned by the constructor represents the output
# of the Constant op.
a = tf.constant(2)
b = tf.constant(3)

# Launch the default graph.
with tf.Session() as sess:
    print(sess.run(a+b))
    print(sess.run(a*b))