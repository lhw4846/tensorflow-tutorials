"""
title: TensorFlow > Develop > GET STARTED > MNIST For ML Beginners
reference: https://www.tensorflow.org/get_started/mnist/beginners
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/data/', one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Implementing the regression
hypothesis = tf.add(tf.matmul(x, W), b)
cost = tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=y)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(cost)

# Training
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
for _ in range(1000):
    batch_x, batch_y = mnist.train.next_batch(100)
    sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

# Evaluating our model
correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))