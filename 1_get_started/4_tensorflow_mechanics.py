"""
title: TensorFlow > Develop > GET STARTED > TensorFlow Mechanics 101
reference: https://www.tensorflow.org/get_started/mnist/mechanics
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import math

# 1. Inputs
mnist = input_data.read_data_sets(train_dir='./../dataset/mnist/', fake_data=False, one_hot=True)

# 2. Placeholders
x = tf.placeholder(tf.float32, shape=(None, 784))
y = tf.placeholder(tf.int32, shape=(None))

# 3. Build the graph
#  3-1. Inference
with tf.name_scope('hidden_1'):
    weights_h1 = tf.Variable(tf.truncated_normal([784, 1024], stddev=1.0 / math.sqrt(float(784)), name='weights'))
    biases_h1 = tf.Variable(tf.zeros([1024]), name='biases')
with tf.name_scope('hidden_2'):
    weights_h2 = tf.Variable(tf.truncated_normal([1024, 1024], stddev=1.0 / math.sqrt(float(1024)), name='weights'))
    biases_h2 = tf.Variable(tf.zeros([1024]), name='biases')
with tf.name_scope('out'):
    weights_out = tf.Variable(tf.truncated_normal([1024, 10], stddev=1.0 / math.sqrt(float(1024)), name='weights'))
    biases_out = tf.Variable(tf.zeros([10]), name='biases')

hidden1 = tf.nn.relu(tf.matmul(x, weights_h1) + biases_h1)
hidden2 = tf.nn.relu(tf.matmul(hidden1, weights_h2) + biases_h2)
hypothesis = tf.matmul(hidden2, weights_out) + biases_out

#  3-2. Loss
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=y, name='cost'))
tf.summary.scalar('cost', cost)

#  3-3. Training
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cost)
correct_pred = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# 5. Train the model
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    summary_writer = tf.summary.FileWriter('./logs/', sess.graph)
    summary = tf.summary.merge_all()
    saver = tf.train.Saver()

    for epoch in range(1000):
        # Feed the graph
        batch_x, batch_y = mnist.train.next_batch(100)
        _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
        # Check the status
        if epoch % 100 == 0:
            print('Step %d: loss = %.2f' % (epoch, c))
        # Visualize the status
        summary_str = sess.run(summary, feed_dict={x: batch_x, y: batch_y})
        summary_writer.add_summary(summary_str, epoch)
        # Save a checkpiont
        saver.save(sess, './restored_model/4', global_step=epoch)
        # saver.restore(sess, './restored_model/')

    # 5. Evaluate the model
    print('Train data eval:',
          sess.run(accuracy, feed_dict={x: mnist.train.images, y: mnist.train.labels}))
    print('Validation data eval:',
          sess.run(accuracy, feed_dict={x: mnist.validation.images, y: mnist.validation.labels}))
    print('Test data eval:',
          sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))