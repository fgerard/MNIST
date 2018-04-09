import tensorflow as tf
import numpy as np
from random import randint
# from matplotlib import pyplot as plt


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 784], name="x")
y_ = tf.placeholder(tf.float32, shape=[None, 10], name="y_")


def weight_variable(shape, name=None):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name=None):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

# ######  Layer 1 Conv


W_conv1 = weight_variable([5, 5, 1, 32], "W_conv1")
b_conv1 = bias_variable([32], "b_conv1")

x_image = tf.reshape(x, [-1, 28, 28, 1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1, name="h_conv1")
print("h_conv1.shape: {}".format(h_conv1))
h_pool1 = max_pool_2x2(h_conv1)
print("h_pool1.shape: {}".format(h_pool1))

# ######  Layer 2 Conv

W_conv2 = weight_variable([5, 5, 32, 64], "W_conv2")
b_conv2 = bias_variable([64], "b_conv2")

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
print("h_conv2.shape: {}".format(h_conv2))
h_pool2 = max_pool_2x2(h_conv2)
print("h_pool2.shape: {}".format(h_pool2))

# ###### Layer 3 FC

W_fc1 = weight_variable([7 * 7 * 64, 1024], "W_fc1")
b_fc1 = bias_variable([1024], "b_fc1")

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
print("h_fc1.shape: {}".format(h_fc1))

# ###### Layer 4 Droput

keep_prob = tf.placeholder(tf.float32, name="keep_prob")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# ###### Layer 5 "Readout"

W_fc2 = weight_variable([1024, 10], "W_fc2")
b_fc2 = bias_variable([10], "b_fc2")

y_conv = tf.add(tf.matmul(h_fc1_drop, W_fc2), b_fc2, name="y_conv")
print("y_conv.shape: {}".format(y_conv))

predict = tf.argmax(y_conv, 1, name="predict")
print("predict.shape: {}".format(predict))

# ###### Cost function
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y_conv))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(predict, tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()

sess.run(tf.global_variables_initializer())

# saver.save(sess,"./chkpt/mnist_model")

epochs = 300  # int(input("Epochs? "))

for i in range(epochs+1):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0],
                                       y_: batch[1],
                                       keep_prob: 1.0})
        # saver.save(sess,"./chkpt/mnist_model",global_step=i)
        print("step %d, training accuracy %g" % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

# W2 = sess.run("W_conv2:0")
# print("W_conv2:0 : "+str(W2))
# print("W_conv2:shape : "+str(W2.shape))

print("W_conv2.sape: {}".format(W_conv2.eval().shape))

print("Saving..")
saver.save(sess, "./chkpt/mnist_model")
print("Done.")


print("test accuracy %g" % accuracy.eval(feed_dict={x: mnist.test.images,
                                                    y_: mnist.test.labels,
                                                    keep_prob: 1.0}))

for i in range(0, 50):
    num = randint(0, mnist.test.images.shape[0])
    img = mnist.test.images[num]
    # print("Shape1: {}".format(img.shape))
    classification = sess.run(predict, feed_dict={x: [img], keep_prob: 1})
    # plt.imshow(img.reshape(28, 28), cmap=plt.cm.binary)
    # plt.show()
    out = classification[0]
    labeled = np.argmax(mnist.test.labels[num], 0)
    mark = " *****" if out == labeled else ""
    print('NN predicted', out, " label:", labeled, mark)

a = 1
print("hello: {}".format(a))
