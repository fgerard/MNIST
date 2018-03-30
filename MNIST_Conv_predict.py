import tensorflow as tf
from random import randint
import numpy as np
#from matplotlib import pyplot as plt

# cargamos los datos de train y test (solo necesitamos test para hacer predicts)
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

saver = tf.train.import_meta_graph('./chkpt/mnist_model.meta')

with tf.Session() as sess:
    # Restore model
    saver.restore(sess,"./chkpt/mnist_model")
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    keep_prob = graph.get_tensor_by_name("keep_prob:0")
    predict = graph.get_tensor_by_name("predict:0")
    for i in range(0,100):
        num = randint(0, mnist.test.images.shape[0])
        img = mnist.test.images[num]
        #print("Shape1: {}".format(img.shape))
        classification = sess.run(predict, feed_dict={x: [img], keep_prob: 1})
        #plt.imshow(img.reshape(28, 28), cmap=plt.cm.binary)
        #plt.show()
        out=classification[0]
        labeled=np.argmax(mnist.test.labels[num],0)
        mark = " *****" if out == labeled else ""
        print('NN predicted test: %d, label: %d, prediction: %d %s'%(num,labeled,out,mark))
