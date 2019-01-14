# Image Recognition Example

import numpy as np
import tensorflow as tf
from sklearn.datasets import load_sample_images
import matplotlib
import matplotlib.pyplot as plt

# Load sample images
dataset = np.array(load_sample_images().images, dtype=np.float32)
batch_size, height, width, channels = dataset.shape

dataset

# Dataset.shape: batch_size, height, width, channels
dataset.shape



# run only the zero matrix and the image is showed in dark
# Create filters 
filters_test = np.zeros(shape=(7, 7, channels, 2), dtype=np.float32)

#filters_test[:, 3, :, 1] = 1 # vertical line
#filters_test[3, :, :, 1] = 1 # horizontal line

# Create a graph with input X plus a convolutional layer applying the 2 filters
X = tf.placeholder(tf.float32, shape=(None, height, width, channels))
convolution = tf.nn.conv2d(X, filters_test, strides=[1,2,2,1], padding="SAME")
with tf.Session() as sess:
    output = sess.run(convolution, feed_dict={X: dataset})
plt.imshow(output[0, :, :, 1]) # plot 1st image's 2nd feature map
plt.show()



# Run only the first filter, the image shows as the parameter changes
# Create 2 filters
# vertical line
filters_test = np.zeros(shape=(7, 7, channels, 2), dtype=np.float32)
filters_test[:, 3, :, 1] = 1 # vertical line 
# if the 1 in [:,3,:1] change into 0, the picture show in all blank
#filters_test[3, :, :, 1] = 1 # horizontal line

# Create a graph with input X plus a convolutional layer applying the 2 filters
X = tf.placeholder(tf.float32, shape=(None, height, width, channels))
convolution = tf.nn.conv2d(X, filters_test, strides=[1,2,2,1], padding="SAME")
with tf.Session() as sess:
    output = sess.run(convolution, feed_dict={X: dataset})
plt.imshow(output[0, :, :, 1]) # plot 1st image's 2nd feature map
plt.show()



# Run only the second filter, the image shows as parameter changes
# Create 2 filters
# horizontal line
filters_test = np.zeros(shape=(7, 7, channels, 2), dtype=np.float32)
filters_test[3, :, :, 1] = 1 # horizontal line
#filters_test[:, 3, :, 1] = 1 # vertical line

# Create a graph with input X plus a convolutional layer applying the 2 filters
X = tf.placeholder(tf.float32, shape=(None, height, width, channels))
convolution = tf.nn.conv2d(X, filters_test, strides=[1,2,2,1], padding="SAME")
with tf.Session() as sess:
    output = sess.run(convolution, feed_dict={X: dataset})
plt.imshow(output[0, :, :, 1]) # plot 1st image's 2nd feature map
plt.show()

# Create 2 filters
filters_test = np.zeros(shape=(7, 7, channels, 2), dtype=np.float32)
filters_test[:, 3, :, 1] = 1 # vertical line
filters_test[3, :, :, 1] = 1 # horizontal line



# To see what the filters change the matrix
filters_test


np.shape(filters_test)



# Run two filters and the image shows
# Create 2 filters
# vertical & horizontal
filters_test = np.zeros(shape=(7, 7, channels, 2), dtype=np.float32)
filters_test[:, 3, :, 1] = 1 # vertical line
filters_test[3, :, :, 1] = 1 # horizontal line

# Create a graph with input X plus a convolutional layer applying the 2 filters
X = tf.placeholder(tf.float32, shape=(None, height, width, channels))
convolution = tf.nn.conv2d(X, filters_test, strides=[1,2,2,1], padding="SAME")
with tf.Session() as sess:
    output = sess.run(convolution, feed_dict={X: dataset})
plt.imshow(output[0, :, :, 1]) # plot 1st image's 2nd feature map
plt.show()
