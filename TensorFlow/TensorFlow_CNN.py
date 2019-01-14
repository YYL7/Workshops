# Convolution: 1D operation with Python (Numpy/Scipy)

import numpy as np
h = [2,1,0]
x = [3,4,5]

y = np.convolve(x,h)
y


print("Compare with the following values from Python: y[0] = {0} ; y[1] = {1}; y[2] = {2}; y[3] = {3}; y[4] = {4}".format(y[0],y[1],y[2],y[3],y[4]))



# There are three methods to apply kernel on the matrix, 
# with padding (full), with padding(same) and without padding(valid):

# 1. CNN One Dimension with Padding (full) in Python

import numpy as np

x= [6,2]
h= [1,2,5,4]
# now, because of the zero padding, 
# the final dimension of the array is bigger
y= np.convolve(x,h,"full") 
y  

# 2. CNN One Dimension with Padding (same) in Python
import numpy as np

x= [6,2]
h= [1,2,5,4]
# it is same as zero padding, but with generates same 
y= np.convolve(x,h,"same") 
y  


# 3. CNN One Dimension with Padding (valid) in Python
import numpy as np

x= [6,2]
h= [1,2,5,4]
# we will understand why we used the argument valid in the next example
y= np.convolve(x,h,"valid")  
y  



# CNN Two Dimension with/without Padding in Python
import scipy
import scipy.signal as sg
I= [[255,   7,  3],
    [212, 240,  4],
    [218, 216, 230],]

g= [[-1,1]]

print ('Without zero padding \n')
print ('{0} \n'.format(sg.convolve( I, g, 'valid')))
# The 'valid' argument states that the output 
# consists only of those elements 
# that do not rely on the zero-padding.

print ('With zero padding \n')
print (sg.convolve(I, g))


# Coding with TensorFlow
import tensorflow as tf

# Building graph

input = tf.Variable(tf.random_normal([1,10,10,1]))
filter = tf.Variable(tf.random_normal([3,3,1,1]))
op = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='VALID')
op2 = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME')

# Initialization and session
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    print("Input \n")
    print('{0} \n'.format(input.eval()))
    print("Filter/Kernel \n")
    print('{0} \n'.format(filter.eval()))
    print("Result/Feature Map with valid positions \n")
    result = sess.run(op)
    print(result)
    print('\n')
    print("Result/Feature Map with padding \n")
    result2 = sess.run(op2)
    print(result2)
