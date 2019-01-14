# TensorFlow

# Introduction

TensorFlow does the same work as NumPy, but instead of returning to Python every time, it creates all the operations in the form of graphs and execute them once with the highly optimized backend.

Convolutional neural network contains multiple copies of the same neuron, and all the neurons share the same weights, bias, and activation function. Each neuron is only connected to a few nearby local neurons in the previous layer, and the neurons share the same weights and biases. So, for an image, it would make sense to apply a small window around a set of pixels, to look for some image feature. By using the same weights, the net assumes that the feature is equally likely to occur at every input position. So, this window can search all over the image, and can be rotated and scaled.

# Problem Statement
To build a Convolutional Neural Network with very simple dataset.

# Approach
There are three methods to apply kernel on the matrix, with padding (full, which shows the sliding procedure with zero padding), with padding (same, which is with padding zero padding but with generates same. And the operation can be verified by Python NumPy.) and without padding (valid, from which the kernel only applies on the second to fourth steps from the “full” (the first) method).

Conclusion
Built a CNN model by simple dataset.
