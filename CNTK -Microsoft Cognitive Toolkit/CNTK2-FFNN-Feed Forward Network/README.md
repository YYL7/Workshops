# CNTK2-FFNN-Feed Forward Network

# Introduction
CNTK (Microsoft Cognitive Toolkit), is a deep learning framework developed by Microsoft Research. Microsoft Cognitive Toolkit describes neural networks as a series of computational steps via a directed graph.

Feed Forward Network is an artificial neural network wherein connections between the nodes do not form a cycle.

# Problem Statement
This project is going to help the hospital to determine if a patient has a fatal malignant cancer vs. a benign growth by building a Feed Forward Network model. (have the same dataset as CNTK1-Logistiic Regression )
 
# Data Summary
Cancer dataset, icludes the age and the size of the tumor. Intuitively, one can imagine that younger patients and/or patient with small tumor size are less likely to have malignant cancer.

# Algorithms
To train a Feed Forward Network, which is an artificial neural network where connections between the units do not form a cycle. In this network, the information moves in only one direction, forward, from the input nodes, through the hidden nodes (if any) and to the output nodes. There are no cycles or loops in the network.

# Approach
1. Data reading: We generate simulated data sets with each sample having two features (plotted below) indicative of the age and 2 tumor size. 

2. Data preprocessing: Often, the individual features such as size or age need to be scaled. Typically, one would scale the data between 0 and 1. 

 3. Model creation: Feed Forward Network.
 
4. Evaluation: error rate.

# Conclusion
Feed Forward Network give us error of 0.12 indicating that our model can very effectively deal with previously unseen observations.
