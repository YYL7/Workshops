# CNTK2-FFNN-Feed Forward Network

# Introduction
CNTK (Microsoft Cognitive Toolkit), is a deep learning framework developed by Microsoft Research. Microsoft Cognitive Toolkit describes neural networks as a series of computational steps via a directed graph.

Logistic regression, is trying to build an equation to do classification with the data we. For the logistic regression classifier (equation), we’ll take our features and multiply each one by a weight and then add them up. This result will be put into the sigmoid function, and we’ll get a number between 0 and 1. And we will use optimization algorithms to find these best-fit parameters.

# Problem Statement
This project is going to help the hospital to determine if a patient has a fatal malignant cancer vs. a benign growth by building a logistic regression model. (classification problem. )
 
# Data Summary
Cancer dataset, icludes the age and the size of the tumor. Intuitively, one can imagine that younger patients and/or patient with small tumor size are less likely to have malignant cancer.

# Algorithms
To train a logistic regression model, which is a fundamental machine learning technique that uses a linear weighted combination of features and generates the probability of predicting different classes. In our case, the classifier will generate a probability in [0,1] which can then be compared to a threshold (such as 0.5) to produce a binary label (0 or 1). However, the method shown can easily be extended to multiple classes.

# Approach
1. Data reading: We generate simulated data sets with each sample having two features (plotted below) indicative of the age and 2 tumor size. 

2. Data preprocessing: Often, the individual features such as size or age need to be scaled. Typically, one would scale the data between 0 and 1. 

 3. Model creation: Logistic regression.
 
4. Evaluation: error rate.

# Conclusion
Logistic Regression give us error of 0.12 indicating that our model can very effectively deal with previously unseen observations (during the training process).
