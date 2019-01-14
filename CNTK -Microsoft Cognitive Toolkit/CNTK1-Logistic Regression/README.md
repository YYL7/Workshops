Introduction
For this workshop we have 4 assignments to do. The first assignment demonstrates how to use CNTK to predict future values in a time series using LSTMs. The second assignment demonstrates how to implement a recurrent network to process text. The third assignment is to perform a classification task by using Feed forward network model. The fourth assignment is to train a logistic regression model.
Problem Statement
For the first assignment, I will demonstrate how to use CNTK to predict future values in a time series using LSTMs by simulating dataset.
For the second assignment, we are going to use ATIS dataset, from which we have task of slot tagging (tag individual words to their respective classes, where the classes are provided as labels in the training data set.)
For the third and fourth assignments, they have the same cancer dataset, from which the hospital wants us to determine if a patient has a fatal malignant cancer vs. a benign growth. This is known as a classification problem. 
Data Summary
For the first assignment, time series problem will use simulated data set of a continuous function. For details, we generate multiple such instances of the input signal (by sampling from sin function) each of size N and the corresponding desired output as our training data. 
For the second assignment, we are going to use ATIS dataset, which has 7 columns and each row have a sequence id with 11 entries. This means that sequence 19 consists of 11 tokens; column S0, which contains numeric word indices; the input data is encoded in one-hot vectors. There are 943 words in the vocabulary, so each word is a 943-element vector of all 0 with a 1 at a vector index chosen to represent that word.
 For the third and fourth assignments, they have the same cancer dataset, given their age and the size of the tumor. Intuitively, one can imagine that younger patients and/or patient with small tumor size are less likely to have malignant cancer.
Algorithms
The first assignment demonstrates how to use CNTK to predict future values in a time series using LSTMs, which are well-suited to classifying, processing and making predictions based on time series data, since there can be lags of unknown duration between important events in a time series. 
The second assignment demonstrates how to implement a Recurrent Network to process text. RNN can connect between nodes form a directed graph along a sequence. This allows it to exhibit temporal dynamic behavior for a time sequence. 
The third assignment is to perform a classification task by using Feed forward network model, which is an artificial neural network where connections between the units do not form a cycle. In this network, the information moves in only one direction, forward, from the input nodes, through the hidden nodes (if any) and to the output nodes. There are no cycles or loops in the network.
The fourth assignment is to train a logistic regression model, which is a fundamental machine learning technique that uses a linear weighted combination of features and generates the probability of predicting different classes. In our case, the classifier will generate a probability in [0,1] which can then be compared to a threshold (such as 0.5) to produce a binary label (0 or 1). However, the method shown can easily be extended to multiple classes.
Approach
Any learning algorithm typically has five stages. These are Data reading, Data preprocessing, Creating a model, Learning the model parameters, and Evaluating the model (testing/prediction).
1. Data reading: We generate simulated data sets with each sample having two features (plotted below) indicative of the age and 2 tumor size. 
2. Data preprocessing: Often, the individual features such as size or age need to be scaled. Typically, one would scale the data between 0 and 1. To keep things simple, we are not doing any scaling in this tutorial (for details look here: feature scaling).
 3. Model creation: We introduce a basic linear model in this tutorial. Learning the model: This is also known as training. While fitting a linear model can be done in a variety of ways (linear regression), in CNTK we use Stochastic Gradient Descent a.k.a. SGD. 
4. Evaluation: This is also known as testing, where one evaluates the model on data sets with known labels that were never used for training. This allows us to assess how a model would perform in real-world (previously unseen) observations.
Conclusion
For the first assignment, the time series we build is not perfect but close enough.
For the second assignment, the bidirectional model has 40% less parameters than the lookahead one. However, the lookahead model trained about 30% faster.
 For the third and fourth assignments, FFNN and Logistic Regression give us error of 0.12 indicating that our model can very effectively deal with previously unseen observations (during the training process).
