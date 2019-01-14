# CNTK3-Natural Language-LSTM

# Introduction
CNTK (Microsoft Cognitive Toolkit), is a deep learning framework developed by Microsoft Research. Microsoft Cognitive Toolkit describes neural networks as a series of computational steps via a directed graph.

LSTM, which stand for Long Short-Term Memory, are a particular type of recurrent neural networks that got lot of attention recently within the machine learning community. In a simple way, LSTM networks have some internal contextual state cells that act as long-term or short-term memory cells.

# Problem Statement
This project is going to use ATIS dataset, from which we have task of slot tagging (tag individual words to their respective classes, where the classes are provided as labels in the training data set.)
 
# Data Summary
we are going to use ATIS dataset, which has 7 columns and each row have a sequence id with 11 entries. This means that sequence 19 consists of 11 tokens; column S0, which contains numeric word indices; the input data is encoded in one-hot vectors. There are 943 words in the vocabulary, so each word is a 943-element vector of all 0 with a 1 at a vector index chosen to represent that word.

# Algorithms
LSTM, is a particular type of recurrent neural networks that got lot of attention recently within the machine learning community. In a simple way, LSTM networks have some internal contextual state cells that act as long-term or short-term memory cells.

# Approach
1. Data reading: We generate simulated data sets with each sample having two features (plotted below) indicative of the age and 2 tumor size. 

2. Data preprocessing: Often, the individual features such as size or age need to be scaled. Typically, one would scale the data between 0 and 1. 

 3. Model creation: LSTM.
 
4. Evaluation: measure the model accuracy by going through all the examples in the test set and use the test_minibatch method of the trainer created inside the evaluate function defined above. .

# Conclusion
The bidirectional model has 40% less parameters than the lookahead one. However, the lookahead model trained about 30% faster.


