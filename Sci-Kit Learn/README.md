# Sci-Kit Learn

# Introduction
Sci-Kit Learn, is a simple and efficient tools for data mining and data analysis, built on NumPy, SciPy, and matplotlib.

# Problem Statement
For this project, I will use several machine learning methods to analyze the specific iris databases for classification by Sci-Kit Learn liabary and try to find the most efficient method. 

# Data Summary
The dataset has 50 samples of 3 different species of iris (150 samples total) with Measurements of sepal length, sepal width, petal length and petal width. I split the features and target into a training and testing set using 30% of the data as the testing set. 

# Algorithms
The machine learning methods include SVM, Logistic Regression, K Nearest Neighbor, Naïve Bayes-GaussianNB, Decision Tree, Random Forest, Neuron Networks and Ensemble. 

# Approach
1. Data reading: Load the dataset.

2. Data preprocessing: Distributions and Correlations of the features. Then split the features and target into a training and testing set using 30% of the data as the testing set. 

 3. Model creation: The machine learning methods include SVM, Logistic Regression, K Nearest Neighbor, Naïve Bayes-GaussianNB, Decision Tree, Random Forest, Neuron Networks and Ensemble. 
 
4. Evaluation: This is also known as testing, where one evaluates the model on data sets with known labels that were never used for training. This allows us to assess how a model would perform in real-world (previously unseen) observations. And I use accuaracy for testing and evaluation.

# Conclusion
For the summary based on those machine learning algorithms, we can see that the top performing model here was Naive Bayes, form which the accuracy may be too high to believe. But, you need assurance that the model has got most of the patterns from the data correct. 

Then we can use cross-validation to estimate the accuracy of the models on the iris dataset by splitting the data, ﬁtting a model and computing the score 5 consecutive time.The average performance of the model shows that now SVM and MLP perform best. And GaussNB gets the lowest average accuracy. 

Lastly, we can try the PCA Decomposition to decompose correlated fields into one field. Overall, for the iris dataset, I think the now SVM and MLP perform best. 
