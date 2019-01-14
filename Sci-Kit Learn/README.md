# Sci-Kit Learn

For this project, I use several machine learning methods to analyze the specific image databases for the iris and try to find the most efficient method. The dataset has 50 samples of 3 different species of iris (150 samples total) with Measurements of sepal length, sepal width, petal length and petal width. 

The deep learning methods include SVM, Logistic Regression, K Nearest Neighbor, Naïve Bayes-GaussianNB, Decision Tree, Random Forest, Neuron Networks and Ensemble. I split the features and target into a training and testing set using 30% of the data as the testing set. 

For the summary based on those deep learning methods, we can see that the top performing model here was Naive Bayes, form which the accuracy may be too high to believe. But, you need assurance that the model has got most of the patterns from the data correct. 

Then we can use cross-validation to estimate the accuracy of the models on the iris dataset by splitting the data, ﬁtting a model and computing the score 5 consecutive time.

The average performance of the model shows that now SVM and MLP perform best. And GaussNB gets the lowest average accuracy. Lastly, we can try the PCA Decomposition to decompose correlated fields into one field. 

Overall, for the iris dataset, I think the now SVM and MLP perform best. 
