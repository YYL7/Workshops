# Sci-kit Learn

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sklearn
from IPython.display import Image
import warnings
warnings.filterwarnings("ignore")
#allows graphs to show in our notebook
%matplotlib inline

# Exploratory Data Analysis
iris = pd.read_csv('iris.csv')
iris.drop("Id", axis=1, inplace = True)
iris.head()

set(iris.Species)

iris.Species.value_counts()
# This is a balanced dataset!

# the distribution of each field to see if there are any irregularities
# Sepal Length
sns.kdeplot(iris.SepalLengthCm[iris.Species == 'Iris-setosa'], label='Iris-setosa');
sns.kdeplot(iris.SepalLengthCm[iris.Species == 'Iris-versicolor'], label='Iris-versicolor');
sns.kdeplot(iris.SepalLengthCm[iris.Species == 'Iris-virginica'], label='Iris-virginica');

# Sepal Width
sns.kdeplot(iris.SepalWidthCm[iris.Species == 'Iris-setosa'], label='Iris-setosa');
sns.kdeplot(iris.SepalWidthCm[iris.Species == 'Iris-versicolor'], label='Iris-versicolor');
sns.kdeplot(iris.SepalWidthCm[iris.Species == 'Iris-virginica'], label='Iris-virginica');

# Petal Length
sns.kdeplot(iris.PetalLengthCm[iris.Species == 'Iris-setosa'], label='Iris-setosa');
sns.kdeplot(iris.PetalLengthCm[iris.Species == 'Iris-versicolor'], label='Iris-versicolor');
sns.kdeplot(iris.PetalLengthCm[iris.Species == 'Iris-virginica'], label='Iris-virginica');

# Petal Width
sns.kdeplot(iris.PetalWidthCm[iris.Species == 'Iris-setosa'], label='Iris-setosa');
sns.kdeplot(iris.PetalWidthCm[iris.Species == 'Iris-versicolor'], label='Iris-versicolor');
sns.kdeplot(iris.PetalWidthCm[iris.Species == 'Iris-virginica'], label='Iris-virginica');
#It looks like Petal related information is highly informative of 'Iris-setosa'

# the relationships between the variables
sns.pairplot(iris, hue="Species", size=3, diag_kind="kde");
#This scatter matrix shows that there are clear groupings between the species

# correlations:
corr = iris.corr()
fg, ax = plt.subplots(figsize = (3,3))
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

p = sns.heatmap(corr, mask = mask, linewidths = .5, square = True);
# There are strong correlations between some of the variables, in real projects, highly correlated variables should be removed.


from sklearn.model_selection import train_test_split

feats = list(iris.columns)
feats.remove('Species')

# X will be our features and y will be our target variable
X = iris[feats]
y = iris.Species

# Training and Testing Set Partition
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
X_train.head()
y_train.head()


from sklearn.metrics import *
m_eval = pd.DataFrame(columns = ['method','trainscore','testscore'])


def addeval(method, train, test):
    global m_eval
    d = pd.DataFrame([[method, train, test]],columns = ['method','trainscore','testscore'])
    m_eval = m_eval.append(d)


import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# SVM (Support Vector Machine) 
from sklearn.svm import SVC

list(iris.Species.unique())

#http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
svc = SVC(kernel = 'poly',C=100, gamma=0.01, probability = True)

#fit the model to the training data
svc.fit(X_train, y_train)

#predict the values using the testing data
svc_pred = svc.predict(X_test)
#display the accuracy or the training set, the testing set and the difference (to see if it is overfitting)
print(svc.score(X_train,y_train), svc.score(X_test, y_test),svc.score(X_train,y_train)-svc.score(X_test,y_test))

#display the confusion matrix for the test set
mtrx = confusion_matrix(y_test,svc_pred)

#add metrics to table
addeval('SVM',svc.score(X_train,y_train), svc.score(X_test, y_test))
class_names = list(iris.Species.unique())
# sns.heatmap(mtrx, linewidths = .5, square = True, annot=True, xticklabels= class_names,
#            yticklabels= class_names);

plot_confusion_matrix(mtrx, classes=class_names ,
                       title='Confusion matrix')


# Logistic Regression
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(penalty = 'l2', dual = True)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
print(lr.score(X_train,y_train), lr.score(X_test, y_test),lr.score(X_train,y_train)-lr.score(X_test,y_test))

mtrx = confusion_matrix(y_test,lr_pred)

addeval('Log Reg',lr.score(X_train,y_train), lr.score(X_test, y_test))

class_names = list(iris.Species.unique())
plot_confusion_matrix(mtrx, classes=class_names ,
                       title='Confusion matrix')


# KNN (K-Nearest-Neighbors)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 20, algorithm = 'auto', weights = 'uniform')
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)
print(knn.score(X_train,y_train), knn.score(X_test,y_test), knn.score(X_train,y_train)-knn.score(X_test,y_test))

mtrx = confusion_matrix(y_test,knn_pred)

addeval('KNN',knn.score(X_train,y_train), knn.score(X_test, y_test))

class_names = list(iris.Species.unique())
plot_confusion_matrix(mtrx, classes=class_names ,
                       title='Confusion matrix')

# Naive Bayes (GaussianNB)
from sklearn.naive_bayes import GaussianNB

gauss = GaussianNB()
gauss.fit(X_train, y_train)
gauss_pred = gauss.predict(X_test)
print(gauss.score(X_train,y_train), gauss.score(X_test,y_test), gauss.score(X_train,y_train)-gauss.score(X_test,y_test))

mtrx = confusion_matrix(y_test,gauss_pred)

addeval('GaussNB',gauss.score(X_train,y_train), gauss.score(X_test, y_test))

class_names = list(iris.Species.unique())
plot_confusion_matrix(mtrx, classes=class_names ,
                       title='Confusion matrix')

# Decision Tree
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
print(dt.score(X_train,y_train), dt.score(X_test,y_test), dt.score(X_train,y_train)-dt.score(X_test,y_test))

mtrx = confusion_matrix(y_test,dt_pred)

addeval('Dec Tree',dt.score(X_train,y_train), dt.score(X_test, y_test))

class_names = list(iris.Species.unique())
plot_confusion_matrix(mtrx, classes=class_names ,
                       title='Confusion matrix')

# Random Forest
from sklearn.ensemble import RandomForestClassifier 

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
print(rf.score(X_train,y_train), rf.score(X_test,y_test), rf.score(X_train,y_train)-rf.score(X_test,y_test))

mtrx = confusion_matrix(y_test,rf_pred)

addeval('Random Forest',rf.score(X_train,y_train), rf.score(X_test, y_test))

class_names = list(iris.Species.unique())
plot_confusion_matrix(mtrx, classes=class_names ,
                       title='Confusion matrix')

# Neural Networks
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

scaler = StandardScaler() 
scaler.fit(X_train)  
sX_train = scaler.transform(X_train)  
sX_test = scaler.transform(X_test)

#1 hidden layer, 15 neurons, 300 epoches, 86.6% accuracy
'''
Multilayer perceptron model, with one hidden layer.
input layer : 4 neurons, represents the feature of Iris
hidden layer : 15 neurons, activation using ReLU
output layer : 3 neurons, represents the class of Iris, Softmax Layer
optimizer =  a stochastic gradient-based optimizer
loss function = categorical cross entropy
activation function = ‘relu’, the rectified linear unit function
epoch = 300
'''
nn = MLPClassifier(hidden_layer_sizes=(15),solver='adam', activation = 'relu', random_state=1
                  ,max_iter=300)
# The ith element represents the number of neurons in the ith hidden layer.
nn.fit(sX_train,y_train)
nn_pred = nn.predict(sX_test)
print(nn.score(sX_train,y_train), nn.score(sX_test,y_test), nn.score(sX_train,y_train)-nn.score(sX_test,y_test))

mtrx = confusion_matrix(y_test,nn_pred)

#addeval('Neural Net',nn.score(sX_train,y_train), nn.score(sX_test, y_test))

class_names = list(iris.Species.unique())
plot_confusion_matrix(mtrx, classes=class_names ,
                       title='Confusion matrix')


#2 hidden layers, 15 neurons each, 300 epoches, 95.5% accuracy
'''
Multilayer perceptron model, with two hidden layer.
input layer : 4 neurons, represents the feature of Iris
hidden layer 0 : 15 neurons, activation using ReLU
hidden layer 1 : 15 neurons, activation using ReLU
output layer : 3 neurons, represents the class of Iris, Softmax Layer
optimizer =  a stochastic gradient-based optimizer
loss function = categorical cross entropy
epoch = 300
'''
nn = MLPClassifier(hidden_layer_sizes=(15, 15),solver='adam', activation = 'relu', random_state=1
                  ,max_iter=300)

nn.fit(sX_train,y_train)
nn_pred = nn.predict(sX_test)
print(nn.score(sX_train,y_train), nn.score(sX_test,y_test), nn.score(sX_train,y_train)-nn.score(sX_test,y_test))

mtrx = confusion_matrix(y_test,nn_pred)

#addeval('Neural Net',nn.score(sX_train,y_train), nn.score(sX_test, y_test))

class_names = list(iris.Species.unique())
plot_confusion_matrix(mtrx, classes=class_names ,
                       title='Confusion matrix')



#3 hidden layers, 15 neurons each, 300 epoches, 95.5% accuracy
'''
Multilayer perceptron model, with two hidden layer.
input layer : 4 neurons, represents the feature of Iris
hidden layer 0 : 15 neurons, activation using ReLU
hidden layer 1 : 15 neurons, activation using ReLU
hidden layer 2 : 15 neurons, activation using ReLU
output layer : 3 neurons, represents the class of Iris, Softmax Layer
optimizer =  a stochastic gradient-based optimizer
loss function = categorical cross entropy
epoch = 300
'''
nn = MLPClassifier(hidden_layer_sizes=(15, 15, 15),solver='adam', activation = 'relu', random_state=1
                  ,max_iter=300)

nn.fit(sX_train,y_train)
nn_pred = nn.predict(sX_test)
print(nn.score(sX_train,y_train), nn.score(sX_test,y_test), nn.score(sX_train,y_train)-nn.score(sX_test,y_test))

mtrx = confusion_matrix(y_test,nn_pred)

#addeval('Neural Net',nn.score(sX_train,y_train), nn.score(sX_test, y_test))

class_names = list(iris.Species.unique())
plot_confusion_matrix(mtrx, classes=class_names ,
                       title='Confusion matrix')


#4 hidden layers, 15 neurons each, 300 epoches, 97.7% accuracy
'''
Multilayer perceptron model, with two hidden layer.
input layer : 4 neurons, represents the feature of Iris
hidden layer 0 : 15 neurons, activation using ReLU
hidden layer 1 : 15 neurons, activation using ReLU
hidden layer 2 : 15 neurons, activation using ReLU
hidden layer 3 : 15 neurons, activation using ReLU
output layer : 3 neurons, represents the class of Iris, Softmax Layer
optimizer =  a stochastic gradient-based optimizer
loss function = categorical cross entropy
epoch = 300
'''
nn = MLPClassifier(hidden_layer_sizes=(15, 15, 15, 15),solver='adam', activation = 'relu', random_state=1
                  ,max_iter=300)

nn.fit(sX_train,y_train)
nn_pred = nn.predict(sX_test)
print(nn.score(sX_train,y_train), nn.score(sX_test,y_test), nn.score(sX_train,y_train)-nn.score(sX_test,y_test))

mtrx = confusion_matrix(y_test,nn_pred)

addeval('MLP',nn.score(sX_train,y_train), nn.score(sX_test, y_test))

class_names = list(iris.Species.unique())
plot_confusion_matrix(mtrx, classes=class_names ,
                       title='Confusion matrix')


#5 hidden layers, 15 neurons each, 400 epoches, 97.7% accuracy
'''
Multilayer perceptron model, with two hidden layer.
input layer : 4 neurons, represents the feature of Iris
hidden layer 0 : 15 neurons, activation using ReLU
hidden layer 1 : 15 neurons, activation using ReLU
hidden layer 2 : 15 neurons, activation using ReLU
output layer : 3 neuron, represents the class of Iris, Softmax Layer
optimizer =  a stochastic gradient-based optimizer
loss function = categorical cross entropy
epoch = 400
'''
nn = MLPClassifier(hidden_layer_sizes=(15, 15, 15, 15),solver='adam', activation = 'relu', random_state=1
                  ,max_iter=400)

nn.fit(sX_train,y_train)
nn_pred = nn.predict(sX_test)
print(nn.score(sX_train,y_train), nn.score(sX_test,y_test), nn.score(sX_train,y_train)-nn.score(sX_test,y_test))

mtrx = confusion_matrix(y_test,nn_pred)

#addeval('Neural Net',nn.score(sX_train,y_train), nn.score(sX_test, y_test))

class_names = list(iris.Species.unique())
plot_confusion_matrix(mtrx, classes=class_names ,
                       title='Confusion matrix')


#5 hidden layers, 400 neurons, 400 epoches, 97.7% accuracy
'''
Multilayer perceptron model, with one hidden layer.
input layer : 4 neurons, represents the feature of Iris
hidden layer : 400 neurons, activation using ReLU
output layer : 3 neurons, represents the class of Iris, Softmax Layer
optimizer =  a stochastic gradient-based optimizer
loss function = categorical cross entropy
epoch = 400
'''
nn = MLPClassifier(hidden_layer_sizes=(400),solver='adam', activation = 'relu', random_state=1,
                  max_iter=400)

nn.fit(sX_train,y_train)
nn_pred = nn.predict(sX_test)
print(nn.score(sX_train,y_train), nn.score(sX_test,y_test), nn.score(sX_train,y_train)-nn.score(sX_test,y_test))

mtrx = confusion_matrix(y_test,nn_pred)

#addeval('Neural Net',nn.score(sX_train,y_train), nn.score(sX_test, y_test))

class_names = list(iris.Species.unique())
plot_confusion_matrix(mtrx, classes=class_names ,
                       title='Confusion matrix')



from sklearn.ensemble import VotingClassifier

ens = VotingClassifier(estimators=[('SVC', svc), ('NB', gauss), ('dt',dt)], 
                       voting='soft', weights=[1,1,1])
ens.fit(X_train, y_train)
ens_pred = knn.predict(X_test)
print(ens.score(X_train,y_train), ens.score(X_test,y_test), ens.score(X_train,y_train)-ens.score(X_test,y_test))

mtrx = confusion_matrix(y_test,ens_pred)

addeval('Ensemble',ens.score(X_train,y_train), ens.score(X_test, y_test))

class_names = list(iris.Species.unique())
plot_confusion_matrix(mtrx, classes=class_names ,
                       title='Confusion matrix')

mm_eval = pd.melt(m_eval[['method','trainscore','testscore']], "method", var_name="Measurement")
m_eval

# The top performing model here was Naive Bayes (The accuracy is too high to believe). 

p = sns.pointplot(x="method", y="value", hue="Measurement", data=mm_eval)
labs = list(m_eval['method'])
p.set_xticklabels(labs, rotation=45);

# Cross Validation
from sklearn.cross_validation import cross_val_score # K-fold cross validation

# Built the testset
k_range = range(1, 31)

k_scores = []
# 10 fold cross-validation
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
    k_scores.append(scores.mean())
#For regression problems, just make scoring='mean_squared_error'.

plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.show()

# The average performance of the models
scores_ls = []
models = ["SVM", "Logistic Reg", "MLP", "KNN", "Dec Tree", "Random Forest", "GaussNB"]
model_dict = {"SVM": SVC(kernel = 'poly',C=100, gamma=0.01, probability = True),
             "Logistic Reg": LogisticRegression(), "MLP": MLPClassifier(hidden_layer_sizes=(15, 15, 15, 15),
                                                             solver='adam', activation = 'relu', 
                                                             random_state=1,max_iter=400),
             "KNN": KNeighborsClassifier(20), "Dec Tree": DecisionTreeClassifier(),
             "Random Forest": RandomForestClassifier(), "GaussNB": GaussianNB()}
for model in models:
    scores = cross_val_score(model_dict[model], X, y, cv=5, scoring='accuracy')
    scores_ls.append(scores.mean())


Performance_CV = pd.DataFrame({"Model": models, "Accuracy_Avg": scores_ls})
Performance_CV

p = sns.pointplot(x="Model", y="Accuracy_Avg", data=Performance_CV)
labs = list(Performance_CV['Model'])
p.set_xticklabels(labs, rotation=45);


# PCA Decomposition
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca_X_train = pca.fit_transform(X_train)
pca_X_test = pca.transform(X_test)

variances = pca.explained_variance_
PCs = pca.components_


PCs_mat = np.matrix(PCs)
PCs_df = pd.DataFrame(PCs_mat.T, 
                  columns = ["PC" + str(i) for i in range(1, len(pca.explained_variance_) + 1)],
                     index = X.columns)
#components_ : array, shape (n_components, n_features)
cumul_var = np.cumsum(variances) / sum(variances)
pca_score_df = pd.DataFrame(pca_X_train, 
                  columns = ["PC" + str(i) for i in range(1, 
                                    pca_X_train.shape[1] + 1)])

cumul_var   # Two PCs can explain nearly 100% variances.

pca.explained_variance_ratio_   # The 1st PC explained over 92.6% variances.

PCs_df   # PetalLengthCm contributes most to PC1.

pca_score_df['y'] = y_train.tolist()


plt.figure()
colors = ['navy', 'turquoise', 'darkorange']
lw = 2
target_names = list(iris.Species.unique())
for color, i, target_name in zip(colors, target_names, target_names):
    plt.scatter(pca_score_df.loc[pca_score_df.y ==i,"PC1"],
                pca_score_df.loc[pca_score_df.y ==i,"PC2"], color=color, alpha=.8, lw=lw,
                label=target_name)
x = pca_score_df.iloc[:,0]
y = pca_score_df.iloc[:,1]
vectors = np.transpose(PCs[:2, :])
vectors_scaled = vectors * [x.max(), y.max()]
for i in range(vectors.shape[0]):
    plt.annotate("", xy=(vectors_scaled[i, 0], vectors_scaled[i, 1]),
                xycoords='data', xytext=(0, 0), textcoords='data',
                arrowprops={'arrowstyle': '-|>', 'ec': 'b'})

    plt.text(vectors_scaled[i, 0] * 1.05, vectors_scaled[i, 1] * 1.05,
            X.columns[i], color='k', fontsize=12)
plt.xlabel('$1^{st} PC$')
plt.ylabel('$2^{nd} PC$')
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of IRIS dataset')
