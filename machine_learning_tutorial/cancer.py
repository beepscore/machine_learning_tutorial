#!/usr/bin/env python3

import numpy as np
import pandas as pd

# sklearn represents scikit-learn
# http://stackoverflow.com/questions/38733220/difference-between-scikit-learn-and-sklearn
from sklearn import preprocessing, model_selection

# k nearest neighbors
# https://pythonprogramming.net/k-nearest-neighbors-application-machine-learning-tutorial/?completed=/k-nearest-neighbors-intro-machine-learning-tutorial/
from sklearn import neighbors

# svm support vector machine
# https://pythonprogramming.net/support-vector-machine-intro-machine-learning-tutorial/?completed=/final-thoughts-knn-machine-learning-tutorial
# http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
from sklearn import svm


input_directory = "../data/input/"
input_file_name = "breast-cancer-wisconsin-data.csv"
input_file_path = input_directory + input_file_name

# df is a pandas dataframe
# each column is a "feature"
df = pd.read_csv(input_file_path)
# file contains character ? to represent missing data. Change it to -99999
df.replace('?', -99999, inplace=True)
# don't need id column for analysis
df.drop(['id'], 1, inplace=True)

# df.drop returns a new dataframe, use it to make a numpy array
# X features
X = np.array(df.drop(['class'], 1))
# y labels
# data set class values:  2 for benign, 4 for malignant
y = np.array(df['class'])

# divide data into training and testing samples
# use model_selection instead of deprecated cross_validation shown in tutorial.
# train_test_split splits arrays or matrices into random train and test subsets
# http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

# k nearest neighbor classifier
# look at class of k nearest neighbors to predict item class
clf = neighbors.KNeighborsClassifier()

# support vector classifier
clf_svm = svm.SVC()

# train the classifiers
clf.fit(X_train, y_train)
clf_svm.fit(X_train, y_train)

# test
accuracy = clf.score(X_test, y_test)
accuracy_svm = clf_svm.score(X_test, y_test)
# expect accuracy ~ 0.95
print('accuracy', accuracy)
print('accuracy_svm', accuracy_svm)

# fake data, doesn't match any rows in data set
# one row
# example_measures = np.array([4, 2, 1, 1, 1, 2, 3, 2, 1])
# reshape one dimensional fake data to avoid deprecation warning
# DeprecationWarning: Passing 1d arrays as data is deprecated in 0.17 and will raise ValueError in 0.19.
# Reshape your data either using X.reshape(-1, 1) if your data has a single feature
# or X.reshape(1, -1) if it contains a single sample. DeprecationWarning)
# example_measures = example_measures.reshape(1, -1)

# two rows
example_measures = np.array([[4, 2, 1, 1, 1, 2, 3, 2, 1], [4, 2, 1, 1, 1, 2, 3, 2, 1]])
example_measures = example_measures.reshape(len(example_measures), -1)

prediction = clf.predict(example_measures)
prediction_svm = clf_svm.predict(example_measures)
# one row prints prediction [2]
# two rows prints prediction [2 2]
print('prediction', prediction)
print('prediction_svm', prediction_svm)
