#!/usr/bin/env python3

import numpy as np
import pandas as pd

# sklearn represents scikit-learn
# http://stackoverflow.com/questions/38733220/difference-between-scikit-learn-and-sklearn
from sklearn import preprocessing, model_selection, neighbors

"""
k nearest neighbors
https://pythonprogramming.net/k-nearest-neighbors-application-machine-learning-tutorial/?completed=/k-nearest-neighbors-intro-machine-learning-tutorial/
"""

input_directory = "../data/input/"
input_file_name = "breast-cancer-wisconsin-data.txt"
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
y = np.array(df['class'])

# divide data into training and testing samples
# use model_selection instead of deprecated cross_validation shown in tutorial.
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

# classifier
# look at class of k nearest neighbors to predict item class
clf = neighbors.KNeighborsClassifier()

# train the classifier
clf.fit(X_train, y_train)

# test
accuracy = clf.score(X_test, y_test)
# expect accuracy ~ 0.95
print('accuracy', accuracy)