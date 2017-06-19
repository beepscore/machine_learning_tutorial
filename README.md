# Purpose
Record info about machine learning Python tutorial.

# References

## Practical Machine Learning Tutorial with Python Introduction
https://pythonprogramming.net/machine-learning-tutorial-python-introduction/

## quandl
Can use some data for free, limited to ~50 calls per day.
Can get an account and pay for more data.
https://www.quandl.com/
https://github.com/quandl/quandl-python


## UCI machine learning repository - data sets
https://archive.ics.uci.edu/ml/datasets.html

### Breast cancer data set
https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Original%29

# Results

## topics
linear regression
K Nearest Neighbors
Support Vector Machines (SVM)
flat clustering
hierarchical clustering
neural networks


## linear regression
### packages
scikit-learn
quandl
pandas


### supervised machine learning
features and labels
#### features
descriptive attributes, for stock prices it's the "continuous data" (time series?)
#### labels
what you want to predict or forecast

## Appendix - Anaconda
I used Anaconda navigator. Created environment machine_learning.
Installed quandl
https://anaconda.org/anaconda/quandl

matplotlib was in environment root, but not in machine_learning.
Installed matplotlib in machine_learning.

## Appendix - run program

    cd machine_learning_tutorial
    python3 machine_learning_tutorial/stock.py
