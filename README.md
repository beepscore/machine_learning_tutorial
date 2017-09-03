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

##### 2017-09-02
Using Anaconda 4.4.0 which comes with Python 3.6.1

I used Anaconda navigator. Created environment beepscore.
Several packages like scikit-learn and matplotlib were in environment root, but not in beepscore.
In environment beepscore, installed scikit-learn, pandas, matplotlib.
In environment beepscore, installed quandl.
https://anaconda.org/anaconda/quandl

## Appendix - run program

### activate conda environment
In terminal  

    cd machine_learning_tutorial
    source activate beepscore

    python3 machine_learning_tutorial/stock.py
