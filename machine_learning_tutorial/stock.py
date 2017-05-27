#!/usr/bin/env python3

import datetime
import math
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
# import pandas as pd
import pickle
import quandl

# sklearn represents scikit-learn
# http://stackoverflow.com/questions/38733220/difference-between-scikit-learn-and-sklearn
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression

"""
https://pythonprogramming.net/regression-introduction-machine-learning-tutorial/
"""

style.use('ggplot')

# df is a pandas dataframe
df = quandl.get('WIKI/GOOGL')
# print(df.head())
# each column is a "feature"
"""
/Users/stevebaker/anaconda/envs/machine_learning/bin/python /Users/stevebaker/Documents/projects/pythonProjects/machine_learning_tutorial/machine_learning_tutorial/stock.py
              Open    High     Low    Close      Volume  Ex-Dividend  \
Date
2004-08-19  100.01  104.06   95.96  100.335  44659000.0          0.0
2004-08-20  101.01  109.08  100.50  108.310  22834300.0          0.0
2004-08-23  110.76  113.48  109.05  109.400  18256100.0          0.0
2004-08-24  111.24  111.60  103.57  104.870  15247300.0          0.0
2004-08-25  104.76  108.00  103.88  106.000   9188600.0          0.0

            Split Ratio  Adj. Open  Adj. High   Adj. Low  Adj. Close  \
Date
2004-08-19          1.0  50.159839  52.191109  48.128568   50.322842
2004-08-20          1.0  50.661387  54.708881  50.405597   54.322689
2004-08-23          1.0  55.551482  56.915693  54.693835   54.869377
2004-08-24          1.0  55.792225  55.972783  51.945350   52.597363
2004-08-25          1.0  52.542193  54.167209  52.100830   53.164113

            Adj. Volume
Date
2004-08-19   44659000.0
2004-08-20   22834300.0
2004-08-23   18256100.0
2004-08-24   15247300.0
2004-08-25    9188600.0

Process finished with exit code 0
"""

# just use adjusted data
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
# video showed incorrect equation, he fixed it in notes
# Add new column HL_PCT. high and low are highly correlated.
df['HL_PCT'] = 100.0 * (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close']
df['PCT_CHANGE'] = 100.0 * (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open']

df = df[['Adj. Close', 'HL_PCT', 'PCT_CHANGE', 'Adj. Volume']]
# print(df.head())
"""
            Adj. Close    HL_PCT  PCT_CHANGE  Adj. Volume
Date
2004-08-19   50.322842  8.072956    0.324968   44659000.0
2004-08-20   54.322689  7.921706    7.227007   22834300.0
2004-08-23   54.869377  4.049360   -1.227880   18256100.0
2004-08-24   52.597363  7.657099   -5.726357   15247300.0
2004-08-25   53.164113  3.886792    1.183658    9188600.0
"""

forecast_col = 'Adj. Close'

# handle missing data
# "You can't just pass a NaN (Not a Number) datapoint to a machine learning classifier..."
# so specify a large negative value. It will be treated as an outlier.
df.fillna(-99999, inplace=True)

# extrapolate ~ 10% past end of data range
# forecast_out = int(math.ceil(0.1*len(df)))
# after this, the video changed from 0.1 to 0.01
forecast_out = int(math.ceil(0.01*len(df)))

# specify the "label" column
# label is forecast stock price adjusted close 1% past end of range
# pandas shift method takes a column and shifts its values by an amount
df['label'] = df[forecast_col].shift(-forecast_out)

# print(df.head())

# the first columns are "features", the "label" column is extrapolating into the future.
# with forecast_out = int(math.ceil(0.1*len(df)))
"""
            Adj. Close    HL_PCT  PCT_CHANGE  Adj. Volume       label
Date
2004-08-19   50.322842  8.072956    0.324968   44659000.0  208.879792
2004-08-20   54.322689  7.921706    7.227007   22834300.0  212.084685
2004-08-23   54.869377  4.049360   -1.227880   18256100.0  214.973603
2004-08-24   52.597363  7.657099   -5.726357   15247300.0  212.395645
2004-08-25   53.164113  3.886792    1.183658    9188600.0  202.394773
"""

# print(df.tail())
# print(df.head())
"""
            Adj. Close    HL_PCT  PCT_CHANGE  Adj. Volume      label
Date                                                                
2004-08-19   50.322842  8.072956    0.324968   44659000.0  69.399229
2004-08-20   54.322689  7.921706    7.227007   22834300.0  68.752232
2004-08-23   54.869377  4.049360   -1.227880   18256100.0  69.639972
2004-08-24   52.597363  7.657099   -5.726357   15247300.0  69.078238
2004-08-25   53.164113  3.886792    1.183658    9188600.0  67.839414
"""

# It is a typical standard with machine learning in code to define X (capital x), as the features,
# and y (lowercase y) as the label that corresponds to the features
# features X is numpy array, using entire dataframe except for label column
X = np.array(df.drop(['label'], 1))
# scale features from -1 to 1
X = preprocessing.scale(X)

# X_lately contains the most recent features
X_lately = X[-forecast_out:]

X = X[:-forecast_out]

df.dropna(inplace=True)

# label y is a numpy array from dataframe label column
y = np.array(df['label'])

# use model_selection instead of deprecated cross_validation shown in video.
# X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
# use 20% of data for testing, 80% for training
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

# define the classifier
# Choosing the right estimator
# http://scikit-learn.org/stable/tutorial/machine_learning_map/


# Use SVR support vector regression
# use defaults, e.g. default kernel is rbf
# http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
# doesn't have n_jobs parameter, so can't run in multiple threads
# classifier_svm = svm.SVR()

#  Use SVR support vector regression with several different kernels
# for k in ['linear', 'poly', 'rbf', 'sigmoid']:
#     classifier_svm = svm.SVR(kernel=k)
#     # train the classifier
#     classifier_svm.fit(X_train, y_train)
#     # test the classifier
#     confidence = classifier_svm.score(X_test, y_test)
#     # print(k, confidence)
#     """
#     linear 0.96698805392
#     poly 0.711236066716
#     rbf 0.815715492338
#     sigmoid 0.896009413564
#     """


# Use LinearRegression with n_jobs parameter to run in multiple threads
# -1 use all available threads
# http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
classifier_linear = LinearRegression(n_jobs=-1)

# train the classifier
classifier_linear.fit(X_train, y_train)

# test the classifier
confidence = classifier_linear.score(X_test, y_test)
# print(confidence)
# linear confidence is higher than svm
# 0.971044562584

forecast_set = classifier_linear.predict(X_lately)
print(forecast_set, confidence, forecast_out)
"""
[ 857.56897904  858.86638568  847.97894709  843.53917183  846.90307581
  849.3825905   858.81564451  858.89407076  857.10184499  865.38685965
  861.82648018  857.46670623  854.01976444  851.27589453  851.0825656
  849.03368375  850.80996207  849.55907194  863.96872717  863.34746195
  865.75466149  869.28435532  868.26344175  887.43874297  897.20895289
  898.05773845  900.25665875  932.01004505  940.76638004  945.38434036
  956.53850889  962.86946649  958.62109155] 0.969901311727 33
"""

# stock prices are daily 5 weekdays/week.
# for simplicity, ignore weekends

df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day_seconds = 86400
next_unix = last_unix + one_day_seconds

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day_seconds
    # set all first columns to nan, set final column to i
    df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]

# graph
df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
# plot image appears in a pop up window
plt.show()

# save the trained classifier.
# training a classifier may take a long time and be expensive.
# By saving a trained classifier, we don't have to re-run the training.
# save python classifier object by serializing it a pickle and writing to a file
with open('linearregression.pickle', 'wb') as f:
    pickle.dump(classifier_linear, f)

# read file and deserialize the object
pickle_in = open('linearregression.pickle', 'rb')
classifier_linear = pickle.load(pickle_in)

