#!/usr/bin/env python3

import pandas as pd
import quandl

# dataframe
df = quandl.get('WIKI/GOOGL')

# print(df.head())
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
