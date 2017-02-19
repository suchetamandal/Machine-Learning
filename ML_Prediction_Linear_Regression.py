
# coding: utf-8

# In[ ]:

# Linear Regression -- Install sklearn and quandl library
get_ipython().system('pip install sklearn')
get_ipython().system('pip install quandl')
get_ipython().system('pip install pandas')


# In[12]:

import pandas as pd
import quandl
import math, datetime
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

df = quandl.get('WIKI/GOOGL')
#print(df.head())


# In[13]:

#Feature Selection
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
#Calculated Price Growth in percentage
df['HL_PCT'] = (df['Adj. High']-df['Adj. Close'])/df[('Adj. Close')] * 100
df['PCT_change'] = (df['Adj. Close']-df['Adj. Open'])/df[('Adj. Open')] * 100

df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]


# In[14]:

#Features and Lebel Making Decesion
forecast_col = 'Adj. Close'
#Fill all NaN with a specific value
df.fillna(-99999, inplace = True)

forecast_out = int(math.ceil(0.01*len(df)))

#Make df columns
df['label'] = df[forecast_col].shift(-forecast_out)

#Feature X and lebel y
X = np.array(df.drop(['label'],1))
X = preprocessing.scale(X)
X = X[:-forecast_out]
X_lately = X[-forecast_out:]

df.dropna(inplace = True)
y = np.array(df['label'])

X_train,X_test,y_train,y_test = cross_validation.train_test_split(X, y, test_size = 0.2)

#Make classifier and fit or train the clasifier with test data set
clf = LinearRegression()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test,y_test)
print(accuracy)


# In[15]:

# Regression forecasting and predicting
forecast_set = clf.predict(X_lately) #X_lately contains data of last 30 days
print (forecast_set, accuracy, forecast_out)

df['Forecast'] = np.nan
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]
    
df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()


# In[ ]:



