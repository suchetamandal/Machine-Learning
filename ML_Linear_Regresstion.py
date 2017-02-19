{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# Linear Regression -- Install sklearn and quandl library\n",
    "!pip install sklearn\n",
    "!pip install quandl\n",
    "!pip install pandas\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 85,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import quandl\n",
    "import math\n",
    "import numpy as np\n",
    "from sklearn import preprocessing, cross_validation, svm\n",
    "from sklearn.linear_model import LinearRegression\n",
    "\n",
    "df = quandl.get('WIKI/GOOGL')\n",
    "#print(df.head())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 86,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "#Feature Selection\n",
    "df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]\n",
    "#Calculated Price Growth in percentage\n",
    "df['HL_PCT'] = (df['Adj. High']-df['Adj. Close'])/df[('Adj. Close')] * 100\n",
    "df['PCT_change'] = (df['Adj. Close']-df['Adj. Open'])/df[('Adj. Open')] * 100\n",
    "\n",
    "df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 93,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.963696672746\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\suche\\Anaconda3\\lib\\site-packages\\sklearn\\preprocessing\\data.py:167: UserWarning: Numerical issues were encountered when centering the data and might not be solved. Dataset may contain too large values. You may need to prescale your features.\n",
      "  warnings.warn(\"Numerical issues were encountered \"\n"
     ]
    }
   ],
   "source": [
    "#Features and Lebel Making Decesion\n",
    "forecast_col = 'Adj. Close'\n",
    "#Fill all NaN with a specific value\n",
    "df.fillna(-99999, inplace = True)\n",
    "\n",
    "forecast_out = int(math.ceil(0.01*len(df)))\n",
    "\n",
    "#Make df columns\n",
    "df['label'] = df[forecast_col].shift(-forecast_out)\n",
    "df.dropna(inplace = True)\n",
    "\n",
    "#Feature X and lebel y\n",
    "X = np.array(df.drop(['label'],1))\n",
    "y = np.array(df['label'])\n",
    "\n",
    "X = preprocessing.scale(X)\n",
    "df.dropna(inplace = True)\n",
    "y = np.array(df['label'])\n",
    "\n",
    "X_train,X_test,y_train,y_test = cross_validation.train_test_split(X, y, test_size = 0.2)\n",
    "\n",
    "#Make classifier and fit or train the clasifier with test data set\n",
    "clf = LinearRegression()\n",
    "clf.fit(X_train, y_train)\n",
    "accuracy = clf.score(X_test,y_test)\n",
    "print(accuracy)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 95,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.778090860439\n"
     ]
    }
   ],
   "source": [
    "#Change Algorithm\n",
    "clf = svm.SVR() # In accurate algo\n",
    "clf.fit(X_train, y_train)\n",
    "accuracy = clf.score(X_test,y_test)\n",
    "print(accuracy) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# Regression forecasting and predicting\n"
   ]
  }
 ],
 "metadata": {
  "anaconda-cloud": {},
  "kernelspec": {
   "display_name": "Python [conda root]",
   "language": "python",
   "name": "conda-root-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.5.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 1
}
