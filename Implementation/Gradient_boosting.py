# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 14:29:25 2018
@author: pravin
"""

import pickle
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

def implement():
    dataset=pd.read_csv('../dataset.csv',parse_dates=['date'])#parse dates separately gives date, month and year of the given date format

    dataset['ordinal_dt']=dataset['date']
    del dataset['date']#Because linear regression is showing a score of 0.01 if attribute's name is "date". What might be the reason for this behaviour?
    dataset['ordinal_dt']=pd.to_datetime(dataset['ordinal_dt'])
    dataset['ordinal_dt']=dataset['ordinal_dt'].map(dt.datetime.toordinal)

    remove=['id','condition','yr_renovated','zipcode']#'yr_built' is not being deleted because when it was considered, the score of the regression incresed by 1%
    for ele in remove:
        del dataset[ele]

    view=pd.get_dummies(dataset['view'])
    del dataset['view']
    dataset['view_0']=view[0]
    dataset['view_1']=view[1]
    dataset['view_2']=view[2]
    dataset['view_3']=view[3]
    dataset['view_4']=view[4]
        
    X=dataset.iloc[:,1:].values
    y=dataset.iloc[:,0].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .20, random_state=42)
  
# =============================================================================
#     gdb=GradientBoostingRegressor(n_estimators=1000,max_depth=4, min_samples_split=2,learning_rate=0.09, loss='ls')
#     gdb.fit(X_train, y_train)
#     pickl='GradientBoost.pickle'
#     pickle.dump(gdb,open(pickl,'wb'))
# =============================================================================
    gdb=pickle.load(open('GradientBoost.pickle','rb'))
    print("Score of Gradient Boosting Regression: %.4f" % gdb.score(X_test,y_test))
    predicted_test = gdb.predict(X_test)
    plt.plot(X_test,y_test,'g')
    plt.plot(X_test,predicted_test,'r')
    plt.show()
    
implement()