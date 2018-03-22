# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 13:38:55 2018

@author: pravin
"""

import pandas as pd
import datetime as dt
from sklearn import linear_model
from sklearn.cross_validation import train_test_split

def implement():
    dataset=pd.read_csv('../dataset.csv',parse_dates=['date'])#parse dates separately gives date, month and year of the given date format

    remove=['id','date','condition','yr_built','yr_renovated','zipcode']
    for ele in remove:
        del dataset[ele]
    wat=pd.get_dummies(dataset['waterfront'])
    del dataset['waterfront']
    view=pd.get_dummies(dataset['view'])
    del dataset['view']
    dataset['waterfront_0']=wat[0]
    dataset['waterfront_1']=wat[1]
    dataset['view_0']=view[0]
    dataset['view_1']=view[1]
    dataset['view_2']=view[2]
    dataset['view_3']=view[3]
    dataset['view_4']=view[4]
    #print(dataset.describe())
    
    X=dataset.iloc[:,1:].values
    y=dataset.iloc[:,0].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .20, random_state=42)
    linReg=linear_model.LinearRegression()
    linReg.fit(X_train,y_train)
    print("Score of Linear Regression: %.2f" % linReg.score(X_test,y_test))
    
implement()