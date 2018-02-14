# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 14:42:59 2018

@author: pravin
"""

import pickle
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from scipy.stats import spearmanr, pearsonr

def implement():
    dataset=pd.read_csv('../dataset.csv',parse_dates=['date'])#parse dates separately gives date, month and year of the given date format

    dataset['ordinal_dt']=dataset['date']
    del dataset['date']#Because linear regression is showing a score of 0.01 if attribute's name is "date". What might be the reason for this behaviour?
    dataset['ordinal_dt']=pd.to_datetime(dataset['ordinal_dt'])
    dataset['ordinal_dt']=dataset['ordinal_dt'].map(dt.datetime.toordinal)

    remove=['id','condition','yr_renovated','zipcode']#'yr_built' is not being deleted because when it was considered, the score of the regression incresed by 1%
    for ele in remove:
        del dataset[ele]

    #Adding dummy variables
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
        
    X=dataset.iloc[:,1:-1].values
    y=dataset.iloc[:,0].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .20, random_state=42)
#    scaler = StandardScaler().fit(X_train)
#    X_train_scaled = pd.DataFrame(scaler.transform(X_train), index=X_train.index.values, columns=X_train.columns.values)
#    X_test_scaled = pd.DataFrame(scaler.transform(X_test), index=X_test.index.values, columns=X_test.columns.values)
    
    rf = RandomForestRegressor(n_estimators=500, oob_score=True, random_state=0)
#    pickl='RandomForest.pickle'
#    rf=pickle.load(open(pickl,'rb'))
    rf.fit(X_train, y_train)
#    pickl='RandomForest.pickle'
#    pickle.dump(rf,open(pickl,'wb'))
    
    print("Score of Random Forest Regression: %.2f" % rf.score(X_test,y_test))
    predicted_train = rf.predict(X_train)
    predicted_test = rf.predict(X_test)
    test_score = r2_score(y_test, predicted_test)
    spearman = spearmanr(y_test, predicted_test)
    pearson = pearsonr(y_test, predicted_test)
    print(f'Out-of-bag R-2 score estimate: {rf.oob_score_:>5.3}')
    print(f'Test data R-2 score: {test_score:>5.3}')
    print(f'Test data Spearman correlation: {spearman[0]:.3}')
    print(f'Test data Pearson correlation: {pearson[0]:.3}')
    plt.plot(X_test,y_test,'g')
    plt.plot(X_test,predicted_test,'r')
    plt.show()
    
# =============================================================================
#     linReg=linear_model.LinearRegression(fit_intercept=False)#fit_intercept=False avoids dummy variable traps
#     linReg.fit(X_train,y_train)
#     print("Score of Linear Regression: %.2f" % linReg.score(X_test,y_test))
# =============================================================================
    
implement()