# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 14:11:34 2018

@author: pravin
"""

import pandas as pd
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt

def visualize():
    dataset=pd.read_csv('../dataset.csv',parse_dates=['date'])#parse dates separately gives date, month and year of the given date format
    #print(dataset['date'])
    #print(dataset.describe())
    
#    times=dataset['date'].dt.time
#    print(times) shows that time is 00:00:00 for all records
    dataset['date']=pd.to_datetime(dataset['date'])
    dataset['date']=dataset['date'].map(dt.datetime.toordinal)#Converting to ordinal dates to be able to plot graph and train models
    sns.regplot(x='date',y='price',data=dataset,scatter=True,marker='x')
    plt.title('Date vs Price')
    plt.xlabel('Ordinal Dates')
    plt.ylabel('Price')
    plt.show()
    
visualize()