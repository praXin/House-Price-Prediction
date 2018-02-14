# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 21:15:51 2018

@author: pravi
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#since heatmap shows condition is not very useful for predicting the price (i.e. condition does not influence the value of price)
def visualize():
    dataset=pd.read_csv('../dataset.csv')
    del dataset['id'] #as we know id of a house is not important in predicting its price
    sns.regplot(x='long',y='price',data=dataset,scatter=True,marker='x')
    plt.title('Longitude vs Price')
    plt.xlabel('Longitude')
    plt.ylabel('Price')
    plt.show()
visualize()