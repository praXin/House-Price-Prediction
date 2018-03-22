# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 00:41:28 2018

@author: pravin
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def visualize():
    dataset=pd.read_csv('../dataset.csv')
    del dataset['id'] #as we know id of a house is not important in predicting its price
    sns.regplot(x='bathrooms',y='price',data=dataset,scatter=True,marker='x')
    plt.title('Bathrooms vs Price')
    plt.xlabel('Number of Bathrooms')
    plt.ylabel('Price')
	plt.show()

visualize()