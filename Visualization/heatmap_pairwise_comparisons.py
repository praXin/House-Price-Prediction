# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 22:29:17 2018

@author: pravin
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def visualize():
    dataset=pd.read_csv('../dataset.csv')
    del dataset['id'] #as we know id of a house is not important in predicting its price
    corr=dataset.corr()
    sns.heatmap(data=corr,xticklabels='auto',yticklabels='auto')
    plt.title('Heatmap of Pairwise Comparisons')
    plt.xticks(rotation=45)#to be able to clearly read the labels on the x and y axes
    plt.yticks(rotation=0)
    plt.show()
visualize()