# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 00:31:52 2018

@author: pravin
"""

import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime as dt
import test1
import post_prediction as pp


def acceptVals(dated,no_beds,no_baths,area_house,area_lot,no_floors,water,view,condition,grade,
                       area_above,area_base,year_built,year_ren,zipcode,lat,long):
    print("Got the data from the GUI\n")
    print("Sending to predictor\n")
    view=int(view)
    view0=0
    view1=0
    view2=0
    view3=0
    view4=0
    if view==0:
        view0=1
    elif view==1:
        view1=1
    elif view==2:
        view2=1
    elif view==3:
        view3=1
    elif view==4:
        view4=1
        
    dat=dt.strptime(dated,'%Y-%m-%d').date()
    ordinaldt=dat.toordinal()
    pred(int(no_beds),float(no_baths),int(area_house),int(area_lot),float(no_floors),int(water),
         int(condition),int(grade),int(area_above),int(area_base),int(year_built),int(year_ren),
         int(zipcode),float(lat),float(long),ordinaldt,view0,view1,view2,view3,view4)
    
def pred(no_beds,no_baths,area_house,area_lot,no_floors,water,condition,grade,
                       area_above,area_base,year_built,year_ren,zipcode,lat,long,
                       ordinaldt,view0,view1,view2,view3,view4):
    
    l=[[no_beds,no_baths,area_house,area_lot,no_floors,water,condition,grade,
                       area_above,area_base,year_built,year_ren,zipcode,lat,long,
                       ordinaldt,view0,view1,view2,view3,view4]]
    print("Predictor received the attributes\n")
    print("Predicting...\n")
    gdb=pickle.load(open('GradientBoost2.pickle','rb'))
    pred_price=gdb.predict(l)
    print("Price: $",pred_price)
    postpred(pred_price)
    
def postpred(pred_price):
    pp.vp_start_gui(pred_price)

if __name__ == '__main__':
    test1.vp_start_gui()
     