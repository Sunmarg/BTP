# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 01:00:34 2020

@author: Sunmarg Das
"""
import pandas as pd
import numpy as np
df=pd.read_excel("Thermocouple.xlsx")
df.drop(["Year"],axis=1,inplace=True)
df["Deviation"]=df["Deviation"].fillna(df["Deviation"].mean())

df=df.rename(columns={'Tempy':'Temp_year'})
X=df.copy()
X.drop(["Temperature"],axis=1,inplace=True)
X.set_index('Temp_year',inplace=True)
import matplotlib.pylab as plt
import seaborn as sns

# Use seaborn style defaults and set the default figure size
sns.set(rc={'figure.figsize':(10,3)})
y=X["Deviation"].plot(linewidth=0.5);
y.set_ylabel('Deviation')

axes = X["Deviation"].plot(marker='.', alpha=0.5, linestyle='None', figsize=(10, 3), subplots=True)
for ax in axes:
    ax.set_ylabel('Deviation')
    
y=X["Deviation"].iloc[448:560].plot()
y=X["Deviation"].iloc[363:484].plot()
y.set_ylabel('Deviation');
df=pd.read_excel("Size.xlsx")
sns.boxplot(data=df, x='Year', y='Deviation')
