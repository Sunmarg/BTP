# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 02:00:15 2020

@author: Sunmarg Das
"""

import pandas as pd
import numpy as np
df=pd.read_excel("Thermocouple.xlsx")
df.drop(["Year"],axis=1,inplace=True)
df["Deviation"]=df["Deviation"].fillna(df["Deviation"].mean())
X=df.copy()
X.drop(["Temperature"],axis=1,inplace=True)
X.set_index('Tempy',inplace=True)

import matplotlib.pylab as plt
 
plt.rcParams['figure.figsize']=(10,5)
plt.style.use('ggplot')
X.plot()
pd.plotting.lag_plot(X['Deviation'])
pd.plotting.autocorrelation_plot(X['Deviation'])

results={}
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from math import sqrt
from statsmodels.tsa.ar_model import AR

train_data = X[:len(X)-55]
test_data = X[len(X)-55:]

#train the autoregression model
means={}
meana={}
r2={}
rmean={}
model = AR(train_data)
model_fitted = model.fit()
print('The lag value chose is: %s' % model_fitted.k_ar)

print('The coefficients of the model are:\n %s' % model_fitted.params)
predictions = model_fitted.predict(
    start=len(train_data), 
    end=len(train_data) + len(test_data)-1, 
    dynamic=False)

# create a comparison dataframe
compare_df = pd.concat(
    [df['Deviation'].tail(12),
    predictions], axis=1).rename(
    columns={'stationary': 'actual', 0:'predicted'})

#plot the two values
compare_df.plot()
test_data=test_data.reset_index()
predictions=predictions.reset_index()
predictions.drop(["index"],axis=1,inplace=True)
result=0
for i in range(0,len(test_data)):
    result=result+abs(abs(test_data["Deviation"][i]-predictions[0][i])/test_data["Deviation"][i])
result=1-result/len(test_data)
AR_forecast=predictions.copy()
AR_forecast.columns=['AR_forecast']
results['AR']=result
means['AR']=mean_squared_error( test_data["Deviation"], AR_forecast)
meana['AR']=mean_absolute_error( test_data["Deviation"], AR_forecast)
rmean['AR']=sqrt(mean_absolute_error( test_data["Deviation"], AR_forecast))
r2['AR']=r2_score(test_data["Deviation"], AR_forecast)
#AR 93.6

from statsmodels.tsa.arima_model import ARMA
model = ARMA(train_data, order=(0, 2))
model_fit = model.fit(disp=False)
yhat = model_fit.predict(len(train_data), len(X)-1)
print(yhat)
yhat=yhat.reset_index()
yhat.drop(["index"],axis=1,inplace=True)
result=0
for i in range(0,len(test_data)):
    result=result+abs(abs(test_data["Deviation"][i]-yhat[0][i])/test_data["Deviation"][i])
result=1-result/len(test_data)
MA_forecast=yhat.copy()
MA_forecast.columns=['MA_forecast']
results['MA']=result
means['MA']=mean_squared_error( test_data["Deviation"], MA_forecast)
meana['MA']=mean_absolute_error( test_data["Deviation"], MA_forecast)
rmean['MA']=sqrt(mean_absolute_error( test_data["Deviation"], MA_forecast))
r2['MA']=r2_score(test_data["Deviation"], MA_forecast)
#MA 21.2

model = ARMA(train_data, order=(1,1))
model_fit = model.fit(disp=False)
yhat = model_fit.predict(len(train_data), len(df)-1)
print(yhat)
yhat=yhat.reset_index()
yhat.drop(["index"],axis=1,inplace=True)
result=0
for i in range(0,len(test_data)):
    result=result+abs(abs(test_data["Deviation"][i]-yhat[0][i])/test_data["Deviation"][i])
result=1-result/len(test_data)
ARMA_forecast=yhat.copy()
ARMA_forecast.columns=['ARMA_forecast']
results['ARMA']=result
means['ARMA']=mean_squared_error( test_data["Deviation"], ARMA_forecast)
meana['ARMA']=mean_absolute_error( test_data["Deviation"], ARMA_forecast)
rmean['ARMA']=sqrt(mean_absolute_error( test_data["Deviation"], ARMA_forecast))
r2['ARMA']=r2_score(test_data["Deviation"], ARMA_forecast)
#arma 93.37
#0,1,1,120 1 1 1
from pmdarima.arima import auto_arima

stepwise_model = auto_arima(train_data, start_p=1, start_q=1,
                           max_p=3, max_q=3, m=1,
                           start_P=0, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)
print(stepwise_model.aic())
stepwise_model.fit(train_data)
future_forecast = stepwise_model.predict(n_periods=55)
data=train_data
yhat=future_forecast.copy()
result=0
for i in range(0,len(test_data)):
    result=result+abs(abs(test_data["Deviation"][i]-yhat[i])/test_data["Deviation"][i]) if test_data["Deviation"][i]!=0 else 0
result=1-result/len(test_data)
results['ARIMA']=result
yhat1=[]
for i in range(0,len(yhat)):
    yhat1.append(yhat[i])
   
yhat=pd.DataFrame(yhat1,columns=['0'])

ARIMA_forecast=yhat.copy()
ARIMA_forecast.columns=['ARIMA_forecast']
means['ARIMA']=mean_squared_error( test_data["Deviation"], ARIMA_forecast)
meana['ARIMA']=mean_absolute_error( test_data["Deviation"], ARIMA_forecast)
rmean['ARIMA']=sqrt(mean_absolute_error( test_data["Deviation"], ARIMA_forecast))
r2['ARIMA']=r2_score(test_data["Deviation"], ARIMA_forecast)

from statsmodels.tsa.statespace.sarimax import SARIMAX
model = SARIMAX(data, order=(1, 1, 1), seasonal_order=(0, 1, 1, 2))
model_fit = model.fit(disp=False)
# make prediction
yhat = model_fit.predict(len(train_data), len(df)-1)
yhat=yhat.reset_index()
yhat.drop(["index"],axis=1,inplace=True)
result=0
for i in range(0,len(test_data)):
    result=result+abs(abs(test_data["Deviation"][i]-yhat[0][i])/test_data["Deviation"][i])
result=1-result/len(test_data)
SARIMA_forecast=yhat.copy()
SARIMA_forecast.columns=['SARIMA_forecast']
results['SARIMA']=result
means['SARIMA']=mean_squared_error( test_data["Deviation"], SARIMA_forecast)
meana['SARIMA']=mean_absolute_error( test_data["Deviation"], SARIMA_forecast)
rmean['SARIMA']=sqrt(mean_absolute_error( test_data["Deviation"], SARIMA_forecast))
r2['SARIMA']=r2_score(test_data["Deviation"], SARIMA_forecast)
#85.47
'''
from statsmodels.tsa.statespace.sarimax import SARIMAX
model = SARIMAX(data, exog=df["Temperature"].iloc[:len(X)-55], order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))
model_fit = model.fit(disp=False)
exog2 = df['Temperature'].iloc[:len(X)-55]
yhat = model_fit.predict(len(data1), len(data1), exog=[exog2])
print(yhat)

from statsmodels.tsa.vector_ar.var_model import VAR
model = VAR(data)
model_fit = model.fit()
# make prediction
yhat = model_fit.forecast(model_fit.y, steps=1)
print(yhat)

# VARMA example
from statsmodels.tsa.statespace.varmax import VARMAX
from random import random
# contrived dataset with dependency
data = list()

model = VARMAX(data, order=(1, 1))
model_fit = model.fit(disp=False)
# make prediction
yhat = model_fit.forecast()
print(yhat)

from statsmodels.tsa.statespace.varmax import VARMAX
model = VARMAX(data, order=(1, 1))
model_fit = model.fit(disp=False)
# make prediction
yhat = model_fit.forecast()
print(yhat)
model = VARMAX(data, exog=data_exog, order=(1, 1))
model_fit = model.fit(disp=False)
# make prediction
data_exog2 = [[100]]
yhat = model_fit.forecast(exog=data_exog2)
print(yhat)
'''
data=train_data.copy()
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
model = SimpleExpSmoothing(data)
model_fit = model.fit()
# make prediction
yhat = model_fit.predict(len(data), len(df)-1)
print(yhat)
yhat=yhat.reset_index()
yhat.drop(["index"],axis=1,inplace=True)
result=0
for i in range(0,len(test_data)):
    result=result+abs(abs(test_data["Deviation"][i]-yhat[0][i])/test_data["Deviation"][i]) if test_data["Deviation"][i]!=0 else 0
result=1-result/len(test_data)
SETS_forecast=yhat.copy()
SETS_forecast.columns=['SETS_forecast']
results['SETS']=result
means['SETS']=mean_squared_error( test_data["Deviation"], SETS_forecast)
meana['SETS']=mean_absolute_error( test_data["Deviation"], SETS_forecast)
rmean['SETS']=sqrt(mean_absolute_error( test_data["Deviation"], SETS_forecast))
r2['SETS']=r2_score(test_data["Deviation"], SETS_forecast)
from statsmodels.tsa.holtwinters import ExponentialSmoothing
model = ExponentialSmoothing(data)
model_fit = model.fit()
# make prediction
yhat = model_fit.predict(len(data), len(df)-1)
print(yhat)
yhat=yhat.reset_index()
yhat.drop(["index"],axis=1,inplace=True)
result=0
for i in range(0,len(test_data)):
    result=result+abs(abs(test_data["Deviation"][i]-yhat[0][i])/test_data["Deviation"][i]) if test_data["Deviation"][i]!=0 else 0
result=1-result/len(test_data)
ETS_forecast=yhat.copy()
ETS_forecast.columns=['ETS_forecast']
results['ETS']=result
means['ETS']=mean_squared_error( test_data["Deviation"], ETS_forecast)
meana['ETS']=mean_absolute_error( test_data["Deviation"], ETS_forecast)
rmean['ETS']=sqrt(mean_absolute_error( test_data["Deviation"], ETS_forecast))
r2['ETS']=r2_score(test_data["Deviation"], ETS_forecast)
from statsmodels.tsa.holtwinters import Holt
model = Holt(data)
model_fit = model.fit()
# make prediction
yhat = model_fit.predict(len(data), len(df)-1)
print(yhat)
yhat=yhat.reset_index()
yhat.drop(["index"],axis=1,inplace=True)
result=0
for i in range(0,len(test_data)):
    result=result+abs(abs(test_data["Deviation"][i]-yhat[0][i])/test_data["Deviation"][i]) if test_data["Deviation"][i]!=0 else 0
result=1-result/len(test_data)
HOLT_forecast=yhat.copy()
HOLT_forecast.columns=['HOLT_forecast']
results['HOLT']=result
means['HOLT']=mean_squared_error( test_data["Deviation"], HOLT_forecast)
meana['HOLT']=mean_absolute_error( test_data["Deviation"], HOLT_forecast)
rmean['HOLT']=sqrt(mean_absolute_error( test_data["Deviation"], HOLT_forecast))
r2['HOLT']=r2_score(test_data["Deviation"], HOLT_forecast)
from tbats import TBATS

estimator = TBATS(seasonal_periods=(5, 12))
model = estimator.fit(train_data)
# Forecast 365 days ahead
y_forecast = model.forecast(steps=55)

data=train_data
yhat=y_forecast.copy()
yhat=pd.DataFrame(yhat,columns=['0'])
result=0
for i in range(0,len(test_data)):
    result=result+abs(abs(test_data["Deviation"][i]-y_forecast[i])/test_data["Deviation"][i]) if test_data["Deviation"][i]!=0 else 0
result=1-result/len(test_data)
TBATS_forecast=yhat.copy()
TBATS_forecast.columns=['TBATS_forecast']
results['TBATS']=result
means['TBATS']=mean_squared_error( test_data["Deviation"], TBATS_forecast)
meana['TBATS']=mean_absolute_error( test_data["Deviation"], TBATS_forecast)
rmean['TBATS']=sqrt(mean_absolute_error( test_data["Deviation"], TBATS_forecast))
r2['TBATS']=r2_score(test_data["Deviation"], TBATS_forecast)
from statsmodels.tsa.vector_ar.var_model import VAR
m=df.copy()
n=df.copy()
m.set_index('Tempy',inplace=True)

mtrain_data = m[:len(X)-55]
ntest_data = m[len(X)-55:]
model = VAR(endog=mtrain_data)
model_fit = model.fit()
yhat = model_fit.forecast(model_fit.y, steps=55)

VAR_forecast=yhat.copy()
print(yhat)

result=0
for i in range(0,len(test_data)):
    result=result+abs(abs(test_data["Deviation"][i]-yhat[i][1])/test_data["Deviation"][i]) if test_data["Deviation"][i]!=0 else 0
result=1-result/len(test_data)
yhat1=[]
results['VAR']=result
for i in range(0,len(yhat)):
    yhat1.append(yhat[i][1])
   
yhat=pd.DataFrame(yhat1,columns=['0'])
VAR_forecast=yhat.copy()
VAR_forecast.columns=['VAR_forecast']
means['VAR']=mean_squared_error( test_data["Deviation"], VAR_forecast)
meana['VAR']=mean_absolute_error( test_data["Deviation"], VAR_forecast)
rmean['VAR']=sqrt(mean_absolute_error( test_data["Deviation"], VAR_forecast))
r2['VAR']=r2_score(test_data["Deviation"], VAR_forecast)
from statsmodels.tsa.statespace.varmax import VARMAX

model = VARMAX(mtrain_data, order=(2,3))
model_fit = model.fit(disp=False)
# make prediction
yhat = model_fit.predict(len(train_data), len(df)-1)
print(yhat)
yhat=yhat.reset_index()
yhat.drop(["index"],axis=1,inplace=True)
result=0
for i in range(0,len(test_data)):
    result=result+abs(abs(test_data["Deviation"][i]-yhat["Deviation"][i])/test_data["Deviation"][i]) if test_data["Deviation"][i]!=0 else 0
result=1-result/len(test_data)
yhat1=[]

for i in range(0,len(yhat)):
    yhat1.append(yhat["Deviation"][i])
   
yhat=pd.DataFrame(yhat1,columns=['0'])
VARMA_forecast=yhat.copy()
VARMA_forecast.columns=['VARMA_forecast']
results['VARMA']=result

means['VARMA']=mean_squared_error( test_data["Deviation"], VARMA_forecast)
meana['VARMA']=mean_absolute_error( test_data["Deviation"], VARMA_forecast)
rmean['VARMA']=sqrt(mean_absolute_error( test_data["Deviation"], VARMA_forecast))
r2['VARMA']=r2_score(test_data["Deviation"], VARMA_forecast)
k=df["Tempy"].iloc[len(X)-55:]

import operator
sorted_x = sorted(results.items(), key=operator.itemgetter(1),reverse=True)
import collections

sorted_dict = collections.OrderedDict(sorted_x)

j=0
s=0
for i in sorted_dict.items():
    if(j<3):
        s+=i[1]
    j=j+1
    
s=s/3
print(s)
    
k=k.to_frame()
k=k.reset_index()
k.drop(["index"],axis=1,inplace=True)

f1=k.join(AR_forecast, lsuffix="_left", rsuffix="_right")
f1=f1.join(MA_forecast, lsuffix="_left", rsuffix="_right")
f2=f1.join(ARMA_forecast, lsuffix="_left", rsuffix="_right")
f3=f2.join(ARIMA_forecast, lsuffix="_left", rsuffix="_right")
f4=f3.join(SARIMA_forecast, lsuffix="_left", rsuffix="_right")
f5=f4.join(ETS_forecast, lsuffix="_left", rsuffix="_right")
f6=f5.join(SETS_forecast, lsuffix="_left", rsuffix="_right")
f7=f6.join(HOLT_forecast, lsuffix="_left", rsuffix="_right")
f8=f7.join(VAR_forecast, lsuffix="_left", rsuffix="_right")
f9=f8.join(VARMA_forecast, lsuffix="_left", rsuffix="_right")
f10=f9.join(TBATS_forecast, lsuffix="_left", rsuffix="_right")

f10.to_excel("output2.xlsx",
             sheet_name='Sheet_name_1')

frames = [k,AR_forecast,MA_forecast,ARMA_forecast,ARIMA_forecast,SARIMA_forecast,VAR_forecast,VARMA_forecast,TBATS_forecast,SETS_forecast,ETS_forecast,HOLT_forecast]

result_plot = pd.concat(frames)

