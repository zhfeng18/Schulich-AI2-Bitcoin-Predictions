#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"></ul></div>

# In[1]:


from pandas import read_csv
from matplotlib import pyplot
from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from sklearn.model_selection import train_test_split
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import f1_score
import math
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMA
from pmdarima.arima import auto_arima
from pandas.plotting import autocorrelation_plot
from joblib import dump, load

import warnings
warnings.filterwarnings('ignore')

from pytrends.request import TrendReq
from datetime import datetime
from sklearn.linear_model import LinearRegression


# In[2]:


# loads and prprocesses the data (I'm using the daily one)
# and X2 Google Trends data
def preprocess(data_file):
    global x_train
    global x_valid
    global x2
   
    # read csv file
    series = pd.read_csv(data_file)
    
    # use 'close'
    # fill the missing by replacing them with the latest available price
    series['close'] = series['close'].ffill()

    # percentage change differencing
    x =  np.diff(series['close']) / series['close'][:-1]
    
    N = series.shape[0]
    split_t = int(N * 0.8)  # 0.8 to training & 0.2 to validation
    x_train = x[:split_t].to_numpy()  # first 0.8
    x_valid = x[split_t:].to_numpy()  # last 0.2
    
    
    # Google Trends bitcoin
    # Eastern standard time UTC-05 in minutes -300
    # I want '2018-03-10' to '2019-10-23' to match hw1 predictions (because of differencing)
    # Google Trends weekly data captures every Sunday
    # '2018-03-10' is Saturday, use '2018-03-04' and drop the first 6 obs
    # '2019-10-23' is Wednesday, use '2019-10-27' and drop the last 5 obs
    start_date = '2018-03-04'
    end_date = '2019-10-27'
    pytrends = TrendReq(hl='en-US', tz=-300)
    kw_list=['bitcoin']
    pytrends.build_payload(kw_list, cat=0, timeframe=start_date + ' ' + end_date, geo='', gprop='')
    bitcoin_trends = pytrends.interest_over_time()
    # resampling to daily
    bitcoin_daily = bitcoin_trends.resample('1d').interpolate(method='linear').drop(['isPartial'], axis='columns')
    bitcoin_daily = bitcoin_daily[6:len(bitcoin_daily)-4]
    x2 = bitcoin_daily['bitcoin'].to_numpy()
    x2 = np.diff(x2) / x2[:-1] # differencing
    
    return x_train, x_valid, x2
    # x_train = x1, x_valid = y


# In[3]:


# input_file
# x1, y, x2 = preprocess('BTCUSD_1d.csv')



# In[4]:


def model(x_train, x_valid, order=(2, 0, 2)):
    global y1_hat
    forecasts = np.zeros(x_valid.shape)
    history = list(x_train)

    for i in range(x_valid.shape[0]):

        model = ARIMA(history, order=order)
        model_fit = model.fit(disp=0)
        y_hat = model_fit.forecast()[0]
        forecasts[i] = y_hat
        y = x_valid[i]
        history.append(y)
        if (i % 20) == 0:
            print('predicted=%f, expected=%f' % (y_hat, y))

    y1_hat = forecasts
    return y1_hat


# In[5]:


# y1_hat = model(x1, y)


# In[4]:


# hw1_forecast = pd.read_csv('arima_forecasts.csv')
# y1_hat = hw1_forecast['forecast'].to_numpy()


# In[25]:


def combiner(y1_hat, x2, filename_comb=('ai2_reg1.joblib')):
    global y2_valid
    global y2_hat
    reg = load(filename_comb)
    
    y1_hat = y1_hat.reshape(len(y1_hat), 1)
    x2 = x2.reshape(len(x2), 1)
    
    # linear regression independent variable array
    iv = np.concatenate((y1_hat, x2), axis=1)
    
#     N = iv.shape[0] # 592
#     split_t = int(N * 0.8)  # value = 473, 0.8 to training & 0.2 to validation
#     iv_train = iv[:split_t]  # first 0.8, from index 0 to 472
#     iv_valid = iv[split_t:]  # last 0.2, from index 473
#     y2_train = y[:split_t] # len=473
#     y2_valid = y[split_t:] # len=119
#     reg = LinearRegression().fit(iv_train, y2_train)
    
    y2_hat = reg.predict(iv)
    
    return y2_hat


# In[26]:


# y2_hat = combiner(y1_hat, x2)
# mse_y2 = mean_squared_error(y, y2_hat)
# print('validation MSE: %.9f' % mse_y2)
# # validation MSE: 0.001391665
