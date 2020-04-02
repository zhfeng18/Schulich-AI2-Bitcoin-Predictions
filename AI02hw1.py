#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"></ul></div>

# In[ ]:


# I used daily 'close'
# the 3rd function def ma20_mse is only used for calculating the baseline MA20 mse
# The ARIMA (order = 2, 0, 2) mse (0.0014) beats the baseline mse (0.0023)
# one can use codes at the bottom part to test


# In[1]:


import pandas as pd
import numpy as np
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMA

import warnings
warnings.filterwarnings('ignore')


# In[2]:


# loads and prprocesses the data
def preprocess(data_file):
    # read csv file
    series = pd.read_csv(data_file)
    
    # use 'close'
    # fill the missing by replacing them with the latest available price
    series['close'] = series['close'].ffill()

    # pct change differencing
    x =  np.diff(series['close']) / series['close'][:-1]
    
    # train test split for inputs and targets
    N = series.shape[0]
    split_t = int(N * 0.8)  # 0.8 to training & 0.2 to validation
    x_train = x[:split_t].to_numpy()  # first 0.8 as inputs X
    x_valid = x[split_t:].to_numpy()  # last 0.2 as targets y
    
    return x_train, x_valid # X and y


# In[3]:


# there is no model file
# the ARIMA model weights (parameters) are (2, 0, 2)

def model(x_train, x_valid, order=(2, 0, 2)):
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

    return forecasts


# In[4]:


# example test code

# # use your own file name
# X, y = preprocess('BTCUSD_1d.csv')

# # wait for forecasts array
# forecasts = model(X, y)
# forecasts


# In[ ]:





# In[5]:


# def ma20_mse(data_file):
#     series = pd.read_csv(data_file)
#     series['close'] = series['close'].ffill()
#     x =  np.diff(series['close']) / series['close'][:-1]
#     ma20_all = x.rolling(window=20).mean()
#     ma20_mse = mean_squared_error(x[19:], ma20_all.dropna())
#     return ma20_mse


# In[6]:


# ma20_mse = ma20_mse('BTCUSD_1d.csv')
# ma20_mse


# In[7]:


# My MA20 mse is 0.0022758139173868143
# arima_mse = mean_squared_error(y, forecasts)
# print('ARIMA validation MSE: ', arima_mse)
# print('My model result MSE beats the baseline MA20 MSE:', arima_mse < ma20_mse)


# In[ ]:




