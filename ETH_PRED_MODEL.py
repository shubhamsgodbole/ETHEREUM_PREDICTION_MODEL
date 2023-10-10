#!/usr/bin/env python
# coding: utf-8

# ## ETHEREUM PREDICTION MODEL

# In[1]:


# Ethereum Prediction Model


# In[2]:


# Importing the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import datetime as dt
from sklearn.svm import SVR


# ### Data Preparation and Preprocessing

# In[3]:


# Preparing and Preprocessing the dataset
start = dt.datetime(2015,1,1)
end = dt.datetime.now().date().isoformat()
symbol = 'ETH-USD'
ETH_data = yf.download(symbol, start, end)
ETH_data #the dataframe containing the entire data of ethereum


# In[4]:


# checking for null values
ETH_data.isnull().sum()


# In[29]:


# setting a new variable to calculate the number of days in
# the future whose ETH price is to be predicted
future_days = 14


# In[30]:


# Here, we have shifted the close price up by 5 rows in the '5_Day_Price_Forecast' column
# for a time series analysis

ETH_data[str(future_days) + '_Day_Price_Forecast'] = ETH_data['Close'].shift(-future_days)
ETH_data


# ### Splitting the data into Training and Testing Sets

# In[31]:


# Defining X as the independent variable
# We are considering only the 'Close' column
# Storing the values inside a numpy array instead of a pandas series for further use
# Also, we need to remove the last 5 rows of data as it contains null values now

X = np.array(ETH_data[['Close']]) #creating 2D array as '.fit' accepts only 2D numpy arrays as input
X = X[ : ETH_data.shape[0]-future_days]
X.shape


# In[32]:


# Defining Y as the dependent variable
# Storing the values inside a numpy array instead of a pandas series for further use
# Also, we need to remove the last 5 rows of data as it contains null values now

Y = np.array(ETH_data[[str(future_days) + '_Day_Price_Forecast']])
Y = Y[ : ETH_data.shape[0]-future_days]
Y.shape


# In[33]:


# Splitting the dataset into training and testing sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.15, random_state = 42, shuffle = False)

# test_size = 15 is giving max score
# Also, shuffle has to be false because you have to check the data of last few days, not random


# ### Training the model

# In[34]:


# Importing the model

from sklearn.svm import SVR


# In[35]:


# Training the SVR model

# The value of C is set to 1e3 because a smaller C encourages a smoother boundary
# A low value like 0.0000001 suggests a broader influence, implying that points farther apart are considered similar.

svr_rbf = SVR(kernel = 'rbf', C = 1e3, gamma = 0.0000001)
svr_rbf.fit(x_train, y_train)


# In[36]:


svr_rbf_confidence = svr_rbf.score(x_test, y_test)
svr_rbf_confidence


# In[37]:


ETH_pred = svr_rbf.predict(x_test)
ETH_pred


# In[38]:


y_test


# In[39]:


# For labels on the x-axis

import datetime as dt
from datetime import date, timedelta
today = date.today()
today


# In[40]:


# Visualizing the prediction

# Initializing variables used for plotting
var_font = 'Calibri'
var_heading_font_size = 14
var_title_font_size = 20
var_label_font_size =10

plt.figure(figsize = (12,4.5))
plt.plot(ETH_pred[-90:], label = 'Prediction', lw = 2, alpha = 0.7, color = 'y')
plt.plot(y_test[-90:], label = 'Actual', lw = 2, alpha = 0.7, color = 'g')
plt.title('Prediction vs Actual', font = var_font, fontsize = var_title_font_size, pad=20)
plt.ylabel('Price in USD', font = var_font, fontsize = var_heading_font_size, labelpad=20)
y = np.array([str(today-timedelta(days = 90)), str(today-timedelta(days = 60)), str(today-timedelta(days = 30)), str(today-timedelta(days = 0))])
plt.xlabel('Time', font = var_font, fontsize = var_heading_font_size, labelpad=20)
plt.legend()
plt.xticks(ticks = [0,30,60,90], labels = y, rotation = 45, font = var_font, fontsize = var_label_font_size)
plt.yticks(rotation = 0, font = var_font, fontsize = var_label_font_size)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




