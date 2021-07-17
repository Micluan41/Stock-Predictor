import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from helper import ExtractData
from regression_model import StockPredictor

# Default constants 

#number of data to use for linear, decisiontree, svm regression model (1 year)
num_days=252
future_days=1

#number of data to use for LSTM model (over 10 year)
start=dt.datetime(2011, 1, 1)
end=dt.datetime.now()

#use previous 60 day data as input for LSTM model
prediction_days=60
# User input stock ticker of the company (this list can be expanded as necessary)
stock_tickers=['AAPL', 'AMZN', 'FB', 'INTC', 'MSFT', 'HPQ', 'ZM', 'NFLX', 'TWTR', 'GS',\
               'MS', 'NVDA', 'MRNA', 'GOOG', 'HPQ', 'BA', 'AMC', 'DIS', 'SBUX', 'TSLA']


# Sidebar: stock selection
st.sidebar.header('User Input Features')
selected_ticker=st.sidebar.selectbox('Stock Ticker', stock_tickers)

# Sidebar: model selection
model_name=['Linear', 'DecisionTree', 'SVM', 'LSTM']
selected_model=st.sidebar.multiselect('Model', model_name, model_name[0])

#change the company name in the below text
st.write(
"""
# Stock Predictor for the next day

* **Data source:** [Yahoo Finance](https://finance.yahoo.com/).

## **Stock historical closing price**

""")

# Web scraping historical stock data for selected company
@st.cache
def load_data(ticker):
    data=ExtractData(selected_ticker, start, end)
    return data
data=load_data(selected_ticker)

#visualize stock history
st.line_chart(data.Close)

st.write(
"""
## **Model Prediction**

Price change and trading advice based on machine learning model 

""")

# Show the prediction result and trading direction of the selected model
for model_name in selected_model:
    if model_name=='LSTM':
        warning_txt='This might take a few minutes!'
        st.text(warning_txt)
    model, prediction=StockPredictor(model_name, data)
    



