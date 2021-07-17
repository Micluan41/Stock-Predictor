import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from sklearn.preprocessing import MinMaxScaler

def ExtractData(ticker, start='2011-01-01', end=dt.datetime.now()):
    '''
    Extract historical data for a company's stock 
    
    Args:
        ticker (str): the stock ticker for the company
        start (datetime): the starting time of the stock history
        end (datetime): the end time of the stock history   
        
    Returns:
        data (pandas dataframe): the stock data history
    '''
    ticker=yf.Ticker(ticker)
    data=ticker.history(start=start, end=end, interval="1d")
    
    return data

def ScaleData(data):
    '''
    Normalize the data to a value between (0,1)
    
    Args:
        data (numpy array): the dataset to normalize 
    
    Returns:
        scaler: scale object 
        scaled_data (numpy array): normalized data
    '''
    
    data=data.reshape(-1,1)
    scaler=MinMaxScaler(feature_range=(0,1))
    scaled_data=scaler.fit_transform(data)
    
    return scaler, scaled_data

def InverseScale(scaler, data):
    '''
    Scale a (0,1) value back to stock price
    
    Args:
        scaler: scale object
        data(numpy array): data to scale back to price
    
    Returns:
        price (numpy array): scaled back to price value
    '''
    
    price=scaler.inverse_transform(data)
    return price


