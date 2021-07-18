import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import sklearn
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from helper import ScaleData, InverseScale

#LSTM model parameter
lstm_size=64
dropout=0.2
dense_size=10
epoch=20
batch_size=64


def StockPredictor(model_name, data, num_days=252, future_days=1, prediction_days=60):
    
    '''
    Create and train the requested model and make the prediction for the next day stock price
    
    Args:
        model_name (str): user input model name
        data (pandas dataframe): historical data
        num_days (int): number of previous data to build regression model
        future_days (int): predict the future nth day stock price
        prediction_days (int): number of input features for LSTM model
        
    Returns:
        model (model object): the model after training
        prediction (float): next day stock price prediction
    '''
    
    data_close=data.Close.values.reshape(-1,1)
    
    if model_name in ['Linear', 'DecisionTree', 'SVM']:
        
        # select the recent num_days data for regression model
        data_reg=data_close[-num_days:]
        
        X=data_reg[:-future_days]
        y=data_reg[future_days:]
        
        #input to predict next day price
        x_next_day=data_reg[-future_days].reshape(-1,1)       
        

        # train test split, set shuffle to false
        x_train, x_test, y_train, y_test=train_test_split(X, y, test_size=0.2, shuffle=False)
        
        # create model and feed the training dataset
        if model_name=='Linear':
            model=LinearRegression().fit(x_train, y_train)
            
        if model_name=='DecisionTree':
            model=DecisionTreeRegressor(max_depth=8).fit(x_train, y_train)
            
        if model_name=='SVM':
            model=SVR(kernel='poly', C=0.001).fit(x_train, y_train.ravel())
        
        # predict next day stock price
        prediction=model.predict(x_next_day)
        prediction=prediction.ravel()[0]
        
        # print the performance of the model on training and testing dataset
        y_fit, y_pred=print_metrics(model_name, model, x_train, y_train, x_test, y_test)
        pred_last=y_pred.ravel()[-1]
       
        
    elif model_name=='LSTM':
        
        # normalize the data between (0,1)
        scaler, scaled_data=ScaleData(data_close)
        # train test split for lstm model in time sequence
        x_train, x_test, y_train, y_test=train_test_split_lstm(scaled_data, prediction_days)
        # get the input data required to predict next day price
        x_next_day=np.array([scaled_data[len(scaled_data)-prediction_days:,0]])
        x_next_day=np.reshape(x_next_day, (x_next_day.shape[0], x_next_day.shape[1], 1))
        
        # create LSTM model and feed the training set
        input_size=(x_train.shape[1], x_train.shape[2])
        model=create_model(input_size, lstm_size, dropout, dense_size)
        model.fit(x_train, y_train, epochs=epoch, batch_size=batch_size, validation_data=(x_test, y_test), verbose=0)
        
        # predict next day stock price
        prediction=model.predict(x_next_day)
        prediction=InverseScale(scaler, prediction)
        prediction=prediction.ravel()[0]
        
        # print the performance of the model on training and testing dataset
        y_fit, y_pred=print_metrics(model_name, model, x_train, y_train, x_test, y_test, scaler)
        pred_last=y_pred.ravel()[-1]
    
    else:
        print('Error! Please select an applicable model name!')
        return None, None
    
    # give trading suggesting
    Trade_suggest(pred_last, prediction)
    
    # visualize recent trend (recent two months)
    Visualize_recent_trend(model_name, data, y_pred.ravel())
    
    return model, prediction
        

def train_test_split_lstm(data, prediction_days, train_size=0.9):
    
    '''Split train and test data for LSTM model
    
    Split train and test data according to the ratio in time order
    
    Args:
        data (numpy array): historical stock price data
        prediction_days: number of previous day price as input 
        train_size: the ratio of training set
        
    Returns:
        x_trian (numpy array): training set input
        x_test (numpy array): test set input
        y_trian (numpy array): training set output
        y_test (numpy array): test set output
    '''
    
    n_train=int(len(data)*train_size)
    x_train, y_train, x_test, y_test=[], [], [], []
    
    for i in range(prediction_days, n_train):
        x_train.append(data[i-prediction_days:i, 0])
        y_train.append(data[i, 0])
    
    for j in range(n_train, len(data)):
        x_test.append(data[j-prediction_days:j, 0])
        y_test.append(data[j, 0])
    
    x_train, y_train=np.array(x_train), np.array(y_train)
    x_test, y_test=np.array(x_test), np.array(y_test)
    x_train=np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test=np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    
    return x_train, x_test, y_train, y_test

def create_model(input_size, lstm_size=64, dropout=0.2, dense_size=10):
    
    '''
    Create a lstm model with two lstm layers, two dropout layers 
    and followed by two dense layers
    
    Args:
        input_size (tuple): input shape to feed the first lstm layer
        lstm_size (int): dimension of lstm layer output
        dropout (float): dropout probability
        dense_size (int): dimension of dense layer output
    
    Returns:
        model: lstm model
    
    '''
    
    model=Sequential()
    model.add(LSTM(units=lstm_size, return_sequences=True, input_shape=input_size))
    model.add(Dropout(dropout))
    model.add(LSTM(units=lstm_size, return_sequences=False))
    model.add(Dropout(dropout))
    model.add(Dense(units=dense_size))
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model

def print_metrics(model_name, model, x_train, y_train, x_test, y_test, scaler=None):
    
    '''
    print the performance of the model on training set and test set
    
    Args:
        model_name (str): the name of the model
        model: the model object after training
        x_trian (numpy array): training set input
        x_test (numpy array): test set input
        y_trian (numpy array): training set output
        y_test (numpy array): test set output
        scaler: scaler for LSTM model
        
    Returns:
        y_fit (numpy array): fitted results on training data set
        y_pred (numpy array): predicted results on testing data set
    '''
        
    y_fit=model.predict(x_train)
    y_pred=model.predict(x_test)
    
    if model_name=='LSTM':
        y_fit=InverseScale(scaler, y_fit)
        y_pred=InverseScale(scaler, y_pred)
        y_train=InverseScale(scaler, y_train.reshape(-1,1))
        y_test=InverseScale(scaler, y_test.reshape(-1,1))
    
    model_txt='{} model: '.format(model_name)
    train_metric_txt='For training dataset: mean square error is {}, r2_score is {}'.format(mean_squared_error(y_train, y_fit),\
                                                                                            r2_score(y_train, y_fit))

    test_metric_txt='For test dataset: mean square error is {}, r2_score is {}'.format(mean_squared_error(y_test, y_pred),\
                                                                                      r2_score(y_test, y_pred))
    
    percentage=np.sqrt(mean_squared_error(y_test, y_pred))/np.mean(y_test)*100.0
    percentage_txt='The prediction is within {}% of the actual price on average'.format(percentage)
    
    print(model_txt)
    print(train_metric_txt)
    print(test_metric_txt)
    print(percentage_txt)
    
    return y_fit, y_pred

def Trade_suggest(today, tomorrow):
    
    '''
    Give trade suggestion based on the prediction stock price of today and the next day
    
    Args:
        today (float): predicted price for current day
        tomorrow (float): predicted price for the next day
    
    Returns:
        None
    '''
    
    pred_txt='The next day stock price is {}. '.format(round(tomorrow, 4))
    print(pred_txt)
    
    if today<=tomorrow:
        change_percent=(tomorrow-today)/today*100.0
        suggestion='The stock price will increase by {}%. In the direction of LONG'.format(round(change_percent,3))
        print(suggestion)
        
    else:
        change_percent=(today-tomorrow)/today*100.0
        suggestion='The stock price will decrease by {}%. In the direction of SHORT'.format(round(change_percent,3))
        print(suggestion)
        
def Visualize_recent_trend(model_name, data, predicted_data, days=60):
    
    '''
    Print the actual stock price and the predicted price in recent days to show the trend
    
    Args:
        model_name (str): the name of the predict model
        data (pandas dataframe)： the historical stock price
        predicted_data (numpy array): predicted price on test set
        days: recent days to plot
    
    Returns:
        None
    '''
    if predicted_data.shape[0]<60:
        days=predicted_data.shape[0]
    
    # select a subset of data to plot
    data_plot=data[-days:]
    data_plot['Prediction']=predicted_data[-days:]
    
    fig, ax=plt.subplots(figsize=(8, 4))
    ax.set_title('Actual vs Predicted Stock Price in Recent 60 Days')
    ax.plot(data_plot['Close'])
    ax.plot(data_plot['Prediction'], '-.')
    ax.set_xlabel('Date')
    ax.set_ylabel('Closing Price $')
    ax.legend(['Actual Price', model_name+' Regression'])
    plt.show()
    
    
    
    
    
    