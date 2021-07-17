# Stock-Predictor
Captone Project for Udacity Data Science Nanodegree 

## **Table of Contents**
1. Installation 
2. Project Motivation
3. Project Definition
4. File Description
5. Running Instruction
6. Analysis and Results
7. Conclusion
8. Licensing, Authors, Acknowledgements

## **Installation**
To run the scripts/notebooks of this project, you'll need the following python libraries installed.
* **streamlit, yfinance, pandas, numpy, matplotlib, tensorflow, sklearn, datetime**

## **Project Motivation**
Quantitative methods and financial models have been widely studied and used to better understand market behavior and make profitable investments and trades. With the development of technology and Internet, it is easy and fast to access historical stock prices and company performance data, which is suitable for machine learning algorithms to process.

What if we have a stock predictor that utilize historical data of a company, process the data and create a model to learn its market behavior and make prediction for the future? Is the prediction of the price going to be accurate enough to give us the correct trading guide? If so, that stock predictor could worth a lot and bring you wealth.
In the project, I want to apply what I've learned from this program to build such a stock predictor. A few different models and approaches will be implemented and compared to get the better results. 

## **Project Definition**
Develop a stock predictor that predict any company's future stock given its historical stock price. Build an web application that shows the stock history, future price, trading direction according to user selected company stock ticker and model names. 

Some solutions including regression (linear, polynomial) models are provided as well as deep learning model (LSTM). Compare all these models and find the best fit (the least mean squred error).

## **File Description**
To run the notebook or the web application, you have to first install the necessary python libraries in requirements.txt

There are 2 notebooks and 3 python scripts available to showcase work related to the above questions. 'Stock_predictor' is the notebook that does all the studies on the different models. It starts with web scraping, data wrangling, visualization, modeling and evaluation. Some markdown cells assists to understand the working process.

Then, I translated all the work in the above notebook to two python scripts 'helper.py' and 'regression_model.py'/'regression_model_old.py'. 'helper.py' helps scape data from Yahoo finance API and preprocess data. 'regression_model.py' contains all the model creation, training, evaluation and printing the metrics and the outcome. 

'User_interface' is a notebook that's user-friendly to check the predicted stock price for your choice. The other purpopse of this notebook is to see how some parameters affect the final outcome of the models. It's easier to get the results here instead of the web app. This notebook uses functions in 'regression_model_old.py'.

Finally, 'stock_app.py' is the python script to enable the web app and showing resutls. Different from the 'User_interface', it uses function of 'regression_model.py'.

## **Running Instruction**
First, install all necessary libraries in requirements.txt. I recommend creating a virtual environment and install by command: pip install -r requirements.txt
If you want to run the notebook, you can also install in the notebook by: import sys, then !{sys.executable} -m pip install -r requirements.txt

To run the web app, open the command prompt in the directory, then use command: streamlit run stock_app.py
In the web app, choose the company sticker and models (multiple choices) from the left sidebar.

## **Analysis and Results**

### - **Data**
The data for this project is simply historical stock daily closing price of a specific company. I query from Yahoo finance API for the data starting from an original time to the most up-to-date time. Thus, there is no external data file needed. The data is then splitted to training and testing datasets for training model and evaluate performance. 
For the deep learning model, there is an extra preprocessing which normalize the price data to a value between (0,1).

### - **Methodology**
To predict the price value, it is genuine to think about regression and neural network to give predictions. For the first part, I've chosen to use linear regression, decision tree regression and SVM for the modelling. Linear model is very straightforward and easy to implement, although it will give you bad fit if the underlying relation is significantly non-linear. SVM, on the other hand, could provide decent fit for non-linear relationship in spite of slow learning rate and risk of overfit. Decision tree is generally more suitable for problems with discrete or categorical features, regression tree could also handle continuous features. However, the tree can be non-robust and 
prone to overfitting. The input data for these models are the stock price of the previous day. Normally, the price won't change dramatically from day to day and the daily stock price moving direction has little connection with stock price a long time ago. For these reasons, I decided to use the data tracing back to a year from now. 

The other approach is neural network. To use larger dataset that tracing back to 10 years ago, we probably keep track of arbitrary long-term dependencies in the input sequences. Recurrent nueral network, more specifically, long short-term memory(LSTM) keeps feeding the hidden state features to the next step and train the model with new input together. It is suitable to make market prediction given the time series stock data. The dataset I used was more than 10 years. The input data contains recent 60-days stock price as the recent representation. 

### -**Evaluation and Refinement**
The models will be trained by feeding the training dataset into them, which are the early data in time series. Then, evaluate the models by comparing prediction on testset to its actual price. Since the price is a continuous variable, I use mean squared error (MSE) to decide which model fits the data best. Also, compute the R2_score for reference. For a case study, I used APPLE INC stock and did a gridsearch on a few models to get the best parameters. 

### -**Results**
For the case study, I run experiments on APPLE stock. The stock has a big jump in 2020 and recently it bounces back and forth while remain an increasing trend. The linear regression and SVM has similar MSE and R2_score, while DecisionTree overfit the training data and got larger MSE and smaller R2_score for the test data. After a GridSearch, the MSE reduces to slightly larger than the other two models. The LSTM model used more data and ended up overfiting the training data and has large MSE compare to the other regression models. Overall, LSTM prediction is smoother and might be used to study the long time trend of the stock change.

### -**Future Improvement**
The model can be improved to generate better useful trading advice. Although it's hard to predict price right, it's easier to predict if the stock price will increase or decrease and convert it into a classification problem. In that way, we can get wrong price but correct direction which is giving the trading direction. Another way to improve the model is by feature engineering the input. For instance, we can use a moving average of the previous 7 days price instead of only the previous day as input. We could also consider some variables like returns and categorical variables like if the company pays dividend etc.

## **Conclusion**
The idea of this project is to use machine learning model to learn the historical progression of a company's stock price and predict it for the next day and so on. A few models are built and compared together. Since different stocks have their own pattern, there is no one best model to fit all. In general, linear regression and SVM have similar performance and smaller MSE. Decisiontree is not stable and can overfit. The LSTM model uses more historical data and also has overfitting problem, but it provides a smooth plot that represents the trend.

The web app does not use grid search to get the best parameters because of the time cost. It gives the users results from four different models and the users can evaluate based on the outcome and plot of the stock price to make trading actions. It should be stated that this stock predictor is more of a showing application of machine learning model in finance rather than real stock trading advice. To make the predictor more practical and meaningful, it is necessary to use some financial knowledge to feature engineering the data and feed into the model.

## **Licensing, Authors, Acknowledgements**
Dataset source: [Yahoo Finance API](https://finance.yahoo.com/)

Udacity MLND Capstone Project Description - Investment and Trading [docs](https://docs.google.com/document/d/1ycGeb1QYKATG6jvz74SAMqxrlek9Ed4RYrzWNhWS-0Q/pub)
