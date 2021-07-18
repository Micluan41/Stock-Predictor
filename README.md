# Stock-Predictor
Captone Project for Udacity Data Science Nanodegree 

## **Table of Contents**
1. Installation 
2. Project Motivation
3. Project Definition
4. File Description
5. Running Instruction
6. Analysis 
7. Results
8. Conclusion
9. Licensing, Authors, Acknowledgements

## **Installation**
To run the scripts/notebooks of this project, you'll need the following python libraries installed.
* **streamlit, yfinance, pandas, numpy, matplotlib, tensorflow, sklearn, datetime**

## **Project Motivation**
Quantitative methods and financial models have been widely studied and used to better understand market behavior and make profitable investments and trades. With the development of technology and Internet, it is easy and fast to access historical stock prices and company performance data, which is suitable for machine learning algorithms to process.

What if we have a stock predictor that utilizes historical data of a company, process the data and create a model to learn its market behavior and make prediction for the future? Is the prediction of the price going to be accurate enough to give us the correct trading guide? If so, that stock predictor could worth a lot and bring you wealth.
In the project, I want to apply what I've learned from this program to build such a stock predictor. A few different models and approaches will be implemented and compared to get the better results. 

## **Project Definition**
Develop a stock predictor that predict any company's future stock given its historical stock price. Build an web application that shows the stock history, future price, trading direction according to user selected company stock ticker and model names. 

Some solutions including regression (linear, polynomial) models are provided as well as deep learning model (LSTM). Compare all these models and find the best fit.

**Metrics:** To evaluate the performance of the model, I decide to compare mean squared error (MSE) and r2_score. MSE measures the average squared difference between the estimated values and the actual value, which is a good measure of the quality of an estimator. r2_score is the amount of the variation in the output dependent attribute which is predictable from the input independent variables. Typically, a smaller MSE and a larger r2_score indicates a better performance.
Another metric to measure the model is to check what percentage the prediction is within the actual price on average. The formula I use is 'rmse/average price' where rmse is the root MSE of the test data and average price is the average actual price of the test data. If it is within 5-10%, the model is considered acceptable.  

## **File Description**
To run the notebook or the web application, you have to first install the necessary python libraries in requirements.txt

There are 2 notebooks and 3 python scripts available to showcase work related to the above questions. 'Stock_predictor' is the notebook that does all the studies on the different models. It starts with web scraping, data wrangling, visualization, modeling and evaluation. Some markdown cells assists to understand the working process.

Then, I packaged some of the work in the above notebook to two python scripts 'helper.py' and 'regression_model.py'/'regression_model_old.py'. 'helper.py' helps scape data from Yahoo finance API and preprocess data. 'regression_model.py' contains all the model creation, training, evaluation and printing the metrics and the outcome. 

'User_interface' is a notebook that's user-friendly to check the predicted stock price for your choice. The other purpopse of this notebook is to see how some parameters might affect the final outcome of the models. It's easier to get the results here instead of the web app. This notebook uses functions in 'regression_model_old.py'.

Finally, 'stock_app.py' is the python script to enable the web app and showing resutls. It uses function of 'regression_model.py', which has streamlit functions to display text or plots in web app.

## **Running Instruction**
First, install all necessary libraries in requirements.txt. I recommend creating a virtual environment and install by command: pip install -r requirements.txt
If you want to run the notebook, you can also install in the notebook by: import sys, then !{sys.executable} -m pip install -r requirements.txt

To run the web app, open the command prompt in the directory, then use command: streamlit run stock_app.py
In the web app, choose the company sticker and models (multiple choices) from the left sidebar.

## **Analysis**

### - **Data**
**Input description:** I query from Yahoo finance API for the data starting from an original time to the most up-to-date data. The dataframe includes daily open, close, high, low price, volume, whether the company pays dividends and splits stock. For regression model, I decide to simplify the input and use closing price only. The input for regression model is simply the **previous day closing price**. Since recent prices are more relevant to predict the next price, I used one-year data from now which contains 252 data points. For the deep learning model, the goal is to study possible long term dependencies. Therefore, the dataset contains around 2600 data points (from 2011-1-1 to now). The input contains the **previous 60-day closing price** and has a dimension of (1, 60). In addition, there is a preprocessing step which normalizes the closing price to (0,1) before training the deep learning model. 

**Visualization:** The closing price history of APPLE INC is plotted. It is clear that the recent data has a quite different trend compare to earlier days. So for regression model, it's reasonable to use one year from now data. For the deep learning model, use the whole history to study long tern dependencies. 
![alt text](https://github.com/Micluan41/Stock-Predictor/blob/main/appl_closing.png?raw=true)

### - **Methodology**
To predict the price value, it is genuine to think about regression and neural network to give predictions. For the first part, I've chosen to use linear regression, decision tree regression and SVM for the modelling. Linear model is very straightforward and easy to implement, although it will give you bad fit if the underlying relation is significantly non-linear. SVM, on the other hand, could provide fit for non-linear relationship in spite of slow learning rate and risk of overfitting. Decision tree is generally more suitable for problems with discrete or categorical features, regression tree could also handle continuous features. However, the tree can be non-robust and 
prone to overfit. The other approach is neural network. To use larger dataset that tracing back to 10 years ago, we probably keep track of arbitrary long-term dependencies. Recurrent nueral network, more specifically, long short-term memory(LSTM) keeps feeding the hidden state features to the next step and train the model with new input together. It is suitable to make market prediction given the time series stock data. 

**Data Preprocessing:**
For regression models (Linear, DecisionTree, SVM), there isn't a preprocessing step. For LSTM, normalize the data to (0,1) using MinMaxScaler from sklearn.preprocessing. Then extract previous 60-day price at each time to form a (n, 60) dimension input.

**Implementation:**
For regression models (Linear, DecisionTree, SVM), recent one-year data were splitted with a testsize of 0.2. Build those models using LinearRegression, SVR,  DecisionTreeRegressor in python sklearn library with default parameters. Feed the training dataset to the model and make prediction on the test dataset. Metrics for both training data and test data are printed. The comparison of predicted and actual price on test data is also plotted. Later, run the gridsearch on DecisionTree and SVM to optimize parameter. Finally, predict the next day price using the final model.

For LSTM, the entire history data is used (from 2011-1-1). A customized train_test_split function was defined to split data with a default testsize of 0.1. The split will not shuffle the data so that the training data are all before test data. The function also reshape the input to the correct dimension as mentioned above. Use tensorflow to build the LSTM model with a few predefined parameters and train the model. The model consists of **two lstm layers**，**two dropout layers** and **two dense layers**. Watch the loss change during each epochs to make sure the model is trained sufficiently. Print metrics on training and testing data and plot test predicton with actual price. Run grid search to optimize key hyperparameters (lstm layer size, dropout, dense layer size).

Finally, subplot the prediction of all four models with actual price of the test data and compare them. Make the next day prediction for LSTM model.

**Evaluation and Refinement:**
For the case study, I used APPLE stock. The metrics (MSE, r2_score) in Project Definition are calculated after the models are trained with default parameters. For DecisionTree, SVM and LSTM, run GridSearch on some parameters to see if the metrics are improved. The final last day prediction should use the optimized model to generate.
Finally, calculate what percentage the prediction is within the actual price. 

## **Results**
### - **Model Evaluation**
The results of final models are shown in the table below, including metrics in training and testing dataset and next day prediction.

Model name   | train mse | test mse | train r2_score | test r2_score | prediction within actual price | next day prediction 
------------ | --------- | ---------| -------------- | ------------- | ------------------------------ | ------------------- 
Linear       | 7.4256    | 3.2717   | 0.9284         | 0.9408        | 1.37%                          | 145.2818            
DecisionTree | 5.6167    | 9.3628   | 0.9458         | 0.8307        | 2.32%                          | 141.6075            
SVM          | 9.1637    | 3.8813   | 0.9116         | 0.9298        | 1.50%                          | 149.3126            
LSTM         | 1.4049    | 25.5688  | 0.9950         | 0.8338        | 4.15%                          | 141.4464            

In general, all four models get prediction within 5% of the actual price and the next day prediction are quite close. 
The linear model has the smallest mse and the best test r2_score using default parameters. 
DecisionTree model overfits with default parameters. After optimization, the mse for test data decreases from 11.3606 to 9.3628 and r2_score increase from  0.7946 to 0.8307.
SVM does not have improvements after the grid search. It ranks the second best behind linear model for this specific stock.
LSTM uses a lot more past data and overfits the training data. Thus it has the largest mse for test data. But the r2_score and the prediction are still acceptable after optimization. **NOTE: The grid search process took over 2 hours to run. The best parameters are set as default to showcase the training and printing metrics in the notebook.**

### - **Model Validation**
In 'User_Interface' notebook, I run the process (without grid search) on other stocks (AMAZON, FACEBOOK, TWITTER) by changing the stock sticker in the third cell. The results validate the model can provide acceptable prediction (5-10%) on different stocks. 

### - **Justification**
The linear model and SVM performs better in most cases than DecisionTree. This is expected as DecisionTree regressor has more parameters that could affect the fit and the risk of overfitting training data is higher. SVM could perform better than linear model when the pattern of the stock is highly non-linear. LSTM, like DecisionTree, tends to overfit the training data. LSTM model also uses 60-day price as features of the input which assumes that previous price has contribution to the next day price. In reality, this might not necessary be related. By doing grid search could improve the r2_score and reduce mse. For example, changing the lstm layer size and the final dense layer size 
could improve the LSTM model. Increase max_depth of DecisionTree regressor improve the overfitting issue.

## **Conclusion**
### - **Reflection**
The idea of this project is to use machine learning model to learn the historical progression of a company's stock price and predict it for the next day and future. A few regression models are built and trained by feeding previous day price as input. A LSTM deep learning model is also built and trained by feeding previous 60-day price as input. The metrics for both training data and testing data are printed and compared to find the better model. Optimize some models by conducting a grid search could improve the performance. Finally, a web application is built by putting everything in the notebook together. The web app will show the next day prediction of the user selected stock and models as well as plot that compares prediction to actual price in recent days.

The insteresting part of this project is that the prediction actually can be very close to the actual price. Despite some difference on the value, the prediction has a good representation of the trend of the price change. Therefore, we can also convert it to a classification problem by predicting whether the price will increase or decrease for the next day. One of the difficulties is to get useful features. This might require some financial knowledge to get more practical features rather than only the stock price.

### - **Future Improvement**
The data preprocessing could be explored more to get better input features. For example, using the 60-day moving average instead of 60-day stock prices for the LSTM or including categorical variable such as whether the company pays dividends or split stock. This model could do better for those stocks that have a sudden jump or plummet at a certain time due to these finiancial decisions. Another thing is discussed in Reflection that we can convert output to categorical variable (increase/decrease) and use accuracy as the metrics. This might be better since people make trading decisions depending on the price changing direction. 

## **Licensing, Authors, Acknowledgements**
Dataset source: [Yahoo Finance API](https://finance.yahoo.com/)

Udacity MLND Capstone Project Description - Investment and Trading [docs](https://docs.google.com/document/d/1ycGeb1QYKATG6jvz74SAMqxrlek9Ed4RYrzWNhWS-0Q/pub)
