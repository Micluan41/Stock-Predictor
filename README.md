# Stock-Predictor
Captone Project for Udacity Data Science Nanodegree 

## **Table of Contents**
1. Installation 
2. Project Motivation
3. Project Definition
4. File Description
5. Results Analysis
6. Conclusion
7. Licensing, Authors, Acknowledgements

## **Installation**
To run the scripts/notebooks of this project, you'll need the following python libraries installed.
* **streamlit, yfinance, pandas, numpy, matplotlib, tensorflow, sklearn, datetime**

## **Project Motivation**
Quantitative methods and financial models have been widely studied and used to better understand market behavior and make profitable investments and trades. With the development of technology and Internet, it is easy and fast to access historical stock prices and company performance data, which is suitable for machine learning algorithms to process.

What if we have a stock predictor that utilize historical data of a company, process the data and create a model to learn its market behavior and make prediction for the future? Is the prediction of the price going to be accurate enough to give us the correct trading guide? If so, that stock predictor could worth a lot and bring you wealth.
In the project, I want to apply what I've learned from this program to build such a stock predictor. A few different models and approaches will be implemented and compared to get the better results. 


## **Project Definition**


## **File Description**
There are 2 notebooks and 3 python scripts available to showcase work related to the above questions. 'Stock_predictor' is the notebook that does all the studies on the different models. It starts with web scraping, data wrangling, visualization, modeling and evaluation. Some markdown cells assists to understand the working process.

Then, I translated all the work in the above notebook to two python scripts 'helper.py' and 'regression_model.py'. 'helper.py' helps scape data from Yahoo finance API and preprocess data. 'regression_model.py' contains all the model creation, training, evaluation and printing the metrics and the outcome. 'User_interface' is a notebook that's user-friendly to check the predicted stock price for your choice. 

Finally, 'stock_app.py' is the python script to enable the web app and showing resutls.

## **Results**
The main results and analysis can be found at this [post](https://micluan41.medium.com/how-does-the-covid-progress-in-the-us-a75d18f477de)

## **Licensing, Authors, Acknowledgements**
Dataset source: Data.CDC.gov
Find the dataset [here](https://data.cdc.gov/Case-Surveillance/United-States-COVID-19-Cases-and-Deaths-by-State-o/9mfq-cb36)
