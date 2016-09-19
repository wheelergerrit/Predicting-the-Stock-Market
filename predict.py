#This script takes stock market data from the S&P500 and uses linear regression to predict 
#closing costs based on metrics we created

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#Import data from Yahoo financial data online
sphist = pd.read_csv("sphist.csv")

sphist["Date"] = pd.to_datetime(sphist["Date"])

sphist = sphist.sort_values("Date", ascending=True)

sphist['index'] = range(0,sphist.shape[0],1)

sphist.set_index(['index'])

close = sphist["Close"]
volume = sphist["Volume"]

#Creating date after april 1, 2015

sphist['date_after_april1_2015'] = sphist.Date > datetime(year=2015, month=4, day=1)

#Creating 12 new columns based on historical data, each metric is unique and fabricated

data_mean_5day = close.rolling(center=False, window=5).mean()

data_mean_365day = close.rolling(center=False, window=365).mean()

data_mean_ratio = data_mean_5day/data_mean_365day


data_std_5day = close.rolling(center=False, window=5).std()

data_std_365day = close.rolling(center=False, window=365).std()

data_std_ratio = data_std_5day/data_std_365day

data_mean_vol_5day = volume.rolling(window=5, center=False).mean()

data_mean_vol_365day = volume.rolling(window=365, center=False).mean()

vol_ratio = data_mean_vol_5day/data_mean_vol_365day

data_std_vol_5day = volume.rolling(window=5, center=False).std()

data_std_vol_365day = volume.rolling(window=365, center=False).std()

sphist["5day_mean"] = data_mean_5day

sphist["mean_ratio"] = data_mean_ratio

sphist["std_ratio"] = data_std_ratio

sphist['365day_mean'] = data_mean_365day

sphist['5day_std'] = data_std_5day

sphist["365day_std"] = data_std_365day

sphist["5day_volume"] = data_mean_vol_5day

sphist["365day_volume"] = data_mean_vol_365day

sphist["vol_ratio"] = vol_ratio

sphist["5day_vol_std"] = data_std_vol_5day

sphist["365day_vol_std"] = data_std_vol_365day

sphist["vol_std_ratio"] = data_std_vol_5day/data_std_vol_365day

#Dropping NA columns that we don't have historical data for and creating training and test sets

sphist = sphist.dropna(axis=0)

train = sphist[sphist["Date"] < datetime(year=2013, month=1, day=1)]

test = sphist[sphist["Date"] >= datetime(year=2013, month=1, day=1)]



print("Size of Training Set:", train.shape)

print("Size of Test Set:", test.shape)

print('Columns:', train.columns)

#We will be using the Mean Squared Error for our metric

#Now we will train our LinearRegression model and make predictions

lr = LinearRegression()

features = ["5day_mean","mean_ratio","std_ratio", '365day_mean','5day_std','365day_std',"5day_volume", "365day_volume",
	"vol_ratio","5day_vol_std","365day_vol_std", "vol_std_ratio"]

lr.fit(train[features], train["Close"])


prediction = lr.predict(test[features])


mse = mean_squared_error(test["Close"], prediction)

print("MSE:", mse)

#The MSE with 6 predictors was 268.74

#The MSE with all 12 predictors was 270.15

. Our error went up, meaning we might be overfitting

#Some further metrics to increase accuracy of model:

#The year component of the date.

#The ratio between the lowest price in the past year and the current price.

#The ratio between the highest price in the past year and the current price.

#The year component of the date.

#The month component of the date.

#The day of week.

#The day component of the date.

#The number of holidays in the prior month.

'''There's a lot of improvement still to be made on the indicator side, and we will think of better indicators that we could use for prediction.



We can also make significant structural improvements to the algorithm, and pull in data from other sources.



Accuracy would improve greatly by making predictions only one day ahead. For example, train a model using data from 1951-01-03 to 2013-01-02,
	make predictions for 2013-01-03, and then train another model using data from 1951-01-03 to 2013-01-03, make predictions for 2013-01-04, 
	and so on. This more closely simulates what you'd do if you were trading using the algorithm.



We can also improve the algorithm used significantly. Try other techniques, like a random forest, and see if they perform better.



We can also incorporate outside data, such as the weather in New York City (where most trading happens) the day before, 
	and the amount of Twitter activity around certain stocks.

We can also make the system real-time by writing an automated script to download the latest data when the market closes, 
	and make predictions for the next day.



Finally, we can make the system "higher-resolution". We're currently making daily predictions, but we could make hourly, minute-by-minute, 
	or second by second predictions. This will require obtaining more data, though. could also make predictions for individual stocks instead 
	of the S&P500.'''