#Load Packages
import pandas as pd
from sklearn import linear_model
import yfinance as yf
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller	#Trend, Seasonality, Cylical and Irregularity
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.python.keras.backend import get_session
from datetime import datetime, timedelta
import tensorflow.keras.backend
from tensorflow.keras import backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

tensor = tf.constant([1, 2, 3])
numpy_array = tensor.numpy()



#fetching data
def download_data(tickers,start_date,end_date):
    stock_data={}
    for stock in tickers:
        ticker = yf.Ticker(stock)
        stock_data[stock]=ticker.history(start=start_date,end=end_date)['Close']
    return pd.DataFrame(stock_data)

#----Developer Mode---# 
#INPUT_stocks=stocks=['AAPL','WMT','TSLA','AMD']
NUM_TRADING_DAYS=252
NUM_PORTFOLIOS=10000
#Historical date for data fecthing:
#Input_start_date=start_date='2014-12-01'
#Input_end_date=end_date='2020-01-01'

#Inital function to download data and align the date time format
#df=download_data(stocks,start_date,end_date).tail(504)
#df.index = pd.DatetimeIndex(df.index).to_period('D')

FORECAST_PERIOD=30
#FORECAST_INDEX = pd.period_range(start=df.index[-1], periods=FORECAST_PERIOD+1,freq='D')

#ARIMA MODEL PRED
def arima_pred(df):
    pred_stock_arima={}
    for i in range(0,df.shape[1]):
        current_stock=df.iloc[:,i]
        model_fit = ARIMA(current_stock,order=(4,2,1)).fit()
        forecast = model_fit.forecast(steps=FORECAST_PERIOD)
        pred_stock_arima[df.columns[i]]=forecast
    return pd.DataFrame(pred_stock_arima)


# Prepare the data for the LSTM model
def prepare_data(data, time_step=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(np.array(data).reshape(-1, 1))
    
    train_data, test_data = scaled_data[0:int(len(data)*0.8)], scaled_data[int(len(data)*0.8):]
    
    def create_dataset(dataset, time_step=1):
        X, y = [], []
        for i in range(len(dataset) - time_step - 1):
            X.append(dataset[i:(i + time_step), 0])
            y.append(dataset[i + time_step, 0])
        return np.array(X), np.array(y)
    
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)
    
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    return X_train, y_train, X_test, y_test, scaler


# Build and train the LSTM model
def build_and_train_model(X_train, y_train, epochs=5, batch_size=256):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    
    return model


def plot_results(original_data, train_predict, test_predict, scaler, time_step=60):
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    
    # Shift train predictions for plotting
    train_predict_plot = np.empty_like(original_data)
    train_predict_plot[:, :] = np.nan
    train_predict_plot[time_step:len(train_predict) + time_step, :] = train_predict
    
    # Shift test predictions for plotting
    test_predict_plot = np.empty_like(original_data)
    test_predict_plot[:, :] = np.nan
    test_predict_plot[len(train_predict) + (time_step * 2) + 1:len(original_data) - 1, :] = test_predict
    
    plt.figure(figsize=(12, 6))
    plt.plot(scaler.inverse_transform(original_data), label='Original Data')
    plt.plot(train_predict_plot, label='Train Predict')
    plt.plot(test_predict_plot, label='Test Predict')
    plt.title('Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()



def LSTM_pred(df):
    pred_stock_lstm={}
    for i in range(0,df.shape[1]):
        current_stock=df.iloc[:,i]
        # Prepare the data
        X_train, y_train, X_test, y_test, scaler = prepare_data(current_stock)
        model = build_and_train_model(X_train, y_train)
        #Make predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)
        #plot_results(np.array(current_stock).reshape(-1, 1), train_predict, test_predict, scaler)
        #--------
        pred_stock_lstm[df.columns[i]]=pd.Series(map(lambda x: x[0], test_predict))
    return pd.DataFrame(pred_stock_lstm)



#Time Series Regression
def RandomForest_pred(df):
    pred_stock_RandomForest={}
    for i in range(0,df.shape[1]):
        current_stock=pd.DataFrame({})
        current_stock['TEMP']=df.iloc[:,i]
        # Split the data into training and testing sets
        current_stock['MA10']=df.iloc[:,i].rolling(window=10).mean()
        current_stock['MA50']=df.iloc[:,i].rolling(window=50).mean()
        current_stock=current_stock.dropna()
        X_train, X_test, y_train, y_test = train_test_split(current_stock[['MA10','MA50']], current_stock[['TEMP']], test_size=30, random_state=42)
        # Create a pipeline that standardizes the data then applies linear regression
        # Train the model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        # Predict the stock prices on the test set
        pred_stock_RandomForest[df.columns[i]]=model.predict(X_test)
    return pd.DataFrame(pred_stock_RandomForest)





def MLapply(df,ML_algos):
    #breakpoint()
    #Empty Data Frame Declared
    lstm_df=pd.DataFrame({},columns=df.columns)
    randomforest_df=pd.DataFrame({},columns=df.columns)
    arima_df=pd.DataFrame({},columns=df.columns)

    df.index = pd.DatetimeIndex(df.index).to_period('D')
    FORECAST_INDEX = pd.period_range(start=df.index[-1], periods=FORECAST_PERIOD+1,freq='D')

    if 'RNN_LSTM' in ML_algos:
        lstm_df=LSTM_pred(df).tail(30)
        lstm_df.index=FORECAST_INDEX[1:]
    
    if 'RandomForest' in ML_algos:
        randomforest_df=RandomForest_pred(df).tail(30)
        randomforest_df.index=FORECAST_INDEX[1:]

    if 'TimeSeries_ARIMA' in ML_algos:
        arima_df = arima_pred(df).tail(30)
        arima_df.index=FORECAST_INDEX[1:]
    #breakpoint()
    return arima_df,randomforest_df,lstm_df

