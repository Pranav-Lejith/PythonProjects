import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import datetime

# Function to download historical stock data
def get_stock_data(stock_symbol, start_date, end_date):
    stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
    return stock_data

# Fetch historical data for a specific stock (e.g., Apple)
stock_symbol = 'AAPL'
stock_data = get_stock_data(stock_symbol, '2015-01-01', datetime.datetime.today().strftime('%Y-%m-%d'))

# Preprocess data
data = stock_data['Close'].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Split into training and test sets
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

# Create dataset function for LSTM
def create_dataset(data, time_step):
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

time_step = 60
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# Reshape input to be [samples, time steps, features] for LSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Predict on test data
predicted_stock_price = model.predict(X_test)
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)
real_stock_price = scaler.inverse_transform(y_test.reshape(-1, 1))

# Visualize results
plt.plot(real_stock_price, color='red', label='Real AAPL Stock Price')
plt.plot(predicted_stock_price, color='blue', label='Predicted AAPL Stock Price')
plt.title('AAPL Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('AAPL Stock Price')
plt.legend()
plt.show()

# Function to get real-time data for prediction
def get_latest_data(stock_symbol):
    today = datetime.datetime.today().strftime('%Y-%m-%d')
    stock_data = yf.download(stock_symbol, period='1d', interval='1m')  # Fetch 1-minute interval data for today
    latest_close_price = stock_data['Close'][-1]
    return latest_close_price

# Predict the next stock price using real-time data
def predict_next_day(stock_symbol):
    real_time_data = yf.download(stock_symbol, period='60d', interval='1d')  # Get the last 60 days of data
    last_60_days = real_time_data['Close'].values[-60:].reshape(-1, 1)  # Get only the last 60 days of Close prices
    last_60_days_scaled = scaler.transform(last_60_days)

    X_input = last_60_days_scaled.reshape(1, -1, 1)  # Reshape for LSTM input
    predicted_stock_price = model.predict(X_input)
    predicted_stock_price = scaler.inverse_transform(predicted_stock_price)
    return predicted_stock_price[0][0]

# Fetch real-time stock price and make predictions
latest_price = get_latest_data(stock_symbol)
print(f"Latest price of {stock_symbol}: {latest_price}")

predicted_next_price = predict_next_day(stock_symbol)
print(f"Predicted next day price of {stock_symbol}: {predicted_next_price}")
