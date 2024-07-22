import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
import sqlite3
import requests
import sys
import pytz
import matplotlib.pyplot as plt
import pickle
from keras.models import load_model
from keras.losses import MeanSquaredError



# kwargs
parser = argparse.ArgumentParser()
parser.add_argument("ticker", type=str, help="First input argument")
parser.add_argument("margin", type=int, help="Second input argument")
args = parser.parse_args()
ticker = args.ticker
margin = args.margin

# Load model
  # Attempt to load model with error handling & caching
try:
    model = load_model('model.h5', compile=False)
    model = load_model('model.h5', custom_objects={'mse': MeanSquaredError()})

except Exception as e:
    print(f"Error loading model.h5: {e}")


# Connect to SQlite database
try:
    db = 'historical.db'
    sqliteConnection = sqlite3.connect(db)
    cursor = sqliteConnection.cursor()
    print(f'SQlite connected with {db}')

except:
    sys.stderr.write("Failed to connect to database")

# Create Prediction Stock dataset
try:
  query = f"SELECT * FROM {ticker};" #
  cursor.execute(query)
  if cursor.fetchone() is None:
    raise Exception("No results")
  print(f"Success querying EVGN_Predict")
except:
  sys.stderr.write(f"Failed to select EVGN_Predict")

# Turn SQlite Database into Pandas Dataframe
predict_data_whole = pd.read_sql_query(query, sqliteConnection)

# Split into known vs unknown, where unknown is the last 78 entries
split_index_whole = len(predict_data_whole) - 78
known_data_whole = predict_data_whole.iloc[:-78]
unknown_data_whole = predict_data_whole.iloc[-78:]

# # Features
# known_features = known_data_whole[['volume', 'volume_weighted_average', 'open', 'close', 'high', 'low']].values
known_features = known_data_whole[['close', 'volume', 'time']].values
known_closes = np.array([known_features[:, 0]]).T
known_volume = np.array([known_features[:, 1]]).T


 # Normalize features
predict_close_scaler = MinMaxScaler(feature_range=(0, 1))
predict_volume_scaler = MinMaxScaler(feature_range=(0, 1))
scaled_close = predict_close_scaler.fit_transform(known_closes)
scaled_volume = predict_volume_scaler.fit_transform(known_volume)

scaled_known_features = known_features
known_features[:, 0] = scaled_close[:, 0]
known_features[:, 1] = scaled_volume[:, 0]


# Create dates column from miliseconds
dates = pd.to_datetime(known_data_whole['time'], unit='ms')
tickers = known_data_whole['ticker']
dates = dates.dt.tz_localize('UTC').dt.tz_convert('US/Pacific')
dates = dates.dt.tz_localize(None)

# Create Prediction Sequences
sequence_length = 78
prediction_length = 78

def create_predict_sequences(data, sequence_length):
    index = len(data) - sequence_length
    xs = []
    xs.append(data[index:])  # Use the last sequence_length data points
    return np.array(xs)
x_predict = create_predict_sequences(scaled_known_features, sequence_length)
x_predict.shape

# Predict with model

predicted_stock_sequence = model.predict(x_predict)

# Unscale
final_predictions = predict_close_scaler.inverse_transform(predicted_stock_sequence)

# Evaluate the model
# model.evaluate(x_predict, final_predictions)

# Plot results
import matplotlib.pyplot as plt

# Create the combined index for plotting
combined_index = np.arange(len(predict_data_whole))

# Create
plt.figure(figsize=(14, 7))

# Plot
plt.plot(combined_index, predict_data_whole['close'], label='Example "Prediction" Data', color='black', linewidth=3.0)
plt.plot(combined_index[:split_index_whole], known_data_whole['close'], label='"Known" Data', color='cyan', linewidth=2.0, linestyle="--")
plt.plot(combined_index[split_index_whole:], unknown_data_whole['close'], label='"Unknown" Data', color='orange', linewidth=2.0, linestyle="--")

plt.plot(combined_index[split_index_whole:], final_predictions[0], label='Model Prediction', color='purple', linewidth=2.0, linestyle="-")



# Format
plt.title('Comparison of Historical, Actual, and Predicted Data')
plt.xlabel('Time Steps')
plt.ylabel('Stock Price')
plt.legend()

# Adjust y-axis limits based on the data range
# plt.ylim(predict_data_whole.iloc[-200:]['close'].min() - 0.2, predict_data_whole.iloc[-200:]['close'].max() + 0.2)
plt.xlim(len(predict_data_whole) - 200, len(predict_data_whole))  # Crop view to just the very end

# Set y-tick locations with a step of 0.05
# yticks = np.arange(min(predict_data_whole['close']) // 0.05 * 0.05, (max(predict_data_whole['close']) // 0.05 + 1) * 0.05, 0.05)
# plt.yticks(yticks)

plt.savefig('plot.png')
plt.close()









