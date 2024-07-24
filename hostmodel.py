from flask import Flask, request, jsonify
from flask_cors import CORS
from flask import send_from_directory, make_response


from keras.models import load_model
import numpy as np
import requests
import os
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
import matplotlib
import matplotlib.pyplot as plt
import pickle
from keras.models import load_model
from keras.losses import MeanSquaredError
matplotlib.use('Agg')

# Load your Keras model
#model = load_model('path_to_your_model.h5')

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/plot.png')
def serve_plot():
    response = make_response(send_from_directory('.', 'plot.png'))
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'  # HTTP 1.1.
    response.headers['Pragma'] = 'no-cache'  # HTTP 1.0.
    response.headers['Expires'] = '0'  # Proxies.
    return response

@app.route('/process', methods=['POST'])
def process():
    if os.path.exists('plot.png'):
        os.remove('plot.png')
    scrape = request.form
    ticker = str(scrape['ticker'])
    margin = int(scrape['margin'])
    print("Ticker:", ticker, type(ticker))
    print("Margin:", margin, type(margin))
    predict(ticker, margin)
    print("WE FINISHED THE IMAGEEEE")
    
    return jsonify({"status": "success"}), 200

def predict(ticker, margin):

    # Load model
    # Attempt to load model with error handling & caching
    # try:
    model = load_model('model.h5', compile=False)
    model = load_model('model.h5', custom_objects={'mse': MeanSquaredError()})

    # except Exception as e:
    #     print(f"Error loading model.h5: {e}")


    # Connect to SQlite database
    # try:
    db = 'historical.db'
    sqliteConnection = sqlite3.connect(db)
    cursor = sqliteConnection.cursor()
    print(f'SQlite connected with {db}')

    # except:
    #     sys.stderr.write("Failed to connect to database")

    # Create Prediction Stock dataset
    # try:
    query = f"SELECT * FROM all_historical WHERE ticker = '{ticker}';" #
    cursor.execute(query)
    # if cursor.fetchone() is None:
        # raise Exception("No results")
    print(f"Success querying EVGN_Predict")
    # except:
    #     sys.stderr.write(f"Failed to select EVGN_Predict")

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

    # Calculate the target price
    last_blue_point = known_data_whole['close'].iloc[-1]
    target_price = last_blue_point * (1 + margin / 100)

    # Find the index in the orange data where the target price is first exceeded
    orange_data = final_predictions[0]
    buy_index = np.argmax(orange_data >= target_price) + split_index_whole  # + split_index_whole to adjust the index
    print("HERE WE GOOOO\n\n\n", buy_index, last_blue_point, orange_data.max(), target_price, "HOTDIGGITYDOG", len(known_data_whole))

    # Create the plot
    plt.figure(figsize=(14, 7))

    # Plot the lines
    plt.plot(combined_index[:split_index_whole], known_data_whole['close'], label='"Known" Data', color='blue', linewidth=2.0)
    plt.plot(combined_index[split_index_whole:], final_predictions[0], label='Model Prediction', color='orange', linewidth=2.0)

    # Shade the area under the lines
    plt.fill_between(combined_index[:split_index_whole], known_data_whole['close'], color='blue', alpha=0.1)
    plt.fill_between(combined_index[split_index_whole:], final_predictions[0], color='darkorange', alpha=0.1)

    # Add vertical line with red dot and "now" label
    plt.vlines(x=split_index_whole - 0.5, ymin=0, ymax=last_blue_point, colors='red', linestyle='--')
    plt.scatter(split_index_whole - 0.5, last_blue_point, color='red', s=100)  # red dot at the last blue point
    plt.text(split_index_whole - 0.5, -max(predict_data_whole['close']) * 0.05, 'now', color='red', ha='center', fontsize=12, verticalalignment='bottom')  # label below the x-axis

    # Add vertical line, red dot, and "buy here" label if a suitable point is found
    if buy_index < len(known_data_whole) + 78 and buy_index != len(known_data_whole):
        print("WE HAVE SOMETHING HERE YESSIR")
        buy_price = final_predictions[0][buy_index - split_index_whole]
        plt.vlines(x=buy_index, ymin=0, ymax=buy_price, colors='red', linestyle='--')
        plt.scatter(buy_index - 0.1, buy_price, color='red', s=100)  # red dot at the buy point
        plt.text(buy_index, -max(predict_data_whole['close']) * 0.05, 'buy here', color='red', ha='center', fontsize=12, verticalalignment='bottom')  # label below the x-axis
    else:
        print("NOT FOUND AT ALL BOZO", )

    # Custom xticks
    times = ["9:00", "9:30", "10:00", "10:30", "11:00", "11:30", "12:00", "12:30", "1:00", "1:30", "2:00", "2:30", "3:00", "3:30", "4:00"]
    xticks_positions = np.linspace(len(predict_data_whole) - 100, len(predict_data_whole), len(times))

    plt.xticks(xticks_positions, times)

    # Format the plot
    plt.title(f'{ticker} Prices for the next day')
    plt.xlabel('Time Steps')
    plt.ylabel('Stock Price')
    plt.legend()

    # Adjust y-axis limits based on the data range
    # plt.ylim(predict_data_whole.iloc[-200:]['close'].min() - 0.2, predict_data_whole.iloc[-200:]['close'].max() + 0.2)
    plt.xlim(len(predict_data_whole) - 100, len(predict_data_whole))  # Crop view to just the very end

    # Save the plot
    plt.savefig('plot.png')
    plt.close()




if __name__ == '__main__':
    app.run(debug=True, port=5000)