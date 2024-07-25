import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask import send_from_directory, make_response
from keras.models import load_model
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
import sqlite3
import matplotlib.pyplot as plt
import matplotlib
import os
from keras.models import load_model
from keras.losses import MeanSquaredError

# constants
SEQUENCE_LENGTH = 78
VIEWFINDER = 120

# format graph
matplotlib.use('Agg')

# connect flask server
app = Flask(__name__)
# Enable CORS for all routes
CORS(app)  


''' Update plot.png on the website end
'''
@app.route('/plot.png')
def serve_plot():
    response = make_response(send_from_directory('.', 'plot.png'))
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'  # HTTP 1.1.
    response.headers['Pragma'] = 'no-cache'  # HTTP 1.0.
    response.headers['Expires'] = '0'  # Proxies.
    return response

''' Receive user input from website and calls prediction procedure on them
Arguments:
    - None

Returns:
    - Flask success status
'''
@app.route('/process', methods=['POST'])
def process():
    # remove preexisting plot.png to make room for new one
    if os.path.exists('plot.png'):
        os.remove('plot.png')

    # scrape form data and store in variables
    scrape = request.form
    ticker = str(scrape['ticker'])
    margin = int(scrape['margin'])

    # run predict function based on form data
    predict(ticker, margin)
    
    # return success status
    return jsonify({"status": "success"}), 200


''' Runs model prediction on user input information and generates results plot
Arguments:
    - ticker: String, stock ticker. Ex: 'PLUG'
    - margin: Integer, percent return margin. Ex: 180

Returns:
    - plot.png: Results image of the prediction on the ticker's most recent day
'''
def predict(ticker, margin):

    # load model (model.h5)
    model = load_model('model.h5', custom_objects={'mse': MeanSquaredError()})

    # connect to SQlite database
    db = 'historical.db'
    sqliteConnection = sqlite3.connect(db)
    cursor = sqliteConnection.cursor()
    print(f'SQlite connected with {db}')

    # NOTE* Once we implement Finnhub API: call on ticker 
    # query desired stock information from database, 
    query = f"SELECT * FROM all_historical WHERE ticker = '{ticker}';"
    cursor.execute(query)
    if cursor.fetchone() is None:
        raise Exception("No results")

    # turn SQlite Database into Pandas Dataframe
    predict_data_whole = pd.read_sql_query(query, sqliteConnection)

    # isolate the last day
    split_index_whole = len(predict_data_whole) - SEQUENCE_LENGTH
    current_day_data = predict_data_whole.iloc[-SEQUENCE_LENGTH:]
  
    # features
    current_day_features = current_day_data[['close', 'volume']].values
    current_day_close = np.array([current_day_features[:, 0]]).T

    # normalize close prices
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_close = scaler.fit_transform(current_day_close)

    # replace close prices with scaled close prices
    current_day_features[:, 0] = scaled_close[:, 0]

    # predict next day with model
    predicted_stock_sequence = model.predict(np.array([current_day_features]))

    # reverse close price scaling
    final_predictions = scaler.inverse_transform(predicted_stock_sequence)

    # Plot results
    # Create the combined index for plotting
    combined_index = np.arange(len(predict_data_whole))

    # Calculate the target price
    last_blue_point = current_day_data['close'].iloc[-1]
    target_price = last_blue_point * (1 + margin / 100)

    # Find the index in the orange data where the target price is first exceeded, if multiple exist, return one with highest profit
    orange_data = final_predictions[0]
    buy_index = np.argmax(orange_data >= target_price) + split_index_whole  # + split_index_whole to adjust the index

    # Create the plot
    plt.figure(figsize=(14, 7))

    # Plot the lines
    plt.plot(combined_index[split_index_whole:]-SEQUENCE_LENGTH, current_day_data['close'], label='"Known" Data', color='blue', linewidth=2.0)
    plt.plot(combined_index[split_index_whole:], final_predictions[0], label='Model Prediction', color='orange', linewidth=2.0)

    # Shade the area under the lines
    plt.fill_between(combined_index[split_index_whole:]-SEQUENCE_LENGTH, current_day_data['close'], color='blue', alpha=0.1)
    plt.fill_between(combined_index[split_index_whole:], final_predictions[0], color='darkorange', alpha=0.1)


    # Add vertical line, red dot, and "sell now" label if a suitable point is found
    if buy_index < len(current_day_data) + SEQUENCE_LENGTH and buy_index != len(current_day_data):
        buy_price = final_predictions[0][buy_index - split_index_whole]
        plt.vlines(x=buy_index, ymin=0, ymax=buy_price, colors='red', linestyle='--')
        plt.scatter(buy_index - 0.1, buy_price, color='red', s=100)  # red dot at the buy point
        plt.text(buy_index, -max(predict_data_whole['close']) * 0.05, 'sell now', color='red', ha='center', fontsize=12, verticalalignment='bottom')  # label below the x-axis

        # Add vertical line with red dot and "buy now" label
        plt.vlines(x=split_index_whole - 0.5, ymin=0, ymax=last_blue_point, colors='red', linestyle='--')
        plt.scatter(split_index_whole - 0.5, last_blue_point, color='red', s=100)  # red dot at the last blue point
        plt.text(split_index_whole - 0.5, -max(predict_data_whole['close']) * 0.05, 'buy now', color='red', ha='center', fontsize=12, verticalalignment='bottom')  # label below the x-axis
    else:
        # Add vertical line with red dot and "do not buy" label
        plt.vlines(x=split_index_whole - 0.5, ymin=0, ymax=last_blue_point, colors='red', linestyle='--')
        plt.scatter(split_index_whole - 0.5, last_blue_point, color='red', s=100)  # red dot at the last blue point
        plt.text(split_index_whole - 0.5, -max(predict_data_whole['close']) * 0.05, 'do not buy', color='red', ha='center', fontsize=12, verticalalignment='bottom')  # label below the x-axis

    # Custom xticks
    times = ["9:30", "10:00", "10:30", "11:00", "11:30", "12:00", "12:30", "1:00", "1:30", "2:00", "2:30", "3:00", "3:30", "4:00"]
    xticks_positions = np.linspace(len(predict_data_whole) - VIEWFINDER, len(predict_data_whole), len(times))

    plt.xticks(xticks_positions, times)

    # Format the plot
    plt.title(f'{ticker} Prices for the next day')
    plt.xlabel('Time Steps')
    plt.ylabel('Stock Price')

    # crop view to show both sides   
    plt.xlim(len(predict_data_whole) -VIEWFINDER, len(predict_data_whole))  

    # save the plot
    plt.savefig('plot.png')
    plt.close()




if __name__ == '__main__':

    # run the flask server
    app.run(debug=True, port=5000)