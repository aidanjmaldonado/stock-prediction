from flask import Flask, request, jsonify
from keras.models import load_model
import numpy as np
import requests

# Load your Keras model
#model = load_model('path_to_your_model.h5')

app = Flask(__name__)

@app.route('/process', methods=['POST'])
def process():
    scrape = request.form
    ticker = scrape['ticker']
    margin = int(scrape['margin'])
    print("Ticker:", ticker, type(ticker))
    print("Margin:", margin, type(margin))


    return jsonify({"error": "No input data provided"}), 400

def predict(ticker, margin):
    # Call realtime with Finnhub

    # Predict

    # Send image to HTML
    pass


if __name__ == '__main__':
    app.run(debug=True, port=5000)