from flask import Flask, request, jsonify
from flask_cors import CORS
from model import StockPredictor
import json
from datetime import datetime
import numpy as np
import os
import traceback
import gc
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import time  # Added for rate limiting

app = Flask(__name__)

# Configure CORS for production - allow all origins
CORS(app, resources={
    r"/*": {
        "origins": "*",  # Allow all origins
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Handle preflight requests
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

predictor = StockPredictor()

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return _build_cors_preflight_response()
    
    try:
        print("Received prediction request")
        data = request.get_json()
        print("Request data:", data)
        
        ticker = data['ticker'].upper()
        model_name = data.get('model', 'Linear Regression')
        days = int(data.get('days', 1))
        
        # Validate input - limit to prevent overload
        if days > 10:
            days = 10
            print(f"Days limited to 10 for performance")
        
        print(f"Processing: {ticker}, {model_name}, {days} days")
        
        # Fetch and process data
        stock_data, error_message = predictor.fetch_data(ticker)
        if error_message and "mock data" not in error_message:
            print("Error fetching data:", error_message)
            return jsonify({'error': error_message, 'status': 'error'}), 400
        
        print("Data fetched successfully")
            
        processed_data = predictor.preprocess_data(stock_data)
        model, r2, mse = predictor.train_model(processed_data, model_name)
        
        print("Model trained successfully")
        
        # Make predictions
        last_data_point = np.array(processed_data.drop(['Prediction'], axis=1))[-1:]
        predictions = predictor.predict(model, last_data_point, days)
        
        print("Predictions generated")
        
        # Generate plots
        plot_url1, plot_url2 = predictor.get_plots(stock_data, predictions, ticker)
        
        print("Plots generated")
        
        # Create response
        response = {
            'ticker': ticker,
            'current_price': round(float(stock_data['Close'][-1]), 2),
            'predictions': [round(float(p), 2) for p in predictions],
            'model_metrics': {
                'model': model_name,
                'r2_score': round(float(r2), 4),
                'mse': round(float(mse), 4)
            },
            'plot1': plot_url1,
            'plot2': plot_url2,
            'status': 'success'
        }
        
        # Add warning if using mock data
        if error_message and "mock data" in error_message:
            response['warning'] = error_message
        
        # Clean up memory
        del stock_data, processed_data, model, predictions
        gc.collect()
        
        print("Sending response")
        return _corsify_actual_response(jsonify(response))
        
    except Exception as e:
        print("Error in prediction:", str(e))
        traceback.print_exc()
        return _corsify_actual_response(jsonify({
            'error': 'Internal server error during prediction',
            'status': 'error'
        })), 500

@app.route('/popular', methods=['GET', 'OPTIONS'])
def popular_stocks():
    if request.method == 'OPTIONS':
        return _build_cors_preflight_response()
    
    popular = [
        {'ticker': 'AAPL', 'name': 'Apple Inc.'},
        {'ticker': 'MSFT', 'name': 'Microsoft Corporation'},
        {'ticker': 'GOOGL', 'name': 'Alphabet Inc.'},
        {'ticker': 'AMZN', 'name': 'Amazon.com Inc.'},
        {'ticker': 'TSLA', 'name': 'Tesla Inc.'},
        {'ticker': 'META', 'name': 'Meta Platforms Inc.'},
        {'ticker': 'NVDA', 'name': 'NVIDIA Corporation'},
        {'ticker': 'JPM', 'name': 'JPMorgan Chase & Co.'}
    ]
    return _corsify_actual_response(jsonify(popular))

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for Render"""
    return jsonify({'status': 'healthy', 'message': 'Server is running'})

@app.route('/test', methods=['GET'])
def test_endpoint():
    """Simple test endpoint without ML"""
    try:
        return jsonify({
            'message': 'Test successful', 
            'status': 'success',
            'test_data': [100, 105, 110, 115, 120]  # Mock predictions
        })
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/debug/<ticker>', methods=['GET'])
def debug_ticker(ticker):
    """Debug endpoint to check yfinance data"""
    try:
        print(f"Debugging ticker: {ticker}")
        
        # Add rate limiting
        time.sleep(1)  # Wait 1 second between debug requests
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)  # Shorter period for debug
        
        print(f"Testing yfinance for {ticker} from {start_date} to {end_date}")
        
        # Method 1: Direct download with timeout
        try:
            data1 = yf.download(ticker, start=start_date, end=end_date, 
                               progress=False, auto_adjust=True, timeout=5)
            shape1 = data1.shape if data1 is not None else 'None'
        except Exception as e:
            shape1 = f"Error: {str(e)}"
        
        # Don't try method 2 if method 1 failed to avoid more rate limiting
        shape2 = "Skipped to avoid rate limiting"
        
        return jsonify({
            'ticker': ticker,
            'method1_shape': str(shape1),
            'method2_shape': shape2,
            'status': 'success',
            'note': 'Rate limiting protection active'
        })
        
    except Exception as e:
        print(f"Debug error: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e), 'status': 'error'}), 500

# CORS helper functions
def _build_cors_preflight_response():
    response = jsonify({'message': 'CORS preflight'})
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "*")
    response.headers.add("Access-Control-Allow-Methods", "*")
    return response

def _corsify_actual_response(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)