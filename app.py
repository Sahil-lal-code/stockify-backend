from flask import Flask, request, jsonify
from flask_cors import CORS
from model import StockPredictor
import json
from datetime import datetime
import numpy as np
import os
import traceback
import gc

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
        stock_data, error = predictor.fetch_data(ticker)
        if error:
            print("Error fetching data:", error)
            return jsonify({'error': error, 'status': 'error'}), 400
        
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