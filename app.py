from flask import Flask, request, jsonify
from flask_cors import CORS
from model import StockPredictor
import numpy as np
import os
import traceback
import gc
import yfinance as yf
from datetime import datetime, timedelta
import time
import requests

app = Flask(__name__)

# Configure CORS properly
CORS(app)

predictor = StockPredictor()

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        response = jsonify({'message': 'CORS preflight'})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type")
        response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
        return response
    
    try:
        print("Received prediction request")
        data = request.get_json()
        print("Request data:", data)
        
        ticker = data['ticker'].upper()
        model_name = data.get('model', 'Linear Regression')
        days = int(data.get('days', 1))
        
        if days > 10:
            days = 10
            print(f"Days limited to 10 for performance")
        
        print(f"Processing: {ticker}, {model_name}, {days} days")
        
        stock_data = predictor.fetch_data(ticker)
        print("Data fetched successfully")
            
        processed_data = predictor.preprocess_data(stock_data)
        model, r2, mse = predictor.train_model(processed_data, model_name)
        
        print("Model trained successfully")
        
        last_data_point = np.array(processed_data.drop(['Prediction'], axis=1))[-1:]
        predictions = predictor.predict(model, last_data_point, days)
        
        print("Predictions generated")
        
        plot_url1, plot_url2 = predictor.get_plots(stock_data, predictions, ticker)
        
        print("Plots generated")
        
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
        
        del stock_data, processed_data, model, predictions
        gc.collect()
        
        print("Sending response")
        return jsonify(response)
        
    except Exception as e:
        print("Error in prediction:", str(e))
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/popular', methods=['GET'])
def popular_stocks():
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
    return jsonify(popular)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'message': 'Server is running'})

@app.route('/test', methods=['GET'])
def test_endpoint():
    try:
        return jsonify({
            'message': 'Test successful', 
            'status': 'success',
            'test_data': [100, 105, 110, 115, 120]
        })
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/proxy-test', methods=['GET'])
def proxy_test():
    """Test proxy functionality"""
    try:
        # Test direct connection
        direct_response = requests.get('https://httpbin.org/ip', timeout=10)
        direct_ip = direct_response.json().get('origin', 'Unknown')
        
        # Test with a proxy
        proxy = predictor.get_random_proxy()
        proxy_response = requests.get(
            'https://httpbin.org/ip',
            proxies={'http': proxy, 'https': proxy},
            timeout=10
        )
        proxy_ip = proxy_response.json().get('origin', 'Unknown')
        
        return jsonify({
            'direct_ip': direct_ip,
            'proxy_ip': proxy_ip,
            'proxy_used': proxy,
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'message': 'Proxy test failed',
            'status': 'error'
        }), 500

@app.route('/debug/<ticker>', methods=['GET'])
def debug_ticker(ticker):
    try:
        print(f"Debugging ticker: {ticker}")
        time.sleep(2)
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=True,
            timeout=10
        )
        
        response_data = {
            'ticker': ticker,
            'has_data': data is not None and not data.empty,
            'data_points': len(data) if data is not None else 0,
            'columns': list(data.columns) if data is not None else [],
            'latest_data': data.tail(3).to_dict() if data is not None and not data.empty else {},
            'status': 'success'
        }
        
        print(f"Debug response: {response_data}")
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Debug error: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e), 'status': 'error'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)