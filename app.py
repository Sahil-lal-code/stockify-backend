from flask import Flask, request, jsonify
from flask_cors import CORS
from model import StockPredictor
import json
import numpy as np

app = Flask(__name__)
# Simplified CORS for initial deployment. We will secure this later.
CORS(app) 

predictor = StockPredictor()

# Health check route for Render
@app.route('/health')
def health_check():
    return jsonify({"status": "ok"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        ticker = data['ticker'].upper()
        model_name = data.get('model', 'Linear Regression')
        days = int(data.get('days', 1))
        
        # Fetch and process data
        stock_data, error = predictor.fetch_data(ticker)
        if error:
            return jsonify({'error': error, 'status': 'error'}), 400
            
        processed_data = predictor.preprocess_data(stock_data)
        model, r2, mse = predictor.train_model(processed_data, model_name)
        
        # Make predictions
        last_data_point = np.array(processed_data.drop(['Prediction'], axis=1))[-1:]
        predictions = predictor.predict(model, last_data_point, days)
        
        # Generate plots
        plot_url1, plot_url2 = predictor.get_plots(stock_data, predictions, ticker)
        
        # Create properly formatted response
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
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

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

# This part is now handled by Gunicorn, so we don't need the __main__ block for deployment
# if __name__ == '__main__':
#     app.run(debug=True)