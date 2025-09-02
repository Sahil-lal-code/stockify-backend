import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
import base64
import gc
import time
import requests
import os
import random

class StockPredictor:
    def __init__(self):
        self.models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(n_estimators=50, random_state=42),
        }
        self.current_model = "Linear Regression"
        self.last_request_time = 0
        self.request_delay = 10
        self.proxy_list = self.get_proxy_list()

    def get_proxy_list(self):
        """Get list of proxies - you'll need to replace these with working proxies"""
        # Free public proxies (these change frequently and may not work)
        # For production, use paid proxy services like Luminati, Smartproxy, etc.
        return [
            # HTTP proxies
            'http://138.199.48.1:8443',
            'http://162.0.220.234:80',
            'http://45.95.147.406:8080',
            'http://193.122.71.184:3128',
            # Add more proxies or use a proxy service
        ]

    def get_random_proxy(self):
        """Get a random proxy from the list"""
        if not self.proxy_list:
            return None
        return random.choice(self.proxy_list)

    def test_proxy(self, proxy):
        """Test if a proxy is working"""
        try:
            test_response = requests.get(
                'http://httpbin.org/ip',
                proxies={'http': proxy, 'https': proxy},
                timeout=10
            )
            return test_response.status_code == 200
        except:
            return False

    def fetch_data(self, ticker, days=60):
        # Rate limiting
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < self.request_delay:
            sleep_time = self.request_delay - time_since_last_request
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days * 3)
        
        print(f"Fetching yfinance data for {ticker} for {days} days")

        # Try with multiple proxies
        max_retries = 3
        for attempt in range(max_retries):
            try:
                proxy = self.get_random_proxy()
                proxy_dict = None
                
                if proxy:
                    print(f"Attempt {attempt + 1} with proxy: {proxy}")
                    proxy_dict = {'http': proxy, 'https': proxy}
                else:
                    print(f"Attempt {attempt + 1} without proxy")

                data = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    progress=False,
                    auto_adjust=True,
                    timeout=30,
                    threads=False,
                    proxy=proxy_dict
                )
                
                if data is None or data.empty:
                    raise ValueError(f"No data returned for ticker: {ticker}")
                
                if 'Close' not in data.columns:
                    for col in ['Adj Close', 'Open', 'High', 'Low']:
                        if col in data.columns:
                            data = data[[col]].copy()
                            data.columns = ['Close']
                            print(f"Using {col} as Close price")
                            break
                    else:
                        raise ValueError("No price data columns found")
                
                data = data[['Close']].copy()
                data = data.tail(days)
                
                if len(data) < 30:
                    raise ValueError(f"Insufficient data points ({len(data)}) for training")
                
                print(f"yFinance success: {data.shape[0]} days of data for {ticker}")
                return data
                
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise Exception(f"All proxy attempts failed for {ticker}: {str(e)}")
                time.sleep(2)  # Wait before retrying

    def preprocess_data(self, data):
        data = data[["Close"]]
        data.columns = ["Close"]
        data["Prediction"] = data["Close"].shift(-1)
        data.dropna(inplace=True)
        return data

    def train_model(self, data, model_name="Linear Regression"):
        X = data[["Close"]].values
        y = data["Prediction"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        model = self.models[model_name]
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)

        self.current_model = model_name
        
        del X_train, X_test, y_train, y_test, y_pred
        gc.collect()
        
        return model, r2, mse

    def predict(self, model, data, days=1):
        future_predictions = []
        current_data = data.copy()

        for _ in range(days):
            next_day_pred = model.predict(current_data[-1:])
            pred_value = next_day_pred[0]
            if np.isnan(pred_value):
                raise ValueError("Model produced NaN prediction; insufficient data.")
            future_predictions.append(pred_value)

            new_row = np.array([[pred_value]])
            current_data = np.append(current_data, new_row, axis=0)

        return future_predictions

    def get_plots(self, data, predictions, ticker):
        # Plot 1: Historical vs Predicted
        plt.figure(figsize=(12, 6))
        plt.plot(data.index, data["Close"], label="Historical Prices", color="#034F68", linewidth=2)
        future_dates = [data.index[-1] + timedelta(days=i) for i in range(1, len(predictions) + 1)]
        plt.plot(future_dates, predictions, label="Predicted Prices", color="orange", linewidth=2)
        plt.scatter(future_dates, predictions, color="red", s=50)
        plt.title(f"{ticker} Stock Price Prediction")
        plt.xlabel("Date")
        plt.ylabel("Price ($)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        img1 = io.BytesIO()
        plt.savefig(img1, format="png", dpi=80, bbox_inches="tight")
        img1.seek(0)
        plot_url1 = base64.b64encode(img1.getvalue()).decode()
        plt.close()

        # Plot 2: Only Predicted Prices
        plt.figure(figsize=(10, 6))
        plt.plot(future_dates, predictions, label="Predicted Prices", color="orange", linewidth=3)
        plt.scatter(future_dates, predictions, color="red", s=60)
        plt.title(f"{ticker} - Future Price Predictions")
        plt.xlabel("Future Dates")
        plt.ylabel("Predicted Price ($)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        img2 = io.BytesIO()
        plt.savefig(img2, format="png", dpi=80, bbox_inches="tight")
        img2.seek(0)
        plot_url2 = base64.b64encode(img2.getvalue()).decode()
        plt.close()

        del img1, img2
        gc.collect()

        return plot_url1, plot_url2