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

class StockPredictor:
    def __init__(self):
        self.models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(n_estimators=50, random_state=42),
        }
        self.current_model = "Linear Regression"

    def fetch_data(self, ticker, days=60):
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        print(f"Fetching data for {ticker} from {start_date} to {end_date}")

        try:
            # Try multiple approaches to fetch data
            data = yf.download(
                ticker, 
                start=start_date, 
                end=end_date, 
                progress=False,
                auto_adjust=True
            )
            
            print(f"Downloaded data shape: {data.shape if data is not None else 'None'}")
            
            if data is None or data.empty:
                print("Data is empty, trying alternative approach...")
                # Try with a ticker object instead
                ticker_obj = yf.Ticker(ticker)
                data = ticker_obj.history(period=f"{days}d")
                print(f"Ticker object data shape: {data.shape if data is not None else 'None'}")
            
            if data is None or data.empty:
                print("Both methods failed, generating mock data...")
                # If yfinance fails, generate mock data
                dates = pd.date_range(end=end_date, periods=days, freq='D')
                mock_prices = np.random.normal(150, 20, days).cumsum()  # Random walk
                data = pd.DataFrame({'Close': mock_prices}, index=dates)
                return data, "Using mock data (yfinance failed)"
            
            # Ensure we have the right columns
            if 'Close' not in data.columns:
                if 'Adj Close' in data.columns:
                    data = data[['Adj Close']].copy()
                    data.columns = ['Close']
                    print("Using Adj Close instead of Close")
                else:
                    print("No Close or Adj Close found, using first column")
                    data = data.iloc[:, [0]].copy()
                    data.columns = ['Close']
            else:
                data = data[['Close']].copy()
            
            print(f"Final data shape: {data.shape}")
            print(f"Data range: {data.index.min()} to {data.index.max()}")
            print(f"Sample data: {data.head().to_dict()}")
            
            return data, None

        except Exception as e:
            error_msg = f"Error fetching data for {ticker}: {str(e)}"
            print(error_msg)
            # Generate mock data on error
            dates = pd.date_range(end=end_date, periods=days, freq='D')
            mock_prices = np.random.normal(150, 20, days).cumsum()
            data = pd.DataFrame({'Close': mock_prices}, index=dates)
            return data, "Using mock data (yfinance error)"

    def preprocess_data(self, data):
        # Ensure flat column name
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
        
        # Clean up memory
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
        plt.figure(figsize=(10, 6))
        plt.plot(data.index, data["Close"], label="Historical Prices", color="#034F68")
        future_dates = [data.index[-1] + timedelta(days=i) for i in range(1, len(predictions) + 1)]
        plt.plot(future_dates, predictions, label="Predicted Prices", color="orange")
        plt.scatter(future_dates, predictions, color="red")
        plt.title(f"{ticker} - Historical vs Predicted Prices")
        plt.xlabel("Date")
        plt.ylabel("Price ($)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        img1 = io.BytesIO()
        plt.savefig(img1, format="png", dpi=80, bbox_inches="tight")
        img1.seek(0)
        plot_url1 = base64.b64encode(img1.getvalue()).decode()
        plt.close()

        # Plot 2: Only Predicted Prices
        plt.figure(figsize=(10, 6))
        plt.plot(future_dates, predictions, label="Predicted Prices", color="orange", linewidth=2)
        plt.scatter(future_dates, predictions, color="red", s=50)
        plt.title(f"{ticker} - Future Price Predictions")
        plt.xlabel("Future Dates")
        plt.ylabel("Predicted Price ($)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        img2 = io.BytesIO()
        plt.savefig(img2, format="png", dpi=80, bbox_inches="tight")
        img2.seek(0)
        plot_url2 = base64.b64encode(img2.getvalue()).decode()
        plt.close()

        # Clean up memory
        del img1, img2
        gc.collect()

        return plot_url1, plot_url2  # Now returns two different plots