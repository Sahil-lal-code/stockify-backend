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

class StockPredictor:
    def __init__(self):
        self.models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        }
        self.current_model = "Linear Regression"

    def fetch_data(self, ticker, days=100):
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        try:
            data = yf.download(ticker, start=start_date, end=end_date)
            if data.empty:
                return None, "No data found for this ticker symbol."

            # Flatten MultiIndex columns if present
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            # Ensure single-level column and return only Close
            data = data[["Close"]].copy()
            return data, None

        except Exception as e:
            return None, f"Error fetching data: {str(e)}"

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
        return model, r2, mse

    def predict(self, model, data, days=1):
        future_predictions = []
        current_data = data.copy()

        for _ in range(days):
            next_day_pred = model.predict(current_data[-1:])
            # NaN guard
            pred_value = next_day_pred[0]
            if np.isnan(pred_value):
                raise ValueError("Model produced NaN prediction; insufficient data.")
            future_predictions.append(pred_value)

            # Append prediction for multi-day forecasting
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
        plt.title(f"{ticker} Stock Price Prediction (Historical vs Predicted)")
        plt.xlabel("Date")
        plt.ylabel("Price ($)")
        plt.legend()
        plt.grid(True)
        plt.gcf().autofmt_xdate()  # Rotate date labels
        img1 = io.BytesIO()
        plt.savefig(img1, format="png", bbox_inches="tight")
        img1.seek(0)
        plot_url1 = base64.b64encode(img1.getvalue()).decode()
        plt.close()

        # Plot 2: Only Predicted
        plt.figure(figsize=(10, 6))
        plt.plot(future_dates, predictions, label="Predicted Prices", color="orange")
        plt.scatter(future_dates, predictions, color="red")
        plt.title(f"{ticker} Stock Price Prediction (Only Predicted)")
        plt.xlabel("Date")
        plt.ylabel("Price ($)")
        plt.legend()
        plt.grid(True)

        # Skip dates in x-axis ticks to avoid overlapping
        skip = max(1, len(future_dates) // 10)  # Skip every 'skip' dates
        plt.xticks(future_dates[::skip], rotation=45)  # Rotate labels for better readability

        img2 = io.BytesIO()
        plt.savefig(img2, format="png", bbox_inches="tight")
        img2.seek(0)
        plot_url2 = base64.b64encode(img2.getvalue()).decode()
        plt.close()

        return plot_url1, plot_url2