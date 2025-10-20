from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle
from datetime import datetime

# ========================
# Load Saved Models
# ========================
reg_model = pickle.load(open("best_regression_model.pkl", "rb"))
class_model = pickle.load(open("best_bitcoin_classification_model.pkl", "rb"))
scaler = pickle.load(open("scaler_class.pkl", "rb"))

app = Flask(__name__)

# ========================
# Feature Engineering Functions
# ========================

def create_technical_indicators(df):
    df = df.copy()
    df['price_range'] = df['High'] - df['Low']
    df['price_change'] = df['Close'] - df['Open']
    df['body_ratio'] = abs(df['Close'] - df['Open']) / (df['High'] - df['Low']).replace(0, 0.001)

    df['returns'] = df['Close'].pct_change()
    df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))

    df['MA_7'] = df['Close'].rolling(window=7).mean()
    df['MA_21'] = df['Close'].rolling(window=21).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()

    df['MA_ratio_7_21'] = df['MA_7'] / df['MA_21']
    df['MA_ratio_7_50'] = df['MA_7'] / df['MA_50']

    df['volatility_7'] = df['returns'].rolling(window=7).std()
    df['volatility_21'] = df['returns'].rolling(window=21).std()

    def calculate_rsi(prices, window=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    df['RSI_14'] = calculate_rsi(df['Close'])

    exp1 = df['Close'].ewm(span=12).mean()
    exp2 = df['Close'].ewm(span=26).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_histogram'] = df['MACD'] - df['MACD_signal']

    df['BB_middle'] = df['Close'].rolling(20).mean()
    bb_std = df['Close'].rolling(20).std()
    df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
    df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
    df['BB_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])

    df['volume_MA_7'] = df['Volume'].rolling(window=7).mean()
    df['volume_ratio'] = df['Volume'] / df['volume_MA_7']

    df['price_position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low']).replace(0, 0.001)
    return df


def create_time_features(df):
    df = df.copy()
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.set_index('Timestamp', inplace=True)

    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['day_of_month'] = df.index.day
    df['week_of_year'] = df.index.isocalendar().week
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df['year'] = df.index.year

    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week']/7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week']/7)
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)

    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_month_end'] = (df.index.is_month_end).astype(int)
    df['is_month_start'] = (df.index.is_month_start).astype(int)
    return df


# ========================
# Route: Home Page
# ========================
@app.route('/')
def home():
    return render_template('index.html')


# ========================
# Route: Predict
# ========================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        open_price = float(request.form['open'])
        high = float(request.form['high'])
        low = float(request.form['low'])
        close = float(request.form['close'])
        volume = float(request.form['volume'])

        # Create a dataframe with current timestamp
        data = pd.DataFrame([{
            'Timestamp': datetime.now(),
            'Open': open_price,
            'High': high,
            'Low': low,
            'Close': close,
            'Volume': volume
        }])

        # Apply feature engineering
        df = create_technical_indicators(data)
        df = create_time_features(df)

        # Fill all missing values
        df = df.fillna(method='bfill').fillna(method='ffill').fillna(0)

        # Drop unnecessary columns
        drop_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        X = df.drop(columns=drop_cols, errors='ignore')

        # Just in case any leftover NaNs remain
        X = X.replace([np.inf, -np.inf], 0).fillna(0)

        # Scale features for classification
        X_scaled = scaler.transform(X)

        # Predictions
        reg_pred = reg_model.predict(X)[0]
        class_pred = class_model.predict(X_scaled)[0]
        class_label = "📈 Price Up" if class_pred == 1 else "📉 Price Down"

        return render_template(
            'result.html',
            reg_pred=round(reg_pred, 2),
            class_label=class_label
        )

    except Exception as e:
        return render_template('result.html', error=f"Error: {str(e)}")


if __name__ == '__main__':
    app.run(debug=True)
