🪙 Bitcoin Price Prediction — Regression & Classification
Overview

This project predicts Bitcoin’s future price and price direction (Up/Down) using advanced feature engineering, machine learning models, and a Flask web app for real-time prediction.
It combines both regression and classification tasks, enabling a complete market forecasting pipeline.

🚀 Features

Predicts next-day closing price using Regression

Predicts price direction (up/down) using Classification

Extensive feature engineering with 60+ technical indicators

Handles class imbalance with SMOTEENN

Supports real-time prediction through Flask web interface

Model persistence with Pickle for deployment

🧠 Machine Learning Workflow
1. Data Preparation

Loaded Bitcoin OHLCV data (Open, High, Low, Close, Volume, Timestamp)

Cleaned missing and invalid values

Converted timestamp to datetime and sorted chronologically

2. Feature Engineering

Implemented three major feature engineering pipelines:

-Technical Indicators
price_range, RSI_14, MACD, Bollinger Bands, MA_7, MA_21, MA_50,
volatility_7, volatility_21, MA_ratio_7_21, MA_ratio_7_50, volume_ratio, etc.

-Time-based Features
hour, day_of_week, month, quarter, year,
hour_sin, month_cos, is_weekend, is_month_start, is_month_end

-Lag & Rolling Features
close_lag_1, close_lag_2, close_lag_7, rolling_mean_7, rolling_std_14,
price_change_1d, price_change_7d, price_change_30d

-Target Variables

target_close: Next-day closing price (Regression)

target_direction: Up (1) or Down (0) (Classification)

target_return: Percentage change

Models Used
Regression Models
Model	Description
Linear Regression ✅	Best performing model (selected)
Decision Tree	Baseline non-linear model
Random Forest	Ensemble regressor
Gradient Boosting	Robust boosting model
Classification Models
Model	Description
Logistic Regression	Strong baseline classifier
Random Forest	Handles feature interactions
Gradient Boosting	Captures complex patterns
XGBoost ✅	High-performance boosting model
⚙️ Model Selection

Best Regression: Linear Regression

Best Classification: XGBoost

Metrics used:

Regression → RMSE, MAE, R²

Classification → Accuracy, Precision, Recall, F1

💾 Model Persistence

Models are saved for production using pickle:

best_regression_model.pkl
best_bitcoin_classification_model.pkl
scaler_class.pkl

🌐 Flask Web App
App Features

User inputs: Open, High, Low, Close, Volume

Predicts:

Next-day Price

Next-day Trend (Up/Down)

Displays prediction and status on the web interface

Run Locally
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Run Flask app
python app.py

# Step 3: Open in browser
http://127.0.0.1:5000/

 Example Inputs
Open	High	Low	Close	Volume
61000	61500	60500	61250	3200
61200	61300	60000	60200	4100
60800	62000	60500	61800	6800
📈 Example Output
Predicted Close Price: 61645.23
Predicted Direction: 📈 UP

Tech Stack

Python

Scikit-Learn

XGBoost

Imbalanced-learn

Flask

Pandas / NumPy

Matplotlib / Seaborn

Future Improvements

Integrate LSTM / Transformer models for sequence forecasting

Use real-time API (Binance / CoinGecko) for live predictions

Build a dashboard with Plotly or Streamlit

Deploy on Render / Hugging Face Spaces / AWS Lambda
