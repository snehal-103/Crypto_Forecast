# Import Libraries
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import zscore
from sklearn.feature_selection import RFECV

# ✅ Download Data (Ensure Raw Prices)
crypto_data = yf.download('BTC-USD', start='2008-01-01', end='2025-03-22', auto_adjust=False)

# ✅ Flatten MultiIndex if necessary
if isinstance(crypto_data.columns, pd.MultiIndex):
    crypto_data.columns = crypto_data.columns.droplevel(1)

# ✅ Debug: Check Available Columns
print("Columns in crypto_data:", crypto_data.columns)

# ✅ Ensure Column Names are Accessible
crypto_data = crypto_data[['Close', 'High', 'Low', 'Open', 'Volume']]
crypto_data['Prev_Close'] = crypto_data['Close'].shift(1)

# ✅ Convert 'Volume' to Numeric (Handling Potential Errors)
crypto_data['Volume'] = pd.to_numeric(crypto_data['Volume'], errors='coerce')

# ✅ Moving Averages (Short & Long-Term Trends)
crypto_data['MA_5'] = crypto_data['Close'].rolling(window=5).mean()
crypto_data['MA_10'] = crypto_data['Close'].rolling(window=10).mean()
crypto_data['EMA_50'] = crypto_data['Close'].ewm(span=50, adjust=False).mean()
crypto_data['EMA_100'] = crypto_data['Close'].ewm(span=100, adjust=False).mean()

# ✅ Bollinger Bands
crypto_data['BB_Upper'] = crypto_data['Close'].rolling(window=10).mean() + (crypto_data['Close'].rolling(window=10).std() * 2)
crypto_data['BB_Lower'] = crypto_data['Close'].rolling(window=10).mean() - (crypto_data['Close'].rolling(window=10).std() * 2)

# ✅ Relative Strength Index (RSI)
delta = crypto_data['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
crypto_data['RSI'] = 100 - (100 / (1 + rs))

# ✅ MACD (Moving Average Convergence Divergence)
crypto_data['EMA_12'] = crypto_data['Close'].ewm(span=12, adjust=False).mean()
crypto_data['EMA_26'] = crypto_data['Close'].ewm(span=26, adjust=False).mean()
crypto_data['MACD'] = crypto_data['EMA_12'] - crypto_data['EMA_26']

# ✅ Williams %R (Momentum Indicator)
crypto_data['High_14'] = crypto_data['High'].rolling(window=14).max()
crypto_data['Low_14'] = crypto_data['Low'].rolling(window=14).min()
crypto_data['Williams_%R'] = -100 * ((crypto_data['High_14'] - crypto_data['Close']) / (crypto_data['High_14'] - crypto_data['Low_14']))

# ✅ ATR (Average True Range for Volatility Measurement)
crypto_data['ATR'] = crypto_data['Close'].rolling(window=14).std()

# ✅ Log transformation of volume to reduce skewness
crypto_data['Log_Volume'] = np.log1p(crypto_data['Volume'])

# ✅ Rate of Change (ROC)
crypto_data['ROC'] = ((crypto_data['Close'] - crypto_data['Close'].shift(10)) / crypto_data['Close'].shift(10)) * 100

# ✅ Drop Rows with Excessive NaN Values (Adjust Threshold)
crypto_data.dropna(inplace=True)

# ✅ Prepare Features & Labels
X = crypto_data[['Prev_Close', 'Open', 'MA_5', 'MA_10', 'EMA_50', 'EMA_100', 'BB_Upper', 'BB_Lower', 'RSI', 'MACD',
                 'Williams_%R', 'ATR', 'Log_Volume', 'ROC']]
y = crypto_data['Close']

# ✅ Ensure No Empty DataFrame (Handling Edge Case)
if X.empty or y.empty:
    raise ValueError("❌ Error: Feature dataset (X) or target variable (y) is empty!")

# ✅ Train/Test Split (Using TimeSeriesSplit)
tscv = TimeSeriesSplit(n_splits=5)
train_indices, test_indices = list(tscv.split(X))[-1]
X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]

# ✅ Scale Features & Target
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()

# ✅ Feature Selection Using Recursive Feature Elimination (RFE)
svr_rfe = SVR(kernel='linear')
selector = RFECV(svr_rfe, step=1, cv=TimeSeriesSplit(n_splits=5), scoring='r2')
X_selected_train = selector.fit_transform(X_train_scaled, y_train_scaled)
X_selected_test = selector.transform(X_test_scaled)

# ✅ Train Optimized SVM Model with Fine-Tuned Hyperparameters
svm_model = SVR(kernel='rbf', C=1000, gamma=0.003, epsilon=0.0005)  # 🔥 Further Optimized
svm_model.fit(X_selected_train, y_train_scaled)

# ✅ Make Predictions
y_pred_scaled = svm_model.predict(X_selected_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

# ✅ Calculate Metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100  # Mean Absolute Percentage Error

# ✅ Print Results
print(f"✅ Final Optimized Mean Absolute Error (MAE): {mae:.2f}")
print(f"✅ Final Optimized Mean Squared Error (MSE): {mse:.2f}")
print(f"✅ Final Optimized R² Score: {r2:.4f}")
print(f"✅ Final Optimized Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
