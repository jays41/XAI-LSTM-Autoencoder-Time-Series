import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal).mean()
    return macd, signal_line

sp500 = yf.download('^GSPC', start='2018-01-01', end='2020-01-01', interval='1d')

df = sp500[['Close', 'Volume']].copy()
df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
df['Realised_Volatility'] = df['Log_Returns'].rolling(window=20).std()

df['RSI'] = calculate_rsi(df['Close'])

df['MACD'], df['MACD_Signal'] = calculate_macd(df['Close'])

df['Normalised_Volume'] = (df['Volume'] - df['Volume'].rolling(window=20).mean()) / df['Volume'].rolling(window=20).std()

df['Momentum'] = df['Close'].pct_change(10)

df = df.dropna()

# check for infinite values
if df.isin([np.inf, -np.inf]).any().any():
    print("Warning: Infinite values detected, replacing with NaN")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

# check reasonable ranges
if (df['RSI'] < 0).any() or (df['RSI'] > 100).any():
    print("Warning: RSI values outside [0, 100] range")
if abs(df['Normalised_Volume']).max() > 10:
    print(f"Warning: Normalized volume has extreme values (max abs: {abs(df['Normalised_Volume']).max():.2f})")

df.to_csv('sp500_data.csv')