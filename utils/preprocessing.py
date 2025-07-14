
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def preprocess_data(df):
    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    df["Returns"] = df["Close"].pct_change().fillna(0)
    df["Volume"] = pd.to_numeric(df["Volume"], errors='coerce').fillna(1)
    df["Log_Volume"] = np.log(df["Volume"].clip(lower=1))

    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    df["RSI"] = 100 - (100 / (1 + rs))

    ema_fast = df["Close"].ewm(span=12, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=26, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    df["MACD"] = signal_line

    df.dropna(inplace=True)
    return df
