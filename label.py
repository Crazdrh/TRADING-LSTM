import os
import pandas as pd
import numpy as np
import ta

INPUT_DIR = 'C:/Users/Hayden/Lambda/LSTM/Lstm/data/polygon'  # Change to your input folder
OUTPUT_DIR = 'C:/Users/Hayden/Lambda/LSTM/Lstm/data/polygon/1'  # Change to your output folder
os.makedirs(OUTPUT_DIR, exist_ok=True)


def process_csv(file_path, save_path):
    df = pd.read_csv(file_path)

    # --- Trend Indicators ---
    df['ma_10'] = df['close'].rolling(10).mean()
    df['ma_20'] = df['close'].rolling(20).mean()
    df['ma_30'] = df['close'].rolling(30).mean()
    df['ma_40'] = df['close'].rolling(40).mean()
    df['dist_from_ma_10'] = (df['close'] - df['ma_10']) / df['ma_10']
    df['dist_from_ma_20'] = (df['close'] - df['ma_20']) / df['ma_20']

    # --- Volume Indicators ---
    df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
    df['relative_volume'] = df['volume'] / df['volume'].rolling(20).mean()
    df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()

    # --- Volatility Indicators ---
    bb = ta.volatility.BollingerBands(df['close'], window=20)
    df['bb_width'] = bb.bollinger_hband() - bb.bollinger_lband()
    df['bb_position'] = (df['close'] - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband())
    df['atr_normalized'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'],
                                                          window=14).average_true_range() / df['close']

    # --- Momentum Indicators ---
    df['rsi_14'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    macd = ta.trend.MACD(df['close'])
    df['macd_diff'] = macd.macd_diff()
    df['stoch_k'] = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close']).stoch()
    df['roc_10'] = ta.momentum.ROCIndicator(df['close'], window=10).roc()

    # --- Market Structure ---
    df['price_to_vwap'] = (df['close'] - df['vwap']) / df['vwap']
    df['hl_spread'] = df['high'] - df['low']
    df['pivot_position'] = (df['close'] - (df['high'] + df['low'] + df['close']) / 3) / (
                (df['high'] + df['low'] + df['close']) / 3)

    # --- Time Features ---
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['hour'] = df['datetime'].dt.hour
        df['dow'] = df['datetime'].dt.dayofweek
        df['mins_since_open'] = df['hour'] * 60 + df['datetime'].dt.minute

        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dow_sin'] = np.sin(2 * np.pi * df['dow'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['dow'] / 7)
    else:
        # If there's no datetime, fill with zeros (or skip as needed)
        df['hour_sin'] = df['hour_cos'] = df['dow_sin'] = df['dow_cos'] = df['mins_since_open'] = 0

    # --- Engineered Features ---
    df['zscore_20'] = (df['close'] - df['close'].rolling(20).mean()) / df['close'].rolling(20).std()
    df['efficiency_ratio'] = np.abs(df['close'] - df['close'].shift(19)) / df['close'].rolling(20).apply(
        lambda x: np.sum(np.abs(np.diff(x))), raw=True)
    df['price_volume_corr'] = df['close'].rolling(20).corr(df['volume'])

    df = df.reset_index(drop=True)
    df.to_csv(save_path, index=False)
    print(f"Processed and saved: {save_path}")


for fname in os.listdir(INPUT_DIR):
    if fname.endswith('.csv'):
        in_path = os.path.join(INPUT_DIR, fname)
        out_path = os.path.join(OUTPUT_DIR, fname)
        try:
            process_csv(in_path, out_path)
        except Exception as e:
            print(f"Failed to process {fname}: {e}")

print("All files processed!")
