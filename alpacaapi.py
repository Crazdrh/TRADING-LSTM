import pandas as pd
from datetime import datetime, timedelta, date
import time

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

API_KEY = ""
API_SECRET = ""
client = StockHistoricalDataClient(API_KEY, API_SECRET)

SYMBOLS = ["AAPL", "GOOGL", "NVDA"]  # <-- Edit this list for your stocks
start_date = date(2020, 7, 18)
end_date   = date(2025, 6, 18)
chunk_days = 15  # 15 days at a time

save_dir = r"C:/Users/Hayden/Lambda/LSTM/Lstm/data/alpaca/"

for symbol in SYMBOLS:
    print(f"\n=== Downloading: {symbol} ===")
    bars = []
    cur_start = start_date
    while cur_start < end_date:
        window_end = min(cur_start + timedelta(days=chunk_days-1), end_date)
        print(f"Fetching {symbol} {cur_start} to {window_end}")
        try:
            request_params = StockBarsRequest(
                symbol_or_symbols=[symbol],
                timeframe=TimeFrame(15, TimeFrameUnit.Minute),
                start=datetime.combine(cur_start, datetime.min.time()),
                end=datetime.combine(window_end, datetime.max.time()),
                adjustment=None,
                feed=None,
                limit=1000
            )
            response = client.get_stock_bars(request_params)
            df = response.df
            if df.empty:
                print("No data returned for this range.")
            else:
                df = df[df.index.get_level_values("symbol") == symbol]
                bars.append(df)
        except Exception as e:
            print(f"Error: {e}")
            print("Sleeping for 60 seconds due to possible rate limit...")
            time.sleep(60)
            continue
        cur_start = window_end + timedelta(days=1)
        time.sleep(1.5)  # Rate limiting

    if not bars:
        print(f"No data downloaded for {symbol} in your range! Skipping.")
        continue

    full_df = pd.concat(bars).reset_index()
    mask = (
        full_df[["open", "high", "low", "close"]].notnull().any(axis=1) &
        (full_df["volume"] > 0)
    )
    filtered_df = full_df[mask]
    if filtered_df.empty:
        print(f"No valid bars after cleaning for {symbol}. Skipping.")
        continue

    download_path = f"{save_dir}{symbol}_15min.csv"
    filtered_df.to_csv(download_path, index=False)
    print(f"Data for {symbol} saved to {download_path}")

print("\nDone downloading all symbols.")

