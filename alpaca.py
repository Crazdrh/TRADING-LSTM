import pandas as pd
from datetime import datetime, timedelta, date
import time

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# REPLACE with your real keys!
API_KEY = ""
API_SECRET = ""
client = StockHistoricalDataClient(API_KEY, API_SECRET)

symbol = "NVDA"
start_date = date(2021, 7, 15)   # Change this! Alpaca only supports recent years for minute data
end_date   = date(2024, 7, 15)
bars = []

# Alpaca's API limit: 1000 bars per request for minute timeframe
# At 390 bars per trading day, ~2-3 trading days per chunk
chunk_days = 1

while start_date < end_date:
    window_end = min(start_date + timedelta(days=chunk_days-1), end_date)
    print(f"Fetching {start_date} to {window_end}")
    try:
        request_params = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=TimeFrame.Minute,
            start=datetime.combine(start_date, datetime.min.time()),
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
            # Filter to just our symbol
            df = df[df.index.get_level_values("symbol") == symbol]
            bars.append(df)
    except Exception as e:
        print(f"Error: {e}")
        print("Sleeping for 60 seconds due to possible rate limit...")
        time.sleep(60)
        continue
    start_date = window_end + timedelta(days=1)
    time.sleep(1.5)  # Respect API limits

if not bars:
    print("No data downloaded for your range! Try more recent dates.")
    exit(1)

full_df = pd.concat(bars)
full_df = full_df.reset_index()

# Drop rows with all OHLCV missing or 0 volume
mask = (
    full_df[["open", "high", "low", "close"]].notnull().any(axis=1) &
    (full_df["volume"] > 0)
)
filtered_df = full_df[mask]

if filtered_df.empty:
    print("No valid bars after cleaning. Likely your date range is too old.")
    exit(1)
# Save to CSV
download_path = r"C:/Users/hayden/LSTM/alpaca/nvda_5min.csv"
filtered_df.to_csv(download_path, index=False)
print(f"Data saved to {download_path}")
