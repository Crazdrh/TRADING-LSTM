from polygon import RESTClient
import pandas as pd
from datetime import date, timedelta
import time

client = RESTClient(api_key="")

start_date = date(2023, 7, 15)
end_date   = date(2025, 7, 14)
bars = []

while start_date < end_date:
    window_end = min(start_date + timedelta(days=128), end_date)
    print(f"Fetching {start_date} to {window_end}")
    try:
        chunk = list(client.list_aggs(
            ticker="META",
            multiplier=5,
            timespan="minute",
            from_=start_date,
            to=window_end,
            limit=5000000
        ))
        bars.extend(chunk)
    except Exception as e:
        print(f"Error: {e}")
        print("Sleeping for 60 seconds due to possible rate limit...")
        time.sleep(60)
        continue
    start_date = window_end + timedelta(days=1)
    time.sleep(12)  # 5 calls per minute safe limit

print(f"Downloaded {len(bars)} bars")

# Convert to DataFrame and save
df = pd.DataFrame([bar.__dict__ for bar in bars])

# Save to Downloads folder
download_path = r"C:/Users/Hayden/Lambda/LSTM/Lstm/data/polygon/meta_5min.csv"
df.to_csv(download_path, index=False)
print(f"Data saved to {download_path}")
