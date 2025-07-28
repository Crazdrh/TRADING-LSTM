import pandas as pd
from datetime import datetime, timedelta, date
import time

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

API_KEY = ""
API_SECRET = ""
client = StockHistoricalDataClient(API_KEY, API_SECRET)

SYMBOLS = sp500_tickers = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "GOOG", "META", "BRK.B", "UNH", "LLY", "JPM", "V", "XOM", "TSLA", "AVGO",
    "PG", "MA", "JNJ", "COST", "MRK", "HD", "ADBE", "ABBV", "CRM", "CVX", "PEP", "TMO", "NFLX", "KO", "ACN", "LIN",
    "AMD", "WMT", "ORCL", "MCD", "ABT", "DHR", "CSCO", "QCOM", "DIS", "VZ", "WFC", "INTU", "TXN", "HON", "NEE", "LOW",
    "PM", "MS", "AMGN", "AMAT", "CAT", "UNP", "IBM", "BA", "GS", "RTX", "SPGI", "GE", "ISRG", "MDT", "PLD", "LMT",
    "ADP", "T", "CVS", "MU", "NOW", "BLK", "C", "SYK", "ELV", "DE", "SCHW", "EL", "GILD", "REGN", "BDX", "ZTS", "TGT",
    "VRTX", "MO", "CB", "SO", "USB", "MMC", "CI", "CL", "DUK", "PNC", "BKNG", "AON", "FISV", "SLB", "ADSK", "ETN",
    "FDX", "ITW", "TJX", "GM", "PGR", "NSC", "EQIX", "EOG", "LRCX", "MPC", "AEP", "EW", "SRE", "NOC", "SHW", "APD",
    "CSX", "ROP", "GD", "FIS", "IDXX", "WMB", "EMR", "STZ", "PSA", "EXC", "D", "AIG", "FCX", "PAYX", "KMB", "AZO",
    "OXY", "TRV", "HUM", "AFL", "PH", "ORLY", "MCK", "PXD", "PSX", "CNC", "MSCI", "COF", "ECL", "HLT", "MAR", "CDNS",
    "BAX", "HCA", "CTSH", "ROP", "SPG", "KMI", "O", "ALL", "DLR", "DOV", "A", "BKR", "CNP", "AVB", "BBY", "BXP", "CDW",
    "CF", "CHD", "CINF", "CMA", "CMS", "CNP", "COO", "CPB", "CRL", "CTAS", "DG", "DHI", "DLTR", "DRI", "EIX", "ESS",
    "F", "FANG", "FAST", "FMC", "FRC", "FTNT", "GL", "GLW", "GRMN", "GWW", "HAS", "HIG", "HOLX", "HST", "HSY", "IFF",
    "ILMN", "INCY", "IP", "IPG", "IR", "IRM", "JBHT", "JCI", "JKHY", "JNPR", "K", "KEY", "KEYS", "KIM", "KMX", "KR",
    "LEG", "LEN", "LH", "LKQ", "LNT", "LUV", "LW", "MAS", "MKC", "MKTX", "MLM", "MNST", "MOS", "MPWR", "MTB", "MTCH",
    "NDAQ", "NDSN", "NEM", "NI", "NUE", "NVR", "NWL", "NWS", "NWSA", "ODFL", "OGN", "OKE", "OMC", "ON", "OTIS", "PARA",
    "PAYC", "PBCT", "PCAR", "PEAK", "PKG", "PODD", "POOL", "PPL", "PRGO", "PRU", "PTC", "PVH", "QRVO", "RCL", "RJF",
    "RMD", "ROL", "RSG", "SBAC", "SBNY", "SBUX", "SEDG", "SEE", "SJM", "SMG", "SNA", "SNPS", "SNY", "SPSC", "STE",
    "SWK", "SWKS", "SYY", "TAP", "TDY", "TECH", "TER", "TFC", "TPR", "TROW", "TSCO", "TTWO", "UAL", "UDR", "UHS", "ULTA",
    "URI", "VFC", "VICI", "VLO", "VMC", "VTR", "VTRS", "VZ", "WAB", "WAT", "WDC", "WELL", "WLTW", "WRB", "WY", "WYN",
    "XEL", "XYL", "YUM", "ZBH", "ZBRA"
]# <-- Edit this list for your stocks
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

