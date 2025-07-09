import os
import pandas as pd

# === CONFIGURATION ===
csv_dir = "C:/Users/Hayden/Downloads/csv/"      # <-- Change to your actual directory
lookahead = 5                       # bars ahead for 'future' labeling
strong_buy_thresh = 0.01
buy_thresh = 0.002
strong_sell_thresh = -0.001
sell_thresh = -0.002

label_to_class = {
    "strong_buy": 0,
    "buy": 1,
    "hold": 2,
    "sell": 3,
    "strong_sell": 4
}

def compute_future(df, lookahead):
    df = df.copy()
    df["future_close"] = df["close"].shift(-lookahead)
    df["future_return"] = (df["future_close"] - df["close"]) / df["close"]
    return df

def get_signal(future_return):
    if pd.isnull(future_return):
        return "hold"
    if future_return >= strong_buy_thresh:
        return "strong_buy"
    elif future_return >= buy_thresh:
        return "buy"
    elif future_return <= strong_sell_thresh:
        return "strong_sell"
    elif future_return <= sell_thresh:
        return "sell"
    else:
        return "hold"

def label_dataframe(df):
    df = compute_future(df, lookahead)
    df["signal"] = df["future_return"].apply(get_signal)
    df["signal_class"] = df["signal"].map(label_to_class)
    return df

def label_all_csvs_in_directory(csv_dir):
    files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
    for fname in files:
        fpath = os.path.join(csv_dir, fname)
        print(f"Labeling: {fname}")
        df = pd.read_csv(fpath)
        df = label_dataframe(df)
        # Optionally drop last N rows with no future label
        df = df.iloc[:-lookahead] if lookahead > 0 else df
        df.to_csv(fpath, index=False)
    print(f"Done! All {len(files)} files in '{csv_dir}' labeled.")

if __name__ == "__main__":
    label_all_csvs_in_directory(csv_dir)
