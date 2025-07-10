import os
import pandas as pd

# === CONFIGURATION ===
csv_dir = "C:/Users/Hayden/Downloads/csv/"  # <-- Change to your actual directory
lookahead = 5
buy_thresh = 0.002      # 0.2%
sell_thresh = -0.002    # -0.2%

label_to_class = {
    "buy": 1,
    "hold": 0,
    "sell": 2
}

def compute_future(df, lookahead):
    df = df.copy()
    df["future_close"] = df["close"].shift(-lookahead)
    df["future_return"] = (df["future_close"] - df["close"]) / df["close"]
    return df

def get_signal(future_return):
    if pd.isnull(future_return):
        return "hold"
    if future_return >= buy_thresh:
        return "buy"
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
        df = df.iloc[:-lookahead] if lookahead > 0 else df
        df.to_csv(fpath, index=False)
    print(f"Done! All {len(files)} files in '{csv_dir}' labeled.")

if __name__ == "__main__":
    label_all_csvs_in_directory(csv_dir)
