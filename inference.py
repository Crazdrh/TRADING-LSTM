import pandas as pd
import numpy as np
import torch
from LSTM import ComplexLSTMModel

# --------- CONFIG ---------
CSV_PATH = "/home/hayden/LSTM/Data/CSV/BATS_AAPL, 5.csv"   # <-- Update
MODEL_PATH = "/home/hayden/LSTM/Ckpt/final_model_weights.pth"
SEQ_LEN = 50
FEATURE_COLS = ['open', 'high', 'low', 'close', 'MA', 'MA.1', 'MA.2', 'MA.3', 'MA.4']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path, n_classes):
    model = ComplexLSTMModel(input_dim=9, output_dim=n_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def load_and_preprocess(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = [c.lower() for c in df.columns]
    for col in [c.lower() for c in FEATURE_COLS]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    features = df[[c.lower() for c in FEATURE_COLS]].astype(float).values
    means = features.mean(axis=0)
    stds = features.std(axis=0) + 1e-8
    features_norm = (features - means) / stds
    return df, features_norm

def predict_over_csv(model, features, df, seq_len=SEQ_LEN):
    results = []
    for i in range(len(features) - seq_len):
        seq = features[i:i+seq_len]
        x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)  # (1, seq_len, 9)
        with torch.no_grad():
            logits = model(x)
            pred_class = logits.argmax(dim=1).item()
        row_num = i + seq_len
        date = df['date'].iloc[row_num] if 'date' in df.columns else row_num
        actual = df['signal_class'].iloc[row_num] if 'signal_class' in df.columns else None
        results.append((row_num, date, pred_class, actual))
    return results

if __name__ == "__main__":
    df, features = load_and_preprocess(CSV_PATH)
    n_classes = int(df['signal_class'].nunique()) if 'signal_class' in df.columns else 3
    model = load_model(MODEL_PATH, n_classes)
    results = predict_over_csv(model, features, df, seq_len=SEQ_LEN)
    print("row_num\tdate\t\tpredicted\tactual")
    for row_num, date, pred, actual in results:
        print(f"{row_num}\t{date}\t{pred}\t\t{actual}")

