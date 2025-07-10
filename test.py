import os
import numpy as np
import pandas as pd
import torch
from LSTM import ComplexLSTMModel  # Make sure this matches your model file

# ==== CONFIGURATION ====
CSV_DIR = "C:/Users/Hayden/Downloads/csv/"  # Directory with your 20 test CSVs
MODEL_PATH = "C:/Users/Hayden/Downloads/trained_lstm_model2.pth"
SEQ_LEN = 50
INPUT_COLS = ['open', 'high', 'low', 'close']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== LOAD MODEL ====
def load_model(model_path, n_classes):
    model = ComplexLSTMModel(input_dim=4, output_dim=n_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# ==== PREPROCESSING ====
def preprocess_features(df):
    # Lowercase column names for consistency
    df.columns = [c.lower() for c in df.columns]
    features = df[INPUT_COLS].astype(float).values
    # Normalize per-file as in your train/inference scripts
    means = features.mean(axis=0)
    stds = features.std(axis=0) + 1e-8
    features_norm = (features - means) / stds
    return features_norm

# ==== PREDICT AND EVALUATE ====
def test_on_file(model, df, features, seq_len=SEQ_LEN):
    preds = []
    trues = []
    for i in range(len(features) - seq_len):
        seq = features[i:i+seq_len]
        x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)  # (1, seq_len, 4)
        with torch.no_grad():
            logits = model(x)
            pred_class = logits.argmax(dim=1).item()
        true_class = int(df['signal_class'].iloc[i+seq_len])
        preds.append(pred_class)
        trues.append(true_class)
    preds = np.array(preds)
    trues = np.array(trues)
    accuracy = (preds == trues).mean() if len(preds) > 0 else np.nan
    return accuracy, len(preds), preds, trues

# ==== MAIN LOOP ====
if __name__ == "__main__":
    files = [f for f in os.listdir(CSV_DIR) if f.endswith('.csv')]
    if not files:
        print(f"No CSV files found in '{CSV_DIR}'!")
        exit(1)

    # Assume all files use the same class mapping (0=buy, 1=hold, 2=sell)
    # Read one file to infer class count
    temp_df = pd.read_csv(os.path.join(CSV_DIR, files[0]))
    n_classes = int(temp_df['signal_class'].nunique())
    model = load_model(MODEL_PATH, n_classes)

    all_preds = []
    all_trues = []
    total_samples = 0

    print(f"Testing model on {len(files)} files...\n")
    for fname in files:
        fpath = os.path.join(CSV_DIR, fname)
        df = pd.read_csv(fpath)
        features = preprocess_features(df)
        acc, n, preds, trues = test_on_file(model, df, features, seq_len=SEQ_LEN)
        total_samples += n
        all_preds.append(preds)
        all_trues.append(trues)
        print(f"{fname}: {acc*100:.2f}% accuracy on {n} samples.")

    # ==== OVERALL ACCURACY ====
    all_preds = np.concatenate(all_preds)
    all_trues = np.concatenate(all_trues)
    overall_acc = (all_preds == all_trues).mean() if len(all_preds) > 0 else np.nan
    print("\n====================")
    print(f"Overall accuracy: {overall_acc*100:.2f}% ({len(all_preds)} samples across {len(files)} files)")
    print("====================")
