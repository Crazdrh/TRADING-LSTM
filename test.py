import os
import numpy as np
import pandas as pd
import torch
from LSTM import ComplexLSTMModel  # Make sure this matches your model file

# ==== CONFIGURATION ====
CSV_DIR = "C:/Users/Hayden/Downloads/5mincsvma"  # Directory with your 20 test CSVs
MODEL_PATH = "C:/Users/Hayden/Downloads/trained_lstm_model_with_ma.pth"
SEQ_LEN = 50
INPUT_COLS = ['open', 'high', 'low', 'close', 'ma', 'ma.1', 'ma.2', 'ma.3', 'ma.4']  # Fixed: lowercase

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==== LOAD MODEL ====
def load_model(model_path, n_classes):
    model = ComplexLSTMModel(input_dim=9, output_dim=n_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


# ==== PREPROCESSING ====
def preprocess_features(df):
    # Lowercase column names for consistency
    df.columns = [c.lower() for c in df.columns]

    # Check if all required columns exist
    missing_cols = [col for col in INPUT_COLS if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing columns after lowercasing: {missing_cols}")

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

    # Make sure we have enough data
    if len(features) <= seq_len:
        print(f"Warning: Not enough data points ({len(features)}) for sequence length {seq_len}")
        return np.nan, 0, np.array([]), np.array([])

    for i in range(len(features) - seq_len):
        seq = features[i:i + seq_len]
        x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)  # (1, seq_len, 9)
        with torch.no_grad():
            logits = model(x)
            pred_class = logits.argmax(dim=1).item()
        true_class = int(df['signal_class'].iloc[i + seq_len])
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

    # Assume all files use the same class mapping (0=hold, 1=buy, 2=sell based on your data)
    # Read one file to infer class count
    temp_df = pd.read_csv(os.path.join(CSV_DIR, files[0]))
    n_classes = int(temp_df['signal_class'].nunique())
    print(f"Detected {n_classes} classes in the data")

    model = load_model(MODEL_PATH, n_classes)
    print(f"Model loaded successfully from {MODEL_PATH}")

    all_preds = []
    all_trues = []
    total_samples = 0
    successful_files = 0

    print(f"\nTesting model on {len(files)} files...\n")

    for fname in files:
        fpath = os.path.join(CSV_DIR, fname)
        try:
            df = pd.read_csv(fpath)
            features = preprocess_features(df)
            acc, n, preds, trues = test_on_file(model, df, features, seq_len=SEQ_LEN)

            if n > 0:  # Only count files with valid predictions
                total_samples += n
                all_preds.append(preds)
                all_trues.append(trues)
                successful_files += 1
                print(f"{fname}: {acc * 100:.2f}% accuracy on {n} samples")
            else:
                print(f"{fname}: No valid predictions (insufficient data)")

        except Exception as e:
            print(f"{fname}: ERROR - {str(e)}")
            continue

    # ==== OVERALL ACCURACY ====
    if successful_files > 0:
        all_preds = np.concatenate(all_preds)
        all_trues = np.concatenate(all_trues)
        overall_acc = (all_preds == all_trues).mean()

        print("\n" + "=" * 50)
        print(f"Overall Results:")
        print(f"Files processed successfully: {successful_files}/{len(files)}")
        print(f"Total samples: {len(all_preds)}")
        print(f"Overall accuracy: {overall_acc * 100:.2f}%")

        # Class distribution
        unique_classes, counts = np.unique(all_trues, return_counts=True)
        print(f"\nClass distribution in test data:")
        for cls, count in zip(unique_classes, counts):
            print(f"  Class {cls}: {count} samples ({count / len(all_trues) * 100:.1f}%)")

        # Per-class accuracy
        print(f"\nPer-class accuracy:")
        for cls in unique_classes:
            mask = all_trues == cls
            cls_acc = (all_preds[mask] == all_trues[mask]).mean()
            print(f"  Class {cls}: {cls_acc * 100:.2f}%")

        print("=" * 50)
    else:
        print("\nNo files processed successfully!"
