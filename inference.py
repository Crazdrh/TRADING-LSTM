import pandas as pd
import numpy as np
import torch
from LSTM import ComplexLSTMModel  # Make sure this matches your training model

# --------- CONFIG ---------
CSV_PATH = "C:/Users/Hayden/Downloads/YOUR_FILE.csv"   # <-- Update
MODEL_PATH = "C:/Users/Hayden/Downloads/trained_lstm_model2.pth"
SEQ_LEN = 50

# --------- DEVICE ---------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------- LOAD MODEL ---------
def load_model(model_path, n_classes):
    model = ComplexLSTMModel(input_dim=4, output_dim=n_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# --------- LOAD & PREP DATA ---------
def load_and_preprocess(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = [c.lower() for c in df.columns]
    for col in ['open', 'high', 'low', 'close']:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    # Normalize as in training (fit only on inference file)
    features = df[['open', 'high', 'low', 'close']].astype(float).values
    means = features.mean(axis=0)
    stds = features.std(axis=0) + 1e-8
    features_norm = (features - means) / stds
    return df, features_norm

# --------- INFERENCE ---------
def predict_over_csv(model, features, df, seq_len=SEQ_LEN):
    results = []
    n_classes = model.output_dim if hasattr(model, "output_dim") else None
    for i in range(len(features) - seq_len):
        seq = features[i:i+seq_len]
        x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)  # (1, seq_len, 4)
        with torch.no_grad():
            logits = model(x)
            pred_class = logits.argmax(dim=1).item()
        # Print row number, date if available, and predicted class
        row_num = i + seq_len
        date = df['date'].iloc[row_num] if 'date' in df.columns else row_num
        actual = df['signal_class'].iloc[row_num] if 'signal_class' in df.columns else None
        results.append((row_num, date, pred_class, actual))
    return results

# --------- MAIN ---------
if __name__ == "__main__":
    # 1. Load and preprocess data
    df, features = load_and_preprocess(CSV_PATH)
    n_classes = int(df['signal_class'].nunique()) if 'signal_class' in df.columns else 3  # Default 3 classes

    # 2. Load model
    model = load_model(MODEL_PATH, n_classes)

    # 3. Run predictions
    results = predict_over_csv(model, features, df, seq_len=SEQ_LEN)

    # 4. Print output
    print("row_num\tdate\t\tpredicted\tactual")
    for row_num, date, pred, actual in results:
        print(f"{row_num}\t{date}\t{pred}\t\t{actual}")

