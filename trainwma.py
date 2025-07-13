import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from LSTM import ComplexLSTMModel

# --- CONFIG ---
DATA_DIR = "/home/hayden/LSTM/Data/CSV"  # Update path
SEQ_LEN = 50
BATCH_SIZE = 1000
EPOCHS = 1
LR = 0.001
MODEL_SAVE_PATH = "/home/hayden/LSTM/Ckpt/trained_lstm_model_with_ma.pth"

# --- DEVICE ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device} | Precision: FP32")

# --- LOAD ALL CSVs ---
def load_all_csvs(data_dir):
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    df_list = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            df_list.append(df)
        except Exception as e:
            print(f"Error loading {f}: {e}")
    if not df_list:
        raise ValueError("No CSV files found in directory or all failed to load.")
    combined = pd.concat(df_list, ignore_index=True)
    return combined

# --- DATASET CLASS ---
class PriceDataset(Dataset):
    def __init__(self, df, seq_len=SEQ_LEN):
        feature_cols = ['open', 'high', 'low', 'close', 'MA', 'MA.1', 'MA.2', 'MA.3', 'MA.4']
        df.columns = [c.strip() for c in df.columns]
        for col in feature_cols + ['signal_class']:
            if col not in df.columns:
                raise ValueError(f"Missing column: {col}")

        features = df[feature_cols].values.astype(np.float32)
        means = features.mean(axis=0)
        stds = features.std(axis=0) + 1e-8
        features = (features - means) / stds

        self.X = []
        self.y = []
        for i in range(len(features) - seq_len):
            seq = features[i:i+seq_len]
            label = df['signal_class'].iloc[i+seq_len]
            self.X.append(seq)
            self.y.append(int(label))
        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y), dtype=torch.long)

    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# --- MAIN TRAINING ---
if __name__ == "__main__":
    df = load_all_csvs(DATA_DIR)
    print(f"Loaded data: {len(df)} rows from {DATA_DIR}")

    dataset = PriceDataset(df)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    n_classes = int(df['signal_class'].dropna().nunique())
    model = ComplexLSTMModel(input_dim=9, output_dim=n_classes).to(device)
    model = model.float()  # Ensure FP32 model
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, dtype=torch.float32)  # Always float32
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_x.size(0)
        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.5f}")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")
