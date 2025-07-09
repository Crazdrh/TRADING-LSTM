import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from LSTM import ComplexLSTMModel

# ---------------- CONFIGURATION ----------------
DATA_DIR = "C:/Users/Hayden/Downloads/csv"  # Directory with your CSV files
SEQ_LEN = 50
EPOCHS = 10
LR = 0.001
SHOW_BATCHES = False       # Set to False to suppress batch printing
SHOW_GRADIENTS = False   # Set to True to print parameter gradients

# ---------------- DEVICE SETUP ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------- UTILITIES ----------------
label_map = {-2: 0, -1: 1, 0: 2, 1: 3, 2: 4}
label_map_inv = {0: -2, 1: -1, 2: 0, 3: 1, 4: 2}
label_names = {-2: "strong_sell", -1: "sell", 0: "hold", 1: "buy", 2: "strong_buy"}

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

# ---------------- DATASET ----------------
class PriceDataset(Dataset):
    def __init__(self, df, seq_len=SEQ_LEN):
        col_candidates = ['close', 'Close', 'CLOSE']
        close_col = next((col for col in col_candidates if col in df.columns), None)
        if close_col is None:
            raise ValueError(f"None of the expected close price columns {col_candidates} found in CSV files.")
        prices = df[close_col].values.astype(float)
        prices = (prices - np.mean(prices)) / (np.std(prices) + 1e-8)
        self.X = []
        self.y = []
        for i in range(len(prices) - seq_len - 1):
            seq = prices[i:i+seq_len]
            next_price = prices[i+seq_len]
            last_price = prices[i+seq_len-1]
            change = next_price - last_price
            # Your thresholds here; update to match your labeling!
            if change > 0.01:
                label = 2   # strong_buy
            elif change > 0.002:
                label = 1   # buy
            elif change < -0.01:
                label = -2  # strong_sell
            elif change < -0.002:
                label = -1  # sell
            else:
                label = 0   # hold
            self.X.append(seq.reshape(-1, 1))
            self.y.append(label_map[label])
        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ---------------- BATCH REPORT FUNCTION ----------------
def show_batch_step(batch_idx, batch_x, batch_y, logits, loss, model=None, show_gradients=False):
    pred_classes = torch.argmax(logits, dim=1).cpu().numpy()
    target_classes = batch_y.cpu().numpy()
    pred_labels = [label_map_inv[int(x)] for x in pred_classes]
    target_labels = [label_map_inv[int(x)] for x in target_classes]
    pred_names = [label_names[l] for l in pred_labels]
    target_names = [label_names[l] for l in target_labels]
    print(f"  Batch {batch_idx}:")
    print(f"    Loss: {loss.item():.5f}")
    print(f"    Predictions: {pred_classes} (labels: {pred_names})")
    print(f"    Targets    : {target_classes} (labels: {target_names})")
    # Optionally print gradient norms
    if show_gradients and model is not None:
        for name, param in model.named_parameters():
            if param.grad is not None:
                print(f"    Grad norm for {name}: {param.grad.norm().item():.5f}")

# ---------------- TRAINING ----------------
if __name__ == "__main__":
    # 1. Load and merge all CSVs
    df = load_all_csvs(DATA_DIR)
    print(f"Loaded data: {len(df)} rows from {DATA_DIR}")

    # 2. Prepare dataset and dataloader
    dataset = PriceDataset(df)
    print(f"Total training samples: {len(dataset)}")
    # For full-batch gradient descent:
    BATCH_SIZE = len(dataset)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 3. Setup model, loss, optimizer
    model = ComplexLSTMModel(input_dim=1, output_dim=1).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # 4. Training loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        for batch_idx, (batch_x, batch_y) in enumerate(loader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_x.size(0)
            if SHOW_BATCHES:
                show_batch_step(
                    batch_idx, batch_x, batch_y, logits, loss,
                    model=model if SHOW_GRADIENTS else None,
                    show_gradients=SHOW_GRADIENTS
                )
        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch+1} | Avg Loss: {avg_loss:.5f}")

    # 5. Save model
    torch.save(model.state_dict(), "C:/Users/Hayden/Downloads/trained_lstm_model2.pth")
    print("Model saved to trained_lstm_model2.pth")
