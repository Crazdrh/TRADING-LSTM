import os
import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from LSTM import ComplexLSTMModel

# --------------- CONFIG ----------------
TEST_DATA_DIR = "C:/Users/Hayden/Downloads/csv"  # change this to your test CSV dir
SEQ_LEN = 50
BATCH_SIZE = 32
MODEL_PATH = "C:/Users/Hayden/Downloads/trained_lstm_model2.pth"

# --------------- DEVICE ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----------- LOAD CSVS -----------
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

# ----------- DATASET -----------
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
            if change > 0.02:
                label = 2   # strong_buy
            elif change > 0.005:
                label = 1   # buy
            elif change < -0.02:
                label = -2  # strong_sell
            elif change < -0.005:
                label = -1  # sell
            else:
                label = 0   # hold
            self.X.append(seq.reshape(-1, 1))
            self.y.append(label)
        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.y = np.array(self.y)  # keep as numpy for easy display
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ----------- LABEL MAPS -----------
idx_to_label = {0: -2, 1: -1, 2: 0, 3: 1, 4: 2}
label_name = {-2: "strong_sell", -1: "sell", 0: "hold", 1: "buy", 2: "strong_buy"}

# ------------- MAIN --------------
if __name__ == "__main__":
    # 1. Load model
    model = ComplexLSTMModel(input_dim=1, output_dim=5)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    # 2. Load test data
    df = load_all_csvs(TEST_DATA_DIR)
    print(f"Loaded {len(df)} rows for testing.")
    dataset = PriceDataset(df)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 3. Inference and print results
    print(f"{'Index':<8} {'Predicted':<12} {'Actual':<12}")
    print("-"*35)
    idx_base = 0
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            for i in range(len(preds)):
                pred_idx = preds[i]
                pred_label = idx_to_label[pred_idx]
                actual_label = batch_y[i]
                print(f"{idx_base + i:<8} {label_name[pred_label]:<12} {label_name[int(actual_label)]:<12}")
            idx_base += len(preds)
