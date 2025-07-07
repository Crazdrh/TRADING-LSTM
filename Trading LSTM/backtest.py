# backtest.py
"""
Backtest the trained model on the test dataset, simulating trading and evaluating performance.
"""
import os
import pandas as pd
import numpy as np
import torch
from train_lstm import LSTMTradingModel, label_to_class  # reuse model definition and label mapping

# Configuration
test_data_dir = "data/test_data"
model_path = "models/lstm_trading_model.pth"
scaler_path = "models/scaler.pkl"

# Load test data (concatenate all test files chronologically)
test_files = [f for f in os.listdir(test_data_dir) if f.lower().endswith(".csv")]
test_files.sort()
test_df_list = [pd.read_csv(os.path.join(test_data_dir, f)) for f in test_files]
if not test_df_list:
    raise FileNotFoundError("No test files found.")
test_df = pd.concat(test_df_list, ignore_index=True)

# Prepare features and labels
feature_cols = ["open", "high", "low", "close", "volume"]
target_col = "signal_class"
if target_col not in test_df.columns:
    test_df[target_col] = test_df["signal"].map(label_to_class)  # map text signal to class if needed
features = test_df[feature_cols].astype(float).values
labels = test_df[target_col].astype(int).values

# Scale features using the saved training scaler
import pickle
with open(scaler_path, "rb") as f:
    scaler_params = pickle.load(f)
mean = scaler_params["mean"]
std = scaler_params["std"]
std[std == 0] = 1.0
features = (features - mean) / std

# Load trained model
input_size = len(feature_cols)
model = LSTMTradingModel(input_size=input_size)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Mapping from class index to label text for interpretation
class_to_label = {v: k for k, v in label_to_class.items()}

# Backtest simulation
initial_balance = 1.0  # starting capital (arbitrary unit)
balance = initial_balance
equity_curve = [balance]
peak = balance
correct_trades = 0
total_trades = 0

seq_length = 50  # must match the sequence length used in training
data_length = len(features)
for idx in range(seq_length, data_length):
    # Prepare input sequence (from idx-seq_length to idx-1)
    seq_x = features[idx-seq_length:idx]
    seq_x_tensor = torch.tensor(seq_x, dtype=torch.float32).unsqueeze(0)  # shape (1, seq_len, features)
    outputs = model(seq_x_tensor)
    _, pred_class_tensor = torch.max(outputs, dim=1)
    pred_class = int(pred_class_tensor.item())
    pred_label = class_to_label.get(pred_class, "")
    # Determine position: long for any 'buy' signal, short for any 'sell' signal
    if pred_label == "hold":
        position = 0
    elif "buy" in pred_label:
        position = 1
    elif "sell" in pred_label:
        position = -1
    else:
        position = 0
    # Calculate profit for this step
    # Use future_return from data if available, otherwise compute from close price
    if "future_return" in test_df.columns:
        step_return = test_df.loc[idx-1, "future_return"]  # future return corresponding to interval (idx-1 -> idx)
    else:
        prev_close = test_df.loc[idx-1, "close"]
        curr_close = test_df.loc[idx, "close"]
        step_return = (curr_close - prev_close) / prev_close if prev_close != 0 else 0.0
    profit_factor = position * step_return
    # Update balance (compound returns)
    balance *= (1 + profit_factor)
    equity_curve.append(balance)
    # Track performance
    if position != 0:
        total_trades += 1
        if profit_factor > 0:
            correct_trades += 1
    # Update peak for drawdown calculation
    if balance > peak:
        peak = balance

final_balance = balance
total_return_pct = (final_balance - initial_balance) / initial_balance * 100
accuracy = (correct_trades / total_trades * 100) if total_trades > 0 else 0.0

print(f"Backtest completed on {len(test_df)} data points.")
print(f"Total trades taken: {total_trades}")
print(f"Directional accuracy: {accuracy:.2f}%")
print(f"Final portfolio value: {final_balance:.4f} (Total Return: {total_return_pct:.2f}%)")

# Calculate maximum drawdown
equity_array = np.array(equity_curve)
roll_max = np.maximum.accumulate(equity_array)
drawdowns = (equity_array - roll_max) / roll_max
max_drawdown_pct = abs(drawdowns.min()) * 100.0
print(f"Max Drawdown: {max_drawdown_pct:.2f}%")
