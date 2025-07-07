# live_test_risk.py
"""
Paper/Live test simulation with risk controls.
Simulates trading on the test dataset with additional risk management rules.
"""
import os
import pandas as pd
import numpy as np
import torch
from train_lstm import LSTMTradingModel, label_to_class

# Configuration
test_data_dir = "data/test_data"
model_path = "models/lstm_trading_model.pth"
scaler_path = "models/scaler.pkl"
max_drawdown_limit = 0.20   # 20% max drawdown allowed
use_weak_signals = False    # whether to execute trades on weak_buy/weak_sell signals or skip them
transaction_cost = 0.001    # 0.1% transaction cost per trade (commission+slippage)

# Load test data
test_files = [f for f in os.listdir(test_data_dir) if f.lower().endswith(".csv")]
test_files.sort()
test_df = pd.concat([pd.read_csv(os.path.join(test_data_dir, f)) for f in test_files], ignore_index=True)
feature_cols = ["open", "high", "low", "close", "volume"]
target_col = "signal_class"
if target_col not in test_df.columns:
    test_df[target_col] = test_df["signal"].map(label_to_class)
features = test_df[feature_cols].astype(float).values

# Scale features using training scaler
import pickle
with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)
mean = scaler["mean"]; std = scaler["std"]; std[std == 0] = 1.0
features = (features - mean) / std

# Load trained model
model = LSTMTradingModel(input_size=len(feature_cols))
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

class_to_label = {v: k for k, v in label_to_class.items()}
initial_balance = 1.0
balance = initial_balance
equity_curve = [balance]
peak_balance = balance
drawdown = 0.0
sequence_length = 50
total_trades = 0
wins = 0

for idx in range(sequence_length, len(features)):
    # Check drawdown and enforce stop trading if exceeded
    current_drawdown = (balance - peak_balance) / peak_balance
    if current_drawdown <= -max_drawdown_limit:
        print(f"Max drawdown limit reached at index {idx}. Stopping trading.")
        break
    # Prepare input sequence for prediction
    seq_x = features[idx-sequence_length:idx]
    seq_tensor = torch.tensor(seq_x, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        output = model(seq_tensor)
        _, pred_class_tensor = torch.max(output, 1)
        pred_class = int(pred_class_tensor.item())
    pred_label = class_to_label.get(pred_class, "")
    # Determine position, skipping weak signals if configured
    if not use_weak_signals and "weak" in pred_label:
        position = 0  # skip trade on weak_buy/weak_sell
    else:
        if "buy" in pred_label:
            position = 1
        elif "sell" in pred_label:
            position = -1
        else:
            position = 0
    # Calculate step return (from idx-1 to idx)
    if idx < len(test_df):
        if "future_return" in test_df.columns:
            step_return = test_df.loc[idx-1, "future_return"]
        else:
            prev_close = test_df.loc[idx-1, "close"]
            curr_close = test_df.loc[idx, "close"]
            step_return = (curr_close - prev_close) / prev_close if prev_close != 0 else 0.0
    else:
        step_return = 0.0
    # Apply transaction cost if a trade is executed
    if position != 0:
        step_return -= transaction_cost
    # Update portfolio balance
    balance *= (1 + position * step_return)
    equity_curve.append(balance)
    # Update trade stats
    if position != 0:
        total_trades += 1
        if position * step_return > 0:
            wins += 1
    # Update peak and drawdown
    if balance > peak_balance:
        peak_balance = balance
    drawdown = min(drawdown, (balance - peak_balance) / peak_balance)

# Final performance metrics
if total_trades > 0:
    win_rate = wins / total_trades * 100.0
else:
    win_rate = 0.0
final_return_pct = (balance - initial_balance) / initial_balance * 100.0
max_drawdown_pct = abs(drawdown) * 100.0

print(f"Trades executed: {total_trades}, Win rate: {win_rate:.2f}%")
print(f"Final portfolio value: {balance:.4f} (Total Return: {final_return_pct:.2f}%)")
print(f"Max drawdown: {max_drawdown_pct:.2f}%")
