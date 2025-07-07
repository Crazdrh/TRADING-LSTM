# generate_signal.py
"""
Generate a trading signal from a single input CSV file using the trained model.
Usage: python generate_signal.py <input_csv_file>
"""
import sys
import pandas as pd
import numpy as np
import torch
from train_lstm import LSTMTradingModel, label_to_class

if len(sys.argv) < 2:
    print("Usage: python generate_signal.py <input_csv_file>")
    sys.exit(1)

input_csv = sys.argv[1]
output_signal_file = "latest_signal.txt"  # preset output file for the signal

# Load the input CSV data
df = pd.read_csv(input_csv)
if df.empty:
    print("Input file is empty or not found.")
    sys.exit(1)

# Ensure required feature columns are present
feature_cols = ["open", "high", "low", "close", "volume"]
for col in feature_cols:
    if col not in df.columns:
        raise ValueError(f"Required feature column '{col}' not found in input data.")

# Use the last `sequence_length` data points for prediction
sequence_length = 50  # must match sequence length used in training
if len(df) < sequence_length:
    seq_df = df.copy()  # if not enough data, use all available (model can handle shorter sequence)
else:
    seq_df = df.iloc[-sequence_length:].copy()

# Drop any columns that won't be used as input features
for col in ["signal", "signal_class", "future_return", "future_close"]:
    if col in seq_df.columns:
        seq_df = seq_df.drop(columns=[col])

# Scale features using the saved scaler from training
import pickle
try:
    scaler_params = pickle.load(open("models/scaler.pkl", "rb"))
    mean = scaler_params["mean"]; std = scaler_params["std"]; std[std == 0] = 1.0
    seq_df[feature_cols] = (seq_df[feature_cols].astype(float) - mean) / std
except FileNotFoundError:
    print("Warning: Scaler file not found. Proceeding without feature scaling.")

# Prepare input tensor for the model
features = seq_df[feature_cols].astype(float).values
seq_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # shape (1, seq_length, feature_dim)

# Load trained model weights
model = LSTMTradingModel(input_size=len(feature_cols))
try:
    model.load_state_dict(torch.load("models/lstm_trading_model.pth", map_location=torch.device('cpu')))
except FileNotFoundError:
    print("Model weights not found. Please train the model before running the signal generation.")
    sys.exit(1)
model.eval()

# Perform prediction
with torch.no_grad():
    output = model(seq_tensor)
    _, pred_class_tensor = torch.max(output, dim=1)
    pred_class = int(pred_class_tensor.item())

# Map predicted class to signal label
class_to_label = {v: k for k, v in label_to_class.items()}
signal_label = class_to_label.get(pred_class, "unknown")
print(signal_label)  # print the predicted signal
# Save the signal to a text file
with open(output_signal_file, "w") as f:
    f.write(signal_label + "\n")
print(f"Signal '{signal_label}' has been saved to {output_signal_file}")
