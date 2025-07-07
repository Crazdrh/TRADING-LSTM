# hyperparameter_tune.py
"""
Hyperparameter tuning for the LSTM model using grid search.
Trains multiple models with different hyperparameters and evaluates on the validation set.
"""
import itertools
import numpy as np
import torch
import torch.nn as nn
from train_lstm import LSTMTradingModel, load_data_from_dir, label_to_class

# Define hyperparameter grid
param_grid = {
    "hidden_size": [32, 64],
    "num_layers": [1, 2],
    "dropout": [0.2, 0.5],
    "learning_rate": [0.001, 0.0005],
    "sequence_length": [30, 50]
}

train_data_dir = "data/train_data"
val_data_dir = "data/val_data"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and prepare data (same process as in train script)
import pandas as pd
train_df, _ = load_data_from_dir(train_data_dir)
val_df, _ = load_data_from_dir(val_data_dir)
feature_cols = ["open", "high", "low", "close", "volume"]
target_col = "signal_class"
if target_col not in train_df.columns:
    train_df[target_col] = train_df["signal"].map(label_to_class)
    val_df[target_col] = val_df["signal"].map(label_to_class)

train_features = train_df[feature_cols].astype(float).values
val_features = val_df[feature_cols].astype(float).values
train_labels = train_df[target_col].astype(int).values
val_labels = val_df[target_col].astype(int).values

# Scale features using train set stats
mean = train_features.mean(axis=0)
std = train_features.std(axis=0)
std[std == 0] = 1.0
train_features = (train_features - mean) / std
val_features = (val_features - mean) / std

# Helper to create sequences given a sequence length and file boundaries
def make_sequences(features, labels, seq_length, file_boundaries):
    seq_X, seq_y = [], []
    for i in range(len(file_boundaries) - 1):
        file_start = file_boundaries[i]
        file_end = file_boundaries[i+1]
        for start in range(file_start, file_end - seq_length):
            end = start + seq_length
            seq_X.append(features[start:end])
            seq_y.append(labels[end-1])
    return np.array(seq_X, dtype=np.float32), np.array(seq_y, dtype=np.int64)

# Determine file boundaries to avoid crossing file edges in sequences
train_file_lengths = []
val_file_lengths = []
import os
for f in os.listdir(train_data_dir):
    if f.lower().endswith(".csv"):
        df_temp = pd.read_csv(os.path.join(train_data_dir, f))
        train_file_lengths.append(len(df_temp))
for f in os.listdir(val_data_dir):
    if f.lower().endswith(".csv"):
        df_temp = pd.read_csv(os.path.join(val_data_dir, f))
        val_file_lengths.append(len(df_temp))
train_boundaries = [0]; cum = 0
for l in train_file_lengths:
    cum += l; train_boundaries.append(cum)
val_boundaries = [0]; cum = 0
for l in val_file_lengths:
    cum += l; val_boundaries.append(cum)

best_params = None
best_val_acc = 0.0

# Iterate through all combinations in the grid
for hidden_size, num_layers, dropout, lr, seq_len in itertools.product(
        param_grid["hidden_size"], param_grid["num_layers"],
        param_grid["dropout"], param_grid["learning_rate"], param_grid["sequence_length"]):
    # Prepare sequence data for this sequence length
    X_train, y_train = make_sequences(train_features, train_labels, seq_len, train_boundaries)
    X_val, y_val = make_sequences(val_features, val_labels, seq_len, val_boundaries)
    if X_train.size == 0 or X_val.size == 0:
        # Skip this combination if sequence length is too large for the data segments
        continue
    # Create PyTorch datasets and loaders
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train_t, y_train_t),
                                               batch_size=64, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_val_t, y_val_t),
                                             batch_size=64, shuffle=False)
    # Initialize model with given hyperparameters
    model = LSTMTradingModel(input_size=len(feature_cols), hidden_size=hidden_size,
                             num_layers=num_layers, num_classes=6, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    # Train for a few epochs (short training for tuning purposes)
    tuning_epochs = 5
    for epoch in range(tuning_epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    # Evaluate on validation set
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            outputs = model(batch_x)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)
    val_acc = correct / total if total > 0 else 0.0
    print(f"Params: hidden_size={hidden_size}, layers={num_layers}, dropout={dropout}, lr={lr}, seq_len={seq_len} -> Val Acc: {val_acc:.4f}")
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_params = {
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "learning_rate": lr,
            "sequence_length": seq_len
        }

if best_params:
    print("Best hyperparameters found:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
else:
    print("No valid hyperparameter combination found.")
