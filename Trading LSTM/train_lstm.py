# train_lstm.py
"""
Train an LSTM model on the training dataset (supervised classification) and evaluate on validation set.
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Configuration
train_data_dir = "data/train_data"
val_data_dir = "data/val_data"
model_save_path = "models/lstm_trading_model.pth"
scaler_save_path = "models/scaler.pkl"
sequence_length = 50   # length of historical sequence for LSTM input
batch_size = 64
num_epochs = 20
learning_rate = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the LSTM model class (from earlier definition)
class LSTMTradingModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, num_classes=6, dropout=0.2):
        super(LSTMTradingModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                             num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]  # output of last timestep
        out = self.fc(out)
        return out

# Label mapping for signals to classes
label_to_class = {
    "strong_sell": 0,
    "sell": 1,
    "weak_sell": 2,
     "hold": 3,
    "weak_buy": 4,
    "buy": 5,
    "strong_buy": 6
}

def load_data_from_dir(data_dir):
    """Load all CSV files from a directory and return a concatenated DataFrame."""
    files = [f for f in os.listdir(data_dir) if f.lower().endswith(".csv")]
    files.sort()
    df_list = []
    for f in files:
        file_path = os.path.join(data_dir, f)
        df = pd.read_csv(file_path)
        df_list.append(df)
    if not df_list:
        raise ValueError(f"No CSV files found in {data_dir}")
    data = pd.concat(df_list, ignore_index=True)
    return data, len(files)

# Load training and validation data
train_df, num_train_files = load_data_from_dir(train_data_dir)
val_df, num_val_files = load_data_from_dir(val_data_dir)

# Identify feature columns and target column
feature_cols = ["open", "high", "low", "close", "volume"]  # adjust if more features are available
target_col = "signal_class"
if target_col not in train_df.columns:
    # Derive numeric class from signal text if needed
    if "signal" in train_df.columns:
        train_df[target_col] = train_df["signal"].map(label_to_class)
        val_df[target_col] = val_df["signal"].map(label_to_class)
    else:
        raise ValueError("No 'signal_class' or 'signal' column found in training data.")

# Prepare feature matrices and label arrays
train_features = train_df[feature_cols].astype(float).values
val_features = val_df[feature_cols].astype(float).values
train_labels = train_df[target_col].astype(int).values
val_labels = val_df[target_col].astype(int).values

# Feature scaling: compute mean and std from training data and apply to train and val
feature_mean = train_features.mean(axis=0)
feature_std = train_features.std(axis=0)
feature_std[feature_std == 0] = 1.0  # avoid division by zero
train_features = (train_features - feature_mean) / feature_std
val_features = (val_features - feature_mean) / feature_std

# Save scaler parameters for future use (e.g., in testing)
import pickle
os.makedirs(os.path.dirname(scaler_save_path), exist_ok=True)
with open(scaler_save_path, "wb") as f:
    pickle.dump({"mean": feature_mean, "std": feature_std}, f)

# Create sequences of length `sequence_length` from the training and validation data.
# We ensure sequences do not cross file boundaries.
def create_sequences(features, labels, file_boundaries, seq_length):
    """
    Generate (sequence, label) pairs given feature array, label array, and file boundary indices.
    `file_boundaries` is a list of cumulative start indices for each file in the concatenated data.
    """
    seq_list = []
    for i in range(len(file_boundaries) - 1):
        file_start = file_boundaries[i]
        file_end = file_boundaries[i+1]
        # Generate sequences within this file
        for start in range(file_start, file_end - seq_length):
            end = start + seq_length
            seq_x = features[start:end]
            seq_y = labels[end-1]  # label at the last time step of the sequence
            seq_list.append((seq_x, seq_y))
    return seq_list

# Determine file boundary indices for train and val sets
train_file_lengths = []
val_file_lengths = []
for f in os.listdir(train_data_dir):
    if f.lower().endswith(".csv"):
        df_temp = pd.read_csv(os.path.join(train_data_dir, f))
        train_file_lengths.append(len(df_temp))
for f in os.listdir(val_data_dir):
    if f.lower().endswith(".csv"):
        df_temp = pd.read_csv(os.path.join(val_data_dir, f))
        val_file_lengths.append(len(df_temp))
train_boundaries = [0]
val_boundaries = [0]
cum = 0
for l in train_file_lengths:
    cum += l; train_boundaries.append(cum)
cum = 0
for l in val_file_lengths:
    cum += l; val_boundaries.append(cum)

# Generate sequence datasets
train_sequences = create_sequences(train_features, train_labels, train_boundaries, sequence_length)
val_sequences = create_sequences(val_features, val_labels, val_boundaries, sequence_length)

# Convert sequences to PyTorch tensors
X_train = torch.tensor([seq for (seq, label) in train_sequences], dtype=torch.float32)
y_train = torch.tensor([label for (seq, label) in train_sequences], dtype=torch.long)
X_val = torch.tensor([seq for (seq, label) in val_sequences], dtype=torch.float32)
y_val = torch.tensor([label for (seq, label) in val_sequences], dtype=torch.long)

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Initialize model, loss function, and optimizer
input_size = X_train.shape[2]  # number of features per time step
model = LSTMTradingModel(input_size=input_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

best_val_acc = 0.0
best_model_state = None
for epoch in range(1, num_epochs+1):
    # Training loop
    model.train()
    total_loss = 0.0
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch_x.size(0)
    avg_loss = total_loss / len(train_dataset)
    # Validation loop
    model.eval()
    correct = 0
    total = 0
    val_loss = 0.0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item() * batch_x.size(0)
            _, predicted = torch.max(outputs, dim=1)
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)
    avg_val_loss = val_loss / len(val_dataset)
    val_acc = correct / total if total > 0 else 0.0
    print(f"Epoch {epoch}/{num_epochs} - Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")
    # Save best model based on validation accuracy
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_state = model.state_dict()

# Save the best model weights
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
if best_model_state is not None:
    torch.save(best_model_state, model_save_path)
    print(f"Best model saved to {model_save_path} (Val Acc = {best_val_acc:.4f})")
else:
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
