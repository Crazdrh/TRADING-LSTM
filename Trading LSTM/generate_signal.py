# generate_signal.py (continuous)
import torch, pickle, time
import pandas as pd
from train_lstm import LSTMTradingModel, label_to_class, class_to_label

feature_cols = ["open", "high", "low", "close", "volume"]
sequence_length = 50
model_path = "models/lstm_trading_model.pth"
scaler_path = "models/scaler.pkl"
csv_path = "live_data.csv"
refresh_interval = 60  # seconds between new signals

model = LSTMTradingModel(input_size=len(feature_cols))
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()
with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)
mean, std = scaler["mean"], scaler["std"]

last_signal = "hold"
while True:
    df = pd.read_csv(csv_path)
    features = df[feature_cols].astype(float).values
    if len(features) < sequence_length:
        print("Not enough data for a prediction.")
        time.sleep(refresh_interval)
        continue
    features = (features - mean) / std
    seq = torch.tensor(features[-sequence_length:], dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        pred_class = model(seq).argmax(dim=1).item()
    signal = class_to_label[pred_class]
    # Output signal, only act on new non-hold
    if signal != "hold" and signal != last_signal:
        print(f"Signal: {signal} (ACTION)")
        last_signal = signal
    else:
        print("Signal: hold")
        last_signal = "hold"
    time.sleep(refresh_interval)
