import torch
import pandas as pd
import numpy as np
from lstm_trading_model import LSTMTradingModel
from env_stock_trading import StockTradingEnv, SIGNALS

# Path to your trained LSTM model weights
LSTM_MODEL_PATH = "./trained_lstm.pth"
DATA_PATH = "./data/AAPL.csv"

# Hyperparameters: update to match your training
input_size = 6  # matches the obs window, you may need to flatten or adapt this
hidden_size = 64
num_layers = 2
num_classes = 7  # for 7 signals

# Load your data
df = pd.read_csv(DATA_PATH)
df = df.sort_values('Date')

# Init environment
env = StockTradingEnv(df)
obs = env.reset()

# Load your model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMTradingModel(input_size, hidden_size, num_layers, num_classes)
model.load_state_dict(torch.load(LSTM_MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

num_episodes = 1
for episode in range(num_episodes):
    obs = env.reset()
    done = False
    step = 0
    while not done and step < 2000:
        # Prepare obs for LSTM [batch, seq, feature]
        # If obs shape is (6,6), treat as one sequence of length 6 with 6 features
        # (may need to transpose depending on your training)
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
        obs_tensor = obs_tensor.unsqueeze(0)  # batch=1
        # [batch, seq, features] => [1, 6, 6]
        action_logits = model(obs_tensor)
        action_idx = torch.argmax(action_logits, dim=-1).item()

        next_obs, reward, done, _ = env.step(action_idx)
        env.render()
        obs = next_obs
        step += 1
