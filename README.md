# TRADING-LSTM

A modular, end-to-end trading pipeline using an LSTM-based neural network for multi-timeframe stock prediction.

Features:

    Chronological data split (by whole CSV files, never inside a file)
    Supervised training with the custom LSTMTradingModel for multi-class buy/sell signals
    Backtest with simulated trading, performance stats, and risk controls
    Hyperparameter tuning with grid search
    Live/paper trading signal generation on new CSV data
    Modular, reusable codebase (each stage is its own script/module)

Project Structure:
   
    
    ├── data/
    │   ├── raw_csv/         # All raw input CSVs (full files, unsplit)
    │   ├── train_data/      # Training set CSVs
    │   ├── val_data/        # Validation set CSVs
    │   └── test_data/       # Test set CSVs
    ├── models/
    │   ├── lstm_trading_model.pth  # Trained LSTM weights
    │   ├── scaler.pkl              # Feature scaler (StandardScaler) params
    │   └── lstm_trading_agent_rl   # (optional) RL agent weights
    ├── scripts/
    │   ├── data_split.py
    │   ├── train_lstm.py
    │   ├── backtest.py
    │   ├── hyperparameter_tune.py
    │   └── generate_signal.py
    ├── latest_signal.txt    # Most recent live/paper trading signal
    ├── README.md
    └── requirements.txt     # Python dependencies

Data Format:

Each CSV file (any frequency: 5min, 10min, etc.) must have columns:

    timestamp,open,high,low,close,volume,future_close,future_return,signal,signal_class
    
signal_class: integer in [0, 5] (strong_sell to strong_buy)
signal: text label (e.g., strong_buy, buy, weak_buy, weak_sell, sell, strong_sell)

Model
LSTMTradingModel is defined in train_lstm.py:

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

  Uses open, high, low, close, volume as features.
  Outputs 6-class signal: strong_sell, sell, weak_sell, weak_buy, buy, strong_buy.

Risk Controls:
    All scripts avoid lookahead bias (no future data leakage).
    Risk management in live/paper tests:
        Skip weak signals (configurable)
        Stop trading if drawdown exceeds 20%
        Optional transaction cost/slippage

Requirements:
    Python 3.8+
    numpy, pandas, torch, scikit-learn
    stable-baselines3 (optional, for RL)
    gym (for RL env)
    


Notes & Troubleshooting:
    
    Always match sequence_length in all scripts!
    Make sure your CSV data is chronologically named for proper splitting.
    For production: add logging, monitoring, and extra input validation as needed.
    Use at your own risk. Trading is inherently risky and this repo is for research/education.
