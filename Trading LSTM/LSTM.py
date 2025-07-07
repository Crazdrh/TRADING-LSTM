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
