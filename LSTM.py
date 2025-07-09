import torch
import torch.nn as nn

class ComplexLSTMModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=5, output_dim=1, dropout=0.3):
        super(ComplexLSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Build LSTM layers manually so input_dim is right for each
        self.lstm_layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            self.lstm_layers.append(
                nn.LSTM(
                    input_size=in_dim,
                    hidden_size=hidden_dim,
                    num_layers=1,  # single-layer for each stack
                    batch_first=True
                )
            )
        
        self.ln = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers - 1)])
        self.dropout = nn.Dropout(dropout)
        
        # Complex output head
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h = x
        for i, lstm in enumerate(self.lstm_layers):
            h, _ = lstm(h)
            if i < self.num_layers - 1:
                # Residual + norm for complexity
                h = self.ln[i](h + (x if i == 0 else h_prev))
                h = self.dropout(h)
            h_prev = h
        # Take output of last timestep
        out = h[:, -1, :]
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

# Example usage
if __name__ == "__main__":
    model = ComplexLSTMModel()
    x = torch.randn(8, 50, 1)  # batch, seq, feature
    out = model(x)
    print(out.shape)  # (8, 1)

