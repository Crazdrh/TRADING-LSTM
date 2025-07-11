import torch
import torch.nn as nn

class ComplexLSTMModel(nn.Module):
    def __init__(self, input_dim=9, hidden_dim=128, num_layers=5, output_dim=3, dropout=0.3):
        super(ComplexLSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.input_proj = nn.Linear(input_dim, hidden_dim)  # Add this line!

        self.lstm_layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = hidden_dim  # Always use hidden_dim for all but first layer
            self.lstm_layers.append(
                nn.LSTM(
                    input_size=in_dim,
                    hidden_size=hidden_dim,
                    num_layers=1,
                    batch_first=True
                )
            )

        self.ln = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers - 1)])
        self.dropout = nn.Dropout(dropout)

        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h = self.input_proj(x)  # Project input to hidden_dim shape!

        for i, lstm in enumerate(self.lstm_layers):
            h, _ = lstm(h)
            if i < self.num_layers - 1:
                # Residual + norm for complexity
                h = self.ln[i](h + h_prev if i > 0 else h)
                h = self.dropout(h)
            h_prev = h
        out = h[:, -1, :]
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

# Example usage
if __name__ == "__main__":
    model = ComplexLSTMModel()
    x = torch.randn(8, 50, 9)  # batch, seq, feature
    out = model(x)
    print(out.shape)  # (8, output_dim)


