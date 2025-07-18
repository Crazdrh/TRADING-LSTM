import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """Add positional encoding to help the model understand sequence positions"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for better pattern recognition"""
    def __init__(self, d_model, num_heads=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.size()
        
        # Multi-head projections
        q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        return self.w_o(context), attn

class BiLSTMBlock(nn.Module):
    """Bidirectional LSTM block with residual connections and attention"""
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size, hidden_size, 
            num_layers=num_layers, 
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False, 
            batch_first=True
        )
        
        # Project bidirectional output back to hidden_size
        self.output_proj = nn.Linear(hidden_size * 2, hidden_size)
        
        # Layer normalization and dropout
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection projection if needed
        self.residual_proj = nn.Linear(input_size, hidden_size) if input_size != hidden_size else None
        
    def forward(self, x):
        # Store input for residual connection
        residual = x
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Project bidirectional output
        out = self.output_proj(lstm_out)
        
        # Residual connection
        if self.residual_proj is not None:
            residual = self.residual_proj(residual)
            
        out = out + residual
        out = self.layer_norm(out)
        out = self.dropout(out)
        
        return out

class TemporalConvBlock(nn.Module):
    """Temporal convolution block for local pattern recognition"""
    def __init__(self, in_channels, out_channels, kernel_sizes=[3, 5, 7], dropout=0.2):
        super().__init__()
        self.convs = nn.ModuleList()
        
        # Calculate output channels per conv to ensure exact division
        channels_per_conv = out_channels // len(kernel_sizes)
        remaining_channels = out_channels % len(kernel_sizes)
        
        for i, kernel_size in enumerate(kernel_sizes):
            # Add remaining channels to the last conv layer
            current_out_channels = channels_per_conv + (remaining_channels if i == len(kernel_sizes) - 1 else 0)
            
            conv = nn.Sequential(
                nn.Conv1d(in_channels, current_out_channels, 
                         kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(current_out_channels),
                nn.GELU()
                nn.Dropout(dropout)
            )
            self.convs.append(conv)
        
        # Fixed: Use out_channels directly instead of in_channels for residual
        self.output_proj = nn.Linear(out_channels, out_channels)
        self.layer_norm = nn.LayerNorm(out_channels)
        
        # Add residual projection if input and output channels don't match
        self.residual_proj = nn.Linear(in_channels, out_channels) if in_channels != out_channels else None
        
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        residual = x
        
        x_conv = x.transpose(1, 2)  # (batch, features, seq_len)
        
        conv_outputs = []
        for conv in self.convs:
            conv_out = conv(x_conv)
            conv_outputs.append(conv_out)
        
        # Concatenate multi-scale features
        combined = torch.cat(conv_outputs, dim=1)
        combined = combined.transpose(1, 2)  # Back to (batch, seq_len, features)
        
        # Project and normalize
        out = self.output_proj(combined)
        
        # Apply residual connection with projection if needed
        if self.residual_proj is not None:
            residual = self.residual_proj(residual)
        
        out = self.layer_norm(out + residual)
        
        return out

class ComplexLSTMModel(nn.Module):
    """
    Extremely complex LSTM model with multiple advanced components:
    - Multi-scale temporal convolutions
    - Bidirectional LSTM layers with residual connections
    - Multi-head attention mechanisms
    - Positional encoding
    - Feature fusion and gating mechanisms
    - Ensemble-like output heads
    """
    def __init__(self, input_dim=9, hidden_dim=256, num_lstm_layers=4, 
                 num_heads=8, output_dim=3, dropout=0.3):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_lstm_layers = num_lstm_layers
        
        # Input projection and embedding
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
            nn.Dropout(dropout * 0.5)
        )
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim)
        
        # Multi-scale temporal convolutions - Fixed dimensions
        self.temp_conv1 = TemporalConvBlock(hidden_dim, hidden_dim, [3, 5, 7], dropout)
        self.temp_conv2 = TemporalConvBlock(hidden_dim, hidden_dim, [3, 5, 9], dropout)
        
        # Bidirectional LSTM layers with progressively increasing complexity
        self.lstm_blocks = nn.ModuleList()
        for i in range(num_lstm_layers):
            lstm_block = BiLSTMBlock(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=2,
                dropout=dropout
            )
            self.lstm_blocks.append(lstm_block)
        
        # Multi-head attention layers
        self.attention_layers = nn.ModuleList([
            MultiHeadAttention(hidden_dim, num_heads, dropout)
            for _ in range(2)
        ])
        
        # Feature fusion gate
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        # Global and local feature extractors
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.local_pool = nn.AdaptiveMaxPool1d(1)
        
        # Multiple output heads for ensemble-like behavior
        self.output_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(hidden_dim // 2, output_dim)
            )
            for _ in range(3)
        ])
        
        # Final ensemble combiner
        self.ensemble_combiner = nn.Sequential(
            nn.Linear(output_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Gradient scaling for stable training
        self.gradient_scale = nn.Parameter(torch.ones(1))
        
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        # Input projection and positional encoding
        h = self.input_proj(x)
        h = self.pos_encoding(h)
        
        # Multi-scale temporal convolutions
        h_conv1 = self.temp_conv1(h)
        h_conv2 = self.temp_conv2(h_conv1)
        
        # Fusion gate for combining conv features
        gate = self.fusion_gate(torch.cat([h_conv1, h_conv2], dim=-1))
        h = gate * h_conv1 + (1 - gate) * h_conv2
        
        # Progressive LSTM processing
        lstm_outputs = []
        for i, lstm_block in enumerate(self.lstm_blocks):
            h = lstm_block(h)
            lstm_outputs.append(h)
            
            # Apply attention every other layer
            if i % 2 == 1 and i // 2 < len(self.attention_layers):
                h_attn, _ = self.attention_layers[i // 2](h)
                h = h + h_attn  # Residual connection
        
        # Multi-scale feature extraction
        # Global features (average over time)
        global_features = self.global_pool(h.transpose(1, 2)).squeeze(-1)
        
        # Local features (max over time)
        local_features = self.local_pool(h.transpose(1, 2)).squeeze(-1)
        
        # Last timestep features
        last_features = h[:, -1, :]
        
        # Combine all features
        combined_features = torch.cat([global_features, local_features, last_features], dim=-1)
        
        # Multiple output heads for ensemble
        head_outputs = []
        for head in self.output_heads:
            head_output = head(combined_features)
            head_outputs.append(head_output)
        
        # Ensemble combination
        ensemble_input = torch.cat(head_outputs, dim=-1)
        final_output = self.ensemble_combiner(ensemble_input)
        
        # Gradient scaling for training stability
        final_output = final_output * self.gradient_scale
        
        return final_output
    
    def get_attention_weights(self, x):
        """Get attention weights for visualization"""
        batch_size, seq_len, _ = x.size()
        
        # Forward pass up to attention layers
        h = self.input_proj(x)
        h = self.pos_encoding(h)
        h = self.temp_conv1(h)
        h = self.temp_conv2(h)
        
        attention_weights = []
        for i, lstm_block in enumerate(self.lstm_blocks):
            h = lstm_block(h)
            if i % 2 == 1 and i // 2 < len(self.attention_layers):
                _, attn_weights = self.attention_layers[i // 2](h)
                attention_weights.append(attn_weights)
        
        return attention_weights
    
    def get_feature_importance(self, x):
        """Get feature importance scores"""
        with torch.no_grad():
            # Get baseline output
            baseline_output = self.forward(x)
            
            importance_scores = []
            for feature_idx in range(x.size(-1)):
                # Zero out one feature at a time
                x_masked = x.clone()
                x_masked[:, :, feature_idx] = 0
                
                # Get output with masked feature
                masked_output = self.forward(x_masked)
                
                # Calculate importance as difference in output
                importance = torch.abs(baseline_output - masked_output).mean()
                importance_scores.append(importance.item())
            
            return importance_scores

# Example usage and testing
if __name__ == "__main__":
    # Create model
    model = ComplexLSTMModel(
        input_dim=9,
        hidden_dim=256,
        num_lstm_layers=4,
        num_heads=8,
        output_dim=3,
        dropout=0.3
    )
    
    # Test with sample data
    batch_size, seq_len, input_dim = 8, 50, 9
    x = torch.randn(batch_size, seq_len, input_dim)
    
    # Forward pass
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test attention weights
    attention_weights = model.get_attention_weights(x)
    print(f"Number of attention layers: {len(attention_weights)}")
    
    # Test feature importance
    importance = model.get_feature_importance(x)
    print(f"Feature importance scores: {importance}")
    
    # Test gradient flow
    loss = F.cross_entropy(output, torch.randint(0, 3, (batch_size,)))
    loss.backward()
    
    # Check for gradient flow
    has_gradients = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    print(f"Gradients flowing: {has_gradients}")