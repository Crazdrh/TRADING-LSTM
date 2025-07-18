import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Dict, List, Tuple, Optional

class FeatureNormalizer(nn.Module):
    """Adaptive feature normalization layer for financial data"""
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        
        # Learnable scale and shift parameters
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        
        # Running statistics for normalization
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.momentum = 0.1
        
    def forward(self, x):
        if self.training:
            # Calculate batch statistics
            batch_mean = x.mean(dim=(0, 1), keepdim=True)
            batch_var = x.var(dim=(0, 1), keepdim=True, unbiased=False)
            
            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean.squeeze()
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var.squeeze()
            
            # Normalize
            x_norm = (x - batch_mean) / torch.sqrt(batch_var + self.eps)
        else:
            # Use running statistics
            x_norm = (x - self.running_mean.view(1, 1, -1)) / torch.sqrt(self.running_var.view(1, 1, -1) + self.eps)
        
        # Apply learnable parameters
        return self.gamma.view(1, 1, -1) * x_norm + self.beta.view(1, 1, -1)

class FeatureProjector(nn.Module):
    """Projects different groups of features to appropriate dimensions"""
    def __init__(self, feature_groups: Dict[str, int], hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.projectors = nn.ModuleDict()
        self.feature_groups = feature_groups
        
        for group_name, input_dim in feature_groups.items():
            if input_dim > 0:
                self.projectors[group_name] = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim // 4),
                    nn.LayerNorm(hidden_dim // 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim // 4, hidden_dim // 4)
                )
        
        # Final projection to combine all groups
        total_dim = len(self.projectors) * (hidden_dim // 4)
        self.final_proj = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        projected_features = []
        
        for group_name, projector in self.projectors.items():
            if group_name in features:
                projected = projector(features[group_name])
                projected_features.append(projected)
        
        # Concatenate all projected features
        combined = torch.cat(projected_features, dim=-1)
        return self.final_proj(combined)

class MultiScaleConvBlock(nn.Module):
    """Multi-scale temporal convolution for different time patterns"""
    def __init__(self, in_channels: int, out_channels: int, kernel_sizes: List[int] = [3, 5, 7, 11], dropout: float = 0.2):
        super().__init__()
        
        self.convs = nn.ModuleList()
        channels_per_conv = out_channels // len(kernel_sizes)
        
        for i, kernel_size in enumerate(kernel_sizes):
            # Dilated convolutions for larger receptive fields
            dilation = 1 if kernel_size <= 5 else 2
            padding = (kernel_size - 1) * dilation // 2
            
            conv = nn.Sequential(
                nn.Conv1d(in_channels, channels_per_conv, kernel_size, 
                         padding=padding, dilation=dilation),
                nn.BatchNorm1d(channels_per_conv),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            self.convs.append(conv)
        
        # Attention-based feature fusion
        self.fusion_attention = nn.Sequential(
            nn.Linear(out_channels, out_channels // 4),
            nn.GELU(),
            nn.Linear(out_channels // 4, len(kernel_sizes)),
            nn.Softmax(dim=-1)
        )
        
        self.output_norm = nn.LayerNorm(out_channels)
        
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        x_conv = x.transpose(1, 2)  # (batch, features, seq_len)
        
        conv_outputs = []
        for conv in self.convs:
            conv_out = conv(x_conv)
            conv_outputs.append(conv_out)
        
        # Stack conv outputs
        stacked = torch.stack(conv_outputs, dim=-1)  # (batch, channels, seq_len, num_convs)
        combined = stacked.view(x.size(0), -1, x.size(1), len(self.convs))  # Reshape
        combined = combined.transpose(1, 2)  # (batch, seq_len, channels, num_convs)
        
        # Attention-based fusion
        attention_input = combined.mean(dim=-1)  # Average across conv outputs
        attention_weights = self.fusion_attention(attention_input)  # (batch, seq_len, num_convs)
        attention_weights = attention_weights.unsqueeze(2)  # (batch, seq_len, 1, num_convs)
        
        # Apply attention weights
        fused = (combined * attention_weights).sum(dim=-1)  # (batch, seq_len, channels)
        
        return self.output_norm(fused)

class UnidirectionalLSTMBlock(nn.Module):
    """Unidirectional LSTM for real-time trading (no future information)"""
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size, hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False,  # Changed from bidirectional
            batch_first=True
        )
        
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection
        self.residual_proj = nn.Linear(input_size, hidden_size) if input_size != hidden_size else None
        
    def forward(self, x):
        residual = x
        
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Residual connection
        if self.residual_proj is not None:
            residual = self.residual_proj(residual)
        
        out = lstm_out + residual
        out = self.layer_norm(out)
        out = self.dropout(out)
        
        return out, (h_n, c_n)

class TemporalAttention(nn.Module):
    """Temporal attention mechanism with position-aware scoring"""
    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        # Learnable position embeddings
        self.max_positions = 100  # Max sequence length
        self.position_embeddings = nn.Parameter(torch.randn(1, self.max_positions, hidden_dim))
        
        # Temporal decay factor (recent timesteps more important)
        self.temporal_decay = nn.Parameter(torch.tensor(0.95))
        
    def forward(self, x, mask: Optional[torch.Tensor] = None):
        batch_size, seq_len, _ = x.size()
        
        # Add position embeddings
        positions = self.position_embeddings[:, :seq_len, :]
        x_with_pos = x + positions
        
        # Project to Q, K, V
        q = self.q_proj(x_with_pos).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_with_pos).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply temporal decay (more recent timesteps get higher weight)
        decay_factors = self.temporal_decay ** torch.arange(seq_len - 1, -1, -1, device=x.device)
        decay_factors = decay_factors.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        scores = scores * decay_factors
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        
        output = self.o_proj(context)
        
        return output, attn_weights

class OptimizedTradingLSTM(nn.Module):
    """
    Optimized LSTM for trading with 70+ indicators
    Key improvements:
    - Unidirectional LSTM (no future information leakage)
    - GELU activation throughout
    - Feature grouping and specialized processing
    - Temporal decay in attention
    - Robust normalization for financial data
    """
    def __init__(
        self,
        feature_config: Dict[str, int] = None,
        hidden_dim: int = 256,
        num_lstm_layers: int = 3,
        num_heads: int = 8,
        output_dim: int = 3,
        dropout: float = 0.3,
        sequence_length: int = 50
    ):
        super().__init__()
        
        # Default feature configuration for 70+ indicators
        if feature_config is None:
            feature_config = {
                'price': 5,      # OHLCV
                'volume': 7,     # Volume indicators
                'volatility': 11, # Volatility indicators
                'momentum': 14,   # Momentum indicators
                'ma': 5,         # Moving averages
                'structure': 10,  # Market structure
                'time': 10,      # Time features
                'engineered': 12  # Engineered features
            }
        
        self.feature_config = feature_config
        self.total_features = sum(feature_config.values())
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        
        # Feature normalization
        self.feature_normalizer = FeatureNormalizer(self.total_features)
        
        # Feature projection for different groups
        self.feature_projector = FeatureProjector(feature_config, hidden_dim, dropout * 0.5)
        
        # Multi-scale temporal convolutions
        self.conv_block1 = MultiScaleConvBlock(hidden_dim, hidden_dim, [3, 5, 7, 11], dropout)
        self.conv_block2 = MultiScaleConvBlock(hidden_dim, hidden_dim, [5, 7, 11, 15], dropout)
        
        # Unidirectional LSTM layers
        self.lstm_blocks = nn.ModuleList()
        for i in range(num_lstm_layers):
            lstm_block = UnidirectionalLSTMBlock(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=2,
                dropout=dropout
            )
            self.lstm_blocks.append(lstm_block)
        
        # Temporal attention layers
        self.attention_layers = nn.ModuleList([
            TemporalAttention(hidden_dim, num_heads, dropout)
            for _ in range(2)
        ])
        
        # Feature extraction heads
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.local_pool = nn.AdaptiveMaxPool1d(1)
        
        # Gated fusion mechanism
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        # Output head with skip connections
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # Confidence estimation head
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights with Xavier/He initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.zeros_(param.data)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def split_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Split input features into groups"""
        features = {}
        start_idx = 0
        
        for group_name, num_features in self.feature_config.items():
            if num_features > 0:
                features[group_name] = x[:, :, start_idx:start_idx + num_features]
                start_idx += num_features
                
        return features
    
    def forward(self, x, return_confidence: bool = False):
        batch_size, seq_len, _ = x.size()
        
        # Normalize features
        x_norm = self.feature_normalizer(x)
        
        # Split and project features by group
        feature_groups = self.split_features(x_norm)
        h = self.feature_projector(feature_groups)
        
        # Multi-scale temporal convolutions
        h_conv1 = self.conv_block1(h)
        h_conv2 = self.conv_block2(h_conv1)
        
        # Combine conv features
        h = h_conv1 + h_conv2
        
        # Progressive LSTM processing
        lstm_states = []
        hidden_states = []
        
        for i, lstm_block in enumerate(self.lstm_blocks):
            h, (h_n, c_n) = lstm_block(h)
            lstm_states.append(h)
            hidden_states.append(h_n[-1])  # Last layer hidden state
            
            # Apply attention every other layer
            if i % 2 == 1 and (i // 2) < len(self.attention_layers):
                h_attn, _ = self.attention_layers[i // 2](h)
                h = h + h_attn  # Residual connection
        
        # Extract different temporal features
        # Global features (average pooling)
        global_features = self.global_pool(h.transpose(1, 2)).squeeze(-1)
        
        # Local features (max pooling)
        local_features = self.local_pool(h.transpose(1, 2)).squeeze(-1)
        
        # Last timestep features (most recent information)
        last_features = h[:, -1, :]
        
        # Combine all features
        combined_features = torch.cat([global_features, local_features, last_features], dim=-1)
        
        # Gated fusion
        gate = self.fusion_gate(combined_features)
        fused_features = gate * global_features + (1 - gate) * last_features
        
        # Final output
        combined_final = torch.cat([fused_features, local_features, last_features], dim=-1)
        output = self.output_projection(combined_final)
        
        if return_confidence:
            confidence = self.confidence_head(combined_final)
            return output, confidence
        
        return output
    
    def get_feature_importance(self, x: torch.Tensor, method: str = 'gradient') -> np.ndarray:
        """Calculate feature importance using gradient-based method"""
        self.eval()
        x.requires_grad_(True)
        
        # Forward pass
        output = self.forward(x)
        
        # Calculate importance for each class
        importance_scores = []
        
        for class_idx in range(output.size(-1)):
            # Backward pass for each class
            self.zero_grad()
            class_output = output[:, class_idx].sum()
            class_output.backward(retain_graph=True)
            
            # Get gradient magnitudes
            grad_magnitudes = x.grad.abs().mean(dim=(0, 1)).cpu().numpy()
            importance_scores.append(grad_magnitudes)
        
        # Average across classes
        return np.array(importance_scores).mean(axis=0)
    
    def predict_with_uncertainty(self, x: torch.Tensor, n_samples: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """Make predictions with uncertainty estimation using MC Dropout"""
        self.train()  # Enable dropout
        
        predictions = []
        confidences = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                pred, conf = self.forward(x, return_confidence=True)
                predictions.append(F.softmax(pred, dim=-1))
                confidences.append(conf)
        
        predictions = torch.stack(predictions)
        confidences = torch.stack(confidences)
        
        # Mean prediction and uncertainty
        mean_pred = predictions.mean(dim=0)
        uncertainty = predictions.var(dim=0).mean(dim=-1, keepdim=True)
        mean_confidence = confidences.mean(dim=0)
        
        # Combine uncertainty and confidence
        final_confidence = mean_confidence * (1 - uncertainty)
        
        return mean_pred, final_confidence


def create_trading_model(num_features: int = 74, **kwargs) -> OptimizedTradingLSTM:
    """
    Factory function to create the trading model with proper feature configuration
    
    Args:
        num_features: Total number of input features (default 74 for all indicators)
        **kwargs: Additional arguments for the model
    """
    # Approximate feature distribution for 74 features
    feature_config = {
        'price': 5,       # OHLCV
        'volume': 7,      # VWAP, OBV, MFI, etc.
        'volatility': 11, # ATR, BB, KC, etc.
        'momentum': 14,   # RSI, MACD, ROC, etc.
        'ma': 5,          # Moving averages
        'structure': 10,  # Market structure indicators
        'time': 10,       # Time-based features
        'engineered': 12  # Z-scores, efficiency, etc.
    }
    
    # Adjust if total doesn't match
    total = sum(feature_config.values())
    if total != num_features:
        # Add difference to engineered features
        feature_config['engineered'] += (num_features - total)
    
    return OptimizedTradingLSTM(feature_config=feature_config, **kwargs)


# Example usage and testing
if __name__ == "__main__":
    # Create model for 74 features
    model = create_trading_model(
        num_features=74,
        hidden_dim=256,
        num_lstm_layers=3,
        num_heads=8,
        output_dim=3,
        dropout=0.3
    )
    
    # Test with sample data
    batch_size, seq_len, num_features = 8, 50, 74
    x = torch.randn(batch_size, seq_len, num_features)
    
    # Forward pass
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test with confidence
    output_with_conf, confidence = model(x, return_confidence=True)
    print(f"Confidence shape: {confidence.shape}")
    
    # Test uncertainty estimation
    predictions, uncertainty = model.predict_with_uncertainty(x, n_samples=10)
    print(f"Prediction shape: {predictions.shape}")
    print(f"Uncertainty shape: {uncertainty.shape}")
    
    # Feature importance
    importance = model.get_feature_importance(x[:1])  # Use single sample
    print(f"Feature importance shape: {importance.shape}")
    
    # Test gradient flow
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    loss = F.cross_entropy(output, torch.randint(0, 3, (batch_size,)))
    loss.backward()
    
    # Check gradient magnitudes
    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_norms.append(grad_norm)
            if grad_norm > 10:
                print(f"Large gradient in {name}: {grad_norm:.4f}")
    
    print(f"Average gradient norm: {np.mean(grad_norms):.4f}")
    print(f"Max gradient norm: {np.max(grad_norms):.4f}")
