# Multi-Head Attention for LSTM Trading Bot

## Overview

The Multi-Head Attention mechanism in this LSTM trading bot serves as a **selective focus system** that allows the model to simultaneously pay attention to different aspects of the trading sequence at different positions. Unlike sequential LSTM processing, attention creates direct connections between any two time points, which is crucial for understanding complex market relationships.

## Complete Code Analysis

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model      # 256 (hidden dimension)
        self.num_heads = num_heads  # 8 (parallel attention heads)
        self.d_k = d_model // num_heads  # 32 (dimension per head)
        
        # Linear transformations for queries, keys, and values
        self.w_q = nn.Linear(d_model, d_model)  # Query projection
        self.w_k = nn.Linear(d_model, d_model)  # Key projection  
        self.w_v = nn.Linear(d_model, d_model)  # Value projection
        self.w_o = nn.Linear(d_model, d_model)  # Output projection
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)  # Scaling factor (√32 ≈ 5.66)
```

## Core Concepts for Trading

### 1. Query, Key, Value Paradigm in Trading Context

In trading applications, the attention mechanism asks three fundamental questions:

```python
# For each timestep in the trading sequence:
q = self.w_q(x)  # QUERY: "What information am I looking for?"
k = self.w_k(x)  # KEY: "What information do I have available?"
v = self.w_v(x)  # VALUE: "What actual information should I retrieve?"
```

**Trading Interpretation**:
- **Query**: "What market conditions am I trying to understand right now?"
- **Key**: "What past market states are available to match against?"
- **Value**: "What actual market information should I extract from relevant past states?"

### 2. Multi-Head Architecture

The model splits the 256-dimensional representation into 8 parallel "heads" of 32 dimensions each:

```python
# Reshaping for multi-head processing
q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
k = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
v = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
# Result: (batch_size, num_heads, seq_len, d_k)
```

**Why 8 heads for trading?**
Each head can specialize in different market aspects:
- **Head 1**: Price momentum patterns
- **Head 2**: Volume-price relationships
- **Head 3**: Support/resistance levels
- **Head 4**: Moving average crossovers
- **Head 5**: Volatility patterns
- **Head 6**: Market sentiment indicators
- **Head 7**: Correlation patterns
- **Head 8**: Long-term trend analysis

## Attention Mechanism Deep Dive

### 1. Scaled Dot-Product Attention

```python
scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
```

**Mathematical breakdown**:
```
For each head h, timestep i, and timestep j:
score[h,i,j] = (q[h,i] · k[h,j]) / √32

Where:
- q[h,i] = query vector for head h at timestep i
- k[h,j] = key vector for head h at timestep j
- · = dot product
- √32 ≈ 5.66 = scaling factor
```

**Trading meaning**: 
- High score between timestep i and j means "the market state at time i is highly relevant to understanding time j"
- The scaling prevents attention weights from becoming too extreme

### 2. Attention Weight Calculation

```python
if mask is not None:
    scores = scores.masked_fill(mask == 0, -1e9)
    
attn = F.softmax(scores, dim=-1)
attn = self.dropout(attn)
```

**Softmax normalization**:
```
For each timestep i:
attention[i,j] = exp(score[i,j]) / Σ(exp(score[i,k])) for all k

Properties:
- Σ(attention[i,j]) = 1 for all j (weights sum to 1)
- 0 ≤ attention[i,j] ≤ 1 (valid probabilities)
```

**Trading interpretation**:
- `attention[i,j]` = "How much should I focus on timestep j when making a decision at timestep i?"
- Higher weights = more relevant past market conditions

### 3. Context Vector Computation

```python
context = torch.matmul(attn, v)
```

**Mathematical operation**:
```
For each head h and timestep i:
context[h,i] = Σ(attention[h,i,j] * value[h,j]) for all j

This creates a weighted average of all value vectors,
where weights are determined by attention scores.
```

**Trading meaning**: The context vector for timestep i is a weighted combination of information from all past timesteps, where the weights reflect relevance to current market conditions.

## Trading-Specific Examples

### Example 1: Support Level Detection

Imagine a 10-day trading sequence where day 8 is testing a support level that was established on day 3:

```
Day:    1    2    3    4    5    6    7    8    9    10
Price: 100  98   95   98  102  104  99   96   ?    ?
              ↑                          ↑
         Support formed              Testing support
```

**Attention behavior**:
- When processing day 8, attention weights might be:
  - `attention[8,3] = 0.7` (high weight to day 3 - support formation)
  - `attention[8,7] = 0.2` (medium weight to day 7 - recent context)
  - `attention[8,1] = 0.1` (low weight to other days)

### Example 2: Trend Reversal Pattern

Consider a sequence where a downtrend (days 1-5) reverses to an uptrend (days 6-10):

```
Days 1-5: Downtrend  |  Days 6-10: Uptrend
Price: 110→100→95→90→85 | 88→92→96→100→105
                    ↑
               Reversal point
```

**Multi-head specialization**:
- **Head 1**: Focuses on trend direction changes
- **Head 2**: Focuses on volume confirmation
- **Head 3**: Focuses on momentum indicators
- **Head 4**: Focuses on support/resistance breaks

## Integration with LSTM Architecture

### Positioning in the Network

```python
# In the main forward pass:
for i, lstm_block in enumerate(self.lstm_blocks):
    h = lstm_block(h)
    lstm_outputs.append(h)
    
    # Apply attention every other layer
    if i % 2 == 1 and i // 2 < len(self.attention_layers):
        h_attn, _ = self.attention_layers[i // 2](h)
        h = h + h_attn  # Residual connection
```

**Strategic placement**:
- **After LSTM layers 2 and 4**: Applied to refined sequential features
- **Residual connections**: Attention refinements are added to LSTM outputs
- **Progressive refinement**: Later attention layers work on more abstract features

### Complementary Roles

**LSTM contributions**:
- Sequential processing of market data
- Maintains hidden state representing market memory
- Captures temporal dependencies and trends

**Attention contributions**:
- Direct connections between any two time points
- Selective focus on relevant past market conditions
- Parallel processing of all time relationships

**Combined power**: LSTM builds sequential understanding, attention adds selective focus.

## Attention Patterns in Trading

### 1. Momentum Patterns

```python
# Example attention weights for momentum detection
attention_weights = [
    # Current day looks at:
    [0.1, 0.1, 0.1, 0.2, 0.5],  # Recent days (momentum continuation)
    [0.3, 0.2, 0.2, 0.2, 0.1],  # Distant days (momentum reversal)
]
```

### 2. Support/Resistance Patterns

```python
# Attention for support/resistance detection
attention_weights = [
    # When price approaches known level:
    [0.8, 0.1, 0.0, 0.0, 0.1],  # High focus on level formation day
    [0.2, 0.2, 0.2, 0.2, 0.2],  # Distributed attention (consolidation)
]
```

### 3. Volatility Patterns

```python
# Attention during high volatility
attention_weights = [
    # Current volatile period looks at:
    [0.4, 0.3, 0.2, 0.1, 0.0],  # Recent volatility causes
    [0.6, 0.1, 0.1, 0.1, 0.1],  # Previous volatility episode
]
```

## Computational Flow for Trading Bot

### Input Processing
```python
# Input: (batch=32, seq_len=60, features=256)
# 32 different stocks, 60 days of data, 256 features per day

batch_size, seq_len, d_model = x.size()  # 32, 60, 256
```

### Multi-Head Transformation
```python
# Transform to queries, keys, values
q = self.w_q(x).view(32, 60, 8, 32).transpose(1, 2)  # (32, 8, 60, 32)
k = self.w_k(x).view(32, 60, 8, 32).transpose(1, 2)  # (32, 8, 60, 32)
v = self.w_v(x).view(32, 60, 8, 32).transpose(1, 2)  # (32, 8, 60, 32)
```

### Attention Score Computation
```python
# For each of 8 heads, compute 60x60 attention matrix
scores = torch.matmul(q, k.transpose(-2, -1)) / 5.66
# Result: (32, 8, 60, 60)
# scores[stock, head, day_i, day_j] = relevance of day_j for day_i
```

### Context Generation
```python
# Generate context vectors
context = torch.matmul(attn, v)  # (32, 8, 60, 32)
# Reshape back to original format
context = context.transpose(1, 2).contiguous().view(32, 60, 256)
```

## Attention Visualization for Trading

### 1. Attention Heatmap Interpretation

```python
# Example attention matrix for one head:
#      Day: 1    2    3    4    5   (keys - what we have)
attn = [[0.1, 0.2, 0.6, 0.1, 0.0],  # Day 1 query
        [0.2, 0.1, 0.4, 0.2, 0.1],  # Day 2 query
        [0.1, 0.1, 0.2, 0.3, 0.3],  # Day 3 query
        [0.0, 0.1, 0.1, 0.2, 0.6],  # Day 4 query
        [0.1, 0.1, 0.2, 0.1, 0.5]]  # Day 5 query
```

**Reading the heatmap**:
- **Row i, Column j**: How much day i focuses on day j
- **Diagonal dominance**: Model focuses on current day
- **Off-diagonal patterns**: Historical dependencies

### 2. Trading Pattern Recognition

**Trend Following**:
```python
# Attention pattern for trend following
# Recent days get higher weights
attention_trend = [0.05, 0.1, 0.15, 0.25, 0.45]  # Increasing weights
```

**Mean Reversion**:
```python
# Attention pattern for mean reversion
# Distant days get higher weights (looking for reversal points)
attention_reversion = [0.4, 0.3, 0.2, 0.1, 0.0]  # Decreasing weights
```

## Advanced Features

### 1. Attention Masking for Trading

```python
# Prevent looking into the future (causal masking)
mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
scores = scores.masked_fill(mask == 1, -1e9)
```

**Purpose**: Ensures the model only uses past information for predictions.

### 2. Feature Importance Analysis

```python
def get_attention_weights(self, x):
    """Extract attention weights for trading analysis"""
    # ... (forward pass until attention)
    return attention_weights

# Usage for trading strategy analysis
weights = model.get_attention_weights(market_data)
# Analyze which historical periods are most important
```

### 3. Multi-Scale Attention Integration

The model uses attention at multiple levels:

```python
# Layer 2 attention: Works on basic LSTM features
# - Focuses on price patterns, volume patterns
attention_layer_2 = MultiHeadAttention(256, 8)

# Layer 4 attention: Works on refined LSTM features  
# - Focuses on complex market relationships
attention_layer_4 = MultiHeadAttention(256, 8)
```

## Trading Strategy Applications

### 1. Entry Signal Generation

```python
# High attention to breakout confirmation days
if attention_weights[current_day, breakout_day] > 0.5:
    # Strong signal - breakout pattern confirmed
    generate_entry_signal()
```

### 2. Risk Management

```python
# Monitor attention to high-volatility periods
volatility_attention = sum(attention_weights[current_day, volatile_days])
if volatility_attention > threshold:
    # Reduce position size due to volatility risk
    adjust_position_size()
```

### 3. Market Regime Detection

```python
# Analyze attention patterns to detect market regimes
if attention_focused_on_recent_days():
    current_regime = "trending"
elif attention_focused_on_distant_days():
    current_regime = "mean_reverting"
```

## Performance Implications

### Memory Usage
```python
# Attention matrix: (batch_size, num_heads, seq_len, seq_len)
memory_per_batch = 32 * 8 * 60 * 60 * 4 bytes = 3.7 MB
# For 60-day sequences, this is manageable
```

### Computational Complexity
```python
# Time complexity: O(seq_len²) per head
# For 60-day sequences: 60² = 3,600 operations per head
# Total: 8 heads × 3,600 = 28,800 operations
```

## Key Advantages for Trading

### 1. **Long-Range Dependencies**
- Can connect current market conditions to relevant historical events
- Overcomes LSTM's limitation with very long sequences

### 2. **Parallel Processing**
- Processes all time relationships simultaneously
- Much faster than sequential LSTM processing

### 3. **Interpretability**
- Attention weights show which historical periods influence decisions
- Valuable for understanding trading strategy logic

### 4. **Selective Focus**
- Automatically learns which historical patterns are most relevant
- Adapts to different market conditions and trading scenarios

### 5. **Multi-Aspect Analysis**
- Different heads can specialize in different market aspects
- Provides comprehensive market understanding

## Integration with Trading Workflow

### 1. Feature Engineering
```python
# Input features that attention can focus on:
features = [
    'price', 'volume', 'volatility',
    'moving_avg_5', 'moving_avg_20',
    'rsi', 'macd', 'bollinger_bands',
    'support_level', 'resistance_level'
]
```

### 2. Signal Generation
```python
# Use attention weights to generate trading signals
if high_attention_to_breakout_pattern():
    signal = "BUY"
elif high_attention_to_reversal_pattern():
    signal = "SELL"
else:
    signal = "HOLD"
```

### 3. Risk Assessment
```python
# Assess risk based on attention patterns
if attention_scattered_across_many_days():
    risk_level = "HIGH"  # Uncertain market conditions
elif attention_focused_on_few_days():
    risk_level = "LOW"   # Clear pattern recognition
```

## Conclusion

The Multi-Head Attention mechanism in this LSTM trading bot serves as a sophisticated **selective focus system** that:

1. **Enhances pattern recognition** by creating direct connections between any two time points
2. **Provides interpretability** through attention weight visualization
3. **Enables multi-aspect analysis** through parallel attention heads
4. **Overcomes LSTM limitations** for long-range dependencies
5. **Supports trading strategy development** through attention pattern analysis

This makes the model particularly powerful for complex trading scenarios where success depends on understanding relationships between current market conditions and relevant historical patterns, regardless of how far apart they occur in time.

The combination of LSTM's sequential processing with attention's selective focus creates a robust foundation for automated trading systems that can adapt to various market conditions and trading strategies.
