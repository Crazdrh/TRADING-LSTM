import os
import pandas as pd
import numpy as np

class TradingPositionLabeler:
    """
    Labels data based on realistic trading positions rather than point-in-time predictions.
    Simulates entering and exiting positions based on trend detection.
    """
    
    def __init__(self, 
                 min_trend_candles=3,      # Minimum candles to confirm trend
                 trend_threshold=0.001,     # 0.1% minimum move to confirm trend
                 stop_loss=0.005,           # 0.5% stop loss
                 take_profit=0.01,          # 1% take profit
                 use_hold_signals=False):   # Whether to use hold signals
        
        self.min_trend_candles = min_trend_candles
        self.trend_threshold = trend_threshold
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.use_hold_signals = use_hold_signals
        
        # Label mapping
        if use_hold_signals:
            self.label_to_class = {"buy": 1, "hold": 0, "sell": 2}
        else:
            self.label_to_class = {"buy": 0, "sell": 1}
    
    def detect_trend_change(self, prices, idx):
        """
        Detect if there's a trend change at the given index.
        Returns: 'up', 'down', or None
        """
        if idx < self.min_trend_candles:
            return None
            
        # Look at recent price movement
        recent_prices = prices[idx - self.min_trend_candles:idx + 1]
        price_change = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
        
        # Check if movement is significant
        if abs(price_change) < self.trend_threshold:
            return None
            
        # Confirm trend by checking direction consistency
        diffs = np.diff(recent_prices)
        
        if price_change > 0 and np.sum(diffs > 0) >= len(diffs) * 0.7:
            return 'up'
        elif price_change < 0 and np.sum(diffs < 0) >= len(diffs) * 0.7:
            return 'down'
            
        return None
    
    def label_with_positions(self, df):
        """
        Label the dataframe with realistic position-based signals.
        """
        df = df.copy()
        prices = df['close'].values
        signals = []
        
        current_position = None  # None, 'long', 'short'
        entry_price = None
        position_start_idx = None
        
        for i in range(len(df)):
            # Default signal
            if self.use_hold_signals:
                signal = 'hold'
            else:
                # If no hold signals, maintain previous position
                signal = signals[-1] if signals else 'buy'
            
            # Check for position exit conditions
            if current_position is not None and entry_price is not None:
                current_price = prices[i]
                price_change = (current_price - entry_price) / entry_price
                
                # Check stop loss and take profit
                if current_position == 'long':
                    if price_change <= -self.stop_loss or price_change >= self.take_profit:
                        # Exit long position
                        current_position = None
                        entry_price = None
                    else:
                        signal = 'buy'  # Continue holding long
                        
                elif current_position == 'short':
                    if price_change >= self.stop_loss or price_change <= -self.take_profit:
                        # Exit short position
                        current_position = None
                        entry_price = None
                    else:
                        signal = 'sell'  # Continue holding short
            
            # Check for new position entry
            if current_position is None:
                trend = self.detect_trend_change(prices, i)
                
                if trend == 'up':
                    current_position = 'long'
                    entry_price = prices[i]
                    position_start_idx = i
                    signal = 'buy'
                    
                elif trend == 'down':
                    current_position = 'short'
                    entry_price = prices[i]
                    position_start_idx = i
                    signal = 'sell'
            
            signals.append(signal)
        
        # Apply position labels retroactively for better training
        # This ensures the model learns to predict at the start of trends
        final_signals = signals.copy()
        
        # Post-process to extend signals backward slightly
        for i in range(1, len(final_signals)):
            if final_signals[i] != final_signals[i-1]:
                # New position started, extend it backward by 1-2 candles
                lookback = min(2, i)
                for j in range(1, lookback + 1):
                    if i - j >= 0:
                        final_signals[i - j] = final_signals[i]
        
        df['signal'] = final_signals
        df['signal_class'] = df['signal'].map(self.label_to_class)
        
        return df
    
    def analyze_positions(self, df):
        """
        Analyze the trading positions for statistics.
        """
        signals = df['signal'].values
        prices = df['close'].values
        
        positions = []
        current_position = None
        entry_idx = None
        entry_price = None
        
        for i in range(len(signals)):
            if i == 0:
                current_position = signals[i]
                entry_idx = i
                entry_price = prices[i]
                continue
                
            if signals[i] != current_position:
                # Position changed
                if current_position != 'hold' and entry_price is not None:
                    exit_price = prices[i]
                    duration = i - entry_idx
                    
                    if current_position == 'buy':
                        pnl = (exit_price - entry_price) / entry_price
                    else:  # sell
                        pnl = (entry_price - exit_price) / entry_price
                    
                    positions.append({
                        'type': current_position,
                        'entry_idx': entry_idx,
                        'exit_idx': i,
                        'duration': duration,
                        'pnl': pnl
                    })
                
                # Start new position
                current_position = signals[i]
                entry_idx = i
                entry_price = prices[i]
        
        return positions


def label_csv_with_positions(file_path, labeler):
    """
    Label a single CSV file with position-based signals.
    """
    print(f"Labeling: {file_path}")
    
    df = pd.read_csv(file_path)
    df = labeler.label_with_positions(df)
    
    # Analyze positions
    positions = labeler.analyze_positions(df)
    
    if positions:
        total_trades = len(positions)
        winning_trades = sum(1 for p in positions if p['pnl'] > 0)
        avg_pnl = np.mean([p['pnl'] for p in positions])
        avg_duration = np.mean([p['duration'] for p in positions])
        
        print(f"  Total trades: {total_trades}")
        print(f"  Win rate: {winning_trades/total_trades*100:.1f}%")
        print(f"  Avg P&L: {avg_pnl*100:.2f}%")
        print(f"  Avg duration: {avg_duration:.1f} candles")
    
    # Show signal distribution
    signal_counts = df['signal'].value_counts()
    print(f"  Signal distribution: {signal_counts.to_dict()}")
    
    return df


def label_all_csvs_with_positions(csv_dir, output_dir=None, **labeler_kwargs):
    """
    Label all CSV files in a directory with position-based signals.
    """
    if output_dir is None:
        output_dir = csv_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create labeler
    labeler = TradingPositionLabeler(**labeler_kwargs)
    
    # Process all CSV files
    files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
    
    for fname in files:
        input_path = os.path.join(csv_dir, fname)
        output_path = os.path.join(output_dir, fname)
        
        try:
            df = label_csv_with_positions(input_path, labeler)
            df.to_csv(output_path, index=False)
        except Exception as e:
            print(f"Error processing {fname}: {e}")
    
    print(f"\nDone! Processed {len(files)} files.")


# Alternative: Momentum-based labeling
def label_with_momentum(df, momentum_window=10, momentum_threshold=0.002):
    """
    Label based on momentum - more responsive to market changes.
    """
    df = df.copy()
    
    # Calculate momentum
    df['momentum'] = df['close'].pct_change(momentum_window)
    
    # Simple momentum-based signals
    df['signal'] = 'hold'
    df.loc[df['momentum'] > momentum_threshold, 'signal'] = 'buy'
    df.loc[df['momentum'] < -momentum_threshold, 'signal'] = 'sell'
    
    # Smooth signals to avoid too frequent changes
    df['signal'] = df['signal'].rolling(3, center=True).apply(
        lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[1]
    )
    
    # Map to classes
    label_to_class = {"buy": 1, "hold": 0, "sell": 2}
    df['signal_class'] = df['signal'].map(label_to_class)
    
    return df


if __name__ == "__main__":
    # Example usage
    csv_dir = "C:/Users/Hayden/Downloads/5mincsvma/"
    output_dir = "C:/Users/Hayden/Downloads/5mincsvma_labeled/"
    
    # Option 1: Position-based labeling (recommended)
    label_all_csvs_with_positions(
        csv_dir=csv_dir,
        output_dir=output_dir,
        min_trend_candles=5,
        trend_threshold=0.0015,  # 0.15%
        stop_loss=0.01,          # 1%
        take_profit=0.02,        # 2%
        use_hold_signals=False   # Just buy/sell for clearer signals
    )
