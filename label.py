import os
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# === CONFIGURATION ===
CSV_DIR = "C:/Users/Hayden/Downloads/5mincsvma/"  # Your directory with CSV files
OUTPUT_DIR = "C:/Users/Hayden/Downloads/5mincsvma_processed/"  # Output directory (optional)

# Signal labeling parameters
LOOKAHEAD = 15          # Look ahead 15 bars (75 minutes)
BUY_THRESHOLD = 0.002   # 0.2% up
SELL_THRESHOLD = -0.002 # 0.2% down
LABEL_TO_CLASS = {
    "buy": 1,
    "hold": 0,
    "sell": 2
}

class TechnicalIndicatorProcessor:
    """Complete technical indicator calculator and signal labeler"""
    
    def __init__(self, lookahead=15, buy_thresh=0.002, sell_thresh=-0.002):
        self.lookahead = lookahead
        self.buy_thresh = buy_thresh
        self.sell_thresh = sell_thresh
        
    def process_file(self, filepath: str, output_path: str = None) -> pd.DataFrame:
        """Process a single CSV file - add indicators and labels"""
        print(f"\nProcessing: {os.path.basename(filepath)}")
        
        # Load data
        df = pd.read_csv(filepath)
        print(f"  - Loaded {len(df)} rows")
        
        # Ensure timestamp is datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            df.set_index('timestamp', inplace=True)
        
        # Add all technical indicators
        print("  - Adding technical indicators...")
        df = self.add_all_indicators(df)
        
        # Add signal labels
        print("  - Adding signal labels...")
        df = self.add_signal_labels(df)
        
        # Clean up NaN values from initial rows (due to rolling calculations)
        initial_nans = df.isna().any(axis=1).sum()
        df = df.dropna()
        print(f"  - Dropped {initial_nans} rows with NaN values")
        
        # Save processed file
        if output_path:
            df.to_csv(output_path)
            print(f"  - Saved to: {output_path}")
        else:
            df.to_csv(filepath, index=True)
            print(f"  - Overwrote original file")
            
        return df
    
    def add_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all technical indicators to the dataframe"""
        
        # 1. Moving Averages
        df = self.add_moving_averages(df)
        
        # 2. Volume Indicators
        df = self.add_volume_indicators(df)
        
        # 3. Volatility Indicators
        df = self.add_volatility_indicators(df)
        
        # 4. Momentum Indicators
        df = self.add_momentum_indicators(df)
        
        # 5. Market Structure
        df = self.add_market_structure(df)
        
        # 6. Time Features
        df = self.add_time_features(df)
        
        # 7. Engineered Features
        df = self.add_engineered_features(df)
        
        return df
    
    def add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add moving averages"""
        periods = [5, 10, 20, 40, 55]
        for period in periods:
            df[f'ma_{period}'] = df['close'].rolling(period).mean()
        return df
    
    def add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based indicators"""
        
        # VWAP
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        df['vwap'] = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
        
        # On-Balance Volume
        df['obv'] = (df['volume'] * np.sign(df['close'].diff())).fillna(0).cumsum()
        
        # Volume Rate of Change
        df['volume_roc'] = df['volume'].pct_change(periods=10) * 100
        
        # Money Flow Index
        money_flow = typical_price * df['volume']
        positive_flow = money_flow.where(typical_price > typical_price.shift(), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(), 0)
        
        positive_mf = positive_flow.rolling(14).sum()
        negative_mf = negative_flow.rolling(14).sum()
        
        mfi_ratio = positive_mf / (negative_mf + 1e-10)
        df['mfi'] = 100 - (100 / (1 + mfi_ratio))
        
        # Accumulation/Distribution Line
        clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + 1e-10)
        df['ad_line'] = (clv * df['volume']).cumsum()
        
        # Relative Volume
        df['relative_volume'] = df['volume'] / df['volume'].rolling(20).mean()
        
        # Chaikin Money Flow
        df['cmf'] = (clv * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
        
        return df
    
    def add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility indicators"""
        
        # Average True Range
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr_14'] = true_range.rolling(14).mean()
        df['atr_normalized'] = df['atr_14'] / df['close'] * 100
        
        # Bollinger Bands
        rolling_mean = df['close'].rolling(20).mean()
        rolling_std = df['close'].rolling(20).std()
        
        df['bb_upper'] = rolling_mean + (rolling_std * 2)
        df['bb_lower'] = rolling_mean - (rolling_std * 2)
        df['bb_middle'] = rolling_mean
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle'] * 100
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
        
        # Keltner Channels
        df['kc_middle'] = df['close'].ewm(span=20).mean()
        df['kc_upper'] = df['kc_middle'] + (df['atr_14'] * 2)
        df['kc_lower'] = df['kc_middle'] - (df['atr_14'] * 2)
        df['kc_position'] = (df['close'] - df['kc_lower']) / (df['kc_upper'] - df['kc_lower'] + 1e-10)
        
        # Historical Volatility
        df['volatility_20'] = df['close'].pct_change().rolling(20).std() * np.sqrt(252 * 78)
        df['volatility_50'] = df['close'].pct_change().rolling(50).std() * np.sqrt(252 * 78)
        
        # Donchian Channels
        df['donchian_high'] = df['high'].rolling(20).max()
        df['donchian_low'] = df['low'].rolling(20).min()
        df['donchian_position'] = (df['close'] - df['donchian_low']) / (df['donchian_high'] - df['donchian_low'] + 1e-10)
        
        return df
    
    def add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators"""
        
        # RSI
        def calculate_rsi(series, period=14):
            delta = series.diff()
            gain = delta.where(delta > 0, 0).rolling(period).mean()
            loss = -delta.where(delta < 0, 0).rolling(period).mean()
            rs = gain / (loss + 1e-10)
            return 100 - (100 / (1 + rs))
        
        df['rsi_14'] = calculate_rsi(df['close'], 14)
        df['rsi_9'] = calculate_rsi(df['close'], 9)
        
        # Stochastic Oscillator
        low_14 = df['low'].rolling(14).min()
        high_14 = df['high'].rolling(14).max()
        df['stoch_k'] = 100 * (df['close'] - low_14) / (high_14 - low_14 + 1e-10)
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()
        
        # MACD
        exp1 = df['close'].ewm(span=12).mean()
        exp2 = df['close'].ewm(span=26).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_diff'] = df['macd'] - df['macd_signal']
        df['macd_diff_normalized'] = df['macd_diff'] / df['close'] * 100
        
        # Rate of Change
        df['roc_5'] = df['close'].pct_change(periods=5) * 100
        df['roc_10'] = df['close'].pct_change(periods=10) * 100
        df['roc_20'] = df['close'].pct_change(periods=20) * 100
        
        # Commodity Channel Index
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        sma = typical_price.rolling(20).mean()
        mad = typical_price.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean())
        df['cci'] = (typical_price - sma) / (0.015 * mad + 1e-10)
        
        # Williams %R
        highest_high = df['high'].rolling(14).max()
        lowest_low = df['low'].rolling(14).min()
        df['williams_r'] = -100 * (highest_high - df['close']) / (highest_high - lowest_low + 1e-10)
        
        # Ultimate Oscillator
        bp = df['close'] - pd.concat([df['low'], df['close'].shift()], axis=1).min(axis=1)
        tr = pd.concat([
            df['high'] - df['low'],
            abs(df['high'] - df['close'].shift()),
            abs(df['low'] - df['close'].shift())
        ], axis=1).max(axis=1)
        
        avg7 = bp.rolling(7).sum() / tr.rolling(7).sum()
        avg14 = bp.rolling(14).sum() / tr.rolling(14).sum()
        avg28 = bp.rolling(28).sum() / tr.rolling(28).sum()
        
        df['ultimate_oscillator'] = 100 * ((4 * avg7) + (2 * avg14) + avg28) / 7
        
        return df
    
    def add_market_structure(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market structure indicators"""
        
        # Price relative to VWAP
        df['price_to_vwap'] = (df['close'] / df['vwap'] - 1) * 100
        
        # High-Low spread
        df['hl_spread'] = (df['high'] - df['low']) / df['close'] * 100
        
        # Close position in daily range
        df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)
        
        # Distance from MAs
        for ma in [5, 10, 20, 40, 55]:
            df[f'dist_from_ma_{ma}'] = (df['close'] - df[f'ma_{ma}']) / df[f'ma_{ma}'] * 100
        
        # MA alignment (trend strength)
        ma_cols = ['ma_5', 'ma_10', 'ma_20', 'ma_40', 'ma_55']
        df['ma_alignment'] = 0
        for i in range(len(ma_cols)-1):
            df['ma_alignment'] += (df[ma_cols[i]] > df[ma_cols[i+1]]).astype(int)
        df['ma_alignment'] = (df['ma_alignment'] - 2) / 2  # Normalize to [-1, 1]
        
        # Pivot Points
        df['pivot'] = (df['high'] + df['low'] + df['close']) / 3
        df['r1'] = 2 * df['pivot'] - df['low']
        df['s1'] = 2 * df['pivot'] - df['high']
        df['pivot_position'] = (df['close'] - df['s1']) / (df['r1'] - df['s1'] + 1e-10)
        
        return df
    
    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        
        if isinstance(df.index, pd.DatetimeIndex):
            # Basic time features
            df['hour'] = df.index.hour
            df['minute'] = df.index.minute
            df['day_of_week'] = df.index.dayofweek
            
            # Cyclical encoding
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 5)
            df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 5)
            
            # Trading session indicators
            df['is_premarket'] = ((df['hour'] < 9) | ((df['hour'] == 9) & (df['minute'] < 30))).astype(int)
            df['is_regular'] = (((df['hour'] == 9) & (df['minute'] >= 30)) | 
                               ((df['hour'] > 9) & (df['hour'] < 16))).astype(int)
            df['is_afterhours'] = (df['hour'] >= 16).astype(int)
            
            # Minutes since market open
            df['mins_since_open'] = np.where(
                df['is_regular'],
                (df['hour'] - 9) * 60 + df['minute'] - 30,
                -1
            )
            
            # First/last 30 minutes of regular session
            df['is_open_30min'] = (df['mins_since_open'].between(0, 30)).astype(int)
            df['is_close_30min'] = (df['mins_since_open'].between(360, 390)).astype(int)
            
        return df
    
    def add_engineered_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add complex engineered features"""
        
        # Price-Volume Correlation
        df['price_volume_corr'] = df['close'].rolling(20).corr(df['volume'])
        
        # Momentum Quality
        returns = df['close'].pct_change()
        df['momentum_quality'] = returns.rolling(20).mean() / (returns.rolling(20).std() + 1e-10)
        
        # Efficiency Ratio
        net_change = abs(df['close'] - df['close'].shift(10))
        total_change = df['close'].diff().abs().rolling(10).sum()
        df['efficiency_ratio'] = net_change / (total_change + 1e-10)
        
        # Mean Reversion Indicators
        df['zscore_20'] = (df['close'] - df['close'].rolling(20).mean()) / (df['close'].rolling(20).std() + 1e-10)
        df['zscore_50'] = (df['close'] - df['close'].rolling(50).mean()) / (df['close'].rolling(50).std() + 1e-10)
        
        # Microstructure proxies
        df['kyle_lambda'] = df['close'].diff().abs() / (df['volume'].rolling(20).mean() + 1e-10)
        df['amihud_illiquidity'] = (returns.abs() / (df['volume'] + 1e-10)).rolling(20).mean()
        
        # Support/Resistance detection (simplified)
        df['near_round_number'] = ((df['close'] % 1).apply(lambda x: min(x, 1-x)) < 0.05).astype(int)
        df['near_half_dollar'] = ((df['close'] % 0.5).apply(lambda x: min(x, 0.5-x)) < 0.05).astype(int)
        
        # Trend exhaustion
        df['rsi_divergence'] = (df['close'].pct_change(5) * 100) - (df['rsi_14'].diff(5))
        
        # Volume patterns
        df['volume_spike'] = (df['volume'] > df['volume'].rolling(20).mean() * 2).astype(int)
        df['volume_dry_up'] = (df['volume'] < df['volume'].rolling(20).mean() * 0.5).astype(int)
        
        return df
    
    def add_signal_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add signal labels based on future returns"""
        
        # Calculate future return
        df['future_close'] = df['close'].shift(-self.lookahead)
        df['future_return'] = (df['future_close'] - df['close']) / df['close']
        
        # Generate signals
        df['signal'] = 'hold'
        df.loc[df['future_return'] >= self.buy_thresh, 'signal'] = 'buy'
        df.loc[df['future_return'] <= self.sell_thresh, 'signal'] = 'sell'
        
        # Convert to numeric classes
        df['signal_class'] = df['signal'].map(LABEL_TO_CLASS)
        
        # Remove rows where we can't calculate future return
        df = df[:-self.lookahead] if self.lookahead > 0 else df
        
        # Drop helper columns
        df = df.drop(['future_close', 'future_return'], axis=1)
        
        return df
    
    def get_feature_columns(self) -> List[str]:
        """Return list of all feature columns for ML model"""
        return [
            # OHLCV
            'open', 'high', 'low', 'close', 'volume',
            
            # Moving Averages
            'ma_5', 'ma_10', 'ma_20', 'ma_40', 'ma_55',
            
            # Volume Indicators
            'vwap', 'obv', 'volume_roc', 'mfi', 'ad_line', 
            'relative_volume', 'cmf',
            
            # Volatility
            'atr_14', 'atr_normalized', 'bb_upper', 'bb_lower', 
            'bb_middle', 'bb_width', 'bb_position', 'kc_position',
            'volatility_20', 'volatility_50', 'donchian_position',
            
            # Momentum
            'rsi_14', 'rsi_9', 'stoch_k', 'stoch_d', 'macd', 
            'macd_signal', 'macd_diff', 'macd_diff_normalized',
            'roc_5', 'roc_10', 'roc_20', 'cci', 'williams_r',
            'ultimate_oscillator',
            
            # Market Structure
            'price_to_vwap', 'hl_spread', 'close_position',
            'dist_from_ma_5', 'dist_from_ma_10', 'dist_from_ma_20',
            'dist_from_ma_40', 'dist_from_ma_55', 'ma_alignment',
            'pivot_position',
            
            # Time Features
            'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
            'is_premarket', 'is_regular', 'is_afterhours',
            'is_open_30min', 'is_close_30min', 'mins_since_open',
            
            # Engineered Features
            'price_volume_corr', 'momentum_quality', 'efficiency_ratio',
            'zscore_20', 'zscore_50', 'kyle_lambda', 'amihud_illiquidity',
            'near_round_number', 'near_half_dollar', 'rsi_divergence',
            'volume_spike', 'volume_dry_up'
        ]


def process_directory(input_dir: str, output_dir: str = None):
    """Process all CSV files in a directory"""
    
    # Create output directory if specified
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Get all CSV files
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return
    
    print(f"Found {len(csv_files)} CSV files to process")
    
    # Initialize processor
    processor = TechnicalIndicatorProcessor(
        lookahead=LOOKAHEAD,
        buy_thresh=BUY_THRESHOLD,
        sell_thresh=SELL_THRESHOLD
    )
    
    # Process each file
    successful = 0
    failed = 0
    
    for filename in csv_files:
        input_path = os.path.join(input_dir, filename)
        
        if output_dir:
            output_path = os.path.join(output_dir, filename.replace('.csv', '_processed.csv'))
        else:
            output_path = None
        
        try:
            df = processor.process_file(input_path, output_path)
            
            # Print summary statistics
            signal_counts = df['signal'].value_counts()
            print(f"    Signal distribution: Buy: {signal_counts.get('buy', 0)}, "
                  f"Hold: {signal_counts.get('hold', 0)}, "
                  f"Sell: {signal_counts.get('sell', 0)}")
            
            successful += 1
            
        except Exception as e:
            print(f"  ERROR processing {filename}: {str(e)}")
            failed += 1
            continue
    
    print(f"\n{'='*50}")
    print(f"Processing complete!")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total features added: {len(processor.get_feature_columns())}")
    
    # Print feature list
    print(f"\nFeature columns for ML model:")
    features = processor.get_feature_columns()
    for i in range(0, len(features), 5):
        print(f"  {', '.join(features[i:i+5])}")


def main():
    """Main function"""
    
    print("="*60)
    print("Technical Indicator and Signal Labeling Processor")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Input directory: {CSV_DIR}")
    print(f"  Output directory: {OUTPUT_DIR if OUTPUT_DIR else 'Overwrite originals'}")
    print(f"  Lookahead period: {LOOKAHEAD} bars ({LOOKAHEAD * 5} minutes)")
    print(f"  Buy threshold: {BUY_THRESHOLD * 100:.1f}%")
    print(f"  Sell threshold: {SELL_THRESHOLD * 100:.1f}%")
    print("="*60)
    
    # Process all files
    process_directory(CSV_DIR, OUTPUT_DIR)
    
    print("\n✓ All files processed successfully!")
    print("\nYour data now includes:")
    print("  - 70+ technical indicators")
    print("  - Signal labels (buy/hold/sell)")
    print("  - Signal class (0/1/2) for ML training")
    print("\nReady for LSTM training!")


if __name__ == "__main__":
    main()
