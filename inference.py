import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import time
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# Import your LSTM model
from LSTM import ComplexLSTMModel


class InferenceDataset(Dataset):
    """Dataset class for inference - processes one sequence at a time"""
    def __init__(self, df, seq_len=50, means=None, stds=None):
        self.seq_len = seq_len
        
        # Feature columns (same as training)
        self.feature_cols = [
            # OHLCV
            'open', 'high', 'low', 'close', 'volume',
            'ma_10', 'ma_20', 'ma_30', 'ma_40', 'dist_from_ma_10', 'dist_from_ma_20',
            # Volume
            'vwap', 'relative_volume', 'obv',
            # Volatility
            'bb_width', 'bb_position', 'atr_normalized',
            # Momentum
            'rsi_14', 'macd_diff', 'stoch_k', 'roc_10',
            # Market Structure
            'price_to_vwap', 'hl_spread', 'pivot_position',
            # Engineered Features
            'zscore_20', 'efficiency_ratio', 'price_volume_corr'
        ]
        
        # Clean column names
        df.columns = [c.strip() for c in df.columns]
        
        # Store raw dataframe for reference
        self.df = df
        
        # Extract features
        features = df[self.feature_cols].values.astype(float)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Normalize using provided or calculated stats
        if means is None or stds is None:
            self.means = features.mean(axis=0)
            self.stds = features.std(axis=0) + 1e-8
        else:
            self.means = means
            self.stds = stds
            
        self.features = (features - self.means) / self.stds
        
        # Extract labels if available
        if 'signal_class' in df.columns:
            self.labels = df['signal_class'].values.astype(int)
        else:
            self.labels = None
            
        # Store timestamps if available
        if 'datetime' in df.columns:
            self.timestamps = pd.to_datetime(df['datetime'])
        else:
            self.timestamps = pd.date_range(start='2023-01-01', periods=len(df), freq='5min')
    
    def __len__(self):
        return len(self.features) - self.seq_len
    
    def __getitem__(self, idx):
        seq = self.features[idx:idx + self.seq_len]
        
        if self.labels is not None:
            label = self.labels[idx + self.seq_len]
            return torch.tensor(seq, dtype=torch.float32), label
        else:
            return torch.tensor(seq, dtype=torch.float32), -1


class LiveInference:
    """Live inference system for LSTM trading model"""
    
    def __init__(self, model_path, checkpoint_path=None, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Class mappings
        self.class_names = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
        self.class_colors = {0: 'yellow', 1: 'green', 2: 'red'}
        
        # Load model
        self.model = self.load_model(model_path, checkpoint_path)
        
        # Performance tracking
        self.predictions_history = deque(maxlen=100)
        self.accuracy_history = deque(maxlen=100)
        
    def load_model(self, model_path, checkpoint_path=None):
        """Load the trained model"""
        print(f"Loading model from {model_path}")
        
        # Initialize model (matching training configuration)
        model = ComplexLSTMModel(
            input_dim=27,  # Updated to match training
            hidden_dim=256,
            num_lstm_layers=3,
            num_heads=8,
            output_dim=3,
            dropout=0.0  # Set to 0 for inference
        )
        
        # Load weights
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading from checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print(f"Loading model weights from: {model_path}")
            model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def predict_single(self, sequence):
        """Make prediction for a single sequence"""
        with torch.no_grad():
            if len(sequence.shape) == 2:
                sequence = sequence.unsqueeze(0)
            
            sequence = sequence.to(self.device)
            logits = self.model(sequence)
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(logits, dim=1)
            
            return pred.item(), probs.cpu().numpy()[0]
    
    def run_live_inference(self, csv_path, start_idx=None, delay=0.5, show_plot=True):
        """Run live inference on CSV data"""
        print(f"\n{'='*60}")
        print(f"Starting Live Inference on: {csv_path}")
        print(f"{'='*60}\n")
        
        # Load data
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} rows of data")
        
        # Create dataset
        dataset = InferenceDataset(df)
        
        # Statistics tracking
        total_predictions = 0
        correct_predictions = 0
        class_predictions = {0: 0, 1: 0, 2: 0}
        class_correct = {0: 0, 1: 0, 2: 0}
        
        # Determine starting index
        if start_idx is None:
            start_idx = max(0, len(dataset) - 100)  # Default to last 100 samples
        
        print(f"\nStarting from index {start_idx}")
        print(f"Press Ctrl+C to stop\n")
        
        # Setup live plot if requested
        if show_plot:
            plt.ion()
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        try:
            for idx in range(start_idx, len(dataset)):
                seq, true_label = dataset[idx]
                
                # Get prediction
                pred_class, probs = self.predict_single(seq)
                
                # Get current row info
                current_idx = idx + dataset.seq_len
                current_time = dataset.timestamps[current_idx]
                current_price = dataset.df['close'].iloc[current_idx]
                
                # Display prediction
                print(f"\n[{current_time}] Price: ${current_price:.2f}")
                print(f"Prediction: {self.class_names[pred_class]} ", end='')
                
                # Color code the output
                if pred_class == 1:
                    print("🟢", end='')
                elif pred_class == 2:
                    print("🔴", end='')
                else:
                    print("🟡", end='')
                
                print(f" (Confidence: {probs[pred_class]:.2%})")
                print(f"Probabilities - BUY: {probs[1]:.2%}, HOLD: {probs[0]:.2%}, SELL: {probs[2]:.2%}")
                
                # If we have true labels, show accuracy
                if true_label != -1:
                    true_class_name = self.class_names[true_label]
                    is_correct = pred_class == true_label
                    
                    print(f"Actual: {true_class_name} ", end='')
                    if is_correct:
                        print("✓ CORRECT")
                        correct_predictions += 1
                        class_correct[true_label] += 1
                    else:
                        print("✗ WRONG")
                    
                    total_predictions += 1
                    class_predictions[pred_class] += 1
                    
                    # Update accuracy
                    current_accuracy = correct_predictions / total_predictions
                    self.accuracy_history.append(current_accuracy)
                    
                    print(f"Running Accuracy: {current_accuracy:.2%} ({correct_predictions}/{total_predictions})")
                
                # Update plot if enabled
                if show_plot and idx % 10 == 0:
                    self.update_plot(fig, ax1, ax2, dataset, idx, start_idx)
                
                # Store prediction
                self.predictions_history.append({
                    'time': current_time,
                    'price': current_price,
                    'prediction': pred_class,
                    'true_label': true_label if true_label != -1 else None,
                    'probabilities': probs
                })
                
                # Delay for live effect
                if delay > 0:
                    time.sleep(delay)
                    
        except KeyboardInterrupt:
            print("\n\nInference stopped by user")
        
        # Final statistics
        self.print_final_statistics(total_predictions, correct_predictions, 
                                   class_predictions, class_correct)
        
        # Show confusion matrix if we have true labels
        if total_predictions > 0:
            self.plot_confusion_matrix(dataset, start_idx)
        
        return self.predictions_history
    
    def update_plot(self, fig, ax1, ax2, dataset, current_idx, start_idx):
        """Update live plot"""
        ax1.clear()
        ax2.clear()
        
        # Plot price
        plot_range = slice(max(0, current_idx - 50), current_idx + 1)
        prices = dataset.df['close'].iloc[plot_range]
        ax1.plot(prices.values, 'b-', label='Price')
        ax1.set_ylabel('Price ($)')
        ax1.set_title('Live Price and Predictions')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot accuracy over time
        if len(self.accuracy_history) > 0:
            ax2.plot(list(self.accuracy_history), 'g-', label='Accuracy')
            ax2.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Random Baseline')
            ax2.set_ylabel('Accuracy')
            ax2.set_xlabel('Predictions')
            ax2.set_title('Model Accuracy Over Time')
            ax2.set_ylim(0, 1)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.pause(0.01)
    
    def print_final_statistics(self, total_predictions, correct_predictions, 
                              class_predictions, class_correct):
        """Print final performance statistics"""
        print(f"\n{'='*60}")
        print("FINAL STATISTICS")
        print(f"{'='*60}")
        
        if total_predictions > 0:
            overall_accuracy = correct_predictions / total_predictions
            print(f"\nOverall Accuracy: {overall_accuracy:.2%} ({correct_predictions}/{total_predictions})")
            
            print("\nPer-Class Performance:")
            for class_idx in range(3):
                class_name = self.class_names[class_idx]
                predictions = class_predictions.get(class_idx, 0)
                
                if predictions > 0:
                    class_acc = class_correct.get(class_idx, 0) / predictions
                    print(f"  {class_name}: {class_acc:.2%} ({class_correct.get(class_idx, 0)}/{predictions})")
                else:
                    print(f"  {class_name}: No predictions made")
            
            print("\nPrediction Distribution:")
            for class_idx in range(3):
                class_name = self.class_names[class_idx]
                count = sum(1 for p in self.predictions_history if p['prediction'] == class_idx)
                percentage = count / len(self.predictions_history) * 100
                print(f"  {class_name}: {count} ({percentage:.1f}%)")
    
    def plot_confusion_matrix(self, dataset, start_idx):
        """Plot confusion matrix"""
        predictions = []
        true_labels = []
        
        for i, pred_info in enumerate(self.predictions_history):
            if pred_info['true_label'] is not None:
                predictions.append(pred_info['prediction'])
                true_labels.append(pred_info['true_label'])
        
        if len(predictions) > 0:
            cm = confusion_matrix(true_labels, predictions)
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['HOLD', 'BUY', 'SELL'],
                       yticklabels=['HOLD', 'BUY', 'SELL'])
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.show()
    
    def batch_inference(self, csv_path, output_path=None):
        """Run batch inference on entire CSV file"""
        print(f"\nRunning batch inference on: {csv_path}")
        
        # Load data
        df = pd.read_csv(csv_path)
        dataset = InferenceDataset(df)
        
        # Create dataloader
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        all_predictions = []
        all_probabilities = []
        all_true_labels = []
        
        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                
                logits = self.model(batch_x)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                
                all_predictions.extend(preds.cpu().numpy())
                all_probabilities.extend(probs.cpu().numpy())
                all_true_labels.extend(batch_y.numpy())
        
        # Create results dataframe
        results_df = df.iloc[dataset.seq_len:].copy()
        results_df['predicted_class'] = all_predictions
        results_df['predicted_signal'] = [self.class_names[p] for p in all_predictions]
        results_df['buy_prob'] = [p[1] for p in all_probabilities]
        results_df['hold_prob'] = [p[0] for p in all_probabilities]
        results_df['sell_prob'] = [p[2] for p in all_probabilities]
        
        # Add accuracy if true labels exist
        if 'signal_class' in results_df.columns:
            results_df['correct'] = results_df['predicted_class'] == results_df['signal_class']
            accuracy = results_df['correct'].mean()
            print(f"Batch Accuracy: {accuracy:.2%}")
            
            # Print classification report
            print("\nClassification Report:")
            print(classification_report(
                results_df['signal_class'], 
                results_df['predicted_class'],
                target_names=['HOLD', 'BUY', 'SELL']
            ))
        
        # Save results if output path provided
        if output_path:
            results_df.to_csv(output_path, index=False)
            print(f"Results saved to: {output_path}")
        
        return results_df


def main():
    """Main function to run inference"""
    import argparse
    
    parser = argparse.ArgumentParser(description='LSTM Trading Model Inference')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model file')
    parser.add_argument('--csv-path', type=str, required=True,
                       help='Path to CSV file for inference')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint file (optional)')
    parser.add_argument('--mode', type=str, choices=['live', 'batch'], default='live',
                       help='Inference mode: live or batch')
    parser.add_argument('--start-idx', type=int, default=None,
                       help='Starting index for live inference')
    parser.add_argument('--delay', type=float, default=0.5,
                       help='Delay between predictions in live mode (seconds)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for batch inference results')
    parser.add_argument('--no-plot', action='store_true',
                       help='Disable live plotting')
    
    args = parser.parse_args()
    
    # Initialize inference system
    inference = LiveInference(
        model_path=args.model_path,
        checkpoint_path=args.checkpoint
    )
    
    # Run inference based on mode
    if args.mode == 'live':
        inference.run_live_inference(
            csv_path=args.csv_path,
            start_idx=args.start_idx,
            delay=args.delay,
            show_plot=not args.no_plot
        )
    else:
        results = inference.batch_inference(
            csv_path=args.csv_path,
            output_path=args.output
        )
        print(f"\nBatch inference completed. Processed {len(results)} samples.")


# Example usage
if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        print("No arguments provided. Using default paths...")
        inference = LiveInference(
            model_path="/lambda/nfs/LSTM/Lstm/ckpt/2/best_model.pth",
            checkpoint_path=None
        )
        
        # Run live inference with defaults
        inference.run_live_inference(
            csv_path="/lambda/nfs/LSTM/Lstm/data/alpaca/done/AAPL_15min.csv",
            start_idx=136886,  # Start near the end for recent data
            delay=0.2,
            show_plot=True
        )
    else:
        main()
