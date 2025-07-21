#train.py
import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from pathlib import Path
import time
import argparse
from sklearn.metrics import classification_report, confusion_matrix
import random
from LSTM import ComplexLSTMModel


class PriceDataset(Dataset):
    def __init__(self, df, seq_len=50, augment=False, noise_level=0.02):
        # Include volume in feature columns
        feature_cols = [
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
        
        # Verify all required columns exist
        missing_cols = []
        for col in feature_cols + ['signal_class']:
            if col not in df.columns:
                missing_cols.append(col)
        
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")
        
        # Extract features and handle NaN/inf values safely
        features = df[feature_cols].values.astype(float)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Robust normalization using percentiles to handle outliers
        self.percentile_5 = np.percentile(features, 5, axis=0)
        self.percentile_95 = np.percentile(features, 95, axis=0)
        features = np.clip(features, self.percentile_5, self.percentile_95)
        
        # Normalize features to [-1, 1] range
        self.min_vals = features.min(axis=0)
        self.max_vals = features.max(axis=0)
        self.range_vals = self.max_vals - self.min_vals + 1e-8
        features = 2 * (features - self.min_vals) / self.range_vals - 1
        
        # Validate and process labels
        labels = df['signal_class'].values
        if np.any(pd.isnull(labels)):
            raise ValueError("signal_class contains NaN values!")
        labels = labels.astype(int)
        
        # Create sequences
        self.X = []
        self.y = []
        for i in range(len(features) - seq_len):
            seq = features[i:i + seq_len]
            label = labels[i + seq_len]
            self.X.append(seq)
            self.y.append(int(label))
        
        self.X = np.array(self.X)
        self.y = np.array(self.y)
        
        # Store parameters
        self.feature_cols = feature_cols
        self.seq_len = seq_len
        self.augment = augment
        self.noise_level = noise_level
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx].copy()
        y = self.y[idx]
        
        # Data augmentation for training
        if self.augment and np.random.random() < 0.5:
            # Add Gaussian noise
            noise = np.random.normal(0, self.noise_level, x.shape)
            x = x + noise
            
            # Random scaling
            scale = np.random.uniform(0.98, 1.02)
            x = x * scale
            
            # Random time shift (small)
            if np.random.random() < 0.3:
                shift = np.random.randint(-2, 3)
                if shift != 0:
                    x = np.roll(x, shift, axis=0)
            
            # Clip to valid range
            x = np.clip(x, -1.5, 1.5)
        
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


def load_all_csvs(data_dir):
    """Load and combine all CSV files from directory"""
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not csv_files:
        raise ValueError(f"No CSV files found in {data_dir}")
    
    df_list = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            df_list.append(df)
            print(f"Loaded {f}: {len(df)} rows")
        except Exception as e:
            print(f"Error loading {f}: {e}")
    
    if not df_list:
        raise ValueError("No CSV files could be loaded successfully.")
    
    combined = pd.concat(df_list, ignore_index=True)
    return combined


def create_time_series_splits(dataset, n_splits=5, val_ratio=0.2):
    """Create proper time series cross-validation splits"""
    n_samples = len(dataset)
    splits = []
    
    for i in range(n_splits):
        # Calculate split points
        test_start = int(n_samples * (i + 1) / (n_splits + 1))
        test_end = int(n_samples * (i + 2) / (n_splits + 1))
        val_start = test_start - int((test_end - test_start) * val_ratio)
        
        train_indices = list(range(0, val_start))
        val_indices = list(range(val_start, test_start))
        test_indices = list(range(test_start, test_end))
        
        splits.append({
            'train': train_indices,
            'val': val_indices,
            'test': test_indices
        })
    
    return splits


def calculate_class_weights(labels, n_classes, device, smoothing=0.1):
    """Calculate class weights with smoothing for handling imbalanced data"""
    class_counts = np.bincount(labels, minlength=n_classes)
    total_samples = len(labels)
    
    # Inverse frequency weighting with smoothing
    class_weights = []
    for i in range(n_classes):
        if class_counts[i] > 0:
            weight = total_samples / (n_classes * class_counts[i])
            # Apply smoothing to prevent extreme weights
            weight = (1 - smoothing) * weight + smoothing
            class_weights.append(weight)
        else:
            class_weights.append(1.0)
    
    # Normalize weights
    class_weights = np.array(class_weights)
    class_weights = class_weights / class_weights.mean()
    
    return torch.tensor(class_weights, dtype=torch.float32, device=device)


def mixup_data(x, y, alpha=0.2):
    """Mixup augmentation for better generalization"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss calculation"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def evaluate_model(model, loader, loss_fn, device, n_classes):
    """Evaluate model and return detailed metrics"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, dtype=torch.float32)
            batch_y = batch_y.to(device)
            
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Calculate metrics
    accuracy = (all_preds == all_targets).mean()
    
    # Per-class accuracy
    class_accuracies = {}
    for cls in range(n_classes):
        mask = all_targets == cls
        if mask.sum() > 0:
            cls_acc = (all_preds[mask] == all_targets[mask]).mean()
            class_accuracies[cls] = cls_acc
    
    # Confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'class_accuracies': class_accuracies,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'targets': all_targets
    }


def run_training_phase(model, dataset, train_indices, val_indices, loss_fn, device,
                      n_classes, phase_config, save_dir, checkpoint_path, args):
    """Run a single training phase with proper regularization"""
    
    # Create subset datasets
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    # Set augmentation for training dataset
    if hasattr(dataset, 'augment'):
        dataset.augment = True
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=phase_config['batch_size'],
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True  # Drop last incomplete batch
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=phase_config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Optimizer with L2 regularization
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=phase_config['lr'],
        weight_decay=phase_config['weight_decay'],
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Learning rate scheduler with warm restarts
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,  # Restart every 10 epochs
        T_mult=2,  # Double the period after each restart
        eta_min=phase_config['lr'] * 0.01
    )
    
    # Early stopping parameters
    patience = 1000
    patience_counter = 0
    best_val_loss = float('inf')
    best_val_acc = 0
    
    # Training history
    training_history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # Mixed precision training
    scaler = GradScaler()
    
    # Label smoothing
    label_smoothing = 0.1
    
    print(f"Starting training with learning rate: {phase_config['lr']}")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    for epoch in range(phase_config['epochs']):
        # Training phase
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        start_time = time.time()
        
        for step, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = batch_x.to(device, dtype=torch.float32)
            batch_y = batch_y.to(device)
            
            # Apply mixup augmentation
            if np.random.random() < 0.3:  # 30% chance of mixup
                batch_x, y_a, y_b, lam = mixup_data(batch_x, batch_y, alpha=0.2)
                
                optimizer.zero_grad()
                with autocast():
                    logits = model(batch_x)
                    loss = mixup_criterion(loss_fn, logits, y_a, y_b, lam)
            else:
                optimizer.zero_grad()
                with autocast():
                    logits = model(batch_x)
                    loss = loss_fn(logits, batch_y)
            
            # Add L1 regularization
            l1_lambda = 1e-5
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            loss = loss + l1_lambda * l1_norm
            
            # Check for NaN loss
            if torch.isnan(loss):
                print(f"NaN loss detected at epoch {epoch}, step {step}")
                return False
            
            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=phase_config['grad_clip'])
            
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            
            # Calculate accuracy
            with torch.no_grad():
                _, predicted = logits.max(1)
                total += batch_y.size(0)
                correct += predicted.eq(batch_y).sum().item()
            
            # Progress logging
            if step % 50 == 0 and step > 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1}/{phase_config['epochs']}, "
                      f"Step {step}/{len(train_loader)}, "
                      f"Loss: {loss.item():.5f}, "
                      f"LR: {current_lr:.6f}")
        
        # Calculate epoch metrics
        train_loss = total_loss / len(train_loader)
        train_acc = 100. * correct / total
        epoch_time = time.time() - start_time
        
        # Validation phase
        val_metrics = evaluate_model(model, val_loader, loss_fn, device, n_classes)
        val_loss = val_metrics['loss']
        val_acc = val_metrics['accuracy'] * 100
        
        # Update learning rate
        #scheduler.step()
        
        # Log epoch results
        print(f"\nEpoch {epoch+1}/{phase_config['epochs']} completed in {epoch_time:.2f}s")
        print(f"Train Loss: {train_loss:.5f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.5f}, Val Acc: {val_acc:.2f}%")
        
        # Check for overfitting
        overfit_gap = train_acc - val_acc
        if overfit_gap > 15:
            print(f"WARNING: Overfitting detected! Gap: {overfit_gap:.2f}%")
            
            # Apply additional dropout dynamically
            if hasattr(model, 'dropout'):
                for module in model.modules():
                    if isinstance(module, nn.Dropout):
                        module.p = min(module.p + 0.05, 0.7)
        
        # Save training history
        training_history['train_loss'].append(train_loss)
        training_history['train_acc'].append(train_acc)
        training_history['val_loss'].append(val_loss)
        training_history['val_acc'].append(val_acc)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
            
            # Save best model
            best_model_path = save_dir / "best_model.pth"
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'optimizer_state_dict': optimizer.state_dict(),
                'training_history': training_history
            }, best_model_path)
            print(f"New best model saved! Val Loss: {val_loss:.5f}, Val Acc: {val_acc:.2f}%")
            
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                print(f"Best Val Loss: {best_val_loss:.5f}, Best Val Acc: {best_val_acc:.2f}%")
                break
        
        # Save checkpoint periodically
        if (epoch + 1) % phase_config['ckpt_freq'] == 0:
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'training_history': training_history
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch+1}")
    
    # Save final model
    final_path = save_dir / "final_model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'training_history': training_history,
        'final_epoch': epoch + 1,
        'best_val_loss': best_val_loss,
        'best_val_acc': best_val_acc
    }, final_path)
    
    print(f"\nPhase training completed!")
    print(f"Best Val Loss: {best_val_loss:.5f}, Best Val Acc: {best_val_acc:.2f}%")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Fixed LSTM Training with Proper Regularization')
    parser.add_argument('--data-dir', type=str, default="/lambda/nfs/LSTM/Lstm/data/alpaca/done/done",
                       help='Directory containing CSV files')
    parser.add_argument('--save-dir', type=str, default="/lambda/nfs/LSTM/Lstm/ckpt/3/",
                       help='Directory to save models and checkpoints')
    parser.add_argument('--seq-len', type=int, default=50, help='Sequence length')
    
    # Model parameters
    parser.add_argument('--hidden-dim', type=int, default=128, help='Hidden dimension size')
    parser.add_argument('--num-layers', type=int, default=3, help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=1024, help='Batch size')
    parser.add_argument('--epochs', type=int, default=70, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.003, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--grad-clip', type=float, default=1.0, help='Gradient clipping')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--n-splits', type=int, default=5, help='Number of time series splits')
    parser.add_argument('--augment', action='store_true', help='Use data augmentation')
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"Loading data from {args.data_dir}")
    df = load_all_csvs(args.data_dir)
    print(f"Loaded {len(df):,} total rows")
    
    # Sort by date if datetime column exists
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime').reset_index(drop=True)
        print("Data sorted by datetime")
    
    # Check class distribution
    class_counts = df['signal_class'].value_counts().sort_index()
    print("\nClass distribution in raw data:")
    for cls, count in class_counts.items():
        print(f"  Class {cls}: {count:,} samples ({count/len(df)*100:.1f}%)")
    
    # Create dataset with augmentation
    dataset = PriceDataset(df, seq_len=args.seq_len, augment=args.augment)
    print(f"\nDataset created: {len(dataset):,} samples after windowing")
    
    # Get number of classes
    n_classes = int(df['signal_class'].dropna().nunique())
    input_dim = 27
    
    # Create time series splits
    splits = create_time_series_splits(dataset, n_splits=args.n_splits)
    
    # Use the first split for training
    split = splits[0]
    train_indices = split['train']
    val_indices = split['val']
    
    print(f"\nUsing first time series split:")
    print(f"  Train: {len(train_indices)} samples")
    print(f"  Val: {len(val_indices)} samples")
    
    # Create model with regularization
    model = ComplexLSTMModel(
        input_dim=input_dim,
        output_dim=2,
        hidden_dim=args.hidden_dim,
        num_lstm_layers=args.num_layers,
        dropout=args.dropout
    )
    model = model.to(device).float()
    
    # Calculate class weights
    train_labels = dataset.y[train_indices]
    class_weights = calculate_class_weights(train_labels, n_classes, device, smoothing=0.1)
    print(f"\nClass weights: {class_weights.tolist()}")
    
    # Loss function with label smoothing
    loss_fn = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    
    # Training configuration
    phase_config = {
        'name': 'REGULARIZED TRAINING',
        'epochs': args.epochs,
        'lr': args.lr,
        'batch_size': args.batch_size,
        'weight_decay': args.weight_decay,
        'grad_clip': args.grad_clip,
        'ckpt_freq': 10,
        'eval_freq': 10
    }
    
    # Define checkpoint path
    checkpoint_path = save_dir / "checkpoint.pth"
    
    # Run training
    print("\n" + "="*60)
    print("STARTING REGULARIZED TRAINING")
    print("="*60)
    
    success = run_training_phase(
        model, dataset, train_indices, val_indices, loss_fn, device,
        n_classes, phase_config, save_dir, checkpoint_path, args
    )
    
    if success:
        print("\nTraining completed successfully!")
        print(f"Models saved to: {save_dir}")
        
        # Evaluate on test set (optional)
        if len(splits) > 1:
            print("\nEvaluating on test set...")
            test_indices = splits[0]['test']
            test_dataset = torch.utils.data.Subset(dataset, test_indices)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
            
            # Load best model
            best_checkpoint = torch.load(save_dir / "best_model.pth",weights_only=False)
            model.load_state_dict(best_checkpoint['model_state_dict'])
            
            test_metrics = evaluate_model(model, test_loader, loss_fn, device, n_classes)
            print(f"Test Accuracy: {test_metrics['accuracy']*100:.2f}%")
            print("\nConfusion Matrix:")
            print(test_metrics['confusion_matrix'])
    else:
        print("\nTraining failed!")


if __name__ == "__main__":
    main()
