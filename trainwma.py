import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
import time
import argparse
from sklearn.metrics import classification_report, confusion_matrix
from LSTM import ComplexLSTMModel


class PriceDataset(Dataset):
    def __init__(self, df, seq_len=50):
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

        # Normalize features
        self.means = features.mean(axis=0)
        self.stds = features.std(axis=0) + 1e-8
        features = (features - self.means) / self.stds

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

        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y), dtype=torch.long)

        # Store normalization parameters for potential inference use
        self.feature_cols = feature_cols
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


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


def calculate_class_weights(labels, n_classes, device):
    """Calculate class weights for handling imbalanced data"""
    class_counts = np.bincount(labels, minlength=n_classes)
    total_samples = len(labels)

    # Inverse frequency weighting
    class_weights = []
    for i in range(n_classes):
        if class_counts[i] > 0:
            weight = total_samples / (n_classes * class_counts[i])
            class_weights.append(weight)
        else:
            class_weights.append(1.0)

    return torch.tensor(class_weights, dtype=torch.float32, device=device)


def save_checkpoint(model, optimizer, scheduler, epoch, step, loss, save_path, metrics=None):
    """Save training checkpoint, replacing any existing checkpoint"""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'epoch': epoch,
        'step': step,
        'loss': loss,
        'metrics': metrics or {}
    }

    # Create a temporary file first, then rename to avoid corruption
    temp_path = str(save_path) + '.tmp'
    torch.save(checkpoint, temp_path)

    # Rename temp file to final path (atomic operation on most filesystems)
    os.rename(temp_path, save_path)

    print(f"Checkpoint saved to {save_path}")


def load_checkpoint(model, optimizer, scheduler, checkpoint_path, device):
    """Load training checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    epoch = checkpoint.get('epoch', 0)
    step = checkpoint.get('step', 0)
    loss = checkpoint.get('loss', float('inf'))
    metrics = checkpoint.get('metrics', {})

    print(f"Loaded checkpoint: epoch={epoch}, step={step}, loss={loss:.5f}")
    return epoch, step, loss, metrics


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

    # Prediction distribution
    unique_preds, pred_counts = np.unique(all_preds, return_counts=True)
    pred_distribution = dict(zip(unique_preds, pred_counts))

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'class_accuracies': class_accuracies,
        'pred_distribution': pred_distribution,
        'predictions': all_preds,
        'targets': all_targets
    }


def get_phase_config(phase_name):
    """Get configuration for each training phase"""
    configs = {
        'initial': {
            'name': 'PHASE 1: INITIAL TRAINING',
            'epochs': 70,
            'lr': 0.003,
            'batch_size': 1500,  # Increased from 1000
            'weight_decay': 1e-5,
            'grad_clip': 1.0,
            'ckpt_freq': 10,
            'eval_freq': 5,
            'description': 'Establish basic patterns with higher learning rate'
        },
        'deep': {
            'name': 'PHASE 2: DEEP LEARNING',
            'epochs': 130,
            'lr': 0.001,
            'batch_size': 1250,  # Increased from 1000
            'weight_decay': 2e-5,
            'grad_clip': 0.8,
            'ckpt_freq': 8,
            'eval_freq': 3,
            'description': 'Deeper learning with reduced learning rate'
        },
        'fine_tune': {
            'name': 'PHASE 3: FINE-TUNING',
            'epochs': 190,
            'lr': 0.0007,
            'batch_size': 1024,  # Increased from 800
            'weight_decay': 3e-5,
            'grad_clip': 0.5,
            'ckpt_freq': 5,
            'eval_freq': 2,
            'description': 'Fine-tuning with lower learning rate'
        },
        'final': {
            'name': 'PHASE 4: FINAL OPTIMIZATION',
            'epochs': 230,
            'lr': 0.0004,
            'batch_size': 1024,  # Increased from 600
            'weight_decay': 5e-5,
            'grad_clip': 0.3,
            'ckpt_freq': 3,
            'eval_freq': 1,
            'description': 'Final optimization with very low learning rate'
        }
    }
    return configs.get(phase_name, None)


def print_phase_header(phase_config):
    """Print phase information header"""
    print(f"\n{'=' * 60}")
    print(f"{phase_config['name']}")
    print(f"{'=' * 60}")
    print(f"Description: {phase_config['description']}")
    print(f"Target Epochs: {phase_config['epochs']}")
    print(f"Learning Rate: {phase_config['lr']} (FIXED - will not be reduced)")
    print(f"Batch Size: {phase_config['batch_size']}")
    print(f"Weight Decay: {phase_config['weight_decay']}")
    print(f"Gradient Clip: {phase_config['grad_clip']}")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(description='Enhanced LSTM Training with Multi-Phase Support')
    parser.add_argument('--data-dir', type=str, default="/lambda/nfs/LSTM/Lstm/data/alpaca/done/done",
                        help='Directory containing CSV files')
    parser.add_argument('--save-dir', type=str, default="/lambda/nfs/LSTM/Lstm/ckpt/2/",
                        help='Directory to save models and checkpoints')
    parser.add_argument('--seq-len', type=int, default=50, help='Sequence length')

    # Training phase arguments
    parser.add_argument('--phase', type=str, choices=['initial', 'deep', 'fine_tune', 'final', 'custom'],
                        default='custom', help='Training phase to use')
    parser.add_argument('--multi-phase', action='store_true',
                        help='Run all phases sequentially (overrides --phase)')

    # Custom hyperparameters (used when phase='custom' or overriding phase defaults)
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size (overrides phase default)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs (overrides phase default)')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate (overrides phase default)')
    parser.add_argument('--weight-decay', type=float, default=None, help='Weight decay (overrides phase default)')
    parser.add_argument('--grad-clip', type=float, default=None,
                        help='Gradient clipping norm (overrides phase default)')
    parser.add_argument('--ckpt-freq', type=int, default=None, help='Checkpoint frequency (overrides phase default)')
    parser.add_argument('--eval-freq', type=int, default=None, help='Evaluation frequency (overrides phase default)')

    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--train-split', type=float, default=0.8, help='Training split ratio')

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

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

    # Check class distribution
    class_counts = df['signal_class'].value_counts().sort_index()
    print("Class distribution in raw data:")
    for cls, count in class_counts.items():
        print(f"  Class {cls}: {count:,} samples ({count / len(df) * 100:.1f}%)")

    # Create dataset
    dataset = PriceDataset(df, seq_len=args.seq_len)
    print(f"Dataset created: {len(dataset):,} samples after windowing")

    # Check class distribution after windowing
    unique_labels, counts = torch.unique(dataset.y, return_counts=True)
    print("Class distribution after windowing:")
    for cls, count in zip(unique_labels.tolist(), counts.tolist()):
        print(f"  Class {cls}: {count:,} samples ({count / len(dataset) * 100:.1f}%)")

    # Split data
    train_size = int(args.train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    print(f"Train samples: {len(train_dataset):,}")
    print(f"Validation samples: {len(val_dataset):,}")

    # Model setup
    n_classes = int(df['signal_class'].dropna().nunique())
    input_dim = 27  # Updated to include volume

    model = ComplexLSTMModel(input_dim=input_dim, output_dim=n_classes)
    model = model.to(device).float()

    # Calculate class weights
    class_weights = calculate_class_weights(dataset.y.numpy(), n_classes, device)
    print(f"Class weights: {class_weights.tolist()}")

    # Loss function with class weights
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    # Define checkpoint path
    checkpoint_path = save_dir / "checkpoint.pth"

    # Multi-phase training
    if args.multi_phase:
        print("\n" + "=" * 60)
        print("STARTING MULTI-PHASE TRAINING")
        print("=" * 60)
        print("Estimated Total Time: 11-22 hours")
        print("=" * 60)

        phases = ['initial', 'deep', 'fine_tune', 'final']
        total_start_time = time.time()

        for i, phase_name in enumerate(phases):
            phase_config = get_phase_config(phase_name)
            print_phase_header(phase_config)

            # Run training for this phase
            success = run_training_phase(
                model, dataset, train_dataset, val_dataset, loss_fn, device,
                n_classes, phase_config, save_dir, checkpoint_path, args
            )

            if not success:
                print(f"\nTraining failed at {phase_config['name']}")
                return False

            print(f"\n{phase_config['name']} completed successfully")

            # Brief pause between phases
            if i < len(phases) - 1:
                print("Pausing for 30 seconds before next phase...")
                time.sleep(30)

        total_time = time.time() - total_start_time
        print(f"\n{'=' * 60}")
        print("COMPLETE MULTI-PHASE TRAINING FINISHED!")
        print(f"Total training time: {total_time / 3600:.1f} hours")
        print(f"Final models saved to: {save_dir}")
        print(f"{'=' * 60}")

    else:
        # Single phase training
        if args.phase != 'custom':
            phase_config = get_phase_config(args.phase)
            print_phase_header(phase_config)
        else:
            phase_config = {
                'name': 'CUSTOM TRAINING',
                'epochs': args.epochs,
                'lr': args.lr or 0.001,
                'batch_size': args.batch_size or 1000,
                'weight_decay': args.weight_decay or 1e-5,
                'grad_clip': args.grad_clip or 1.0,
                'ckpt_freq': args.ckpt_freq or 10,
                'eval_freq': args.eval_freq or 5,
                'description': 'Custom training configuration'
            }
            print_phase_header(phase_config)

        # Override with command line arguments if provided
        if args.batch_size is not None:
            phase_config['batch_size'] = args.batch_size
        if args.lr is not None:
            phase_config['lr'] = args.lr
        if args.weight_decay is not None:
            phase_config['weight_decay'] = args.weight_decay
        if args.grad_clip is not None:
            phase_config['grad_clip'] = args.grad_clip
        if args.ckpt_freq is not None:
            phase_config['ckpt_freq'] = args.ckpt_freq
        if args.eval_freq is not None:
            phase_config['eval_freq'] = args.eval_freq

        success = run_training_phase(
            model, dataset, train_dataset, val_dataset, loss_fn, device,
            n_classes, phase_config, save_dir, checkpoint_path, args
        )

        if success:
            print("\nTraining completed successfully!")
        else:
            print("\nTraining failed!")


def run_training_phase(model, dataset, train_dataset, val_dataset, loss_fn, device,
                       n_classes, phase_config, save_dir, checkpoint_path, args):
    """Run a single training phase"""

    # Create data loaders with phase-specific batch size
    train_loader = DataLoader(
        train_dataset,
        batch_size=phase_config['batch_size'],
        shuffle=True,
        num_workers=20,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=phase_config['batch_size'],
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    # Optimizer with phase-specific parameters - NO SCHEDULER
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=phase_config['lr'],
        weight_decay=phase_config['weight_decay']
    )

    # Remove the scheduler to maintain constant learning rate
    scheduler = None

    # Resume from checkpoint if specified or exists
    start_epoch = 0
    best_val_loss = float('inf')

    if args.resume:
        start_epoch, _, best_val_loss, _ = load_checkpoint(
            model, optimizer, scheduler, args.resume, device
        )
    elif checkpoint_path.exists():
        # Auto-resume from existing checkpoint
        print(f"Found existing checkpoint at {checkpoint_path}, resuming...")
        start_epoch, _, best_val_loss, _ = load_checkpoint(
            model, optimizer, scheduler, checkpoint_path, device
        )

    # Training loop
    print(f"Starting training with FIXED learning rate: {phase_config['lr']}")
    training_history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
    scaler = GradScaler()

    for epoch in range(start_epoch, phase_config['epochs']):
        # Training phase
        model.train()
        total_loss = 0.0
        num_batches = 0
        all_preds = []
        all_targets = []

        start_time = time.time()

        for step, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = batch_x.to(device, dtype=torch.float32)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            with autocast():
                logits = model(batch_x)
                loss = loss_fn(logits, batch_y)

            # Check for NaN loss
            if torch.isnan(loss):
                print(f"NaN loss detected at epoch {epoch}, step {step}")
                print(f"Batch X stats: min={batch_x.min()}, max={batch_x.max()}")
                print(f"Batch Y unique values: {batch_y.unique()}")
                return False

            scaler.scale(loss).backward()

            # Gradient clipping
            if phase_config['grad_clip'] > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=phase_config['grad_clip'])

            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            num_batches += 1

            # Collect predictions
            with torch.no_grad():
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())

            # Progress logging
            if step % 50 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(
                    f"Epoch {epoch + 1}/{phase_config['epochs']}, Step {step + 1}/{len(train_loader)}, Loss: {loss.item():.5f}, LR: {current_lr:.6f}")

        # Calculate training metrics
        train_loss = total_loss / num_batches
        train_accuracy = (np.array(all_preds) == np.array(all_targets)).mean()
        epoch_time = time.time() - start_time

        print(f"Epoch {epoch + 1}/{phase_config['epochs']} completed in {epoch_time:.2f}s")
        print(f"Train Loss: {train_loss:.5f}, Train Accuracy: {train_accuracy:.4f}")

        # Validation phase
        if (epoch + 1) % phase_config['eval_freq'] == 0:
            print("Running validation...")
            val_metrics = evaluate_model(model, val_loader, loss_fn, device, n_classes)

            print(f"Validation Loss: {val_metrics['loss']:.5f}, Accuracy: {val_metrics['accuracy']:.4f}")

            # Per-class validation accuracy
            for cls, acc in val_metrics['class_accuracies'].items():
                samples_count = (val_metrics['targets'] == cls).sum()
                print(f"  Class {cls} accuracy: {acc:.4f} ({samples_count} samples)")

            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                best_model_path = save_dir / "best_model.pth"
                torch.save(model.state_dict(), best_model_path)
                print(f"New best model saved: {best_model_path}")

            # Update training history
            training_history['val_loss'].append(val_metrics['loss'])
            training_history['val_accuracy'].append(val_metrics['accuracy'])

        training_history['train_loss'].append(train_loss)

        # Save checkpoint
        if (epoch + 1) % phase_config['ckpt_freq'] == 0:
            save_checkpoint(model, optimizer, scheduler, epoch + 1, 0, train_loss, checkpoint_path)

    # Save final model
    final_path = save_dir / "final_model.pth"
    torch.save(model.state_dict(), final_path)
    print(f"Final model saved to {final_path}")

    # Save training history
    import json
    history_path = save_dir / "training_history.json"
    with open(history_path, 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        history_json = {}
        for key, values in training_history.items():
            history_json[key] = [float(v) for v in values]
        json.dump(history_json, f, indent=2)

    print(f"Phase training completed! Models saved to {save_dir}")
    return True


if __name__ == "__main__":
    main()
