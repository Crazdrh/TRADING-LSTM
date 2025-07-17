import argparse
import os
import time
import json
from pathlib import Path
from contextlib import contextmanager
from typing import Dict, Any, Optional, Tuple
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
from torch.distributed.fsdp.api import FullStateDictConfig, StateDictType
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import pandas as pd
import numpy as np
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
from LSTM import ComplexLSTMModel
import warnings
warnings.filterwarnings('ignore')

@contextmanager
def rank_ordered(*, should_go_first: bool):
    """Context manager to ensure rank-ordered execution"""
    if should_go_first:
        yield
    if dist.is_initialized():
        dist.barrier()
    if not should_go_first:
        yield
    if dist.is_initialized():
        dist.barrier()

class LocalTimer:
    """High-precision timer for performance monitoring"""
    def __init__(self, device: torch.device):
        if device.type == "cpu":
            self.synchronize = lambda: torch.cpu.synchronize()
        elif device.type == "cuda":
            self.synchronize = lambda: torch.cuda.synchronize(device=device)
        else:
            self.synchronize = lambda: None
        self.measurements = []
        self.start_time = None

    def __enter__(self):
        self.synchronize()
        self.start_time = time.time()
        return self

    def __exit__(self, type, value, traceback):
        if traceback is None:
            self.synchronize()
            end_time = time.time()
            self.measurements.append(end_time - self.start_time)
        self.start_time = None

    def avg_elapsed_ms(self):
        return 1000 * (sum(self.measurements) / len(self.measurements)) if self.measurements else 0

    def reset(self):
        self.measurements = []
        self.start_time = None

class PriceDataset(Dataset):
    """Enhanced dataset with better error handling and validation"""
    def __init__(self, df: pd.DataFrame, seq_len: int = 50, feature_cols: Optional[list] = None):
        if feature_cols is None:
            feature_cols = ['open', 'high', 'low', 'close', 'MA', 'MA.1', 'MA.2', 'MA.3', 'MA.4']
        
        # Clean column names
        df.columns = [c.strip() for c in df.columns]
        
        # Validate required columns
        missing_cols = [col for col in feature_cols + ['signal_class'] if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")
        
        # Process features
        features = df[feature_cols].values.astype(np.float32)
        
        # Handle infinities and NaN values
        features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Robust normalization
        feature_means = np.nanmean(features, axis=0)
        feature_stds = np.nanstd(features, axis=0) + 1e-8
        features = (features - feature_means) / feature_stds
        
        # Validate and process labels
        labels = df['signal_class'].values
        if np.any(pd.isnull(labels)):
            raise ValueError("signal_class contains NaN values!")
        
        labels = labels.astype(np.int64)
        
        # Create sequences
        self.sequences = []
        self.targets = []
        
        for i in range(len(features) - seq_len):
            seq = features[i:i + seq_len]
            target = labels[i + seq_len]
            
            # Additional validation
            if not np.isfinite(seq).all():
                continue
                
            self.sequences.append(seq)
            self.targets.append(target)
        
        # Convert to tensors
        self.sequences = torch.tensor(np.array(self.sequences), dtype=torch.float32)
        self.targets = torch.tensor(np.array(self.targets), dtype=torch.long)
        
        # Store normalization stats
        self.feature_means = feature_means
        self.feature_stds = feature_stds
        self.seq_len = seq_len
        self.feature_cols = feature_cols
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

def load_all_csvs(data_dir: str) -> pd.DataFrame:
    """Load and combine all CSV files with better error handling"""
    import glob
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    if not csv_files:
        raise ValueError(f"No CSV files found in {data_dir}")
    
    df_list = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            if len(df) > 0:
                df_list.append(df)
        except Exception as e:
            print(f"Warning: Failed to load {csv_file}: {e}")
    
    if not df_list:
        raise ValueError("No valid CSV files could be loaded")
    
    combined_df = pd.concat(df_list, ignore_index=True)
    return combined_df

def calculate_class_weights(df: pd.DataFrame, n_classes: int, method: str = "balanced") -> torch.Tensor:
    """Calculate class weights for handling imbalanced datasets"""
    class_counts = df['signal_class'].value_counts().sort_index()
    total_samples = len(df)
    
    if method == "balanced":
        # sklearn-style balanced weights
        weights = []
        for cls in range(n_classes):
            if cls in class_counts:
                weight = total_samples / (n_classes * class_counts[cls])
                weights.append(weight)
            else:
                weights.append(1.0)
    elif method == "inverse_freq":
        # Simple inverse frequency
        weights = []
        for cls in range(n_classes):
            if cls in class_counts:
                weight = total_samples / class_counts[cls]
                weights.append(weight)
            else:
                weights.append(1.0)
    else:
        weights = [1.0] * n_classes
    
    return torch.tensor(weights, dtype=torch.float32)

def save_checkpoint(model: FSDP, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler._LRScheduler,
                   epoch: int, global_step: int, save_path: Path, rank: int, 
                   metrics: Dict[str, float] = None) -> None:
    """Enhanced checkpoint saving with metrics"""
    if rank == 0:
        print(f"Saving checkpoint to {save_path}")
    
    # Configure FSDP to gather full state dict
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, 
                             FullStateDictConfig(offload_to_cpu=True, rank0_only=True)):
        model_state = model.state_dict()
    
    # Only rank 0 saves the checkpoint
    if rank == 0:
        checkpoint = {
            "model": model_state,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "metrics": metrics or {},
            "timestamp": time.time()
        }
        
        # Atomic save
        temp_path = save_path.with_suffix('.tmp')
        torch.save(checkpoint, temp_path)
        temp_path.rename(save_path)
        
        print(f"Checkpoint saved successfully to {save_path}")

def load_checkpoint(model: FSDP, optimizer: torch.optim.Optimizer, 
                   scheduler: torch.optim.lr_scheduler._LRScheduler,
                   checkpoint_path: Path, device: torch.device, rank: int) -> Tuple[int, int, Dict[str, float]]:
    """Enhanced checkpoint loading"""
    if rank == 0:
        print(f"Loading checkpoint from {checkpoint_path}")
    
    # Load checkpoint on all ranks
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state dict
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, 
                             FullStateDictConfig(offload_to_cpu=True, rank0_only=True)):
        model.load_state_dict(checkpoint["model"])
    
    # Load optimizer and scheduler state
    optimizer.load_state_dict(checkpoint["optimizer"])
    if "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    
    epoch = checkpoint.get("epoch", 0)
    global_step = checkpoint.get("global_step", 0)
    metrics = checkpoint.get("metrics", {})
    
    if rank == 0:
        print(f"Loaded checkpoint: epoch={epoch}, step={global_step}")
        if metrics:
            print(f"Previous metrics: {metrics}")
    
    return epoch, global_step, metrics

class TrainingPhaseConfig:
    """Configuration for different training phases"""
    def __init__(self, name: str, epochs: int, lr: float, batch_size: int,
                 weight_decay: float = 1e-5, grad_clip: float = 1.0,
                 ckpt_freq: int = 10, eval_freq: int = 5):
        self.name = name
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.ckpt_freq = ckpt_freq
        self.eval_freq = eval_freq

def get_training_phases() -> Dict[str, TrainingPhaseConfig]:
    """Define training phases with optimized parameters"""
    return {
        "phase1": TrainingPhaseConfig(
            name="Initial Training",
            epochs=50,
            lr=0.001,
            batch_size=1000,
            weight_decay=1e-5,
            grad_clip=1.0,
            ckpt_freq=10,
            eval_freq=5
        ),
        "phase2": TrainingPhaseConfig(
            name="Deep Learning",
            epochs=120,
            lr=0.0007,
            batch_size=1000,
            weight_decay=2e-5,
            grad_clip=0.8,
            ckpt_freq=8,
            eval_freq=3
        ),
        "phase3": TrainingPhaseConfig(
            name="Fine-tuning",
            epochs=180,
            lr=0.0003,
            batch_size=800,
            weight_decay=3e-5,
            grad_clip=0.5,
            ckpt_freq=5,
            eval_freq=2
        ),
        "phase4": TrainingPhaseConfig(
            name="Final Optimization",
            epochs=220,
            lr=0.0001,
            batch_size=600,
            weight_decay=5e-5,
            grad_clip=0.3,
            ckpt_freq=3,
            eval_freq=1
        )
    }

def adjust_batch_size_for_gpu_memory(base_batch_size: int, world_size: int) -> int:
    """Adjust batch size based on available GPU memory"""
    if torch.cuda.is_available():
        # Get GPU memory info
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        
        # Heuristic: adjust batch size based on GPU memory
        if gpu_memory < 8:
            return max(base_batch_size // 2, 64)
        elif gpu_memory > 24:
            return min(base_batch_size * 2, 2048)
    
    return base_batch_size

def calculate_metrics(predictions: np.ndarray, targets: np.ndarray, n_classes: int) -> Dict[str, float]:
    """Calculate comprehensive metrics"""
    accuracy = (predictions == targets).mean()
    
    metrics = {"accuracy": accuracy}
    
    # Per-class metrics
    for cls in range(n_classes):
        mask = targets == cls
        if mask.sum() > 0:
            cls_acc = (predictions[mask] == targets[mask]).mean()
            metrics[f"class_{cls}_accuracy"] = cls_acc
            metrics[f"class_{cls}_samples"] = mask.sum()
        else:
            metrics[f"class_{cls}_accuracy"] = 0.0
            metrics[f"class_{cls}_samples"] = 0
    
    # Prediction distribution
    unique_preds, pred_counts = np.unique(predictions, return_counts=True)
    for cls, count in zip(unique_preds, pred_counts):
        metrics[f"pred_class_{cls}_count"] = count
    
    return metrics

def train_epoch(model: FSDP, loader: DataLoader, optimizer: torch.optim.Optimizer,
                loss_fn: nn.Module, device: torch.device, rank: int, epoch: int,
                grad_clip: float = 1.0) -> Tuple[float, Dict[str, float]]:
    """Train for one epoch with comprehensive monitoring"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    all_predictions = []
    all_targets = []
    
    timer = LocalTimer(device)
    
    for step, (batch_x, batch_y) in enumerate(loader):
        with timer:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            
            # Check for NaN loss
            if torch.isnan(loss) or torch.isinf(loss):
                if rank == 0:
                    print(f"[RANK {rank}] Invalid loss at epoch {epoch} step {step}: {loss.item()}")
                    print(f"Batch X stats: min={batch_x.min():.6f}, max={batch_x.max():.6f}, mean={batch_x.mean():.6f}")
                    print(f"Batch Y unique: {batch_y.unique()}")
                continue
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if grad_clip > 0:
                clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Collect predictions for metrics
            with torch.no_grad():
                preds = torch.argmax(logits, dim=1)
                all_predictions.extend(preds.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())
        
        # Progress logging
        if rank == 0 and (step % 50 == 0 or step == len(loader) - 1):
            avg_time = timer.avg_elapsed_ms()
            print(f"[Epoch {epoch}] Step {step+1}/{len(loader)} | "
                  f"Loss: {loss.item():.5f} | "
                  f"Avg step time: {avg_time:.1f}ms")
    
    avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    
    # Calculate metrics
    predictions = np.array(all_predictions)
    targets = np.array(all_targets)
    n_classes = len(np.unique(targets))
    
    metrics = calculate_metrics(predictions, targets, n_classes)
    metrics["avg_loss"] = avg_loss
    
    return avg_loss, metrics

def run_training_phase(model: FSDP, loader: DataLoader, optimizer: torch.optim.Optimizer,
                      scheduler: torch.optim.lr_scheduler._LRScheduler, loss_fn: nn.Module,
                      device: torch.device, rank: int, world_size: int,
                      phase_config: TrainingPhaseConfig, save_dir: Path,
                      start_epoch: int = 0, global_step: int = 0) -> Tuple[int, Dict[str, float]]:
    """Run a complete training phase"""
    
    if rank == 0:
        print(f"\n{'='*80}")
        print(f"STARTING {phase_config.name.upper()}")
        print(f"Epochs: {start_epoch + 1} -> {phase_config.epochs}")
        print(f"Learning Rate: {phase_config.lr}")
        print(f"Batch Size: {phase_config.batch_size}")
        print(f"Weight Decay: {phase_config.weight_decay}")
        print(f"Gradient Clip: {phase_config.grad_clip}")
        print(f"{'='*80}")
    
    # Update optimizer parameters
    for param_group in optimizer.param_groups:
        param_group['lr'] = phase_config.lr
        param_group['weight_decay'] = phase_config.weight_decay
    
    phase_start_time = time.time()
    best_loss = float('inf')
    
    for epoch in range(start_epoch, phase_config.epochs):
        epoch_start_time = time.time()
        
        # Set epoch for distributed sampler
        if hasattr(loader.sampler, 'set_epoch'):
            loader.sampler.set_epoch(epoch)
        
        # Train epoch
        avg_loss, metrics = train_epoch(
            model, loader, optimizer, loss_fn, device, rank, 
            epoch + 1, phase_config.grad_clip
        )
        
        # Learning rate scheduling
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_loss)
        new_lr = optimizer.param_groups[0]['lr']
        
        epoch_time = time.time() - epoch_start_time
        
        if rank == 0:
            print(f"\nEpoch {epoch + 1}/{phase_config.epochs} Summary:")
            print(f"  Loss: {avg_loss:.5f}")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Time: {epoch_time:.1f}s")
            
            if old_lr != new_lr:
                print(f"  Learning rate: {old_lr:.6f} -> {new_lr:.6f}")
            
            # Per-class accuracy
            n_classes = len([k for k in metrics.keys() if k.startswith('class_') and k.endswith('_accuracy')])
            for cls in range(n_classes):
                if f"class_{cls}_samples" in metrics and metrics[f"class_{cls}_samples"] > 0:
                    cls_acc = metrics[f"class_{cls}_accuracy"]
                    cls_samples = int(metrics[f"class_{cls}_samples"])
                    print(f"  Class {cls}: {cls_acc:.4f} ({cls_samples} samples)")
        
        # Save checkpoint
        if (epoch + 1) % phase_config.ckpt_freq == 0:
            ckpt_path = save_dir / f"checkpoint_epoch_{epoch + 1}.pt"
            save_checkpoint(model, optimizer, scheduler, epoch + 1, global_step, 
                          ckpt_path, rank, metrics)
        
        # Track best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            if rank == 0:
                best_path = save_dir / f"best_model_{phase_config.name.lower().replace(' ', '_')}.pt"
                save_checkpoint(model, optimizer, scheduler, epoch + 1, global_step,
                              best_path, rank, metrics)
        
        global_step += len(loader)
    
    phase_time = time.time() - phase_start_time
    
    if rank == 0:
        print(f"\n{phase_config.name} completed in {phase_time/3600:.1f} hours")
        print(f"Best loss: {best_loss:.5f}")
    
    return global_step, metrics

def main():
    parser = argparse.ArgumentParser(description="Enhanced Multi-Phase FSDP LSTM Training")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing CSV files")
    parser.add_argument("--save-dir", type=str, required=True, help="Directory to save checkpoints")
    parser.add_argument("--seq-len", type=int, default=50, help="Sequence length")
    parser.add_argument("--base-batch-size", type=int, default=1000, help="Base batch size per GPU")
    parser.add_argument("--phase", type=str, choices=["phase1", "phase2", "phase3", "phase4", "all"], 
                       default="all", help="Which phase to run")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--cpu-offload", action="store_true", help="Enable CPU offloading")
    parser.add_argument("--mixed-precision", action="store_true", help="Enable mixed precision training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-workers", type=int, default=2, help="Number of data loading workers")
    parser.add_argument("--class-weight-method", type=str, choices=["balanced", "inverse_freq", "none"], 
                       default="balanced", help="Method for calculating class weights")
    
    args = parser.parse_args()
    
    # Initialize distributed training
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl')
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    # Set device
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    
    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    if rank == 0:
        print(f"Training with {world_size} GPUs")
        print(f"Device: {device}")
        print(f"Mixed precision: {args.mixed_precision}")
        print(f"CPU offload: {args.cpu_offload}")
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and prepare data
    if rank == 0:
        print("Loading data...")
    
    df = load_all_csvs(args.data_dir)
    
    if rank == 0:
        print(f"Loaded {len(df):,} samples from {args.data_dir}")
        class_counts = df['signal_class'].value_counts().sort_index()
        print("Class distribution:")
        for cls, count in class_counts.items():
            print(f"  Class {cls}: {count:,} samples ({count/len(df)*100:.1f}%)")
    
    # Create dataset
    dataset = PriceDataset(df, seq_len=args.seq_len)
    n_classes = len(df['signal_class'].unique())
    
    if rank == 0:
        print(f"Created dataset with {len(dataset):,} sequences")
        print(f"Number of classes: {n_classes}")
    
    # Calculate class weights
    class_weights = calculate_class_weights(df, n_classes, args.class_weight_method)
    class_weights = class_weights.to(device)
    
    if rank == 0:
        print(f"Class weights ({args.class_weight_method}): {class_weights.tolist()}")
    
    # Get training phases
    training_phases = get_training_phases()
    
    # Create model
    model = ComplexLSTMModel(input_dim=9, output_dim=n_classes)
    model = model.to(device)
    
    # Wrap with FSDP
    model = FSDP(
        model,
        device_id=device,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        cpu_offload=args.cpu_offload,
        mixed_precision=None,  # Configure separately if needed
        sync_module_states=True,
        use_orig_params=True,
    )
    
    # Loss function
    loss_fn = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if args.mixed_precision else None
    
    # Synchronize before training
    dist.barrier()
    
    # Run training phases
    global_step = 0
    
    if args.phase == "all":
        phases_to_run = ["phase1", "phase2", "phase3", "phase4"]
    else:
        phases_to_run = [args.phase]
    
    for phase_name in phases_to_run:
        phase_config = training_phases[phase_name]
        
        # Adjust batch size for this phase
        batch_size = adjust_batch_size_for_gpu_memory(phase_config.batch_size, world_size)
        
        # Create data loader for this phase
        sampler = DistributedSampler(dataset, shuffle=True, drop_last=True)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=True if args.num_workers > 0 else False
        )
        
        # Create optimizer for this phase
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=phase_config.lr,
            weight_decay=phase_config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Create scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=True if rank == 0 else False
        )
        
        # Handle resuming
        start_epoch = 0
        if args.resume and phase_name == phases_to_run[0]:
            start_epoch, global_step, _ = load_checkpoint(
                model, optimizer, scheduler, Path(args.resume), device, rank
            )
        
        # Run the phase
        global_step, final_metrics = run_training_phase(
            model, loader, optimizer, scheduler, loss_fn, device, rank, world_size,
            phase_config, save_dir, start_epoch, global_step
        )
        
        # Save phase completion checkpoint
        if rank == 0:
            completion_path = save_dir / f"{phase_name}_complete.pt"
            save_checkpoint(model, optimizer, scheduler, phase_config.epochs, 
                          global_step, completion_path, rank, final_metrics)
        
        # Brief pause between phases
        if phase_name != phases_to_run[-1]:
            if rank == 0:
                print("Pausing 30 seconds before next phase...")
            time.sleep(30)
    
    # Save final model
    if rank == 0:
        print("Saving final model...")
        final_path = save_dir / "final_model.pt"
        
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, 
                                 FullStateDictConfig(offload_to_cpu=True, rank0_only=True)):
            model_state = model.state_dict()
            torch.save({
                "model_state_dict": model_state,
                "model_config": {
                    "input_dim": 9,
                    "output_dim": n_classes,
                    "seq_len": args.seq_len
                },
                "training_completed": True
            }, final_path)
        
        print(f"Final model saved to {final_path}")
        print("Training completed successfully!")
    
    # Cleanup
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
