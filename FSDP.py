import argparse
import os
import time
from pathlib import Path
from contextlib import contextmanager
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
from torch.distributed.fsdp.api import FullStateDictConfig, StateDictType
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import pandas as pd
import numpy as np
import torch.nn as nn
from LSTM import ComplexLSTMModel

@contextmanager
def rank_ordered(*, should_go_first: bool):
    if should_go_first:
        yield
    dist.barrier()
    if not should_go_first:
        yield
    dist.barrier()

class LocalTimer:
    def __init__(self, device: torch.device):
        if device.type == "cpu":
            self.synchronize = lambda: torch.cpu.synchronize(device=device)
        elif device.type == "cuda":
            self.synchronize = lambda: torch.cuda.synchronize(device=device)
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
        return 1000 * (sum(self.measurements) / len(self.measurements))

    def reset(self):
        self.measurements = []
        self.start_time = None

# ----- Dataset Definition -----
class PriceDataset(Dataset):
    def __init__(self, df, seq_len=50):
        feature_cols = ['open', 'high', 'low', 'close', 'MA', 'MA.1', 'MA.2', 'MA.3', 'MA.4']
        df.columns = [c.strip() for c in df.columns]
        for col in feature_cols + ['signal_class']:
            if col not in df.columns:
                raise ValueError(f"Missing column: {col}")
        features = df[feature_cols].values.astype(float)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)  # SAFE!
        means = features.mean(axis=0)
        stds = features.std(axis=0) + 1e-8
        features = (features - means) / stds
        # Validate labels are int, no NaN:
        labels = df['signal_class'].values
        if np.any(pd.isnull(labels)):
            raise ValueError("signal_class has NaN!")
        labels = labels.astype(int)
        self.X = []
        self.y = []
        for i in range(len(features) - seq_len):
            seq = features[i:i+seq_len]
            label = labels[i+seq_len]
            self.X.append(seq)
            self.y.append(int(label))
        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y), dtype=torch.long)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_all_csvs(data_dir):
    import glob
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    df_list = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            df_list.append(df)
        except Exception as e:
            print(f"Error loading {f}: {e}")
    if not df_list:
        raise ValueError("No CSV files found in directory or all failed to load.")
    combined = pd.concat(df_list, ignore_index=True)
    return combined

def save_checkpoint(model, optimizer, epoch, global_step, save_path, rank):
    """Save checkpoint using FSDP's full state dict"""
    if rank == 0:
        print(f"Saving checkpoint to {save_path}")
    
    # Configure FSDP to gather full state dict
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, FullStateDictConfig(offload_to_cpu=True, rank0_only=True)):
        model_state = model.state_dict()
    
    # Only rank 0 saves the checkpoint
    if rank == 0:
        checkpoint = {
            "model": model_state,
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "step": global_step
        }
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved successfully to {save_path}")

def load_checkpoint(model, optimizer, checkpoint_path, device, rank):
    """Load checkpoint for FSDP model"""
    if rank == 0:
        print(f"Loading checkpoint from {checkpoint_path}")
    
    # Load checkpoint on all ranks
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state dict
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, FullStateDictConfig(offload_to_cpu=True, rank0_only=True)):
        model.load_state_dict(checkpoint["model"])
    
    # Load optimizer state
    optimizer.load_state_dict(checkpoint["optimizer"])
    
    epoch = checkpoint.get("epoch", 0)
    global_step = checkpoint.get("step", 0)
    
    if rank == 0:
        print(f"Loaded checkpoint: epoch={epoch}, step={global_step}")
    
    return epoch, global_step

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--seq-len", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ckpt-freq", type=int, default=1, help="Save checkpoint every N epochs")
    parser.add_argument("--ckpt-steps", type=int, default=0, help="Save checkpoint every N steps (batches)")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Initialize distributed training
    dist.init_process_group()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.manual_seed(args.seed)

    if rank == 0:
        print(f"Using {world_size} GPUs, device {device}")

    # ----- Data Loading -----
    df = load_all_csvs(args.data_dir)
    if rank == 0:
        print(f"Loaded dataframe: {len(df):,} rows from {args.data_dir}")
        
        # Check class distribution
        class_counts = df['signal_class'].value_counts().sort_index()
        print(f"Class distribution in raw data:")
        for cls, count in class_counts.items():
            print(f"  Class {cls}: {count:,} samples ({count/len(df)*100:.1f}%)")
    
    dataset = PriceDataset(df, seq_len=args.seq_len)
    if rank == 0:
        print(f"PriceDataset: {len(dataset):,} samples after sequence windowing (seq_len={args.seq_len})")
        print(f"Label classes in dataset: {dataset.y.unique().tolist()}")
        
        # Check class distribution after windowing
        unique_labels, counts = torch.unique(dataset.y, return_counts=True)
        print(f"Class distribution after windowing:")
        for cls, count in zip(unique_labels.tolist(), counts.tolist()):
            print(f"  Class {cls}: {count:,} samples ({count/len(dataset)*100:.1f}%)")
    
    sampler = DistributedSampler(dataset, shuffle=True, drop_last=True)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=2,
        pin_memory=True,
    )

    # ----- Model Setup -----
    n_classes = int(df['signal_class'].dropna().nunique())
    model = ComplexLSTMModel(input_dim=9, output_dim=n_classes)
    model = model.to(device).float()
    
    # Calculate class weights to handle imbalance
    class_counts = df['signal_class'].value_counts().sort_index()
    total_samples = len(df)
    class_weights = []
    
    for cls in range(n_classes):
        if cls in class_counts:
            weight = total_samples / (n_classes * class_counts[cls])
            class_weights.append(weight)
        else:
            class_weights.append(1.0)
    
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
    
    if rank == 0:
        print(f"Class weights: {class_weights.tolist()}")
    
    # Wrap with FSDP
    model = FSDP(
        model,
        device_id=device,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        cpu_offload=args.cpu_offload,
        sync_module_states=True,
        use_orig_params=True,  # This helps with state dict consistency
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    
    # Fixed: Remove verbose parameter which is not supported in newer PyTorch versions
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # Create save directory
    exp_dir = Path(args.save_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    global_step = 0

    # ----- Resume from checkpoint -----
    if args.resume:
        # Look for epoch checkpoints first
        ckpts = sorted(exp_dir.glob("checkpoint_epoch*.pt"))
        if not ckpts:
            # Then look for step checkpoints
            ckpts = sorted(exp_dir.glob("checkpoint_step*.pt"))
        
        if ckpts:
            last_ckpt = ckpts[-1]
            start_epoch, global_step = load_checkpoint(model, optimizer, last_ckpt, device, rank)
        else:
            if rank == 0:
                print("No checkpoint found, starting fresh.")

    # Synchronize after potential checkpoint loading
    dist.barrier()

    # ----- Training Loop -----
    for epoch in range(start_epoch, args.epochs):
        model.train()
        sampler.set_epoch(epoch)
        total_loss = 0.0
        num_batches = 0
        
        # Track predictions for epoch metrics
        all_preds = []
        all_targets = []

        for step, (batch_x, batch_y) in enumerate(loader):
            batch_x = batch_x.to(device, dtype=torch.float32)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)

            # Check for NaN loss
            if torch.isnan(loss):
                print(f"[RANK {rank}] NaN loss at epoch {epoch} step {step}. Exiting.")
                print("Batch X stats:", batch_x.min().item(), batch_x.max().item())
                print("Batch Y unique:", batch_y.unique())
                dist.destroy_process_group()
                exit(1)

            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            global_step += 1
            
            # Collect predictions for metrics
            with torch.no_grad():
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())

            # Progress logging
            if rank == 0 and (step % 10 == 0 or step == len(loader) - 1):
                print(f"[Epoch {epoch+1}] Step {step+1}/{len(loader)} (Global step {global_step}) | Loss: {loss.item():.5f}")

            # Step-based checkpointing
            if args.ckpt_steps > 0 and global_step % args.ckpt_steps == 0:
                save_path = exp_dir / f"checkpoint_step{global_step}.pt"
                save_checkpoint(model, optimizer, epoch, global_step, save_path, rank)

        # Calculate epoch metrics
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        # Calculate accuracy and per-class metrics
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        accuracy = (all_preds == all_targets).mean()
        
        if rank == 0:
            print(f"Epoch {epoch+1}/{args.epochs} completed | Average Loss: {avg_loss:.5f} | Accuracy: {accuracy:.4f}")
            
            # Per-class accuracy
            for cls in range(n_classes):
                mask = all_targets == cls
                if mask.sum() > 0:
                    cls_acc = (all_preds[mask] == all_targets[mask]).mean()
                    print(f"  Class {cls} accuracy: {cls_acc:.4f} ({mask.sum()} samples)")
            
            # Prediction distribution
            unique_preds, pred_counts = np.unique(all_preds, return_counts=True)
            print(f"  Prediction distribution: {dict(zip(unique_preds, pred_counts))}")
            
            # Learning rate scheduling - print current LR when it changes
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step(avg_loss)
            new_lr = optimizer.param_groups[0]['lr']
            if old_lr != new_lr:
                print(f"  Learning rate reduced from {old_lr:.6f} to {new_lr:.6f}")
        else:
            # Non-rank 0 processes still need to step the scheduler
            scheduler.step(avg_loss)

        # Epoch-based checkpointing
        if (epoch + 1) % args.ckpt_freq == 0:
            save_path = exp_dir / f"checkpoint_epoch{epoch+1}.pt"
            save_checkpoint(model, optimizer, epoch + 1, global_step, save_path, rank)

    # ----- Save final model -----
    if rank == 0:
        print("Saving final model...")
    
    final_path = exp_dir / "final_model_weights.pth"
    
    # Save full model state dict
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, FullStateDictConfig(offload_to_cpu=True, rank0_only=True)):
        model_state = model.state_dict()
    
    if rank == 0:
        torch.save(model_state, final_path)
        print(f"Final model weights saved to {final_path}")

    # Final synchronization
    dist.barrier()
    
    if rank == 0:
        print(f"Training finished successfully. Checkpoints saved to {exp_dir}")

    # Clean up
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
