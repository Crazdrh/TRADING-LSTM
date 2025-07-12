import argparse
import os
import json
from pathlib import Path
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.distributed.fsdp.wrap import default_auto_wrap_policy
from torch.distributed.checkpoint import save, load
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict, StateDictOptions
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import torch.nn as nn
import pandas as pd
import numpy as np
from LSTM import ComplexLSTMModel

# ---- Dataset Definition ----
class PriceDataset(Dataset):
    def __init__(self, df, seq_len=50):
        feature_cols = ['open', 'high', 'low', 'close', 'MA', 'MA.1', 'MA.2', 'MA.3', 'MA.4']
        df.columns = [c.strip() for c in df.columns]
        for col in feature_cols + ['signal_class']:
            if col not in df.columns:
                raise ValueError(f"Missing column: {col}")
        features = df[feature_cols].values.astype(float)
        means = features.mean(axis=0)
        stds = features.std(axis=0) + 1e-8
        features = (features - means) / stds
        self.X = []
        self.y = []
        for i in range(len(features) - seq_len):
            seq = features[i:i+seq_len]
            label = df['signal_class'].iloc[i+seq_len]
            self.X.append(seq)
            self.y.append(int(label))
        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y), dtype=torch.long)

    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ---- Data Loading ----
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

# ---- Main FSDP Training Logic ----
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--seq-len", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # ---- DDP / FSDP Init ----
    dist.init_process_group()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.manual_seed(args.seed)

    if rank == 0:
        print(f"Using {world_size} GPUs, device {device}")

    # ---- Load Data ----
    df = load_all_csvs(args.data_dir)
    dataset = PriceDataset(df, seq_len=args.seq_len)
    sampler = DistributedSampler(dataset, shuffle=True, drop_last=True)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=2,
        pin_memory=True,
    )

    n_classes = int(df['signal_class'].dropna().nunique())
    model = ComplexLSTMModel(input_dim=9, output_dim=n_classes).to(device)

    # ---- Wrap in FSDP ----
    model = FSDP(
        model,
        device_id=device,
        sharding_strategy="FULL_SHARD",
        cpu_offload=args.cpu_offload,
        auto_wrap_policy=default_auto_wrap_policy,
        sync_module_states=True,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    exp_dir = Path(args.save_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    state_path = exp_dir / "fsdp_state.json"

    start_epoch = 0
    if args.resume and state_path.exists():
        sharded_model_state, sharded_optimizer_state = get_state_dict(
            model, optimizer, options=StateDictOptions(full_state_dict=False, cpu_offload=True)
        )
        load(
            dict(model=sharded_model_state, optimizer=sharded_optimizer_state),
            checkpoint_id=str(exp_dir / "checkpoint"),
        )
        set_state_dict(
            model, optimizer, model_state_dict=sharded_model_state,
            optim_state_dict=sharded_optimizer_state,
            options=StateDictOptions(full_state_dict=False, cpu_offload=True)
        )
        with open(state_path) as fp:
            state = json.load(fp)
            start_epoch = state.get("epoch", 0)
        if rank == 0:
            print(f"Resumed training from epoch {start_epoch}")

    # ---- Training Loop ----
    for epoch in range(start_epoch, args.epochs):
        model.train()
        sampler.set_epoch(epoch)
        total_loss = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_x.size(0)
        avg_loss = total_loss / len(dataset)
        if rank == 0:
            print(f"Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.5f}")

        # Save checkpoint every epoch (or set your own freq)
        sharded_model_state, sharded_optimizer_state = get_state_dict(
            model, optimizer, options=StateDictOptions(full_state_dict=False, cpu_offload=True)
        )
        save(
            dict(model=sharded_model_state, optimizer=sharded_optimizer_state),
            checkpoint_id=str(exp_dir / "checkpoint"),
        )
        if rank == 0:
            state = {"epoch": epoch + 1}
            with open(state_path, "w") as fp:
                json.dump(state, fp)
        dist.barrier()

    # Final Save (rank 0)
    if rank == 0:
        print(f"Training finished. Final checkpoint at {exp_dir}")

if __name__ == "__main__":
    main()
