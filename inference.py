import pandas as pd
import numpy as np
import torch
from LSTM import ComplexLSTMModel
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import time

# --------- CONFIG ---------
CSV_PATH = "/lambda/nfs/LSTM/Lstm/data/alpaca/nvda_5minla.csv"
MODEL_PATH = "/lambda/nfs/LSTM/Lstm/ckpt/1/best_model.pth"
SEQ_LEN = 50
FEATURE_COLS = ['open', 'high', 'low', 'close','volume', 'ma_5', 'ma_10', 'ma_20', 'ma_40', 'ma_55']

# GPU Optimization Settings
BATCH_SIZE = 2048  # Increase this based on your GPU memory
NUM_WORKERS = 12    # For faster data loading
PIN_MEMORY = True  # Faster CPU->GPU transfer
USE_MIXED_PRECISION = True  # Set to True for 2x speedup (may reduce accuracy slightly)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Enable GPU optimizations
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True  # Optimize for fixed input sizes
    torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

def load_model(model_path, n_classes):
    model = ComplexLSTMModel(input_dim=10, output_dim=n_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Enable half precision for 2x speedup (if your model supports it)
    # Uncomment the next line if you want to try mixed precision
    # model = model.half()
    
    return model

def load_and_preprocess(csv_path):
    print("Loading and preprocessing data...")
    df = pd.read_csv(csv_path)
    df.columns = [c.lower() for c in df.columns]
    
    for col in [c.lower() for c in FEATURE_COLS]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    features = df[[c.lower() for c in FEATURE_COLS]].astype(np.float32).values
    
    # Vectorized normalization
    means = features.mean(axis=0, keepdims=True)
    stds = features.std(axis=0, keepdims=True) + 1e-8
    features_norm = (features - means) / stds
    
    return df, features_norm.astype(np.float32)

def create_sequences_vectorized(features, seq_len):
    n_samples = len(features) - seq_len
    if n_samples <= 0:
        return np.array([]), np.array([])
    # Shape: (n_samples, seq_len, n_features)
    sequences = np.lib.stride_tricks.sliding_window_view(features, (seq_len, features.shape[1]))
    # Fix extra dimension if present
    if sequences.shape[1] == 1:
        sequences = sequences[:, 0, :, :]
    sequences = np.ascontiguousarray(sequences)
    indices = np.arange(seq_len, len(features))
    return sequences, indices


def predict_batch_optimized(model, features, df, seq_len=SEQ_LEN, batch_size=BATCH_SIZE):
    """Optimized batch prediction with maximum GPU utilization"""
    print("Creating sequences...")
    sequences, indices = create_sequences_vectorized(features, seq_len)
    
    if len(sequences) == 0:
        return []
    
    print(f"Processing {len(sequences)} sequences in batches of {batch_size}")
    print(f"Sequences shape: {sequences.shape}, Indices shape: {indices.shape}")
    
    # Convert to tensors - make sure indices are the right length
    sequences_tensor = torch.from_numpy(sequences).to(device, non_blocking=True)
    indices_tensor = torch.from_numpy(indices).long()  # Ensure indices are long type
    
    # Create DataLoader for efficient batching
    dataset = TensorDataset(sequences_tensor, indices_tensor)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=True if NUM_WORKERS > 0 else False
    )
    
    results = []
    total_batches = len(dataloader)
    
    print("Starting inference...")
    start_time = time.time()
    
    with torch.no_grad():
        for batch_idx, (batch_sequences, batch_indices) in enumerate(dataloader):
            # Move to GPU if not already there
            batch_sequences = batch_sequences.to(device, non_blocking=True)
            
            # Use mixed precision if enabled
            if USE_MIXED_PRECISION:
                with torch.cuda.amp.autocast():
                    logits = model(batch_sequences)
            else:
                logits = model(batch_sequences)
            
            pred_classes = logits.argmax(dim=1).cpu().numpy()
            batch_indices = batch_indices.cpu().numpy()
            
            # Process results
            for i, (row_num, pred_class) in enumerate(zip(batch_indices, pred_classes)):
                date = df['date'].iloc[row_num] if 'date' in df.columns else row_num
                actual = df['signal_class'].iloc[row_num] if 'signal_class' in df.columns else None
                results.append((row_num, date, pred_class, actual))
            
            # Progress update
            if batch_idx % 10 == 0 or batch_idx == total_batches - 1:
                elapsed = time.time() - start_time
                progress = (batch_idx + 1) / total_batches * 100
                samples_processed = (batch_idx + 1) * batch_size
                samples_per_sec = samples_processed / elapsed if elapsed > 0 else 0
                print(f"Progress: {progress:.1f}% | Batch {batch_idx+1}/{total_batches} | "
                      f"Speed: {samples_per_sec:.0f} samples/sec")
    
    total_time = time.time() - start_time
    print(f"\nInference completed in {total_time:.2f} seconds")
    print(f"Average speed: {len(results)/total_time:.0f} samples/sec")
    
    return results

def optimize_gpu_memory():
    """Optimize GPU memory usage"""
    if torch.cuda.is_available():
        # Clear cache
        torch.cuda.empty_cache()
        
        # Set memory fraction if needed (uncomment and adjust if you get OOM errors)
        # torch.cuda.set_per_process_memory_fraction(0.8)
        
        print(f"GPU Memory before: {torch.cuda.memory_allocated()/1e9:.2f} GB")

if __name__ == "__main__":
    # Optimize GPU settings
    optimize_gpu_memory()
    
    # Load and preprocess data
    df, features = load_and_preprocess(CSV_PATH)
    n_classes = int(df['signal_class'].nunique()) if 'signal_class' in df.columns else 3
    
    # Load model
    print("Loading model...")
    model = load_model(MODEL_PATH, n_classes)
    
    # Warm up GPU (optional but helps with consistent timing)
    print("Warming up GPU...")
    dummy_input = torch.randn(BATCH_SIZE, SEQ_LEN, 10).to(device)
    with torch.no_grad():
        try:
            _ = model(dummy_input)
            print("GPU warmup successful!")
        except Exception as e:
            print(f"GPU warmup failed: {e}")
            print("Proceeding without warmup...")
    
    # Run optimized inference
    results = predict_batch_optimized(model, features, df, seq_len=SEQ_LEN, batch_size=BATCH_SIZE)
    
    # Display results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print("row_num\tdate\t\tpredicted\tactual")
    print("-"*70)
    
    for row_num, date, pred, actual in results[:20]:  # Show first 20 results
        print(f"{row_num}\t{date}\t{pred}\t\t{actual}")
    
    if len(results) > 20:
        print(f"\n... and {len(results)-20} more results")
    
    # Memory cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"\nFinal GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        print(f"\nFinal GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")
