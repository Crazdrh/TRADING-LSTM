import pandas as pd
import numpy as np
import torch
from LSTM import ComplexLSTMModel
from torch.utils.data import DataLoader, TensorDataset

# --------- CONFIG ---------
CSV_PATH = "/lambda/nfs/LSTM/Lstm/data/alpaca/nvda_5minla.csv"
MODEL_PATH = "/lambda/nfs/LSTM/Lstm/ckpt/1/best_model.pth"
SEQ_LEN = 50
FEATURE_COLS = ['open', 'high', 'low', 'close', 'volume', 'ma_5', 'ma_10', 'ma_20', 'ma_40', 'ma_55']
BATCH_SIZE = 4096                # <- Make as large as fits in GPU memory
NUM_WORKERS = 4                  # <- Set to num CPU cores for max speed
PIN_MEMORY = True                # <- Accelerates host->GPU transfer
USE_MIXED_PRECISION = True       # <- Use AMP for 2x speed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_model(model_path, n_classes):
    model = ComplexLSTMModel(input_dim=10, output_dim=n_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def load_and_preprocess(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = [c.lower() for c in df.columns]
    for col in [c.lower() for c in FEATURE_COLS]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    features = df[[c.lower() for c in FEATURE_COLS]].astype(np.float32).values
    means = features.mean(axis=0, keepdims=True)
    stds = features.std(axis=0, keepdims=True) + 1e-8
    features_norm = (features - means) / stds
    return df, features_norm

def create_sequences(features, seq_len):
    # Output shape: (num_sequences, seq_len, num_features)
    n_samples = len(features) - seq_len + 1
    if n_samples <= 0:
        return np.array([]), np.array([])
    sequences = np.lib.stride_tricks.sliding_window_view(
        features, (seq_len, features.shape[1])
    )
    # Remove the singleton dimension if present
    if sequences.shape[1] == 1:
        sequences = sequences[:, 0, :, :]
    # Guarantee writeable for torch
    sequences = np.ascontiguousarray(sequences)
    # indices correspond to END of each window (prediction point)
    indices = np.arange(seq_len-1, len(features))
    assert sequences.shape[0] == indices.shape[0], f"{sequences.shape[0]} != {indices.shape[0]}"
    return sequences, indices

def predict_batches(model, features, df, seq_len=SEQ_LEN, batch_size=BATCH_SIZE):
    sequences, indices = create_sequences(features, seq_len)
    if len(sequences) == 0:
        return []

    # Move to torch tensors
    sequences_tensor = torch.from_numpy(sequences)
    print("sequences_tensor.shape:", sequences_tensor.shape)
    print("indices.shape:", torch.from_numpy(indices).shape)
    dataset = TensorDataset(sequences_tensor, torch.from_numpy(indices))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, persistent_workers=True)

    results = []
    print(f"Running inference on {len(sequences)} sequences...")
    with torch.no_grad():
        for batch_sequences, batch_indices in dataloader:
            batch_sequences = batch_sequences.to(device, non_blocking=True)
            if USE_MIXED_PRECISION and torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    logits = model(batch_sequences)
            else:
                logits = model(batch_sequences)
            preds = logits.argmax(dim=1).cpu().numpy()
            batch_indices = batch_indices.cpu().numpy()

            for i, idx in enumerate(batch_indices):
                date = df['date'].iloc[idx] if 'date' in df.columns else idx
                actual = df['signal_class'].iloc[idx] if 'signal_class' in df.columns else None
                results.append((idx, date, preds[i], actual))
    return results

def warmup_gpu(model, seq_len, feature_dim, batch_size):
    # Warm up GPU for consistent performance
    dummy_input = torch.randn(batch_size, seq_len, feature_dim).to(device)
    with torch.no_grad():
        if torch.cuda.is_available():
            with torch.cuda.amp.autocast():
                _ = model(dummy_input)
        else:
            _ = model(dummy_input)
    print("GPU warmup done.")

if __name__ == "__main__":
    import time
    df, features = load_and_preprocess(CSV_PATH)
    n_classes = int(df['signal_class'].nunique()) if 'signal_class' in df.columns else 3
    model = load_model(MODEL_PATH, n_classes)
    warmup_gpu(model, SEQ_LEN, 10, BATCH_SIZE)  # Optional, but recommended

    torch.cuda.empty_cache()
    start = time.time()
    results = predict_batches(model, features, df, seq_len=SEQ_LEN, batch_size=BATCH_SIZE)
    elapsed = time.time() - start

    print("\nrow_num\tdate\t\tpredicted\tactual")
    for row_num, date, pred, actual in results[:20]:
        print(f"{row_num}\t{date}\t{pred}\t\t{actual}")
    print(f"\nProcessed {len(results)} samples in {elapsed:.2f} seconds "
          f"({len(results)/elapsed:.0f} samples/sec)")

    if torch.cuda.is_available():
        print(f"Peak GPU memory usage: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
        torch.cuda.empty_cache()
    import pandas as pd

    OUTPUT_CSV = "inference_results.csv"

    df_out = pd.DataFrame(results, columns=["row_num", "date", "predicted", "actual"])
    df_out.to_csv(OUTPUT_CSV, index=False)

    print(f"\nSaved results to {OUTPUT_CSV}")
