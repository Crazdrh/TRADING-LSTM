# data_split.py
"""
Chronologically split data into training, validation, and test sets (by whole files, not splitting files).
"""
import os
import shutil

# Configuration: input directory and output directories for splits, and split ratios
input_dir = "data/raw_csv"  # directory containing all raw csv files
output_train_dir = "data/train_data"
output_val_dir = "data/val_data"
output_test_dir = "data/test_data"
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Ensure output directories exist
os.makedirs(output_train_dir, exist_ok=True)
os.makedirs(output_val_dir, exist_ok=True)
os.makedirs(output_test_dir, exist_ok=True)

# List all CSV files in the input directory
all_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".csv")]
if not all_files:
    raise FileNotFoundError(f"No CSV files found in {input_dir}")

# Sort files by name (assuming file names contain chronological order, e.g., dates or incremental indices)
all_files.sort()

# Determine split indices (split by file count)
total_files = len(all_files)
train_count = int(total_files * train_ratio)
val_count = int(total_files * val_ratio)
test_count = total_files - train_count - val_count

# Adjust if rounding issues cause any split to be zero
if test_count == 0:
    test_count = max(1, total_files - train_count - val_count)
    val_count = total_files - train_count - test_count
if train_count == 0 or val_count == 0:
    raise ValueError("Not enough files to split into train/val/test with given ratios.")

train_files = all_files[:train_count]
val_files = all_files[train_count:train_count+val_count]
test_files = all_files[train_count+val_count:train_count+val_count+test_count]

print(f"Total files: {total_files}. Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")

# Copy files to respective directories
for fname in train_files:
    shutil.copy(os.path.join(input_dir, fname), os.path.join(output_train_dir, fname))
for fname in val_files:
    shutil.copy(os.path.join(input_dir, fname), os.path.join(output_val_dir, fname))
for fname in test_files:
    shutil.copy(os.path.join(input_dir, fname), os.path.join(output_test_dir, fname))

print("Files have been split chronologically into train, val, and test directories.")
