# shift_tokens.py
import numpy as np
from pathlib import Path

DATA_PATH = Path("data/processed/task1")

# Load data
train_seqs = np.load(DATA_PATH / "train_seqs.npy")
val_seqs = np.load(DATA_PATH / "val_seqs.npy")

print(f"Before shift - Train min: {train_seqs.min()}, max: {train_seqs.max()}")
print(f"Before shift - Val min: {val_seqs.min()}, max: {val_seqs.max()}")

# Shift down by 1 so tokens start at 0
train_seqs_shifted = train_seqs - 1
val_seqs_shifted = val_seqs - 1

print(f"\nAfter shift - Train min: {train_seqs_shifted.min()}, max: {train_seqs_shifted.max()}")
print(f"After shift - Val min: {val_seqs_shifted.min()}, max: {val_seqs_shifted.max()}")

# Save shifted data
np.save(DATA_PATH / "train_seqs_shifted.npy", train_seqs_shifted)
np.save(DATA_PATH / "val_seqs_shifted.npy", val_seqs_shifted)

print("\n✓ Saved shifted data")
print("Update your training script to load these files")