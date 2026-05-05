import numpy as np
from pathlib import Path

# Check training data
train_seqs = np.load("data/processed/task1/train_seqs_fixed.npy")
print(f"Train data token range: {train_seqs.min()} to {train_seqs.max()}")

# Check for note range (21-109)
note_tokens = train_seqs[(train_seqs >= 21) & (train_seqs <= 109)]
print(f"Note tokens (21-109): {len(note_tokens)} out of {train_seqs.size} ({100*len(note_tokens)/train_seqs.size:.2f}%)")

# See unique tokens
unique_tokens = np.unique(train_seqs)
print(f"Unique tokens: {sorted(unique_tokens)[:30]}")  # First 30