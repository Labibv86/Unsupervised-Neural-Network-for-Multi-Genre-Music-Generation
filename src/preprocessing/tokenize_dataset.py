"""
Tokenize MIDI files for LSTM Autoencoder (Task 1)
Fixed for deep nested A-Z folder structure
"""

import os
import json
from pathlib import Path
from tqdm import tqdm
import torch
import numpy as np
from miditok import REMI, TokenizerConfig
from miditoolkit import MidiFile


DATA_PATH = Path("data/raw_midi/lmd_matched") 
OUTPUT_PATH = Path("data/processed/task1")
SEQ_LEN = 512
STRIDE = 256

# Create output directories
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
(OUTPUT_PATH / "tokens").mkdir(exist_ok=True)

def find_all_midi_files(base_path):
    """Recursively find all .mid files in nested structure"""
    all_files = []
    
    # Walk through all directories recursively
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.lower().endswith(('.mid', '.midi')):
                full_path = Path(root) / file
                all_files.append(full_path)
    
    return all_files

def count_files_by_folder(base_path):
    """Count MIDI files in each subfolder to understand structure"""
    folder_counts = {}
    
    # Check first level folders (A-Z)
    for folder in base_path.iterdir():
        if folder.is_dir():
            count = len(list(folder.rglob("*.mid")) + list(folder.rglob("*.MID")))
            if count > 0:
                folder_counts[folder.name] = count
    
    return folder_counts

# Configure REMI tokenizer
config = TokenizerConfig(
    pitch_range=(21, 109),
    beat_res={(0, 4): 8, (4, 12): 4},
    nb_velocities=32,
    use_chords=True,
    use_rests=True,
    use_tempos=True,
    use_time_signatures=True,
    use_programs=True,
)

tokenizer = REMI(config)

def tokenize_midi_file(midi_path):
    """Convert MIDI to token sequence"""
    try:
        midi = MidiFile(midi_path)
        tokens = tokenizer(midi)
        token_ids = tokens.ids
        
        if len(token_ids) < SEQ_LEN:
            return None
        
        # Create overlapping windows
        windows = []
        for start in range(0, len(token_ids) - SEQ_LEN, STRIDE):
            window = token_ids[start:start + SEQ_LEN]
            windows.append(window)

            if len(windows) >= 20:
                break
        
        return windows
    except Exception as e:
        return None

def main():
    print("="*60)
    print("TASK 1: Scanning and Tokenizing MIDI Files")
    print("="*60)
    
    # Check if path exists
    if not DATA_PATH.exists():
        print(f"\n✗ ERROR: Path not found: {DATA_PATH}")
        print("\nPlease update DATA_PATH in the script to your Lakh dataset location")
        print("Example: DATA_PATH = Path('E:/lakh_midi/lmd_matched')")
        return
    
    print(f"\n✓ Data path: {DATA_PATH}")
    
    # First, understand the folder structure
    print("\n Scanning folder structure...")
    folder_counts = count_files_by_folder(DATA_PATH)
    
    if folder_counts:
        print(f"\nFound {len(folder_counts)} top-level folders with MIDI files:")
        for folder_name, count in list(folder_counts.items())[:10]:  # Show first 10
            print(f"  - {folder_name}: {count:,} MIDI files")
        print(f"  ... and {len(folder_counts) - 10} more folders")
    else:
        print("\nNo files found in top-level folders. Checking deeper...")
    
    # Find ALL MIDI files recursively
    print("\n🔍 Finding all MIDI files (this may take a minute)...")
    all_midi_files = find_all_midi_files(DATA_PATH)
    print(f"\n✓ Total MIDI files found: {len(all_midi_files):,}")
    
    if len(all_midi_files) == 0:
        print("\n✗ No MIDI files found! Check if:")
        print("  1. The path is correct")
        print("  2. The Lakh dataset is properly extracted")
        print("  3. Files have .mid or .MID extension")
        return
    


    midi_files = all_midi_files
    
    # Limit for testing (remove for full dataset)
    MAX_FILES = 50000  
    if len(midi_files) > MAX_FILES:
        print(f"\n⚠ Limiting to {MAX_FILES} files for initial processing")
        print("  (Remove MAX_FILES limit for full dataset)")
        midi_files = midi_files[:MAX_FILES]
    
    print(f"\n🎵 Processing {len(midi_files)} MIDI files...")
    
    # Tokenize all files
    all_windows = []
    failed_files = []
    successful_files = 0
    
    for midi_path in tqdm(midi_files, desc="Tokenizing"):
        windows = tokenize_midi_file(midi_path)
        if windows:
            all_windows.extend(windows)
            successful_files += 1
        else:
            failed_files.append(midi_path.name)
    
    print(f"\n✓ Tokenization complete!")
    print(f"  - Successful files: {successful_files}/{len(midi_files)}")
    print(f"  - Total sequences: {len(all_windows):,}")
    print(f"  - Sequence length: {SEQ_LEN} tokens")
    print(f"  - Failed files: {len(failed_files)}")
    
    if len(all_windows) == 0:
        print("\n✗ No sequences generated!")
        print("Possible issues:")
        print("  1. MIDI files might be corrupted")
        print("  2. Files might be too short (<512 tokens)")
        print("  3. Tokenizer configuration might need adjustment")
        return
    
    # Convert to numpy array
    token_sequences = np.array(all_windows, dtype=np.int64)
    print(f"\n📊 Data shape: {token_sequences.shape}")
    
    # Train/validation split (90/10)
    split_idx = int(0.9 * len(token_sequences))
    train_seqs = token_sequences[:split_idx]
    val_seqs = token_sequences[split_idx:]
    
    print(f"\n✓ Train sequences: {len(train_seqs):,}")
    print(f"✓ Validation sequences: {len(val_seqs):,}")
    
    # Save processed data
    np.save(OUTPUT_PATH / "train_seqs_shifted.npy", train_seqs)
    np.save(OUTPUT_PATH / "val_seqs_shifted.npy", val_seqs)
    
    # Estimate vocabulary size from data
    unique_tokens = len(np.unique(token_sequences))
    print(f"✓ Unique tokens found: {unique_tokens}")
    
    # Save metadata
    metadata = {
        "data_source": str(DATA_PATH),
        "total_midi_files": len(all_midi_files),
        "processed_files": successful_files,
        "total_sequences": len(all_windows),
        "sequence_length": SEQ_LEN,
        "stride": STRIDE,
        "num_train": len(train_seqs),
        "num_val": len(val_seqs),
        "vocab_size": unique_tokens + 10,  # Add padding/SOS/EOS tokens
        "files_processed": successful_files,
        "failed_files": failed_files[:20]
    }
    
    with open(OUTPUT_PATH / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nData saved to {OUTPUT_PATH}")
    print(f"\nReady for training!")
    print("\nNext command:")
    print("  python src/training/train_autoencoder.py")

if __name__ == "__main__":
    main()