"""
Generate MIDI samples from trained LSTM Autoencoder (Task 1)
Produces 5 original music compositions
"""

import torch
import numpy as np
from pathlib import Path
import json
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from models.autoencoder import LSTMVAE

# Configuration
DATA_PATH = Path("data/processed/task1")
CHECKPOINT_PATH = Path("checkpoints/task1_autoencoder")
OUTPUT_PATH = Path("outputs/task1_samples")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# Load metadata
with open(DATA_PATH / "metadata.json", "r") as f:
    metadata = json.load(f)

vocab_size = metadata["vocab_size"]
SEQ_LEN = metadata["sequence_length"]

# Hyperparameters (must match training)
EMBED_DIM = 32
HIDDEN_DIM = 64
LATENT_DIM = 32
NUM_LAYERS = 3

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMVAE(vocab_size, EMBED_DIM, HIDDEN_DIM, LATENT_DIM, NUM_LAYERS)
model.load_state_dict(torch.load(CHECKPOINT_PATH / "final_model.pt", map_location=device))
model = model.to(device)
model.eval()

print(f"✓ Model loaded from {CHECKPOINT_PATH}")
print(f"✓ Vocabulary size: {vocab_size}")
print(f"✓ Device: {device}")

def generate_single_sample(model, device, max_len=512, temperature=1.0):
    """Generate one music sequence from random latent vector"""
    with torch.no_grad():
        # Sample random latent vector from prior
        z = torch.randn(1, LATENT_DIM, device=device)
        
        # Generate token IDs
        generated_tokens = model.generate(z, max_len=max_len)
        token_sequence = generated_tokens[0].cpu().numpy()
        
    return token_sequence

def tokens_to_midi(token_sequence, output_path, tokenizer):
    """Convert token sequence back to MIDI file"""
    try:
        from miditok import REMI, TokenizerConfig
        from miditoolkit import MidiFile
        
        # Create tokenizer (same config as training)
        config = TokenizerConfig(
            pitch_range=(21, 109),
            beat_res={(0, 4): 8, (4, 12): 4},
            nb_velocities=32,
            use_chords=True,
            use_rests=True,
            use_tempos=True,
        )
        tokenizer_obj = REMI(config)
        
        # Convert tokens to MIDI
        # Note: Need to reconstruct token object properly
        # For now, save as numpy and we'll convert separately
        np.save(output_path.with_suffix('.npy'), token_sequence)
        print(f"  ✓ Token sequence saved: {output_path.with_suffix('.npy')}")
        
    except Exception as e:
        print(f"  ⚠ Could not convert to MIDI: {e}")
        # Save as numpy as fallback
        np.save(output_path.with_suffix('.npy'), token_sequence)

print("\n🎵 Generating 5 music samples...")
print("-" * 40)

# For Task 1 report, we just need the token sequences
# Full MIDI conversion can be done with miditok's decode function

for i in range(5):
    print(f"\nGenerating sample {i+1}/5...")
    
    # Generate token sequence
    tokens = generate_single_sample(model, device, max_len=SEQ_LEN)
    
    # Save as numpy (will convert to MIDI later)
    output_file = OUTPUT_PATH / f"generated_sample_{i+1}"
    np.save(output_file.with_suffix('.npy'), tokens)
    
    print(f"  ✓ Saved to {output_file}.npy")
    print(f"  ✓ Token range: {tokens.min()} - {tokens.max()}")
    print(f"  ✓ Sequence length: {len(tokens)}")

print("\n" + "="*40)
print(f"5 samples generated and saved to:")
print(f"   {OUTPUT_PATH}")
