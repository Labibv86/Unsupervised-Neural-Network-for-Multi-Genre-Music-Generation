"""
Generate 10 long compositions with Transformer (Task 3 deliverable)
"""

import torch
import numpy as np
from pathlib import Path
import json
import sys

sys.path.append(str(Path(__file__).parent.parent))
from models.transformer import MusicTransformer

def main():
    DATA_PATH = Path("data/processed/task1")
    CHECKPOINT_PATH = Path("checkpoints/task3_transformer")
    OUTPUT_PATH = Path("outputs/task3_samples")
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    
    # Load metadata
    with open(DATA_PATH / "metadata.json", "r") as f:
        metadata = json.load(f)
    
    vocab_size = metadata["vocab_size"]
    
    # Model config (must match training)
    D_MODEL = 256
    NHEAD = 8
    NUM_LAYERS = 4
    DIM_FEEDFORWARD = 1024
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MusicTransformer(
        vocab_size=vocab_size,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD
    )
    
    best_model_path = CHECKPOINT_PATH / "best_model.pt"
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f"✓ Loaded best_model.pt")
    else:
        print(f"✗ No model found!")
        return
    
    model = model.to(device)
    model.eval()
    
    print(f"✓ Transformer model loaded")
    
    # Generate 10 long compositions
    print("\n🎵 Generating 10 long compositions (1024 tokens each)...")
    
    # Create random prompts from validation set
    val_seqs = np.load(DATA_PATH / "val_seqs_fixed.npy")
    prompt_length = 128
    
    for i in range(10):
        # Random prompt from real music
        random_idx = np.random.randint(0, len(val_seqs))
        prompt_tokens = val_seqs[random_idx][:prompt_length]
        prompt = torch.LongTensor(prompt_tokens).unsqueeze(0).to(device)
        
        # Generate continuation
        max_new_tokens = 896  # Total = 1024 (128 prompt + 896 new)
        temperature = 1.0 + (i * 0.1)  # Vary temperature for diversity
        
        with torch.no_grad():
            generated = model.generate(
                prompt, 
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=50
            )
        
        # Save
        generated_numpy = generated[0].cpu().numpy()
        np.save(OUTPUT_PATH / f"transformer_composition_{i+1}.npy", generated_numpy)
        
        # Check for notes
        has_notes = np.any((generated_numpy >= 21) & (generated_numpy <= 109))
        print(f"  ✓ Composition {i+1}/10 | Length: {len(generated_numpy)} | Has notes: {has_notes} | Temperature: {temperature:.1f}")
    
    print(f"\n✅ 10 long compositions saved to {OUTPUT_PATH}")
    print("\n📊 Next: Compute perplexity report")

if __name__ == "__main__":
    main()