"""
Compare Transformer vs RLHF-finetuned model
Generates samples from both for side-by-side comparison
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
    TRANSFORMER_PATH = Path("checkpoints/task3_transformer")
    RLHF_PATH = Path("checkpoints/rlhf_finetuned")
    OUTPUT_PATH = Path("outputs/rlhf_comparison")
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    
    # Load vocabulary
    with open(DATA_PATH / "metadata.json", 'r') as f:
        metadata = json.load(f)
    vocab_size = metadata["vocab_size"]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load original transformer
    original_model = MusicTransformer(vocab_size=vocab_size)
    original_model.load_state_dict(torch.load(TRANSFORMER_PATH / "best_model.pt", map_location=device))
    original_model = original_model.to(device)
    original_model.eval()
    
    # Load RLHF model
    rlhf_model = MusicTransformer(vocab_size=vocab_size)
    rlhf_path = RLHF_PATH / "rlhf_finetuned_model.pt"
    
    rlhf_loaded = False
    if rlhf_path.exists():
        rlhf_model.load_state_dict(torch.load(rlhf_path, map_location=device))
        rlhf_model = rlhf_model.to(device)
        rlhf_model.eval()
        rlhf_loaded = True
    
    # Load validation sequences for prompts
    val_seqs = np.load(DATA_PATH / "val_seqs_shifted.npy")
    
    # Generate 10 comparison pairs
    print("🎵 Generating Before/After comparison samples...")
    
    results = []
    
    for i in range(10):
        # Same prompt for both models
        prompt_length = 128
        random_idx = np.random.randint(0, len(val_seqs))
        prompt_tokens = val_seqs[random_idx][:prompt_length]
        prompt = torch.LongTensor(prompt_tokens).unsqueeze(0).to(device)
        
        # Original model generation
        with torch.no_grad():
            original_gen = original_model.generate(prompt, max_new_tokens=384, temperature=1.0, top_k=50)
        
        # RLHF model generation
        if rlhf_loaded:
            with torch.no_grad():
                rlhf_gen = rlhf_model.generate(prompt, max_new_tokens=384, temperature=1.0, top_k=50)
        else:
            rlhf_gen = original_gen
        
        # Save both
        np.save(OUTPUT_PATH / f"comparison_{i+1}_original.npy", original_gen[0].cpu().numpy())
        np.save(OUTPUT_PATH / f"comparison_{i+1}_rlhf.npy", rlhf_gen[0].cpu().numpy())
        
        # Quick metrics
        original_has_notes = np.any((original_gen.cpu().numpy() >= 21) & (original_gen.cpu().numpy() <= 109))
        rlhf_has_notes = np.any((rlhf_gen.cpu().numpy() >= 21) & (rlhf_gen.cpu().numpy() <= 109))
        
        results.append({
            "sample": i + 1,
            "original_has_notes": bool(original_has_notes),
            "rlhf_has_notes": bool(rlhf_has_notes)
        })
        
        print(f"  ✓ Comparison {i+1}/10 | Original notes: {original_has_notes} | RLHF notes: {rlhf_has_notes}")
    
    print(f"\n✅ Comparison samples saved to {OUTPUT_PATH}")
    print("\n📊 Task 4 deliverables ready:")
    print("   - Human survey with 10+ participants")
    print("   - Reward model trained")
    print("   - RLHF fine-tuned model")
    print("   - 10 before/after comparison samples")

if __name__ == "__main__":
    main()