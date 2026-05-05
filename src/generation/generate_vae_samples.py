"""
Generate diverse multi-genre samples + latent interpolation (Task 2 deliverables)
Fixed for tensor dimension issues
"""

import torch
import numpy as np
from pathlib import Path
import json
import sys

sys.path.append(str(Path(__file__).parent.parent))
from models.vae import MusicVAE

def main():
    DATA_PATH = Path("data/processed/task1")
    CHECKPOINT_PATH = Path("checkpoints/task2_vae")
    OUTPUT_PATH = Path("outputs/task2_samples")
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    # Load metadata
    with open(DATA_PATH / "metadata.json", "r") as f:
        metadata = json.load(f)

    vocab_size = metadata["vocab_size"]

    # Model config (must match training)
    EMBED_DIM = 8       # Reduced from 128
    HIDDEN_DIM = 16     # Reduced from 256
    LATENT_DIM = 8      # Reduced from 64
    NUM_LAYERS = 1

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MusicVAE(vocab_size, EMBED_DIM, HIDDEN_DIM, LATENT_DIM, NUM_LAYERS)
    
    # Try loading best model, if not found use final model
    best_model_path = CHECKPOINT_PATH / "best_model.pt"
    final_model_path = CHECKPOINT_PATH / "final_model.pt"
    
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f"✓ Loaded best_model.pt")
    elif final_model_path.exists():
        model.load_state_dict(torch.load(final_model_path, map_location=device))
        print(f"✓ Loaded final_model.pt")
    else:
        print(f"✗ No model found in {CHECKPOINT_PATH}")
        return
    
    model = model.to(device)
    model.eval()

    print(f"✓ VAE model loaded")
    print(f"✓ Latent dimension: {LATENT_DIM}")

    # 1. Generate 8 diverse samples (Task 2 deliverable)
    print("\n🎵 Generating 8 diverse music samples...")
    for i in range(8):
        z = model.sample_latent(batch_size=1, device=device)
        # z shape: (1, latent_dim) - correct for generation
        tokens = model.generate(z, max_len=512)
        token_numpy = tokens.cpu().numpy()
        np.save(OUTPUT_PATH / f"vae_sample_{i+1}.npy", token_numpy)
        print(f"  ✓ Sample {i+1}/8 generated (unique tokens: {len(np.unique(token_numpy))})")

    # 2. Latent interpolation experiment (Task 2 deliverable)
    print("\n🔮 Latent interpolation between two points...")
    z1 = model.sample_latent(batch_size=1, device=device)
    z2 = model.sample_latent(batch_size=1, device=device)

    # Interpolate (returns tensor of shape [steps, 1, latent_dim])
    interpolated = model.interpolate(z1, z2, steps=8)
    print(f"  Interpolation shape: {interpolated.shape}")

    for i in range(interpolated.shape[0]):
        # Get individual z (shape: [1, latent_dim])
        z_step = interpolated[i]  # Already has batch dimension
        tokens = model.generate(z_step, max_len=512)
        np.save(OUTPUT_PATH / f"interpolation_step_{i+1}.npy", tokens.cpu().numpy())
        print(f"  ✓ Interpolation step {i+1}/8 saved")

    print(f"\n✅ Task 2 deliverables ready!")
    print(f"   - 8 diverse samples: {OUTPUT_PATH}/vae_sample_*.npy")
    print(f"   - 8 interpolation steps: {OUTPUT_PATH}/interpolation_step_*.npy")
    print(f"\n📊 Next step: Run evaluation metrics:")
    print(f"   python src/evaluation/metrics.py")

if __name__ == "__main__":
    main()