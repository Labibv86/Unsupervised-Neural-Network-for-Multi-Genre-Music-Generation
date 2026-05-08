"""
Train VAE for Multi-Genre Music Generation (Task 2)
Numerically stable version with gradient clipping and KL clamping
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

sys.path.append(str(Path(__file__).parent.parent))
from models.vae import MusicVAE

def main():
    # Configuration
    DATA_PATH = Path("data/processed/task1")
    CHECKPOINT_PATH = Path("checkpoints/task2_vae")
    CHECKPOINT_PATH.mkdir(parents=True, exist_ok=True)

    # Hyperparameters - Conservative for stability
    BATCH_SIZE = 64  # Reduced from 64 for stability
    EPOCHS = 20
    LEARNING_RATE = 1e-4  # Reduced from 1e-3
    
    # Model size - Smaller for faster training
    EMBED_DIM = 8       # Reduced from 128
    HIDDEN_DIM = 16     # Reduced from 256
    LATENT_DIM = 8      # Reduced from 64
    NUM_LAYERS = 1
    
    # KL Annealing parameters
    BETA_START = 0.0
    BETA_END = 0.001      # Lower end to prevent collapse
    BETA_WARMUP_EPOCHS = 10
    
    # Regularization
    WEIGHT_DECAY = 1e-4
    DROPOUT = 0.1

    # Load data
    print("Loading data...")
    train_seqs = np.load(DATA_PATH / "train_seqs_shifted.npy")
    val_seqs = np.load(DATA_PATH / "val_seqs_shifted.npy")

    with open(DATA_PATH / "metadata.json", "r") as f:
        metadata = json.load(f)

    vocab_size = metadata["vocab_size"]
    print(f"✓ Vocabulary size: {vocab_size}")
    print(f"✓ Train sequences: {train_seqs.shape}")
    print(f"✓ Val sequences: {val_seqs.shape}")

    # Use only subset of data for faster training
    # Remove these lines if you want full dataset
    max_train = 20000  # Use 20k sequences for faster training
    if len(train_seqs) > max_train:
        train_seqs = train_seqs[:max_train]
        print(f"✓ Using {max_train} train sequences (for speed)")

    # Create dataloaders
    train_dataset = TensorDataset(torch.LongTensor(train_seqs))
    val_dataset = TensorDataset(torch.LongTensor(val_seqs))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Initialize model with stable dropout
    model = MusicVAE(
        vocab_size=vocab_size,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"✓ Model on: {device}")
    print(f"✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"✓ Latent dimension: {LATENT_DIM}")

    # Loss function
    recon_criterion = nn.CrossEntropyLoss(ignore_index=0)

    def stable_kl_loss(mu, logvar):
        """Numerically stable KL divergence calculation"""
        # Clamp logvar to prevent numerical issues
        logvar = torch.clamp(logvar, min=-10, max=10)
        
        # Calculate KL
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Clamp KL to prevent extreme values
        kl = torch.clamp(kl, min=0, max=100)
        
        return kl / mu.size(0)  # Normalize by batch size

    def vae_loss(recon, target, mu, logvar, beta):
        """Stable VAE loss"""
        # Reconstruction loss
        recon_loss = recon_criterion(recon.view(-1, recon.size(-1)), target.view(-1))
        
        # Stable KL loss
        kl_loss = stable_kl_loss(mu, logvar)
        
        # Total loss
        total_loss = recon_loss + beta * kl_loss
        
        return total_loss, recon_loss, kl_loss

    # Optimizer with lower learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    # Training tracking
    train_losses = []
    val_losses = []
    recon_losses = []
    kl_losses = []
    betas_used = []
    
    best_val_loss = float('inf')
    patience_counter = 0

    print("\n" + "="*60)
    print("🚀 Training Stable VAE (KL Annealing Enabled)")
    print("="*60)
    print(f"   Beta warmup: {BETA_START} → {BETA_END} over {BETA_WARMUP_EPOCHS} epochs")
    print(f"   Learning rate: {LEARNING_RATE}")
    print(f"   Batch size: {BATCH_SIZE}")
    print("="*60 + "\n")

    for epoch in range(EPOCHS):
        # Calculate beta for this epoch (KL annealing)
        if epoch < BETA_WARMUP_EPOCHS:
            beta = BETA_START + (BETA_END - BETA_START) * (epoch / BETA_WARMUP_EPOCHS)
        else:
            beta = BETA_END
        
        betas_used.append(beta)
        
        # Training
        model.train()
        epoch_train_loss = 0
        epoch_recon_loss = 0
        epoch_kl_loss = 0

        for batch_idx, (x,) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")):
            x = x.to(device)

            optimizer.zero_grad()
            
            # Forward pass
            recon, mu, logvar = model(x)
            loss, recon_loss, kl_loss = vae_loss(recon, x, mu, logvar, beta)
            
            # Check for NaN
            if torch.isnan(loss):
                print(f"⚠️ NaN detected at batch {batch_idx}, skipping...")
                continue
            
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            epoch_train_loss += loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_kl_loss += kl_loss.item()

        # Validation
        model.eval()
        epoch_val_loss = 0

        with torch.no_grad():
            for batch_idx, (x,) in enumerate(val_loader):
                x = x.to(device)
                recon, mu, logvar = model(x)
                loss, _, _ = vae_loss(recon, x, mu, logvar, beta)
                epoch_val_loss += loss.item()

        # Calculate averages
        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_val_loss = epoch_val_loss / len(val_loader)
        avg_recon = epoch_recon_loss / len(train_loader)
        avg_kl = epoch_kl_loss / len(train_loader)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        recon_losses.append(avg_recon)
        kl_losses.append(avg_kl)

        # Print progress
        print(f"Epoch {epoch+1:3d} | Beta: {beta:.3f} | Train: {avg_train_loss:.4f} | Recon: {avg_recon:.4f} | KL: {avg_kl:.4f} | Val: {avg_val_loss:.4f}")

        # Early stopping (only after warmup)
        if epoch >= BETA_WARMUP_EPOCHS:
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(model.state_dict(), CHECKPOINT_PATH / "best_model.pt")
                print(f"  ✓ Best model saved (val_loss: {best_val_loss:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= 3:
                    print(f"\n🛑 Early stopping at epoch {epoch+1}")
                    break
        
        # Adjust learning rate
        scheduler.step(avg_val_loss)

    # Save final model
    torch.save(model.state_dict(), CHECKPOINT_PATH / "final_model.pt")
    print(f"\n✓ Model saved to {CHECKPOINT_PATH}")

    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss curves
    axes[0, 0].plot(train_losses, label='Train Loss', linewidth=2)
    axes[0, 0].plot(val_losses, label='Validation Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training & Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Reconstruction vs KL
    axes[0, 1].plot(recon_losses, label='Reconstruction Loss', color='green', linewidth=2)
    axes[0, 1].plot(kl_losses, label='KL Loss', color='orange', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Reconstruction vs KL Divergence')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Beta annealing
    axes[1, 0].plot(betas_used, label='Beta (KL Weight)', color='purple', linewidth=2)
    axes[1, 0].axhline(y=BETA_END, color='red', linestyle='--', label=f'Target Beta={BETA_END}')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Beta')
    axes[1, 0].set_title('KL Annealing Schedule')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Final loss distribution
    axes[1, 1].bar(['Train Loss', 'Val Loss', 'Recon Loss', 'KL Loss'], 
                   [train_losses[-1], val_losses[-1], recon_losses[-1], kl_losses[-1]],
                   color=['blue', 'red', 'green', 'orange'])
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].set_title('Final Loss Values')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(CHECKPOINT_PATH / "training_curves.png", dpi=150)
    print(f"✓ Training curves saved to {CHECKPOINT_PATH}/training_curves.png")
    plt.show()

    # Summary
    print("\n" + "="*60)
    print("📊 TRAINING SUMMARY")
    print("="*60)
    print(f"✓ Final Train Loss: {train_losses[-1]:.4f}")
    print(f"✓ Final Val Loss: {val_losses[-1]:.4f}")
    print(f"✓ Final Reconstruction Loss: {recon_losses[-1]:.4f}")
    print(f"✓ Final KL Loss: {kl_losses[-1]:.4f}")
    print(f"✓ Best Validation Loss: {best_val_loss:.4f}")
    
    # VAE success check
    if kl_losses[-1] > 0.1:
        print("\n✅ SUCCESS: KL Loss > 0.1 - VAE is learning a meaningful latent space!")
    elif kl_losses[-1] > 0.05:
        print("\n⚠️ PARTIAL: KL Loss is positive but low - Consider increasing BETA_END to 1.0")
    else:
        print("\n❌ WARNING: KL Loss near zero - VAE may have collapsed to autoencoder")
    
    print("\n✅ Task 2 training complete!")
    print("\nNext steps:")
    print("  1. Run: python src/generation/generate_vae_samples.py")
    print("  2. Run: python src/evaluation/metrics.py")

if __name__ == "__main__":
    main()