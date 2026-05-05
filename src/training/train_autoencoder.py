"""
Train LSTM Autoencoder for Task 1
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add parent directory to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

# Configuration
DATA_PATH = Path("data/processed/task1")
CHECKPOINT_PATH = Path("checkpoints/task1_autoencoder")
CHECKPOINT_PATH.mkdir(parents=True, exist_ok=True)

# Hyperparameters
BATCH_SIZE = 128
EPOCHS = 15  # Start with 30 epochs
LEARNING_RATE = 1e-4
EMBED_DIM = 32
HIDDEN_DIM = 64
LATENT_DIM = 32
NUM_LAYERS = 3
BETA = 0.01

def main():
    # Check if data exists
    if not (DATA_PATH / "train_seqs.npy").exists():
        print("✗ No training data found!")
        print("Please run tokenization first:")
        print("  python src/preprocessing/tokenize_dataset.py")
        return
    
    # Load data
    print("Loading data...")
    train_seqs = np.load(DATA_PATH / "train_seqs_shifted.npy")
    val_seqs = np.load(DATA_PATH / "val_seqs_shifted.npy")
    
    with open(DATA_PATH / "metadata.json", "r") as f:
        metadata = json.load(f)
    
    vocab_size = metadata["vocab_size"]
    print(f"✓ Vocabulary size: {vocab_size}")
    print(f"✓ Train sequences: {train_seqs.shape}")
    print(f"✓ Validation sequences: {val_seqs.shape}")
    
    # Create dataloaders
    train_dataset = TensorDataset(torch.LongTensor(train_seqs))
    val_dataset = TensorDataset(torch.LongTensor(val_seqs))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # Import model (lazy import to avoid circular imports)
    from models.autoencoder import LSTMVAE
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMVAE(vocab_size, EMBED_DIM, HIDDEN_DIM, LATENT_DIM, NUM_LAYERS)
    model = model.to(device)
    print(f"✓ Model on: {device}")
    
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Loss function
    recon_criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    def loss_function(recon, target, mu, logvar, beta=0.01):
        recon_loss = recon_criterion(recon.view(-1, recon.size(-1)), target.view(-1))
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / target.size(0)
        return recon_loss + beta * kl_loss, recon_loss, kl_loss
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Training loop
    train_losses = []
    val_losses = []

    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    print("\n🚀 Starting training...\n")
    
    for epoch in range(EPOCHS):
        # Training
        model.train()
        epoch_train_loss = 0
        epoch_recon_loss = 0
        epoch_kl_loss = 0
        
        for batch_idx, (x,) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")):
            x = x.to(device)
            
            optimizer.zero_grad()
            recon, mu, logvar = model(x)
            loss, recon_loss, kl_loss = loss_function(recon, x, mu, logvar, BETA)
            
            loss.backward()
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
                loss, _, _ = loss_function(recon, x, mu, logvar, BETA)
                epoch_val_loss += loss.item()
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_val_loss = epoch_val_loss / len(val_loader)
        avg_recon = epoch_recon_loss / len(train_loader)
        avg_kl = epoch_kl_loss / len(train_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1:3d} | Train: {avg_train_loss:.4f} | Recon: {avg_recon:.4f} | KL: {avg_kl:.4f} | Val: {avg_val_loss:.4f}")
        
        

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping!")
                break


        scheduler.step(avg_val_loss)
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, CHECKPOINT_PATH / f"checkpoint_epoch_{epoch+1}.pt")
    
    # Save final model
    torch.save(model.state_dict(), CHECKPOINT_PATH / "final_model.pt")
    print(f"\n✓ Model saved to {CHECKPOINT_PATH}")
    
    # Plot loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('LSTM Autoencoder Training - Task 1')
    plt.legend()
    plt.grid(True)
    plt.savefig(CHECKPOINT_PATH / "loss_curve.png")
    print(f"✓ Loss curve saved to {CHECKPOINT_PATH}/loss_curve.png")
    
    print("\n✅ Training complete!")

if __name__ == "__main__":
    main()