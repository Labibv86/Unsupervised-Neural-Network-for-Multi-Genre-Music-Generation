"""
Train Transformer for Long Coherent Music Generation (Task 3)
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
from models.transformer import MusicTransformer

def main():
    # Configuration
    DATA_PATH = Path("data/processed/task1")
    CHECKPOINT_PATH = Path("checkpoints/task3_transformer")
    CHECKPOINT_PATH.mkdir(parents=True, exist_ok=True)

    # Hyperparameters (optimized for RTX 3060 12GB)
    BATCH_SIZE = 16  # Smaller than VAE due to attention
    EPOCHS = 30
    LEARNING_RATE = 1e-4  # Lower LR for transformer
    
    # Model architecture
    D_MODEL = 256      # Embedding dimension
    NHEAD = 8          # Attention heads
    NUM_LAYERS = 4     # Transformer layers
    DIM_FEEDFORWARD = 1024
    DROPOUT = 0.1
    
    # Sequence length for training
    SEQ_LEN = 512
    CONTEXT_LEN = 256  # Shorter for faster training
    
    # Load data
    print("Loading data...")
    train_seqs = np.load(DATA_PATH / "train_seqs_fixed.npy")
    val_seqs = np.load(DATA_PATH / "val_seqs_fixed.npy")
    
    # Use shorter sequences for transformer
    train_seqs = train_seqs[:, :CONTEXT_LEN]
    val_seqs = val_seqs[:, :CONTEXT_LEN]
    
    with open(DATA_PATH / "metadata.json", "r") as f:
        metadata = json.load(f)
    
    vocab_size = metadata["vocab_size"]
    print(f"✓ Vocabulary size: {vocab_size}")
    print(f"✓ Train sequences: {train_seqs.shape}")
    print(f"✓ Val sequences: {val_seqs.shape}")
    print(f"✓ Sequence length: {CONTEXT_LEN}")
    
    # Create dataloaders
    train_dataset = TensorDataset(torch.LongTensor(train_seqs))
    val_dataset = TensorDataset(torch.LongTensor(val_seqs))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MusicTransformer(
        vocab_size=vocab_size,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT
    )
    model = model.to(device)
    print(f"✓ Model on: {device}")
    print(f"✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # Training
    train_losses = []
    val_losses = []
    train_perplexities = []
    val_perplexities = []
    
    best_val_perplexity = float('inf')
    patience_counter = 0
    
    print("\n🚀 Training Transformer...\n")
    
    for epoch in range(EPOCHS):
        # Training
        model.train()
        epoch_train_loss = 0
        
        for batch_idx, (x,) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")):
            x = x.to(device)
            
            # Forward pass
            logits = model(x)
            
            # Shift for next token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_targets = x[:, 1:].contiguous()
            
            loss = criterion(shift_logits.view(-1, vocab_size), shift_targets.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_train_loss += loss.item()
        
        # Validation
        model.eval()
        epoch_val_loss = 0
        
        with torch.no_grad():
            for batch_idx, (x,) in enumerate(val_loader):
                x = x.to(device)
                logits = model(x)
                
                shift_logits = logits[:, :-1, :].contiguous()
                shift_targets = x[:, 1:].contiguous()
                
                loss = criterion(shift_logits.view(-1, vocab_size), shift_targets.view(-1))
                epoch_val_loss += loss.item()
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_val_loss = epoch_val_loss / len(val_loader)
        
        # Compute perplexity
        train_perp = np.exp(avg_train_loss)
        val_perp = np.exp(avg_val_loss)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_perplexities.append(train_perp)
        val_perplexities.append(val_perp)
        
        print(f"Epoch {epoch+1:3d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Train PPL: {train_perp:.2f} | Val PPL: {val_perp:.2f}")
        
        scheduler.step()
        
        # Save best model
        if val_perp < best_val_perplexity:
            best_val_perplexity = val_perp
            patience_counter = 0
            torch.save(model.state_dict(), CHECKPOINT_PATH / "best_model.pt")
            print(f"  ✓ Best model saved (Perplexity: {best_val_perplexity:.2f})")
        else:
            patience_counter += 1
            if patience_counter >= 7:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'train_perplexity': train_perp,
                'val_perplexity': val_perp
            }, CHECKPOINT_PATH / f"checkpoint_epoch_{epoch+1}.pt")
    
    # Save final model
    torch.save(model.state_dict(), CHECKPOINT_PATH / "final_model.pt")
    print(f"\n✓ Model saved to {CHECKPOINT_PATH}")
    
    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    axes[0].plot(train_losses, label='Train Loss')
    axes[0].plot(val_losses, label='Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Transformer Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Perplexity plot
    axes[1].plot(train_perplexities, label='Train Perplexity')
    axes[1].plot(val_perplexities, label='Validation Perplexity')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Perplexity')
    axes[1].set_title('Perplexity (Lower is Better)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(CHECKPOINT_PATH / "training_curves.png", dpi=150)
    print(f"✓ Training curves saved")
    
    # Summary
    print("\n" + "="*60)
    print("📊 TRAINING SUMMARY - Task 3 (Transformer)")
    print("="*60)
    print(f"✓ Best Validation Perplexity: {best_val_perplexity:.2f}")
    print(f"✓ Final Train Loss: {train_losses[-1]:.4f}")
    print(f"✓ Final Val Loss: {val_losses[-1]:.4f}")
    print("\n✅ Task 3 training complete!")
    print("\nNext: Generate long compositions")

if __name__ == "__main__":
    main()