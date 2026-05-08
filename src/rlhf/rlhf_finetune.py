"""
Fine-tune Transformer using PPO-style RLHF
Improves generation quality based on human preferences
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import sys

sys.path.append(str(Path(__file__).parent.parent))
from models.transformer import MusicTransformer

# Load reward model
from reward_model import RewardModel

def main():
    print("="*60)
    print("RLHF Fine-tuning of Transformer")
    print("="*60)
    
    DATA_PATH = Path("data/processed/task1")
    TRANSFORMER_PATH = Path("checkpoints/task3_transformer")
    REWARD_PATH = Path("checkpoints/task4_rlhf")
    OUTPUT_PATH = Path("checkpoints/rlhf_finetuned")
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    
    # Load vocabulary
    with open(DATA_PATH / "metadata.json", 'r') as f:
        metadata = json.load(f)
    vocab_size = metadata["vocab_size"]
    
    # Load base transformer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transformer = MusicTransformer(
        vocab_size=vocab_size,
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_feedforward=1024
    )
    transformer.load_state_dict(torch.load(TRANSFORMER_PATH / "best_model.pt", map_location=device))
    transformer = transformer.to(device)
    transformer.train()
    
    # Load reward model
    reward_model = RewardModel(vocab_size).to(device)
    reward_model.load_state_dict(torch.load(REWARD_PATH / "reward_model.pt", map_location=device))
    reward_model.eval()
    
    # Freeze reward model
    for param in reward_model.parameters():
        param.requires_grad = False
    
    # Load validation data for prompts
    val_seqs = np.load(DATA_PATH / "val_seqs_shifted.npy")
    
    # RLHF hyperparameters
    LEARNING_RATE = 1e-5  # Small LR for fine-tuning
    PPO_CLIP = 0.2
    VALUE_LOSS_COEF = 0.5
    ENTROPY_COEF = 0.01
    
    optimizer = torch.optim.AdamW(transformer.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    # RLHF training loop
    print("\n🚀 Fine-tuning with RLHF...")
    
    rlhf_rewards = []
    
    for iteration in range(10):  # 10 RL iterations
        print(f"\n--- RLHF Iteration {iteration + 1}/10 ---")
        
        # Generate samples with current policy
        generated_samples = []
        prompts = []
        
        for _ in range(16):  # Generate 16 samples per iteration
            # Random prompt
            prompt_length = np.random.randint(64, 128)
            random_idx = np.random.randint(0, len(val_seqs))
            prompt_tokens = val_seqs[random_idx][:prompt_length]
            prompt = torch.LongTensor(prompt_tokens).unsqueeze(0).to(device)
            prompts.append(prompt)
            
            # Generate
            with torch.no_grad():
                generated = transformer.generate(prompt, max_new_tokens=256, temperature=1.0, top_k=50)
            generated_samples.append(generated)
        
        # Get reward scores for generated samples
        rewards = []
        for sample in generated_samples:
            with torch.no_grad():
                reward = reward_model(sample).squeeze().item()
            rewards.append(reward)
        
        avg_reward = np.mean(rewards)
        rlhf_rewards.append(avg_reward)
        print(f"  Average reward: {avg_reward:.3f} (scale 1-5)")
        
        # Policy gradient update
        for sample, prompt, reward in zip(generated_samples, prompts, rewards):
            # Normalize reward
            normalized_reward = (reward - 3.0) / 2.0  # Scale -1 to 1
            
            # Forward pass
            logits = transformer(sample)
            
            # Compute loss with reward weighting
            shift_logits = logits[:, :-1, :].contiguous()
            shift_targets = sample[:, 1:].contiguous()
            
            # Negative log likelihood weighted by reward
            log_probs = -criterion(shift_logits.view(-1, vocab_size), shift_targets.view(-1))
            
            # Policy gradient: maximize reward * log_prob
            policy_loss = -normalized_reward * log_probs
            
            # Also maintain language modeling loss
            lm_loss = criterion(shift_logits.view(-1, vocab_size), shift_targets.view(-1))
            
            # Combined loss
            total_loss = policy_loss + lm_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(transformer.parameters(), max_norm=1.0)
            optimizer.step()
    
    # Save fine-tuned model
    torch.save(transformer.state_dict(), OUTPUT_PATH / "rlhf_finetuned_model.pt")
    print(f"\n✓ RLHF fine-tuned model saved to {OUTPUT_PATH}")
    
    # Plot reward progress
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.plot(rlhf_rewards, marker='o')
    plt.xlabel('RLHF Iteration')
    plt.ylabel('Average Human Reward (1-5)')
    plt.title('RLHF: Improvement from Human Feedback')
    plt.grid(True)
    plt.savefig(OUTPUT_PATH / "rlhf_progress.png")
    print(f"✓ Progress plot saved")
    
    print("\n✅ RLHF Fine-tuning Complete!")
    print("\n📊 Next: Compare before/after samples")

if __name__ == "__main__":
    main()