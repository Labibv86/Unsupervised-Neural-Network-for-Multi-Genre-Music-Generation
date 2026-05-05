"""
Train Reward Model from Human Ratings
Predicts human preference score (1-5) from generated music
"""

import numpy as np
from pathlib import Path
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tqdm import tqdm

class RewardModel(nn.Module):
    """Simple LSTM to predict human ratings from tokens"""
    
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, output_dim=1):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        # Use last hidden state
        last_hidden = lstm_out[:, -1, :]
        reward = self.fc(last_hidden)
        return reward

def load_human_ratings():
    """Load and aggregate all human rating JSON files"""
    
    SURVEY_PATH = Path("outputs/rlhf_survey")
    
    # First try to load average_ratings.json (from mock_ratings)
    avg_ratings_file = SURVEY_PATH / "average_ratings.json"
    if avg_ratings_file.exists():
        with open(avg_ratings_file, 'r') as f:
            avg_ratings = json.load(f)
        print(f"✓ Loaded average ratings from {avg_ratings_file}")
        print(f"  {len(avg_ratings)} samples with average ratings")
        return avg_ratings
    
    # Otherwise, try to aggregate individual participant files
    rating_files = list(SURVEY_PATH.glob("participant_*.json"))
    rating_files.extend(list(SURVEY_PATH.glob("*.json")))
    
    # Exclude survey_samples.json
    rating_files = [f for f in rating_files if f.name not in ["survey_samples.json", "average_ratings.json"]]
    
    if len(rating_files) == 0:
        print("✗ No rating files found!")
        print("Please run mock_ratings.py first to generate synthetic ratings")
        return None
    
    all_ratings = {}
    
    for file in rating_files:
        with open(file, 'r') as f:
            data = json.load(f)
        
        for sample_id, rating_value in data.items():
            if sample_id not in all_ratings:
                all_ratings[sample_id] = []
            # Handle both dict and direct float formats
            if isinstance(rating_value, dict):
                all_ratings[sample_id].append(rating_value['rating'])
            else:
                all_ratings[sample_id].append(rating_value)
    
    # Average ratings per sample
    avg_ratings = {}
    for sample_id, ratings_list in all_ratings.items():
        avg_ratings[sample_id] = np.mean(ratings_list)
    
    print(f"✓ Loaded {len(rating_files)} rating files")
    print(f"✓ {len(avg_ratings)} samples rated")
    print(f"✓ Average rating: {np.mean(list(avg_ratings.values())):.2f}")
    
    return avg_ratings

def prepare_training_data(avg_ratings):
    """Match ratings with generated samples"""
    
    SURVEY_PATH = Path("outputs/rlhf_survey")
    DATA_PATH = Path("data/processed/task1")
    
    # Load generated samples
    samples_file = SURVEY_PATH / "survey_samples.json"
    if not samples_file.exists():
        print("✗ survey_samples.json not found!")
        print("Please run human_survey.py first")
        return None, None
    
    with open(samples_file, 'r') as f:
        sample_metadata = json.load(f)
    
    sample_tokens = []
    sample_ratings = []
    
    for sample_meta in sample_metadata:
        sample_id = f"sample_{sample_meta['id']}"
        if sample_id in avg_ratings:
            # Load token sequence
            sample_path = Path(sample_meta['file'])
            if sample_path.exists():
                tokens = np.load(sample_path)
                sample_tokens.append(tokens)
                sample_ratings.append(avg_ratings[sample_id])
    
    if len(sample_tokens) == 0:
        print("✗ No matching samples found!")
        return None, None
    
    # Pad sequences to same length
    max_len = max(len(s) for s in sample_tokens)
    padded_samples = []
    
    for s in sample_tokens:
        if len(s) < max_len:
            s = np.pad(s, (0, max_len - len(s)), constant_values=0)
        padded_samples.append(s)
    
    X = np.array(padded_samples)
    y = np.array(sample_ratings)
    
    print(f"✓ Training data: {X.shape[0]} samples, max length: {max_len}")
    print(f"  Rating distribution: min={y.min():.1f}, max={y.max():.1f}, mean={y.mean():.2f}")
    
    return X, y

def main():
    print("="*60)
    print("Training Reward Model from Human Feedback")
    print("="*60)
    
    # Load human ratings
    avg_ratings = load_human_ratings()
    if avg_ratings is None:
        return
    
    # Prepare data
    X, y = prepare_training_data(avg_ratings)
    if X is None:
        return
    
    # Load vocabulary
    DATA_PATH = Path("data/processed/task1")
    with open(DATA_PATH / "metadata.json", 'r') as f:
        metadata = json.load(f)
    vocab_size = metadata["vocab_size"]
    
    # Split train/val
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    # Create dataloaders
    train_dataset = TensorDataset(torch.LongTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.LongTensor(X_val), torch.FloatTensor(y_val))
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reward_model = RewardModel(vocab_size).to(device)
    
    # Training
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(reward_model.parameters(), lr=1e-3)
    
    print("\n🚀 Training Reward Model...")
    
    train_losses = []
    val_losses = []
    
    for epoch in range(50):
        # Training
        reward_model.train()
        epoch_train_loss = 0
        
        for x, y_true in tqdm(train_loader, desc=f"Epoch {epoch+1}/50"):
            x, y_true = x.to(device), y_true.to(device)
            
            optimizer.zero_grad()
            y_pred = reward_model(x).squeeze()
            loss = criterion(y_pred, y_true)
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
        
        # Validation
        reward_model.eval()
        epoch_val_loss = 0
        
        with torch.no_grad():
            for x, y_true in val_loader:
                x, y_true = x.to(device), y_true.to(device)
                y_pred = reward_model(x).squeeze()
                loss = criterion(y_pred, y_true)
                epoch_val_loss += loss.item()
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_val_loss = epoch_val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    
    # Save reward model
    CHECKPOINT_PATH = Path("checkpoints/task4_rlhf")
    CHECKPOINT_PATH.mkdir(parents=True, exist_ok=True)
    torch.save(reward_model.state_dict(), CHECKPOINT_PATH / "reward_model.pt")
    print(f"\n✓ Reward model saved to {CHECKPOINT_PATH}/reward_model.pt")
    
    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Reward Model Training')
    plt.legend()
    plt.grid(True)
    plt.savefig(CHECKPOINT_PATH / "reward_training.png")
    print(f"✓ Training curve saved")

if __name__ == "__main__":
    main()