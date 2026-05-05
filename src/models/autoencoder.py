"""
LSTM Autoencoder for Music Generation (Task 1)
- Encoder: Compresses token sequences to latent vector
- Decoder: Reconstructs sequence from latent vector
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMEncoder(nn.Module):
    """Encodes token sequence to latent vector"""
    
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512, latent_dim=256, num_layers=3, dropout=0.2):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        
        # Project bidirectional output to latent vector
        self.fc_mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dim)
        
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len) token indices
        Returns:
            z: (batch, latent_dim) latent vector
        """
        # Embed tokens
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)
        
        # LSTM encoding
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Use last hidden state from both directions
        hidden_forward = hidden[-2]  # Last forward layer
        hidden_backward = hidden[-1]  # Last backward layer
        hidden_concat = torch.cat([hidden_forward, hidden_backward], dim=1)  # (batch, hidden_dim*2)
        
        # Project to latent
        mu = self.fc_mu(hidden_concat)
        logvar = self.fc_logvar(hidden_concat)
        
        return mu, logvar

class LSTMDecoder(nn.Module):
    """Decodes latent vector to token sequence"""
    
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512, latent_dim=256, num_layers=3, dropout=0.2):
        super().__init__()
        
        self.latent_proj = nn.Linear(latent_dim, hidden_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, z, target_seq=None, max_len=512, teacher_forcing_ratio=0.5):
        """
        Args:
            z: (batch, latent_dim) latent vector
            target_seq: (batch, seq_len) for teacher forcing (training)
            max_len: maximum generation length
            teacher_forcing_ratio: probability of using teacher forcing
        Returns:
            output: (batch, seq_len, vocab_size) logits
        """
        batch_size = z.size(0)
        
        # Project latent to initial hidden state
        h0 = self.latent_proj(z).unsqueeze(0).repeat(self.lstm.num_layers, 1, 1)
        c0 = torch.zeros_like(h0)
        
        if self.training and target_seq is not None:
            # Training with teacher forcing
            seq_len = target_seq.size(1)
            embedded = self.embedding(target_seq)  # (batch, seq_len, embed_dim)
            lstm_out, _ = self.lstm(embedded, (h0, c0))
            output = self.fc_out(lstm_out)
            return output
        else:
            # Generation mode (autoregressive)
            outputs = []
            current_token = torch.zeros(batch_size, 1, dtype=torch.long, device=z.device)
            h, c = h0, c0
            
            for t in range(max_len):
                embedded = self.embedding(current_token)  # (batch, 1, embed_dim)
                lstm_out, (h, c) = self.lstm(embedded, (h, c))
                logits = self.fc_out(lstm_out)  # (batch, 1, vocab_size)
                outputs.append(logits)
                
                # Sample next token
                current_token = logits.argmax(dim=-1)  # (batch, 1)
            
            return torch.cat(outputs, dim=1)

class LSTMVAE(nn.Module):
    """Full VAE with LSTM encoder and decoder"""
    
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512, latent_dim=256, num_layers=3, dropout=0.2):
        super().__init__()
        
        self.encoder = LSTMEncoder(vocab_size, embed_dim, hidden_dim, latent_dim, num_layers, dropout)
        self.decoder = LSTMDecoder(vocab_size, embed_dim, hidden_dim, latent_dim, num_layers, dropout)
        
        self.latent_dim = latent_dim
        
    def reparameterize(self, mu, logvar):
        """Reparameterization trick for VAE"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x, teacher_forcing_ratio=0.5):
        """
        Args:
            x: (batch, seq_len) input token sequences
        Returns:
            recon: (batch, seq_len, vocab_size) reconstructed logits
            mu: (batch, latent_dim)
            logvar: (batch, latent_dim)
        """
        # Encode
        mu, logvar = self.encoder(x)
        
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        
        # Decode
        recon = self.decoder(z, target_seq=x, teacher_forcing_ratio=teacher_forcing_ratio)
        
        return recon, mu, logvar
    
    def generate(self, z, max_len=512):
        """Generate sequence from latent vector"""
        self.eval()
        with torch.no_grad():
            generated = self.decoder(z, max_len=max_len, teacher_forcing_ratio=0)
        return generated.argmax(dim=-1)
    
    def sample_latent(self, batch_size=1, device='cuda'):
        """Sample random latent vectors from prior"""
        z = torch.randn(batch_size, self.latent_dim, device=device)
        return z