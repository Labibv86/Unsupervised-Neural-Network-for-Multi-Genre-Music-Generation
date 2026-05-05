"""
Variational Autoencoder for Multi-Genre Music Generation (Task 2)
Fixed architecture with proper attribute access
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class VAEEncoder(nn.Module):
    """Encoder with Gaussian output for VAE"""
    
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, latent_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Project to mean and log variance
        self.fc_mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len) token indices
        Returns:
            mu: (batch, latent_dim)
            logvar: (batch, latent_dim)
        """
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Use last hidden state from both directions
        hidden_forward = hidden[-2]  # Last forward layer
        hidden_backward = hidden[-1]  # Last backward layer
        hidden_concat = torch.cat([hidden_forward, hidden_backward], dim=1)
        
        mu = self.fc_mu(hidden_concat)
        logvar = self.fc_logvar(hidden_concat)
        
        # Clamp logvar for numerical stability
        logvar = torch.clamp(logvar, min=-5, max=5)
        
        return mu, logvar

class VAEDecoder(nn.Module):
    """Decoder for VAE"""
    
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, latent_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        
        self.latent_proj = nn.Linear(latent_dim, hidden_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
    def forward(self, z, target_seq=None, max_len=512, teacher_forcing_ratio=0.5):
        batch_size = z.size(0)
        device = z.device
        
        # Project latent to initial hidden state
        h0 = self.latent_proj(z).unsqueeze(0).repeat(self.num_layers, 1, 1)
        c0 = torch.zeros_like(h0)
        
        if self.training and target_seq is not None:
            # Teacher forcing mode
            embedded = self.embedding(target_seq)
            embedded = self.dropout(embedded)
            lstm_out, _ = self.lstm(embedded, (h0, c0))
            output = self.fc_out(lstm_out)
            return output
        else:
            # Generation mode (autoregressive)
            outputs = []
            current_token = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
            h, c = h0, c0
            
            for t in range(max_len):
                embedded = self.embedding(current_token)
                lstm_out, (h, c) = self.lstm(embedded, (h, c))
                logits = self.fc_out(lstm_out)
                outputs.append(logits)
                
                # Sample with temperature for diversity
                probs = F.softmax(logits / 1.2, dim=-1)
                current_token = torch.multinomial(probs.squeeze(1), 1)
            
            return torch.cat(outputs, dim=1)

class MusicVAE(nn.Module):
    """Full VAE model with reparameterization trick"""
    
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, latent_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        
        self.encoder = VAEEncoder(vocab_size, embed_dim, hidden_dim, latent_dim, num_layers, dropout)
        self.decoder = VAEDecoder(vocab_size, embed_dim, hidden_dim, latent_dim, num_layers, dropout)
        
        self.latent_dim = latent_dim
        
    def reparameterize(self, mu, logvar):
        """Reparameterization trick: z = mu + sigma * epsilon"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x, teacher_forcing_ratio=0.5):
        """Forward pass with KL loss computation"""
        # Encode
        mu, logvar = self.encoder(x)
        
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        
        # Decode
        recon = self.decoder(z, target_seq=x, teacher_forcing_ratio=teacher_forcing_ratio)
        
        return recon, mu, logvar
    
    def generate(self, z, max_len=512):
        """Generate music from latent vector"""
        self.eval()
        with torch.no_grad():
            generated = self.decoder(z, max_len=max_len, teacher_forcing_ratio=0)
        return generated.argmax(dim=-1)
    
    def sample_latent(self, batch_size=1, device='cuda'):
        """Sample random latent vectors from prior N(0,1)"""
        return torch.randn(batch_size, self.latent_dim, device=device)
    
    def interpolate(self, z1, z2, steps=8):
        """Linear interpolation between two latent vectors"""
        alphas = torch.linspace(0, 1, steps)
        interpolated = []
        for alpha in alphas:
            z = (1 - alpha) * z1 + alpha * z2
            interpolated.append(z)
        return torch.stack(interpolated)