"""
Transformer Decoder for Music Generation (Task 3)
Autoregressive generation with causal attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerDecoder(nn.Module):
    """Transformer decoder for autoregressive music generation"""
    
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=4, dim_feedforward=1024, dropout=0.1, max_len=2048):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Embeddings
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output projection
        self.fc_out = nn.Linear(d_model, vocab_size)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def generate_causal_mask(self, sz, device):
        """Create causal mask for autoregressive generation"""
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()
        return mask
    
    def forward(self, x, memory=None, mask=None):
        """
        Args:
            x: (batch, seq_len) input token indices
            memory: Optional memory from encoder (not used in decoder-only)
        Returns:
            output: (batch, seq_len, vocab_size) logits
        """
        device = x.device
        seq_len = x.shape[1]
        
        # Embed and add positional encoding
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        
        # Create causal mask (prevents looking at future tokens)
        if mask is None:
            mask = self.generate_causal_mask(seq_len, device)
        
        # For decoder-only, we use the same sequence as both memory and target
        output = self.transformer_decoder(x, x, tgt_mask=mask)
        
        # Project to vocabulary
        output = self.fc_out(output)
        
        return output
    
    def generate(self, start_tokens, max_new_tokens=512, temperature=1.0, top_k=50):
        """
        Generate new tokens autoregressively
        
        Args:
            start_tokens: (batch, seq_len) initial tokens
            max_new_tokens: number of new tokens to generate
            temperature: sampling temperature (higher = more random)
            top_k: only sample from top k tokens
        Returns:
            generated: (batch, seq_len + max_new_tokens) full sequence
        """
        self.eval()
        device = start_tokens.device
        generated = start_tokens.clone()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Get logits for next token
                logits = self.forward(generated)  # (batch, current_len, vocab_size)
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][:, -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to sequence
                generated = torch.cat([generated, next_token], dim=1)
                
        return generated

class MusicTransformer(nn.Module):
    """Simplified wrapper for Transformer model"""
    
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=4, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        
        self.decoder = TransformerDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
    def forward(self, x):
        """Forward pass for training"""
        return self.decoder(x)
    
    def generate(self, prompt, max_new_tokens=512, temperature=1.0, top_k=50):
        """Generate continuation from prompt"""
        return self.decoder.generate(prompt, max_new_tokens, temperature, top_k)
    
    def compute_perplexity(self, x):
        """
        Compute perplexity on validation set
        Perplexity = exp(average negative log likelihood)
        Lower is better
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)  # (batch, seq_len, vocab_size)
            
            # Shift for next token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_targets = x[:, 1:].contiguous()
            
            # Compute cross entropy
            loss_fn = nn.CrossEntropyLoss(ignore_index=0)
            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_targets.view(-1))
            
            # Perplexity = exp(loss)
            perplexity = torch.exp(loss)
            
        return perplexity.item(), loss.item()