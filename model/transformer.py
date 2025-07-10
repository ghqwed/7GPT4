import torch
import torch.nn as nn
from torch.nn import functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.model.n_heads
        self.dim = config.model.dim
        self.head_dim = self.dim // self.n_heads
        
        self.qkv = nn.Linear(self.dim, 3 * self.dim)
        self.proj = nn.Linear(self.dim, self.dim)
        
    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).split(self.dim, dim=2)
        q, k, v = [y.view(B, T, self.n_heads, self.head_dim).transpose(1, 2) for y in qkv]
        
        attn = (q @ k.transpose(-2, -1)) * (1.0 / torch.sqrt(torch.tensor(self.head_dim)))
        attn = F.softmax(attn, dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.model.dim, 4 * config.model.dim),
            nn.GELU(),
            nn.Linear(4 * config.model.dim, config.model.dim)
        )
        
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.model.dim)
        self.attn = MultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config.model.dim)
        self.ff = FeedForward(config)
        
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.token_embed = nn.Embedding(config.model.vocab_size, config.model.dim)
        self.pos_embed = nn.Embedding(config.model.max_seq_len, config.model.dim)
        
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.model.n_layers)])
        self.ln_f = nn.LayerNorm(config.model.dim)
        self.head = nn.Linear(config.model.dim, config.model.vocab_size, bias=False)
        
    def forward(self, idx, return_hidden=False):
        B, T = idx.shape
        tok_emb = self.token_embed(idx)
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.pos_embed(pos)
        
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        if return_hidden:
            return x
        logits = self.head(x)
        return logits
