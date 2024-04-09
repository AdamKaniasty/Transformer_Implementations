import torch.nn as nn
import torch

from EncoderDecoder.layers.Head import Head


class MultiHead(nn.Module):
    def __init__(self, num_heads, n_embed, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embed, head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, k=None, v=None, mask=None):
        out = torch.cat([h(x, k, v, mask) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))
