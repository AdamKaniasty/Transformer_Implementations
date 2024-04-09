import torch.nn as nn
import torch

from EncoderDecoder.layers.FeedForward import FeedForward
from EncoderDecoder.layers.MultiHead import MultiHead


class BlockDecoder(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.msa_heads = MultiHead(n_head, n_embed, head_size)
        self.sa_heads = MultiHead(n_head, n_embed, head_size)
        self.feed_forward = FeedForward(n_embed)
        self.layer_norm1 = nn.LayerNorm(n_embed)
        self.layer_norm2 = nn.LayerNorm(n_embed)
        self.layer_norm3 = nn.LayerNorm(n_embed)

    def forward(self, x, k, v, seq_len, mask=None):
        x = x + self.msa_heads(self.layer_norm1(x), k, v, mask)
        x = x + self.sa_heads(self.layer_norm2(x), k, v)
        x = x + self.feed_forward(self.layer_norm3(x))
        return x
