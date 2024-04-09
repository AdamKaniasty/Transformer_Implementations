import torch.nn as nn

from EncoderDecoder.layers.FeedForward import FeedForward
from EncoderDecoder.layers.MultiHead import MultiHead


class BlockEncoder(nn.Module):
    def __init__(self, n_embed, n_head, head_size):
        super().__init__()
        self.sa_heads = MultiHead(n_head, n_embed, head_size)
        self.feed_forward = FeedForward(n_embed)
        self.layer_norm1 = nn.LayerNorm(n_embed)
        self.layer_norm2 = nn.LayerNorm(n_embed)

    def forward(self, x, mask):
        x = x + self.sa_heads(self.layer_norm1(x), mask=mask)
        x = x + self.feed_forward(self.layer_norm2(x))
        return x
