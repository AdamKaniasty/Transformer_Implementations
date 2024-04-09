import torch.nn as nn
import torch

from EncoderDecoder.layers.Encoder.EncoderBlock import BlockEncoder


class Encoder(nn.Module):

    def __init__(self, vocab_size, n_embed, num_heads, seq_len):
        super().__init__()
        head_size = n_embed // num_heads
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(seq_len, n_embed)
        self.blocks = nn.ModuleList([BlockEncoder(n_embed, num_heads, head_size) for _ in range(2)])
        self.layer_norm = nn.LayerNorm(n_embed)
        self.key_projection = nn.Linear(n_embed, head_size)
        self.value_projection = nn.Linear(n_embed, head_size)

    def forward(self, idx, prompt_mask=None):
        B, T = idx.shape
        token_embedding = self.token_embedding_table(idx)
        position_embedding = self.position_embedding_table(torch.arange(T))
        x = token_embedding + position_embedding
        for block in self.blocks:
            x = block(x, prompt_mask)
        x = self.layer_norm(x)
        k = self.key_projection(x)
        v = self.value_projection(x)
        return k, v
