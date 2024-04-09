import torch.nn as nn
import torch
import torch.nn.functional as F

from EncoderDecoder.layers.Decoder.DecoderBlock import BlockDecoder


class Decoder(nn.Module):

    def __init__(self, vocab_size, n_embed, num_heads, seq_len):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(seq_len, n_embed)
        self.blocks = [BlockDecoder(n_embed, num_heads) for _ in range(3)]
        self.blocks = nn.ModuleList(self.blocks)
        self.layer_norm = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)
        self.seq_len = seq_len

    def forward(self, idx, k, v, targets=None, mask=None):
        B, T = idx.shape
        token_embedding = self.token_embedding_table(idx)
        position_embedding = self.position_embedding_table(torch.arange(T))
        x = token_embedding + position_embedding
        for block in self.blocks:
            x = block(x, k, v, mask)
        x = self.layer_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, n):
        for _ in range(n):
            idx_crop = idx[:, -self.seq_len:]
            logits, _ = self(idx_crop)
            logits = logits[:, -1, :]
            p = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(p, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx
