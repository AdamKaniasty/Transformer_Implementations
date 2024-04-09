import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
    def __init__(self, n_embed, head_size, decoder=True):
        super().__init__()
        if decoder:
            self.key = nn.Linear(n_embed, head_size, bias=False)
            self.value = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.head_size = head_size
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, k=None, v=None, mask=None):
        B, T, C = x.shape
        if k is None and v is None:
            k = self.key(x)
            v = self.value(x)
        q = self.query(x)
        weights = q @ k.transpose(-1, -2) * C ** -0.5
        if mask is not None:
            weights = weights.masked_fill(mask == 0, float('-inf'))
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)
        return weights @ v
