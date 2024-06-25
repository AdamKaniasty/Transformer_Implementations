import random

import torch
import torch.nn as nn

from seq2seq.layers.decoder import Decoder
from seq2seq.layers.encoder import Encoder


class Seq2Seq(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, y, teacher_forcing_ratio):
        batch_size = y.shape[1]
        y_len = y.shape[0]
        vocab_size = self.decoder.output_dim

        outputs = torch.zeros(y_len, batch_size, vocab_size)

        hidden, cell = self.encoder(x)
        inp = y[0, :]
        for t in range(1, y_len):
            output, hidden, cell = self.decoder(inp, hidden, cell)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            inp = y[t] if teacher_force else top1

        return outputs
