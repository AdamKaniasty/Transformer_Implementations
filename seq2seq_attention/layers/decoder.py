import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, output_dim, embedding_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, inp, hidden, cell):
        inp = inp.unsqueeze(0)
        embedded = self.dropout(self.embedding(inp))
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))

        pred = self.fc(output.squeeze(0))
        return pred, hidden, cell
