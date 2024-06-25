import itertools
from collections import Counter


class Vocabulary:
    def __init__(self, texts, special_tokens, min_freq):
        self.text = texts
        self.special_tokens = special_tokens
        self.min_freq = min_freq
        self.stoi = None
        self.itos = None
        self.special_length = len(special_tokens)
        self._build_default_vocab()
        self._build_special_vocab()

    def _build_default_vocab(self):
        token_counts = Counter(itertools.chain.from_iterable(self.text))
        filtered_tokens = {token: count for token, count in token_counts.items() if count > self.min_freq}
        self.stoi = {token: idx + self.special_length for idx, (token, _) in enumerate(filtered_tokens.items())}
        self.itos = {idx + self.special_length: token for idx, (token, _) in enumerate(filtered_tokens.items())}

    def _build_special_vocab(self):
        for idx, token in enumerate(self.special_tokens):
            self.stoi[token] = idx
            self.itos[idx] = token

    def encode(self, token: str):
        if token not in self.stoi:
            return self.stoi['<unk>']
        return self.stoi[token]

    def decode(self, idx: int):
        return self.itos[idx]

    def encode_seq(self, seq: [str]):
        seq_tokens = []
        for item in seq:
            seq_tokens.append(self.encode(item))
        return seq_tokens

    def decode_seq(self, seq: [int]):
        string = ''
        for item in seq:
            string += self.decode(item)
        return string

    def get_stoi(self):
        return self.stoi

    def __len__(self):
        return max(self.itos.keys()) + 1
