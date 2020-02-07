import math

import torch
import torch.nn as nn


def build_embedding(vocab=32000, d_model=512, drop_rate=0.1, max_len=12):
    return nn.Sequential(
        Embeddings(vocab, d_model),
        PositionalEncoding(d_model, drop_rate, max_len)
    )


class Embeddings(nn.Module):

    # vocab is number of vocabulary.
    def __init__(self, vocab, d_model):
        super(Embeddings, self).__init__()
        # Simple Embedding Layer
        self.emb = nn.Embedding(vocab, d_model)
        self.coefficient = math.sqrt(d_model)

    def forward(self, x: torch.Tensor) -> torch.FloatTensor:
        # multiply sqrt(d_model) cuz paper indicate this
        return self.emb(x) * self.coefficient


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, drop_rate, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(drop_rate)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = 10000 ** (torch.arange(0.0, d_model, 2.0) / d_model)
        pe[:, 0::2] = torch.sin(position / div_term)
        pe[:, 1::2] = torch.cos(position / div_term)
        pe = pe.unsqueeze(0)
        # because of using `resister_buffer`,
        # i wanna save this parameter as state_dict parameter
        self.register_buffer('pe', pe)

    def forward(self, x: torch.FloatTensor) -> torch.Tensor:
        x = x + self.pe[:, x.size(1), :]
        return self.dropout(x)
