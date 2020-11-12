import torch.nn as nn
from transformers import AutoModel

from .layers import SelfAttention, SourceTargetAttention


class Encoder(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.net = AutoModel.from_pretrained(params.model_name)

    def forward(self, source, source_mask=None):
        return self.net(source, source_mask)[0]


class Decoder(nn.Module):
    def __init__(self, params, embedding):
        super().__init__()

        self.n_layers = params.model.n_layers

        self.embedding = embedding

        self.self_attns = nn.ModuleList([SelfAttention(params, is_ffn=False) for _ in range(params.model.n_layers)])
        self.st_attns = nn.ModuleList(
            [SourceTargetAttention(params, is_ffn=True) for _ in range(params.model.n_layers)])
        self.linear = nn.Linear(params.model.channel, params.model.vocab_size)

    def forward(self, x, mem, src_mask=None, tgt_mask=None):
        x = self.embedding(x)
        for i in range(self.n_layers):
            x = self.self_attns[i](x, tgt_mask)
            x = self.st_attns[i](x, mem, src_mask)
        x = self.linear(x)
        return x
