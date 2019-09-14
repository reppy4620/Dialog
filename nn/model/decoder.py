import torch
import torch.nn as nn

from .attention import SourceTargetAttention, SelfAttention
from .ffn import FFN


def build_decoder(N=6, h=8, d_model=512, d_ff=2048, drop_rate=0.1):
    decoder_layers = [DecoderLayer(h, d_model, d_ff, drop_rate) for _ in range(N)]
    decoder = Decoder(nn.ModuleList(decoder_layers), d_model)
    return decoder


class Decoder(nn.Module):

    def __init__(self, layers, d_model):
        super(Decoder, self).__init__()
        # decoder layers
        self.layers = layers
        self.norm = nn.LayerNorm(d_model)

    def forward(self,
                x: torch.FloatTensor, memory: torch.FloatTensor,
                source_mask: torch.Tensor, target_mask: torch.Tensor
                ) -> torch.FloatTensor:
        source_mask = source_mask.unsqueeze(-2)
        # note that memory is passed through encoder
        for layer in self.layers:
            x = layer(x, memory, source_mask, target_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):

    def __init__(self, h=8, d_model=512, d_ff=2048, drop_rate=0.1):
        super(DecoderLayer, self).__init__()

        # Self Attention Layer
        # query key and value come from previous layer.
        self.self_attn = SelfAttention(h, d_model, drop_rate)
        # Source Target Attention Layer
        # query come from encoded space.
        # key and value come from previous self attention layer
        self.st_attn = SourceTargetAttention(h, d_model, drop_rate)
        self.ff = FFN(d_model, d_ff)

    def forward(self, x, mem, source_mask, target_mask):
        # self attention
        x = self.self_attn(x, target_mask)
        # source target attention
        x = self.st_attn(mem, x, source_mask)
        # pass through feed forward network
        return self.ff(x)
