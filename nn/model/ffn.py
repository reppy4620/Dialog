import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


# Paper denote this module to `FFN = max(0, x * W1 + b1) * W2 + b2`
class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, drop_rate: float = 0.1,
                 activation=gelu):
        super(FFN, self).__init__()
        self.drop_rate = drop_rate
        # Linear or Conv1d
        self.l1 = nn.Conv1d(d_model, d_ff, 1)
        self.l2 = nn.Conv1d(d_ff, d_model, 1)

        # relu or gelu
        self.activation = activation

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        out = self.activation(self.l1(x.transpose(1, 2)))
        out = F.dropout(out.transpose(1, 2), p=self.drop_rate, training=self.training)
        out = self.l2(out.transpose(1, 2)).transpose(1, 2)
        out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.norm(x + out)
        return out
