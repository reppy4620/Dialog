import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


# Paper denote this module to `FFN = max(0, x * W1 + b1) * W2 + b2`
# but i use gelu instead of relu about activation
class FFN(nn.Module):
    def __init__(self, d_model=768, d_ff=2048, drop_rate=0.1, activation=gelu):
        super(FFN, self).__init__()
        self.drop_rate = drop_rate
        # Linear or Conv1d
        # self.l1 = nn.Linear(d_model, d_ff)
        # self.l2 = nn.Linear(d_ff, d_model)
        self.l1 = nn.Conv1d(d_model, d_ff, 1)
        self.l2 = nn.Conv1d(d_ff, d_model, 1)

        # relu or gelu
        self.activation = activation

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        # Empirically the implementation according to the paper failed to train.
        # this matter can be said in the attention layer too.
        # i think the cause is the order of normalization.
        # paper's implementation applied normalization last.
        # but i applied normalization first to solve this problem.
        # this idea comes from the implementation of Open-NMT.
        # please refer to the Open-NMT if you want more information.
        out = self.norm(x)
        out = self.activation(self.l1(out.transpose(1, 2)))
        out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.l2(out)
        out = F.dropout(out.transpose(1, 2), p=self.drop_rate, training=self.training)
        return x + out
