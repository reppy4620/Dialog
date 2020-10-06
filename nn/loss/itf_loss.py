from collections import Counter
from typing import List

import torch
import torch.nn as nn


def make_itf(data: List[tuple], voc_size: int):
    counter = Counter()
    for (inp, tgt) in data:
        counter.update(Counter(inp))
        counter.update(Counter(tgt))
    itf = [0] * voc_size
    for k, v in counter.items():
        if v > 1000000:
            itf[k] = 100 / v
        elif 100000 < v < 1000000:
            itf[k] = 10 / v
        else:
            itf[k] = 1 / v
    return torch.FloatTensor(itf)


# Inverse Token Frequency Loss
class ITFLoss(nn.Module):
    def __init__(self, itf: torch.Tensor):
        super(ITFLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss(reduction='none', ignore_index=0)
        self.register_buffer('itf', itf)

    def forward(self, pred, tgt):
        loss = self.loss(pred, tgt)
        itf = self.itf[tgt].type_as(loss)
        return itf * loss
