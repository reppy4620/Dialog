import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.layer = nn.Linear(d_model, vocab)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = F.log_softmax(self.layer(x), dim=-1)
        return x
