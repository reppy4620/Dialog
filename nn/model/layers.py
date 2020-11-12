import math
from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F


def Conv1d(c_in: int, c_out: int, k: int, s: int = 1, d: int = 1):
    p = (k - 1) // 2
    conv = nn.Conv1d(c_in, c_out, k, s, p, dilation=d)
    nn.init.kaiming_normal_(conv.weight)
    return conv


class AttentionLayer(nn.Module, ABC):

    def __init__(self, params, is_ffn):
        super(AttentionLayer, self).__init__()

        channel = params.model.channel

        self.attn = MultiHeadAttention(params)
        self.attn_norm = nn.LayerNorm(params.model.channel)
        self.attn_dropout = nn.Dropout(params.model.dropout)

        self.is_ffn = is_ffn
        if is_ffn:
            self.ffn = nn.Sequential(
                Conv1d(channel, channel * 2, 1),
                nn.GELU(),
                Conv1d(channel * 2, channel, 1)
            )
            self.ffn_norm = nn.LayerNorm(channel)
            self.ffn_dropout = nn.Dropout(params.model.dropout)

    def _attn(self, query, key, value, mask):
        x = self.attn(query, key, value, mask)
        x = query + self.attn_dropout(x)
        x = self.attn_norm(x)
        return x

    def _ffn(self, x):
        residual = x
        # (B, L, C) => (B, C, L)
        x = x.transpose(1, 2)
        x = self.ffn(x)
        # (B, C, L) => (B, L, C)
        x = x.transpose(1, 2)
        x = residual + x
        x = self.ffn_norm(x)
        return x


class SelfAttention(AttentionLayer):

    def forward(self, x, mask):
        x = self._attn(x, x, x, mask)
        if self.is_ffn:
            x = self._ffn(x)
        return x


class SourceTargetAttention(AttentionLayer):

    def forward(self, x, mem, mask):
        x = self._attn(x, mem, mem, mask)
        if self.is_ffn:
            x = self._ffn(x)
        return x


class MultiHeadAttention(nn.Module):

    def __init__(self, params):
        super().__init__()

        channel = params.model.channel

        assert channel % params.model.n_head == 0
        self.d_k = channel // params.model.n_head
        self.n_head = params.model.n_head

        self.scale_factor = math.sqrt(self.d_k)

        # Weight of doing linear transformation
        self.w_q = nn.Linear(channel, channel, bias=False)
        self.w_k = nn.Linear(channel, channel, bias=False)
        self.w_v = nn.Linear(channel, channel, bias=False)

        self.linear = nn.Linear(channel, channel)

        # Just to be sure, hold the attention score.
        self.attn_map = None
        self.dropout = nn.Dropout(params.model.dropout)

    def forward(self, query, key, value, mask):
        if mask is not None:
            mask = mask.unsqueeze(1)
        n_batches = query.size(0)

        # Simple Linear Transformation
        # B x h x d_model x d_k
        query = self.w_q(query).contiguous().view(n_batches, -1, self.n_head, self.d_k).transpose(1, 2)
        key = self.w_k(key).contiguous().view(n_batches, -1, self.n_head, self.d_k).transpose(1, 2)
        value = self.w_v(value).contiguous().view(n_batches, -1, self.n_head, self.d_k).transpose(1, 2)

        x, self.attn_map = self.attention(query, key, value, mask, self.dropout)
        x = x.transpose(1, 2).contiguous().view(n_batches, -1, self.n_head * self.d_k)
        return self.linear(x)

    def attention(self, query, key, value, mask=None, dropout=None):
        # (QK^T)/sqrt(d_k)
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale_factor
        if mask is not None:
            if len(mask.size()) == 3:
                mask = mask.unsqueeze(-2)
            scores = scores.masked_fill(mask, -1e9)
        # softmax((QK^T)/sqrt(d_k))
        attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            attn = dropout(attn)
        # calculate softmax((QK^T)/sqrt(d_k))V and return result and attention score
        return torch.matmul(attn, value), attn
