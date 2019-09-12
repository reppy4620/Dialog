import torch
import torch.nn as nn


# カルバックライブラーによりモデルの出力とターゲットの分布からロスを計算
# その時LabelSmoothingを使用する
class LabelSmoothing(nn.Module):
    def __init__(self, size, pad_id, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.pad_id = pad_id
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, tgt):
        assert x.size(1) == self.size
        t_dist = x.clone()
        t_dist.fill_(self.smoothing / (self.size - 2))
        t_dist.scatter_(1, tgt.unsqueeze(1), self.confidence)
        t_dist[:, self.pad_id] = 0
        mask = torch.nonzero(tgt == self.pad_id)
        if mask.dim() > 0:
            t_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = t_dist
        return self.criterion(x, t_dist.clone())
