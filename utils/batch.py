import torch

from .helper import subsequent_mask


class Batch:

    def __init__(self, source: torch.Tensor, target: torch.Tensor = None, pad: int = 0):
        self.source = source
        self.source_mask = (source != pad)
        if target is not None:
            self.target = target[:, :-1]
            self.target_y = target[:, 1:]
            self.target_mask = self.make_std_mask(self.target, pad)
            self.n_tokens = (self.target != pad).sum()

    @staticmethod
    def make_std_mask(target: torch.Tensor, pad: int) -> torch.Tensor:
        mask = (target != pad).unsqueeze(-2)
        mask = mask & subsequent_mask(target.size(-1)).type_as(mask)
        return mask
