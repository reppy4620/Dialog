from typing import List

import pytorch_lightning as pl
import torch
from torch.utils.data import random_split, DataLoader

from nn.loss import make_itf
from tokenizer import Tokenizer
from utils import AttributeDict
from .collate_fn import collate_fn
from .dataset import DialogDataset


class DialogDataModule(pl.LightningDataModule):

    def __init__(self, data: List[tuple], params: AttributeDict, tokenizer: Tokenizer):
        super().__init__()
        self.data = data
        self.params = params
        self.tokenizer = tokenizer
        # following variable will be initialized in setup function.
        self.train_x = None
        self.valid_x = None

        # _itf is initialized when get itf property
        self._itf = None

    def setup(self, stage=None):
        dataset = DialogDataset(self.data)
        train_size = int(self.params.train_ratio * len(dataset))
        valid_size = len(dataset) - train_size
        self.train_x, self.valid_x = random_split(
            dataset=dataset,
            lengths=[train_size, valid_size],
            generator=torch.Generator().manual_seed(self.params.seed)
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_x,
            batch_size=self.params.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            pin_memory=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.valid_x,
            batch_size=self.params.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            pin_memory=True
        )

    @property
    def itf(self):
        if self._itf is None:
            self._itf = make_itf(self.data, self.tokenizer.vocab_size)
        return self._itf
