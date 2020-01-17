import torch.nn.utils.rnn as rnn
from torch.utils.data import Dataset
from torch.utils.data.sampler import RandomSampler, BatchSampler

from config import Config


class BalancedDataLoader(BatchSampler):

    def __init__(self, data: Dataset, pad_id: int):
        super().__init__(RandomSampler(data), Config.batch_size, True)
        self.pad_id = pad_id
        self.count = 0

    def __iter__(self):
        src_list = list()
        tgt_list = list()
        # sampler is RandomSampler
        for i in self.sampler:
            self.count += 1
            src, tgt = self.sampler.data_source[i]
            src_list.append(src)
            tgt_list.append(tgt)
            if self.count % self.batch_size == 0:
                assert len(src_list) == self.batch_size
                src = rnn.pad_sequence(src_list, batch_first=True, padding_value=self.pad_id)
                tgt = rnn.pad_sequence(tgt_list, batch_first=True, padding_value=self.pad_id)
                src_list.clear()
                tgt_list.clear()
                yield src, tgt
