import torch

from torch.utils.data import Dataset


class DialogDataset(Dataset):

    def __init__(self, train_data, bos_id=4, eos_id=5, pad_id=3, max_length=12):
        # train_data is list of tuple that contains sourcec and tgt pairs
        self.data = train_data
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.max_length = max_length

    def __getitem__(self, idx):
        src, tgt = self.data[idx]
        src, tgt = self.get_sequence(src), self.get_sequence(tgt)
        source = torch.ones(self.max_length) * self.pad_id
        target = torch.ones(self.max_length) * self.pad_id
        source[:len(src)] = torch.LongTensor(src)
        target[:len(tgt)] = torch.LongTensor(tgt)
        return source.long(), target.long()

    def __len__(self):
        return len(self.data)

    def get_sequence(self, x):
        return [self.bos_id] + x + [self.eos_id]
