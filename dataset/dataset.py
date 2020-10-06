import torch
from torch.utils.data import Dataset


class DialogDataset(Dataset):

    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        src, tgt = self.data[idx]
        src = torch.LongTensor(src)
        tgt = torch.LongTensor(tgt)
        return src, tgt

    def __len__(self):
        return len(self.data)
