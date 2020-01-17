import torch
from torch.utils.data import Dataset


class DialogDataset(Dataset):

    def __init__(self, train_data, tokenizer):
        self.data = train_data
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        src, tgt = self.data[idx]
        src = torch.LongTensor(src)
        tgt = torch.LongTensor(tgt)
        return src, tgt

    def __len__(self):
        return len(self.data)
