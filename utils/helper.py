import os
import pickle
import random
from collections import Counter

import numpy as np
import torch
from tqdm import tqdm


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True


def subsequent_mask(size: int) -> torch.Tensor:
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def make_train_data_from_txt(config, tokenizer):
    data = list()
    with open(config.train_data_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for i in tqdm(range(0, len(lines) - 1, 3)):
        data.append(tuple(map(tokenizer.encode, lines[i:i + 2])))
    with open(f'{config.pickle_path}', 'wb') as f:
        pickle.dump(data, f)
    return data


def make_itf(data, voc_size):
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
