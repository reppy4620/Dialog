import pickle
from pathlib import Path

from tqdm import tqdm


def load_from_pkl(params) -> list:
    p = Path(f'{params.data_dir}/{params.train_fn}.pkl')
    if not p.exists():
        raise FileExistsError(f'Please check {str(p)} is exists.')

    print('Loading training data')
    with open(str(p), mode='rb') as f:
        data = pickle.load(f)
    return data


def load_from_txt(params, tokenizer, make_pkl=False):
    p = Path(f'{params.data_dir}/{params.train_fn}.txt')
    if not p.exists():
        raise FileExistsError(f'Please check {str(p)} is exists.')

    print('Loading training data')
    data = list()
    with open(str(p), 'r', encoding='utf-8') as f:
        lines = f.readlines()
    # lines = lines[:3000]
    for i in tqdm(range(0, len(lines) - 1, 3)):
        data.append([tokenizer.encode(x) for x in lines[i:i + 2]])
    if make_pkl:
        print('Make pickle data')
        with open(f'{params.data_dir}/{params.train_fn}.pkl', 'wb') as f:
            pickle.dump(data, f)
    return data
