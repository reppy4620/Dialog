import pickle

from config import Config
from tokenizer import Tokenizer
from utils import make_itf
from utils import make_train_data_from_txt


def test_itf():
    tokenizer = Tokenizer.from_pretrained(Config.model_name)
    if Config.use_pickle:
        with open(f'{Config.pickle_path}', 'rb') as f:
            train_data = pickle.load(f)
    else:
        train_data = make_train_data_from_txt(Config, tokenizer)
    counter, itf = make_itf(train_data, Config.vocab_size, tokenizer)
    # itf = (itf - itf.min()) / (itf.max() - itf.min())
    # for i in range(itf.size(0)):
    #     print(i, itf[i])
    # itf[itf == 0] += 1e-6
    for k, v in counter.most_common(len(counter)):
        print(tokenizer.decode([k]), v)


if __name__ == '__main__':
    test_itf()
