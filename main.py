import logging
import os
import pickle

import torch
import torch.optim as optim

from config import Config
from nn import build_model, get_optimizer, ITFLoss
from tokenizer import Tokenizer
from utils import (DialogDataset, one_cycle, evaluate,
                   seed_everything, BalancedDataLoader,
                   make_train_data_from_txt, make_itf)

logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    logging.info('*** Initializing ***')

    if not os.path.isdir(Config.data_dir):
        os.mkdir(Config.data_dir)

    seed_everything(Config.seed)
    device = torch.device(Config.device)

    start_epoch = 0
    tokenizer = Tokenizer.from_pretrained(Config.model_name)

    logging.info('Preparing training data')
    if Config.use_pickle:
        with open(f'{Config.pickle_path}', 'rb') as f:
            train_data = pickle.load(f)
    else:
        train_data = make_train_data_from_txt(Config, tokenizer)
    itf = make_itf(train_data, Config.vocab_size)
    dataset = DialogDataset(train_data, tokenizer)

    logging.info('Define Models')
    model = build_model(Config).to(device)

    logging.info('Define Loss and Optimizer')
    # criterion = LabelSmoothing(tokenizer.vocab_size, pad_id=tokenizer.pad_token_id, smoothing=Config.smoothing)
    criterion = ITFLoss(itf)
    # criterion = nn.CrossEntropyLoss(reduction='none')
    _opt = optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    optimizer = get_optimizer(_opt, factor=Config.factor, warmup=Config.warmup)

    logging.info('Start Training')
    for epoch in range(start_epoch, Config.n_epoch):
        one_cycle(epoch, Config, model, optimizer, criterion,
                  BalancedDataLoader(dataset, tokenizer.pad_token_id),
                  tokenizer, device)
        evaluate(Config, 'もう疲れたー', tokenizer, model, device)
