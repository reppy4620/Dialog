import logging
import os
import pickle

import torch
import torch.nn as nn
import torch.optim as optim

from config import Config
from nn import build_model
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
    model.unfreeze()

    logging.info('Define Loss and Optimizer')
    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.AdamW(model.parameters(), lr=Config.lr, betas=Config.betas, eps=1e-9)

    if Config.load:
        state_dict = torch.load(f'{Config.data_dir}/{Config.fn}.pth')
        start_epoch = 10
        print(f'Start Epoch: {start_epoch}')
        model.load_state_dict(state_dict['model'])
        optimizer.load_state_dict(state_dict['opt'])

    logging.info('Start Training')
    for epoch in range(start_epoch, Config.n_epoch):
        one_cycle(epoch, Config, model, optimizer, criterion,
                  BalancedDataLoader(dataset, tokenizer.pad_token_id),
                  tokenizer, device)
        evaluate(Config, 'おはよーーー', tokenizer, model, device)
