import logging
import os
import pickle

import torch
import torch.optim as optim

from config import Config
from nn import build_model, LabelSmoothing, get_optimizer
from tokenizer import Tokenizer
from utils import (DialogDataset, train,
                   seed_everything, BalancedDataLoader,
                   make_train_data_from_txt)

logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    logging.info('*** Initializing ***')

    if not os.path.isdir(Config.output_dir):
        os.mkdir(Config.output_dir)

    seed_everything(Config.seed)
    device = torch.device(Config.device)

    start_epoch = 0

    logging.info('Define Models')
    model = build_model(Config).to(device)
    tokenizer = Tokenizer.from_pretrained(Config.model_name)

    logging.info('Define Loss and Optimizer')
    criterion = LabelSmoothing(tokenizer.vocab_size, pad_id=tokenizer.pad_token_id, smoothing=Config.smoothing)
    _opt = optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    optimizer = get_optimizer(_opt, factor=Config.factor, warmup=Config.warmup)

    logging.info('Preparing training data')
    if Config.use_pickle:
        with open(f'{Config.pickle_path}', 'rb') as f:
            train_data = pickle.load(f)
    else:
        train_data = make_train_data_from_txt(Config, tokenizer)
    dataset = DialogDataset(train_data, tokenizer)
    loader = BalancedDataLoader(dataset, tokenizer.pad_token_id)

    logging.info('Start Training')
    train(Config, model, optimizer, criterion, loader,
          tokenizer, device, start_epoch)
