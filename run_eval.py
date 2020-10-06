from argparse import ArgumentParser

import torch

from nn import DialogModule
from tokenizer import Tokenizer
from utils import get_config

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--config_path', type=str, default='configs/config.yaml')
    parser.add_argument('--ckpt_path', type=str, default='data/last.ckpt')
    parser.add_argument('--device_type', type=str, default='cpu')
    args = parser.parse_args()

    config = get_config(args.config_path)

    if args.device_type != 'cpu' and args.device_type != 'cuda':
        raise ValueError('Please set device_type to cpu or cuda')
    device = torch.device(args.device_type)

    tokenizer = Tokenizer(config.model_name)

    model = DialogModule(config, tokenizer)
    model = model.load_from_checkpoint(args.ckpt_path, params=config, tokenizer=tokenizer, itf=None)

    while True:
        s = input('You>')
        if s == 'q':
            break
        print('BOT>', end='')
        text = model(s)
        print(text)
