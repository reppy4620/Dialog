import re
from time import sleep

import requests
import torch

from config import Config
from nn import build_model
from tokenizer import Tokenizer
from utils import evaluate, seed_everything

root = 'https://still-reaches-94354.herokuapp.com'
log = -1

if __name__ == '__main__':
    seed_everything(Config.seed)

    device = torch.device(Config.device)

    state_dict = torch.load(f'{Config.data_dir}/ckpt_1.pth')

    tokenizer = Tokenizer.from_pretrained(Config.model_name)

    model = build_model(Config).to(device)
    model.load_state_dict(state_dict['model'])
    model.eval()
    model.freeze()
    text = ''

    print('Start listening')

    while True:
        response = requests.post(root + '/dialog_receive')
        if response.status_code == 200:
            index = re.search('([-0-9][0-9]*):', response.text).groups()[0]
            index = int(index)
            text = re.search(':(.*)', response.text).groups()[0]
            if log < index:
                print(str(index) + ' ' + text)
                log = index
                out = evaluate(Config, text, tokenizer, model, device, verbose=True)
                requests.post(root + '/dialog_send', data={'message': out})
        sleep(1)
