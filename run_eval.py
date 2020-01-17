import torch

from config import Config
from nn import build_model
from tokenizer import Tokenizer
from utils import evaluate

if __name__ == '__main__':

    device = torch.device(Config.device)

    state_dict = torch.load(f'{Config.data_dir}/{Config.fn}.pth')

    tokenizer = Tokenizer.from_pretrained(Config.model_name)

    model = build_model(Config).to(device)
    model.load_state_dict(state_dict['model'])
    model.eval()
    model.freeze()

    while True:
        s = input('You>')
        if s == 'q':
            break
        print('BOT>', end='')
        text = evaluate(Config, s, tokenizer, model, device, True)
