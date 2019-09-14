import sentencepiece as spm
import torch

from config import Config
from nn import EncoderDecoder
from utils import evaluate, seed_everything

if __name__ == '__main__':

    seed_everything(Config.seed)

    cuda = True
    device = torch.device('cuda' if cuda else 'cpu')

    sp_model = spm.SentencePieceProcessor()
    sp_model.Load('./pretrained/wiki-ja.model')

    model = EncoderDecoder(bert_model_dir='./pretrained').to(device)
    model.load(torch.load('./models/ckpt.pth')['model'])

    while True:
        s = input('You>')
        if s == 'q':
            break
        print('BOT>', end='')
        evaluate(Config, s, sp_model, model, device)
        print()
