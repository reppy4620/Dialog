import sentencepiece as spm
import torch

from config import Config
from nn import Seq2SeqModel
from utils import evaluate, seed_everything

if __name__ == '__main__':

    seed_everything(Config.seed)

    sp_model = spm.SentencePieceProcessor()
    sp_model.Load('./pretrained/wiki-ja.model')

    model = Seq2SeqModel(bert_model_dir='./pretrained').cuda()
    model.load(torch.load('./models/ckpt.pth')['model'])

    while True:
        s = input('あなた>')
        if s == 'q':
            break
        print('BOT>', end='')
        evaluate(Config, s, sp_model, model)
        print()
