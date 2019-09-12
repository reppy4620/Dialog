import torch
import sentencepiece as spm

from nn import Seq2SeqModel
from utils import evaluate
from config import Config


if __name__ == '__main__':

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
