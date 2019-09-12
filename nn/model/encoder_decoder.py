import torch.nn as nn

from .embedding import build_embedding
from .encoder import Encoder
from .decoder import build_decoder
from .generator import Generator


class Seq2SeqModel(nn.Module):
    def __init__(self, bert_model_dir, vocab_size=32000, h=8,
                 d_model=768, N=6, d_ff=2048, drop_rate=0.1, max_len=12):
        super(Seq2SeqModel, self).__init__()

        self.target_emb = build_embedding(vocab_size, d_model, drop_rate, max_len=max_len)

        self.encoder = Encoder.from_pretrained(bert_model_dir)
        # Freeze Pre-Trained Encoder
        # so we'll train only decoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.decoder = build_decoder(N, h, d_model, d_ff)

        self.generator = Generator(d_model, vocab_size)

        for param in self.decoder.parameters():
            if param.dim > 1:
                nn.init.xavier_uniform_(param)
        
        for param in self.generator.parameters():
            if param.dim > 1:
                nn.init.xavier_uniform_(param)

    def forward(self, source, source_mask, target, target_mask):
        x = self.encode(source, source_mask)
        x = self.decode(x, source_mask, target, target_mask)
        x = self.generate(x)
        return x

    def encode(self, source, source_mask):
        return self.encoder(source, attention_mask=source_mask)

    def decode(self, mem, source_mask, target, target_mask):
        return self.decoder(self.target_emb(target), mem, source_mask, target_mask)

    def generate(self, x):
        return self.generator(x)

    def load(self, obj):
        self.load_state_dict(obj)
