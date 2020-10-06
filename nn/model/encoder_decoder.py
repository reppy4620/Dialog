import torch.nn as nn

from .decoder import build_decoder
from .encoder import build_encoder
from .generator import Generator


def build_model(config):
    model = EncoderDecoder(config.model_name, config.decoder.vocab_size, config.decoder.num_head,
                           config.decoder.d_model, config.decoder.num_head, config.decoder.d_ff)
    model.freeze_encoder()
    return model


class EncoderDecoder(nn.Module):
    def __init__(self, bert_model_name, vocab_size=32000, h=8,
                 d_model=768, N=6, d_ff=2048):
        super(EncoderDecoder, self).__init__()

        self.encoder = build_encoder(bert_model_name)
        self.decoder = build_decoder(N, h, d_model, d_ff)

        self.freeze_encoder()

        self.target_emb = self.encoder.embeddings

        if bert_model_name == "ALINEAR/albert-japanese-v2":
            self.embedding_mapping = nn.Linear(128, d_model)
        else:
            self.embedding_mapping = None

        self.generator = Generator(d_model, vocab_size)

        for param in self.decoder.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

        for param in self.generator.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def freeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = False

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad = True

    def forward(self, source, source_mask, target, target_mask):
        x = self.encode(source, source_mask)
        x = self.decode(x, source_mask, target, target_mask)
        x = self.generate(x)
        return x

    def encode(self, source, source_mask):
        return self.encoder(source, attention_mask=source_mask)[0]

    def decode(self, mem, source_mask, target, target_mask):
        target = self.target_emb(target)
        if self.embedding_mapping is not None:
            target = self.embedding_mapping(target)
        return self.decoder(target, mem, source_mask, target_mask)

    def generate(self, x):
        return self.generator(x)

    def load(self, obj):
        self.load_state_dict(obj)
