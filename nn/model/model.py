import torch.nn as nn

from .networks import Encoder, Decoder


class DialogModel(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.encoder = Encoder(params)
        self.freeze(self.encoder)

        self.decoder = Decoder(
            params,
            nn.Sequential(
                self.encoder.net.embeddings,
                nn.Linear(128, params.model.channel)
            )
        )

    def forward(self, source, target, source_mask=None, target_mask=None):
        x = self.encoder(source, source_mask)
        x = self.decoder(target, x, source_mask, target_mask)
        return x

    def encode(self, source, source_mask=None):
        return self.encoder(source, source_mask)

    def decode(self, x, mem, source_mask=None, target_mask=None):
        return self.decoder(x, mem, source_mask, target_mask)

    @staticmethod
    def freeze(net):
        for p in net.parameters():
            p.requires_grad = False
        net.eval()

    @staticmethod
    def unfreeze(net):
        for p in net.parameters():
            p.requires_grad = True
        net.train()
