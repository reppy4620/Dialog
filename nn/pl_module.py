import pytorch_lightning as pl
import torch

from optim import RAdam
from utils import Batch, AttributeDict, subsequent_mask
from .loss import ITFLoss
from .model import build_model


class DialogModule(pl.LightningModule):

    def __init__(self, params, tokenizer, itf=None):
        super().__init__()

        if not isinstance(params, AttributeDict):
            params = AttributeDict(params)
        self.hparams = params

        self.tokenizer = tokenizer

        self.model = build_model(params)

        if itf is None:
            itf = torch.zeros(tokenizer.vocab_size)
        self.criterion = ITFLoss(itf)

    def forward(self, input_seq: str):
        device = next(self.model.parameters()).device
        self.freeze()
        ids = self.tokenizer.encode(input_seq)
        src = torch.tensor(ids, dtype=torch.long, device=device)[None, :]
        src_mask = torch.ones(src.size(), dtype=torch.long, device=device)
        mem = self.model.encode(src, src_mask)
        ys = torch.tensor([self.tokenizer.cls_token_id], dtype=torch.long, device=device)[None, :]
        with torch.no_grad():
            for i in range(self.hparams.max_len - 1):
                out = self.model.decode(mem, src_mask,
                                        ys, subsequent_mask(ys.size(1)).type_as(ys))
                prob = self.model.generate(out[:, -1])
                _, candidate = prob.topk(5, dim=1)
                next_word = torch.tensor([candidate[:, 0]], dtype=torch.long, device=device)[None, :]
                if next_word.item() == self.tokenizer.sep_token_id:
                    break
                ys = torch.cat([ys, next_word], dim=1)
        ys = ys.view(-1).cpu().numpy().tolist()[1:]
        text = self.tokenizer.decode(ys)
        return text

    def training_step(self, _batch, batch_idx):
        src, tgt = _batch
        batch = Batch(src, tgt, self.tokenizer.pad_token_id)
        out = self.model(batch.source, batch.source_mask,
                         batch.target, batch.target_mask)
        loss = self.criterion(out.transpose(1, 2), batch.target_y).mean()

        result = pl.TrainResult(loss)
        result.log_dict({
            'train_loss': loss,
        }, on_epoch=True)

        return result

    def validation_step(self, _batch, batch_idx):
        src, tgt = _batch
        batch = Batch(src, tgt, self.tokenizer.pad_token_id)
        out = self.model(batch.source, batch.source_mask,
                         batch.target, batch.target_mask)
        loss = self.criterion(out.transpose(1, 2), batch.target_y).mean()

        result = pl.EvalResult(checkpoint_on=loss)
        result.log_dict({
            'valid_loss': loss,
        }, prog_bar=True)

        return result

    def configure_optimizers(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        return RAdam(params, lr=self.hparams.optimizer.lr)
