import pytorch_lightning as pl
import torch
import torch.optim as optim

from utils import Batch, AttributeDict
from .loss import ITFLoss
from .model import DialogModel


class DialogModule(pl.LightningModule):

    def __init__(self, params, tokenizer, itf=None):
        super().__init__()

        if not isinstance(params, AttributeDict):
            params = AttributeDict(params)
        self.hparams = params

        self.tokenizer = tokenizer

        self.model = DialogModel(params)

        if itf is None:
            itf = torch.zeros(tokenizer.vocab_size)
        self.criterion = ITFLoss(itf)

    def forward(self, input_seq: str):
        device = next(self.model.parameters()).device
        ids = self.tokenizer.encode(input_seq)
        src = torch.tensor(ids, dtype=torch.long, device=device)[None, :]
        ys = torch.tensor([self.tokenizer.cls_token_id], dtype=torch.long, device=device)[None, :]

        with torch.no_grad():
            mem = self.model.encode(src, None)
            for i in range(self.hparams.max_len - 1):
                out = self.model.decode(ys, mem, None, None)
                _, candidate = out[:, -1].topk(5, dim=-1)
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
        out = self.model(batch.source, batch.target,
                         batch.source_mask, batch.target_mask)
        loss = self.criterion(out.transpose(1, 2), batch.target_y).mean()

        self.log_dict({
            'train_loss': loss,
        }, on_epoch=True)

        return loss

    def validation_step(self, _batch, batch_idx):
        src, tgt = _batch
        batch = Batch(src, tgt, self.tokenizer.pad_token_id)
        out = self.model(batch.source, batch.target,
                         batch.source_mask, batch.target_mask)
        loss = self.criterion(out.transpose(1, 2), batch.target_y).mean()

        self.log_dict({
            'val_loss': loss,
            'step': self.global_step
        }, prog_bar=True)

        self.logger.experiment.add_text('generated_text', self.forward('おはよう'), global_step=self.global_step)

    def configure_optimizers(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        return optim.AdamW(params, lr=self.hparams.optimizer.lr)
