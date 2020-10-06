import pathlib
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.seed import seed_everything

from dataset import DialogDataModule
from nn import DialogModule
from tokenizer import Tokenizer
from utils import get_config, load_from_pkl, load_from_txt

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('-c', '--config_path', type=str, default='configs/config.yaml')
    parser.add_argument('-d', '--data_dir', type=str, default='./data')
    parser.add_argument('-f', '--train_fn', type=str, default='train_data')
    parser.add_argument('-m', '--model_dir', type=str, default='./data/models')
    args = parser.parse_args()

    config = get_config(args.config_path)
    config.data_dir = args.data_dir
    config.train_fn = args.train_fn
    config.model_dir = args.model_dir

    seed_everything(config.seed)

    tokenizer = Tokenizer(config.model_name)

    if config.use_pickle:
        data = load_from_pkl(config)
    else:
        data = load_from_txt(config, tokenizer, make_pkl=True)
    dm = DialogDataModule(data, config, tokenizer)

    model = DialogModule(config, tokenizer, dm.itf)

    model_dir = pathlib.Path(config.model_dir)
    if not model_dir.exists():
        model_dir.mkdir(parents=True)
    save_fn = str(model_dir / 'dialog_{epoch:02d}-{val_loss:.6f}')
    mc = ModelCheckpoint(filepath=save_fn, save_last=True, save_top_k=5)

    trainer = pl.Trainer(
        gpus=1,
        checkpoint_callback=mc,
        max_epochs=config.n_epochs,
        deterministic=True,
        gradient_clip_val=config.gradient_clip,
    )
    trainer.fit(model=model, datamodule=dm)
