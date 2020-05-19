import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from .batch import Batch


def one_cycle(epoch, config, model, optimizer, criterion, data_loader,
              tokenizer, device):
    model.train()
    with tqdm(total=len(data_loader), desc=f'Epoch: {epoch + 1}') as pbar:
        for i, (x, y) in enumerate(data_loader):
            optimizer.zero_grad()
            batch = Batch(x.to(device), y.to(device), pad=tokenizer.pad_token_id)
            out = model(batch.source, batch.source_mask,
                        batch.target, batch.target_mask)
            loss = criterion(out.transpose(1, 2), batch.target_y).mean()
            loss.backward()
            optimizer.step()
            clip_grad_norm_(model.parameters(), config.max_grad_norm)
            pbar.update(1)
            pbar.set_postfix_str(f'Loss: {loss.item():.5f}')
    # always overwrite f'{config.data_dir}/{config.fn}.pth'
    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'opt': optimizer.state_dict()
    }, f'{config.data_dir}/{config.fn}.pth')
    # not overwrite
    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'opt': optimizer.state_dict()
    }, f'{config.data_dir}/{config.fn}_{epoch}.pth')
    print('*** Saved Model ***')
