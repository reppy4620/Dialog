import torch
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
            loss = criterion(out.contiguous().view(-1, out.size(-1)),
                             batch.target_y.contiguous().view(-1)) / batch.n_tokens
            loss.backward()
            optimizer.step()

            pbar.update(1)
            pbar.set_postfix_str(f'Loss: {loss.item():.5f}')
            break
    # Difference is file name.
    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'opt': optimizer.state_dict(),
        'param': optimizer.parameters()
    }, f'{config.data_dir}/{config.fn}.pth')
    print('*** Saved Model ***')
