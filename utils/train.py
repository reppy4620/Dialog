import torch
from tqdm import tqdm

from .batch import Batch
from .eval import evaluate


def train(config, model, optimizer, criterion, data_loader,
          sp_model, device, start_epoch):
    for epoch in range(start_epoch, config.n_epoch):
        model.train()
        with tqdm(total=len(data_loader), desc=f'Epoch: {epoch + 1}') as pbar:
            for i, (x, y) in enumerate(data_loader):
                optimizer.zero_grad()
                batch = Batch(x.to(device), y.to(device), pad=3)
                out = model(batch.source, batch.source_mask,
                            batch.target, batch.target_mask)
                loss = criterion(out.contiguous().view(-1, out.size(-1)),
                                 batch.target_y.contiguous().view(-1)) / batch.n_tokens
                loss.backward()
                optimizer.step()

                pbar.update(1)
                pbar.set_postfix_str(f'Loss: {loss.item():.5f}')

        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'opt': optimizer.state_dict(),
            'param': optimizer.parameters()
        }, f'{config.output_dir}/{config.fn}.pth')
        print('*** Saved Model ***')

        if epoch != 0 and epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'opt': optimizer.state_dict(),
                'param': optimizer.parameters()
            }, f'{config.output_dir}/{config.fn}_{epoch}.pth')

        model.eval()
        evaluate(config, 'おはよう', sp_model, model, device)
