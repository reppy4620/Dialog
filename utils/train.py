import torch

from tqdm import tqdm
from transformer.utils import subsequent_mask
from utils import Batch


def train(config, model, optimizer, criterion, data_loader,
          sp_model, start_epoch=0):
    for epoch in range(start_epoch, config.n_epoch):
        model.train()
        with tqdm(total=len(data_loader), desc=f'Epoch: {epoch + 1}') as pbar:
            for i, (x, y) in enumerate(data_loader):
                optimizer.zero_grad()
                batch = Batch(x.cuda(), y.cuda(), pad=3)
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
        evaluate(config, 'マジで疲れた', sp_model, model)


def evaluate(config, input_seq, sp_model, model):
    ids = sp_model.EncodeAsIds(input_seq)
    src = torch.zeros(1, 12).fill_(config.pad_id)
    src[:, :len(ids)] = torch.LongTensor(ids)
    src_mask = src != config.pad_id
    src, src_mask = src.long().cuda(), src_mask.cuda()
    mem = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(config.bos_id).long().cuda()
    with torch.no_grad():
        for i in range(config.max_len - 1):
            out = model.decode(mem, src_mask,
                               ys, subsequent_mask(ys.size(1)).type_as(ys))
            prob = model.generate(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word[0]
            if next_word == config.eos_id:
                break
            ys = torch.cat([ys, torch.ones(1, 1).type_as(ys).fill_(next_word).long()], dim=1)
    ys = ys.view(-1).detach().cpu().numpy().tolist()[1:]
    print(''.join([sp_model.IdToPiece(i) for i in ys]).replace('▁', ''))
