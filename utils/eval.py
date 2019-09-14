import torch

from .helper import subsequent_mask


def evaluate(config, input_seq, sp_model, model, device):
    ids = sp_model.EncodeAsIds(input_seq)
    src = torch.zeros(1, 12).fill_(config.pad_id)
    src[:, :len(ids)] = torch.LongTensor(ids)
    src_mask = src != config.pad_id
    src, src_mask = src.long().to(device), src_mask.to(device)
    mem = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(config.bos_id).long().to(device)
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
