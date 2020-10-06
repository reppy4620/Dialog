from torch.nn.utils.rnn import pad_sequence


def _pad(x, pad_id):
    return pad_sequence(x, batch_first=True, padding_value=pad_id)


def collate_fn(batch, pad_id: int = 0):
    src, tgt = tuple(zip(*batch))
    src, tgt = _pad(src, pad_id), _pad(tgt, pad_id)
    return src, tgt
