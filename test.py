def partial_test():
    import functools
    from dataset.collate_fn import collate_fn

    _collate_fn = functools.partial(collate_fn, pad_id=1)
    print([list(range(1, 5)), list(range(6, 10))])
    _collate_fn([[list(range(1, 5)), list(range(6, 10))]])


def model_test():
    from utils import get_config
    from nn import DialogModel
    config = get_config('configs/config.yaml')
    model = DialogModel(config)
    print(f'num of params: {len(list(model.parameters()))}')
    params = [p for p in model.parameters() if p.requires_grad]
    print(f'num of learnable params: {len(params)}')
    print(model)


if __name__ == '__main__':
    model_test()
