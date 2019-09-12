import torch.optim as optim


def get_optimizer(optimizer: optim.Optimizer, d_model=2048, factor=2, warmup=4000):
    return WarmupOptimizer(
        optimizer=optimizer,
        factor=factor,
        d_model=d_model,
        warmup=warmup
    )


class WarmupOptimizer:

    def __init__(self, optimizer: optim.Optimizer, factor: int, d_model: int, warmup: int):
        self.optimizer = optimizer
        self.step_num = 0
        self.warmup = warmup
        self.factor = factor
        self.d_model = d_model
        self._rate = 0

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.step_num += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step: int = None) -> float:
        if step is None:
            step = self.step_num
        return self.factor * \
               (self.d_model ** -0.5 * min(step ** -0.5,
                                           step * self.warmup ** -1.5))

    def load(self, opt_state_dict, parameters):
        self.load_state_dict(opt_state_dict)
        self.load_parameters(parameters)

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, obj):
        self.optimizer.load_state_dict(obj)

    def parameters(self):
        return {'warmup': self.warmup,
                'step_num': self.step_num,
                'factor': self.factor,
                '_rate': self._rate}

    def load_parameters(self, obj):
        self.warmup = obj['warmup']
        self.step_num = obj['warmup']
        self.factor = obj['factor']
        self._rate = obj['_rate']
