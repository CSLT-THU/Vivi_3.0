'''A wrapper class for optimizer '''
import numpy as np
from torch import optim

def get_optimizer(model, model_param):
    d_model = int(model_param['d_model'])
    n_warmup_steps = int(model_param['n_warmup_steps'])
    optimizer = ScheduledOptim(
        optim.Adam(
            filter(lambda x: x.requires_grad, model.parameters()), # model是对象 # Jun17
            betas=(0.9, 0.98), eps=1e-09), # adam default setting：alpha=0.001,beta=(0.9,0.999),eps=1e-08
        d_model, n_warmup_steps)
    return optimizer

class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 1 # Jun17
        # self.n_current_steps = 10
        self.init_lr = 8 # Jun17
        # self.init_lr = np.power(d_model, -0.5)

    def step(self):
    # def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()
        # print('init lr, lr scale, lr:', self.init_lr, self._get_lr_scale(), lr) # Jun17

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr

