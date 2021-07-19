import torch
import math
from algs import opt




class grad_scaler(object):
    """
    gradient scaler, generally follows logic of AMP gradient scaler
    """
    def __init__(self, init_scale, min_eps, increase_every, betas, margin):
        self.consec_finite = 0
        self._scale = init_scale
        self.min_eps = min_eps
        self.increase_every = increase_every
        beta1, beta2 = betas
        self.buf_scale =  beta1 / math.sqrt(beta2)
        self.margin = margin


    def scale(self, loss):
        return loss*self._scale

    def can_step(self, optimizer, callback_prestep=None):
        self.last_ok = self.all_finite(optimizer)
        return self.last_ok


    def post_step(self, optimizer):
        if self.last_ok:
            with torch.no_grad():
                self.consec_finite += 1
                if self.consec_finite > self.increase_every:
                    self.consec_finite = 0
                    if self.can_scale_buffer(optimizer, 2.0):
                        self._scale *= 2
                        self.mul_bufs(optimizer, 2.0)
        else:
            with torch.no_grad():
                self.consec_finite = 0
                self.mul_bufs(optimizer, 0.5)
                self._scale *= 0.5


    @torch.no_grad()
    def can_scale_buffer(self, optimizer, factor):
        for group in optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = optimizer.state[p]
                # cannot ask to scale before bufs are initialized
                if 'exp_avg' not in state:
                    continue
                if isinstance(optimizer, opt.num_Adam):
                    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq_num']
                    new_first = self.margin * factor * exp_avg
                    new_second = self.margin * factor * exp_avg_sq
                elif isinstance(optimizer, opt.mixed_precision_Adam) or isinstance(optimizer, torch.optim.Adam):
                    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                    new_first = self.margin * factor * exp_avg
                    new_second = self.margin * (factor **2) * exp_avg_sq
                else:
                    raise Exception('optim not supported')
                if not (torch.isfinite(new_first).all() and torch.isfinite(new_second).all()):
                    return False
        return True


    @torch.no_grad()
    def mul_bufs(self, optimizer, factor):
        for group in optimizer.param_groups:
            group['eps'] = max(factor * group['eps'], self.min_eps)
            for p in group['params']:
                if p.grad is None:
                    continue
                state = optimizer.state[p]
                # cannot ask to scale before bufs are initialized
                if 'exp_avg' not in state:
                    continue
                if isinstance(optimizer, opt.num_Adam):
                    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq_num']
                    exp_avg_sq.mul_(factor)
                elif isinstance(optimizer, opt.mixed_precision_Adam) or isinstance(optimizer, torch.optim.Adam):
                    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                    exp_avg_sq.mul_(factor**2)
                else:
                    raise Exception('optim not supported')
                exp_avg.mul_(factor)

    @torch.no_grad()
    def all_finite(self, optimizer):
        for group in optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if not torch.isfinite(self.margin*self.buf_scale*grad).all():
                    return False
        return True




class grad_scaler_naive(grad_scaler):
    def step(self, optimizer):
        if self.all_finite(optimizer):
            self.unscale_grads(optimizer)
            optimizer.step()
            with torch.no_grad():
                self.consec_finite += 1
                if self.consec_finite > self.increase_every:
                    self.consec_finite = 0
                    self._scale *= 2
        else:
            with torch.no_grad():
                self.consec_finite = 0
                self._scale *= 0.5

    def unscale_grads(self, optimizer):
        """ unscales gradient with self._scale """
        for group in optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                p.grad.div_(self._scale)







