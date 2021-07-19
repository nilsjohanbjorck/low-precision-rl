import torch
from torch import nn

from qtorch.quant import Quantizer, quantizer
from qtorch import FloatingPoint
from algs import gscale
from algs import opt

LARGE_NUM = 10**10



""" 
logic for interfacing with qtorch
"""


class gradScalerQtorch(gscale.grad_scaler):
    """
    override our gradscaler to do qtorch quantization
    """
    @torch.no_grad()
    def all_finite(self, optimizer):
        for group in optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if (self.margin*self.buf_scale*grad >= self.quantizer.max_val).any():
                    return False
        return True


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
                if isinstance(optimizer, opt.num_Adam) or isinstance(optimizer, qtorchNumAdam):
                    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq_num']
                    new_first = self.margin * factor * exp_avg
                    new_second = self.margin * factor * exp_avg_sq
                elif isinstance(optimizer, opt.mixed_precision_Adam) or isinstance(optimizer, torch.optim.Adam):
                    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                    new_first = self.margin * factor * exp_avg
                    new_second = self.margin * (factor **2) * exp_avg_sq
                else:
                    raise Exception('optim not supported')
                if (new_first>=self.quantizer.max_val).any() or (new_second>=self.quantizer.max_val).any():
                    return False
        return True



class qtorchNumAdam(opt.num_Adam):
    def kahan_step(self, state, exp_avg, denom, step_size, p):
        kahan = state['kahan']
        update = self.quantizer(- step_size * exp_avg/denom)
        y = self.quantizer(update - kahan)
        t = self.quantizer(p.data + y)
        kahan_new = self.quantizer(self.quantizer(t - p.data) - y)
        kahan.copy_(kahan_new)
        p.data.copy_(t)




class quantClass(Quantizer):
    def __init__(self, exp, man, rounding):
        bit_rep = FloatingPoint(exp=exp, man=man)
        super().__init__(forward_number=bit_rep, backward_number=bit_rep,
			 forward_rounding=rounding, backward_rounding=rounding)
        self.max_val = self.quantize(torch.Tensor([LARGE_NUM])).cuda()


def qtorch_no_inf(optim, quant):
    for group in optim.param_groups:
        for p in group['params']:
            if p.grad is None:
                continue
            grad = p.grad
            if (grad.abs() >= quant.max_val).any():
                return False         
    return True



class Sequential_quantized(nn.Sequential):
    def __init__(self, *args, quantizer=None):
        super().__init__(*args)
        assert quantizer is not None
        self.quantizer = quantizer
    def forward(self, input):
        for module in self:
            input = module(input)
            input = self.quantizer(input)
        return input



def optim_step_maybe_with_qtorch(optim, quant):
    if quant is not None:
        pre_quant(optim, quant)
    optim.step()
    if quant is not None:
        post_quant(optim, quant)

@torch.no_grad()
def pre_quant(optim, quant):
    # quantize gradient
    for group in optim.param_groups:
        for p in group["params"]:
            if p.grad is not None:
                p.grad.data = quant(p.grad.data)

@torch.no_grad()
def post_quant(optim, quant):
    # quantize weight
    for group in optim.param_groups:
        for p in group["params"]:
            p.data = quant(p.data).data

    # quantize buffers
    for group in optim.param_groups:
        for p in group["params"]:
            param_state = optim.state[p]
            for key in ['exp_avg', 'exp_avg_sq', 'exp_avg_sq_num']:
                if key in optim.state[p]:
                    param_state[key] = quant(param_state[key])


def soft_update_params_with_qtorch(quantizer, net, target_net, target_net_scaled, target_net_kahan, tau, soft_scale):
    with torch.no_grad():
        for param, target_param, target_scaled_param, kahan_param in zip(net.parameters(), target_net.parameters(),
                                                                     target_net_scaled.parameters(),
                                                                     target_net_kahan.parameters()):
            update = quantizer.quantize(tau * (soft_scale * param.data  -  target_scaled_param.data))
            y = quantizer.quantize(update - kahan_param.data)
            t = quantizer.quantize(target_scaled_param.data + y)
            kahan_new = quantizer.quantize(quantizer.quantize(t - target_scaled_param.data) - y)
            kahan_param.data.copy_(kahan_new)
            target_scaled_param.data.copy_(t)
            target_param.data.copy_(quantizer.quantize(target_scaled_param.data/soft_scale))

