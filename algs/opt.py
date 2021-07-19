import math
import torch







class num_Adam(torch.optim.Adam):
    """
    hAdam (we call it num_Adam is the code)
    """
    def __init__(self, *args, **kwargs):
        if 'kahan' in kwargs:
            self.kahan = True
            del kwargs['kahan']
        else:
            self.kahan = False
        super().__init__(*args, **kwargs)
        

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']
                assert not amsgrad

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq_num'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    if self.kahan:
                        state['kahan'] = torch.zeros_like(p, memory_format=torch.preserve_format)


                exp_avg, exp_avg_sq, exp_avg_sq_num = state['exp_avg'], state['exp_avg_sq'], state['exp_avg_sq_num']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)


                # NOTE -- must perform operations in place
                max_g = torch.max((1-beta1) * grad.abs(), math.sqrt(beta2)*exp_avg_sq_num)
                min_g = torch.min((1-beta1) * grad.abs(), math.sqrt(beta2)*exp_avg_sq_num)
                factor = torch.sqrt(1 + (min_g/(max_g+group['eps']))**2)
                exp_avg_sq_num -= exp_avg_sq_num
                exp_avg_sq_num += max_g * torch.sqrt(1 + (min_g/(max_g+group['eps']))**2)
                # a^2 = (beta)b^2 + (1-beta) c^2
                # a =  b times sqrt(beta + (1 - beta)* (c/b)^2) 
                denom = (exp_avg_sq_num / math.sqrt(bias_correction2)).add_(group['eps'])

                step_size = group['lr'] * (1-beta1) / (math.sqrt(1-beta2) * bias_correction1)


                if self.kahan:
                    self.kahan_step(state, exp_avg, denom, step_size, p)
                else:
                    p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss



    def kahan_step(self, state, exp_avg, denom, step_size, p):
        kahan = state['kahan']
        update = - step_size * exp_avg/denom
        y = update - kahan
        t = p.data + y
        kahan.copy_( (t - p.data) - y )
        p.data.copy_(t)





class mixed_precision_Adam(torch.optim.Adam):
    """
    Adam using mixed precision
    """
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                assert not grad.is_sparse, 'no sparse'
                amsgrad = group.get("amsgrad", False)
                assert not amsgrad, "not allowed"


                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["p_data_fp32"] = p.data.float()
                    state["exp_avg"] = torch.zeros_like(state["p_data_fp32"])
                    state["exp_avg_sq"] = torch.zeros_like(state["p_data_fp32"])


                p_data_fp32 = state["p_data_fp32"]
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1
                grad = grad.float()

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                step_size = group["lr"] * math.sqrt(bias_correction2) / bias_correction1

                # update parameters
                p_data_fp32.addcdiv_(exp_avg, denom, value=-step_size)
                p.data.copy_(p_data_fp32)

        return loss



class Adam_enable_kahan(torch.optim.Adam):

    """
    minor modification of Adam which enables kahan updates, used for ablation experiments
    """

    def __init__(self, *args, **kwargs):
        if 'kahan' in kwargs:
            self.kahan = True
            del kwargs['kahan']
        else:
            self.kahan = False
        super().__init__(*args, **kwargs)


    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if self.kahan:
                        state['kahan'] = torch.zeros_like(p, memory_format=torch.preserve_format)


                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                step_size = group['lr'] / bias_correction1


                if self.kahan:
                    self.kahan_step(state, exp_avg, denom, step_size, p)
                else:
                    p.addcdiv_(exp_avg, denom, value=-step_size)
        return loss



    def kahan_step(self, state, exp_avg, denom, step_size, p):
        kahan = state['kahan']
        update = - step_size * exp_avg/denom
        y = update - kahan
        t = p.data + y
        kahan.copy_( (t - p.data) - y )
        p.data.copy_(t)



