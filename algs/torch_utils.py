import torch
import torch.nn as nn
import quant



"""
various torch utilities
"""



def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)




def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None, quantizer=None):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)

    if quantizer is not None:
        trunk = quant.Sequential_quantized(*mods, quantizer=quantizer)
    else:
        trunk = nn.Sequential(*mods)
    return trunk




def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)




def soft_update_params_kahan(net, target_net, target_net_scaled, target_net_kahan, tau, soft_scale):
    for param, target_param, target_scaled_param, kahan_param in zip(net.parameters(), target_net.parameters(),
                                                                     target_net_scaled.parameters(),
                                                                     target_net_kahan.parameters()):
        update = tau * (soft_scale * param.data  -  target_scaled_param.data)
        y = update - kahan_param.data
        t = target_scaled_param.data + y
        kahan_new = (t - target_scaled_param.data) - y
        kahan_param.data.copy_(kahan_new)
        target_scaled_param.data.copy_(t)
        target_param.data.copy_(target_scaled_param.data/soft_scale)






def scale_all_weights(net, scale):
    for param in net.parameters():
        param.data.copy_(param.data * scale)


# remove nan grads
def nan_to_num_grads(optim):
    for group in optim.param_groups:
        for p in group['params']:
            if p.grad is not None:
                p.grad.data = nan_to_num(p.grad.data)

# remove nan weights and bufs
def nan_to_num_weights_bufs(optim):
    for group in optim.param_groups:
        for p in group['params']:
            p.data = nan_to_num(p.data)
            state = optim.state[p]
            if 'exp_avg' in state:
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                exp_avg = nan_to_num(exp_avg)
                exp_avg_sq = nan_to_num(exp_avg_sq)

def nan_to_num(t):
    t[torch.isnan(t)] = 0.0
    pos_inf = torch.logical_and(torch.isinf(t), t>0)
    neg_inf = torch.logical_and(torch.isinf(t), t<0)
    t[pos_inf] = 65504.0
    t[neg_inf] = -65504.0
    return t



