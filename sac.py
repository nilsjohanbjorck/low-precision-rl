import numpy as np
import hydra

import torch
import torch.nn.functional as F
from torch import nn

from utils import utils
import quant
from algs import opt, distributions, torch_utils


"""
SAC agent
"""


class DoubleQCritic(nn.Module):
    """
    critic, outputs two q-values
    """
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth, cast_float, use_qtorch, qtorch_cfg):
        super().__init__()
        quantizer = hydra.utils.instantiate(qtorch_cfg) if use_qtorch else None
        self.Q1 = torch_utils.mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth, quantizer=quantizer)
        self.Q2 = torch_utils.mlp(obs_dim + action_dim, hidden_dim, 1, hidden_depth, quantizer=quantizer)
        self.apply(torch_utils.weight_init)
        self.in_half = False
        self.cast_float = cast_float

    def to_half(self):
        self.in_half = True
        self = self.half()

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)
        if self.in_half:
            obs = obs.half()
            action = action.half()
        else:
            obs = obs.float()
            action = action.float()
        obs_action = torch.cat([obs, action], dim=-1)
        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)
        return q1, q2



class DiagGaussianActor(nn.Module):
    """
    actor, outputs a randomized policy
    """
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth,
                 log_std_bounds, tanh_threshold, cast_float, stable_normal, use_qtorch, qtorch_cfg):
        super().__init__()

        
        self.quantizer = hydra.utils.instantiate(qtorch_cfg) if use_qtorch else None
        self.log_std_bounds = log_std_bounds
        self.trunk = torch_utils.mlp(obs_dim, hidden_dim, 2 * action_dim,
                               hidden_depth, quantizer=self.quantizer)

        self.apply(torch_utils.weight_init)
        self.tanh_threshold = tanh_threshold

        self.in_half = False
        self.cast_float = cast_float
        self.stable_normal = stable_normal

    def forward(self, obs):
        if self.in_half:
            obs = obs.half()

        mu, log_std = self.trunk(obs).chunk(2, dim=-1)

        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std +  1)

        if self.quantizer is not None:
            log_std = self.quantizer(log_std)
        std = log_std.exp()
        if self.quantizer is not None:
            std = self.quantizer(std)

        dist = distributions.SquashedNormal(mu, std, threshold=self.tanh_threshold, stable=self.stable_normal)
        action = dist.rsample()
        log_prob = dist.log_prob(action)

        sample, mean = dist.sample(), dist.mean
        if self.quantizer is not None:
            action, log_prob, sample, mean = map(self.quantizer, (action, log_prob, sample, mean))
        return action, log_prob, sample, mean


    def to_half(self):
        self.in_half = True
        self = self.half()


class SACAgent(object):
    """
    sac agent, contains the critic and actor, and performs the optimization
    """
    def __init__(self, obs_dim, action_dim, action_range, device, critic_cfg,
                 actor_cfg, discount, init_temperature, alpha_lr, betas,
                 actor_lr, actor_update_frequency, critic_lr,
                 critic_tau, critic_target_update_frequency,
                 batch_size, learnable_temperature, use_num_adam, crit_half, 
                 actor_half, use_grad_scaler, adam_eps, 
                 soft_update_scale, alpha_half, alpha_kahan, crit_kahan, 
                 nan_to_num_actions, nan_to_num_grads, 
                 nan_to_num_weights_bufs, mixed_prec_adam, use_grad_scaler_naive, 
                 use_qtorch, grad_scaler_cfg, qtorch_cfg):

        super().__init__()
        self.action_range = action_range
        self.device = torch.device(device)
        self.discount = discount
        self.critic_tau = critic_tau
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency
        self.batch_size = batch_size
        self.learnable_temperature = learnable_temperature
        self.soft_update_scale = soft_update_scale

        # nan coercing
        self.nan_to_num_actions = nan_to_num_actions
        self.nan_to_num_grads = nan_to_num_grads
        self.nan_to_num_weights_bufs = nan_to_num_weights_bufs

        # critic
        self.critic = hydra.utils.instantiate(critic_cfg).to(self.device)
        self.critic_target = hydra.utils.instantiate(critic_cfg).to(
            self.device)
        self.critic_target_scaled = hydra.utils.instantiate(critic_cfg).to(self.device)
        self.critic_target_kahan = hydra.utils.instantiate(critic_cfg).to(self.device)


        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target_scaled.load_state_dict(self.critic.state_dict())
        torch_utils.scale_all_weights(self.critic_target_kahan, 0)
        if soft_update_scale is not None:
            torch_utils.scale_all_weights(self.critic_target_scaled, soft_update_scale)

        # actor
        self.actor = hydra.utils.instantiate(actor_cfg).to(self.device)

        # alpha
        self.log_alpha = torch.tensor(np.log(init_temperature)).to(self.device)
        if alpha_half:
            self.log_alpha = self.log_alpha.half()
        # qtorch only support floats
        elif use_qtorch:
            self.log_alpha = self.log_alpha.float()
        self.log_alpha.requires_grad = True
        self.target_entropy = -action_dim


        # set optim
        if use_num_adam and use_qtorch:
            optim = quant.qtorchNumAdam
        elif use_num_adam:
            optim = opt.num_Adam
        elif mixed_prec_adam:
            optim = opt.mixed_precision_Adam
        else:
            optim = opt.Adam_enable_kahan

        # actor optim
        self.actor_optimizer = optim(self.actor.parameters(),
                                                lr=actor_lr,
                                                betas=betas, eps=adam_eps)

        # crit optim
        crit_kwargs = {'lr' : critic_lr, 'betas' : betas, 'eps' : adam_eps}
        if crit_kahan:
            crit_kwargs['kahan'] = True
        self.critic_optimizer = optim(self.critic.parameters(), **crit_kwargs)

        # alpha optim
        alpha_kwargs = {'lr' : alpha_lr, 'betas' : betas, 'eps' : adam_eps}
        if alpha_kahan:
            alpha_kwargs['kahan'] = alpha_kahan
        self.log_alpha_optimizer = optim([self.log_alpha], **alpha_kwargs)

        # create grad scalers
        assert not (use_grad_scaler and use_grad_scaler_naive), 'use only one grad scaler'
        assert not (use_grad_scaler and self.nan_to_num_grads), 'cannot use both'
        self.critic_scaler = hydra.utils.instantiate(grad_scaler_cfg)
        self.actor_scaler = hydra.utils.instantiate(grad_scaler_cfg)
        self.alpha_scaler = hydra.utils.instantiate(grad_scaler_cfg)
        self.use_grad_scaler = use_grad_scaler or use_grad_scaler_naive

        self.train()
        self.critic_target.train()

        # move to half prec optionally
        if crit_half:
            self.critic.to_half()
            self.critic_target.to_half()
            self.critic_target_scaled.to_half()
            self.critic_target_kahan.to_half()

        if actor_half:
            self.actor.to_half()

        self.quantizer = hydra.utils.instantiate(qtorch_cfg) if use_qtorch else None
        # qtorch

        # give optimizers access to quantizer for kahan step
        if use_qtorch:
            assert use_num_adam and use_grad_scaler, 'only numadam supported'
            self.actor_optimizer.quantizer = self.quantizer
            self.critic_optimizer.quantizer = self.quantizer
            self.log_alpha_optimizer.quantizer = self.quantizer
            assert isinstance(self.critic_scaler, quant.gradScalerQtorch), 'must use qtorch gradscaler'
            self.critic_scaler.quantizer = self.quantizer
            self.actor_scaler.quantizer = self.quantizer
            self.alpha_scaler.quantizer = self.quantizer


    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, obs, sample=False):
        obs = torch.FloatTensor(obs).to(self.device)
        obs = obs.unsqueeze(0)
        _, _, action_sampled, dist_mean = self.actor(obs)
        action = action_sampled if sample else dist_mean
        action = action.clamp(*self.action_range)
        assert action.ndim == 2 and action.shape[0] == 1
        if self.nan_to_num_actions:
            action = torch_utils.nan_to_num(action)
        return utils.to_np(action[0])

    def grad_step(self, loss, optimizer, scaler):
        optimizer.zero_grad()
        if self.use_grad_scaler:
            scaler.scale(loss).backward()
            qtorch_ok = self.quantizer is None or quant.qtorch_no_inf(optimizer, self.quantizer)
            if scaler.can_step(optimizer) and qtorch_ok:
                quant.optim_step_maybe_with_qtorch(optimizer, self.quantizer) 
            scaler.last_ok = scaler.last_ok and qtorch_ok
            scaler.post_step(optimizer)
        else:
            loss.backward()
            if self.nan_to_num_grads:
                torch_utils.nan_to_num_grads(optimizer)
            quant.optim_step_maybe_with_qtorch(optimizer, None)
        if self.nan_to_num_weights_bufs:
           torch_utils.nan_to_num_weights_bufs(optimizer)

    def update_critic(self, obs, action, reward, next_obs, not_done):
        next_action, log_prob, _, _  = self.actor(next_obs)
        log_prob = log_prob.sum(-1, keepdim=True)
        target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
        target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob

        # do not cast types here
        if isinstance(target_V, torch.cuda.HalfTensor):
            target_Q = reward.half() + (not_done.half() * torch.cuda.HalfTensor([self.discount]) * target_V)
        else:
            target_Q = reward + (not_done * self.discount * target_V)
        target_Q = target_Q.detach()

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.grad_step(critic_loss, self.critic_optimizer, self.critic_scaler)


    def update_actor_and_alpha(self, obs):
        action, log_prob, _, _ = self.actor(obs)
        log_prob = log_prob.sum(-1, keepdim=True)
        actor_Q1, actor_Q2 = self.critic(obs, action)
        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()
        self.grad_step(actor_loss, self.actor_optimizer, self.actor_scaler)
        if self.learnable_temperature:
            alpha_loss = (self.alpha * (-log_prob - self.target_entropy).detach()).mean()
            self.grad_step(alpha_loss, self.log_alpha_optimizer, self.alpha_scaler)


    def update(self, replay_buffer, step):
        obs, action, reward, next_obs, not_done, not_done_no_max = replay_buffer.sample(self.batch_size)
        self._update(obs, action, reward, next_obs, not_done, not_done_no_max, step)


    def _update(self, obs, action, reward, next_obs, not_done, not_done_no_max, step):
        """
        main update function
        """
        self.update_critic(obs, action, reward, next_obs, not_done_no_max)

        if step % self.actor_update_frequency == 0:
            self.update_actor_and_alpha(obs)

        if step % self.critic_target_update_frequency == 0:
            if self.soft_update_scale is not None and self.quantizer is None:
                torch_utils.soft_update_params_kahan(self.critic, self.critic_target, self.critic_target_scaled,
                                                self.critic_target_kahan, self.critic_tau, self.soft_update_scale)
            elif self.soft_update_scale is not None and self.quantizer is not None:
                    quant.soft_update_params_with_qtorch(self.quantizer, self.critic, self.critic_target,
                                                        self.critic_target_scaled, self.critic_target_kahan,
                                                        self.critic_tau, self.soft_update_scale)
            else:
               assert self.quantizer is None, 'not supported'
               torch_utils.soft_update_params(self.critic, self.critic_target, self.critic_tau)




