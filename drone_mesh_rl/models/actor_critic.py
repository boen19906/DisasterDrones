"""
MAPPO actor-critic: shared decentralized actor + centralized critic (CTDE).
"""

import os

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


def _layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class MAPPOModel(nn.Module):
    """Parameter-shared Gaussian actor and a global-state value critic."""

    def __init__(self, num_drones, obs_dim, act_dim, hidden=128):
        super().__init__()
        self.num_drones = num_drones
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        state_dim = num_drones * obs_dim

        self.actor = nn.Sequential(
            _layer_init(nn.Linear(obs_dim, hidden)),
            nn.Tanh(),
            _layer_init(nn.Linear(hidden, hidden)),
            nn.Tanh(),
            _layer_init(nn.Linear(hidden, act_dim), std=0.01),
        )
        self.actor_log_std = nn.Parameter(torch.zeros(act_dim))

        self.critic = nn.Sequential(
            _layer_init(nn.Linear(state_dim, hidden)),
            nn.Tanh(),
            _layer_init(nn.Linear(hidden, hidden)),
            nn.Tanh(),
            _layer_init(nn.Linear(hidden, 1), std=1.0),
        )

    def get_value(self, state):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        return self.critic(state).squeeze(-1)

    def get_action_and_value(self, obs, state, action=None, deterministic=False):
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        mean = torch.tanh(self.actor(obs))
        log_std = self.actor_log_std.clamp(-5.0, 2.0)
        std = log_std.exp().expand_as(mean)
        dist = Normal(mean, std)

        if action is None:
            action = mean if deterministic else dist.sample()
            action = action.clamp(-1.0, 1.0)

        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self.get_value(state)

        if value.dim() == 0:
            value = value.unsqueeze(0)
        if value.shape[0] == 1 and log_prob.shape[0] > 1:
            value = value.expand(log_prob.shape[0])
        elif value.shape[0] != log_prob.shape[0] and log_prob.shape[0] % value.shape[0] == 0:
            repeats = log_prob.shape[0] // value.shape[0]
            value = value.repeat_interleave(repeats)

        return action, log_prob, entropy, value

    def save(self, path):
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "num_drones": self.num_drones,
                "obs_dim": self.obs_dim,
                "act_dim": self.act_dim,
            },
            path,
        )

    def load(self, path, map_location="cpu"):
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            self.load_state_dict(ckpt["state_dict"])
        else:
            self.load_state_dict(ckpt)
        return self
