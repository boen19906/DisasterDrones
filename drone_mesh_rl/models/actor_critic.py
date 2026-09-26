"""
MAPPO actor-critic: decentralized Gaussian actor + centralized value critic (CTDE).

Architecture (per PROJECT_ARCHITECTURE_AND_ROADMAP.md):
  - Actor:  obs_dim (25)  -> continuous act_dim (5), tanh-bounded Gaussian
  - Critic: num_drones * obs_dim (global state) -> V(s)
  - Orthogonal weight initialization (CleanRL-style)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    """Orthogonal initialization matching CleanRL / architecture doc."""
    nn.init.orthogonal_(layer.weight, gain=std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class MAPPOModel(nn.Module):
    """Multi-Agent PPO model with shared decentralized actor and centralized critic."""

    def __init__(self, num_drones: int, obs_dim: int, act_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.num_drones = num_drones
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.state_dim = num_drones * obs_dim

        # Decentralized actor: local obs -> action mean (pre-tanh)
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, act_dim), std=0.01),
        )
        # State-independent log std (learned)
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

        # Centralized critic: concatenated global state -> value
        self.critic = nn.Sequential(
            layer_init(nn.Linear(self.state_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )

    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """Return V(s). Accepts (state_dim,) or (B, state_dim); returns matching 1-D values."""
        if state.dim() == 1:
            state = state.unsqueeze(0)
        return self.critic(state).squeeze(-1)

    def _distribution(self, obs: torch.Tensor) -> Normal:
        mean = self.actor(obs)
        logstd = self.actor_logstd.expand_as(mean)
        return Normal(mean, torch.exp(logstd))

    @staticmethod
    def _tanh_log_prob(dist: Normal, raw: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """log π(a|o) for a = tanh(u), u ~ Normal."""
        log_prob = dist.log_prob(raw).sum(dim=-1)
        log_prob = log_prob - torch.log(1.0 - action.pow(2) + 1e-6).sum(dim=-1)
        return log_prob

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        state: torch.Tensor,
        action: torch.Tensor | None = None,
        deterministic: bool = False,
    ):
        """
        Sample (or evaluate) actions and return (action, log_prob, entropy, value).

        obs:   (N, obs_dim) or (B, obs_dim)
        state: (state_dim,) shared, or (B, state_dim) batched
        action: optional stored env actions for PPO ratio update
        """
        dist = self._distribution(obs)

        if action is None:
            if deterministic:
                raw = dist.mean
                action = torch.tanh(raw)
            else:
                raw = dist.rsample()
                action = torch.tanh(raw)
        else:
            raw = torch.atanh(action.clamp(-0.999999, 0.999999))

        log_prob = self._tanh_log_prob(dist, raw, action)
        entropy = dist.entropy().sum(dim=-1)

        value = self.get_value(state)
        # Shared global state during rollout: broadcast V(s) to each agent
        if state.dim() == 1 and value.shape[0] == 1 and obs.shape[0] != 1:
            value = value.expand(obs.shape[0])

        return action, log_prob, entropy, value

    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Convenience: return only actions from local observations."""
        dist = self._distribution(obs)
        if deterministic:
            return torch.tanh(dist.mean)
        return torch.tanh(dist.rsample())

    def evaluate_actions(self, obs: torch.Tensor, state: torch.Tensor, actions: torch.Tensor):
        """Re-evaluate log-probs / entropy / values for a batch of stored actions."""
        _, log_prob, entropy, value = self.get_action_and_value(obs, state, action=actions)
        return log_prob, entropy, value

    def save(self, path: str) -> None:
        torch.save(
            {
                "state_dict": self.state_dict(),
                "num_drones": self.num_drones,
                "obs_dim": self.obs_dim,
                "act_dim": self.act_dim,
            },
            path,
        )

    def load(self, path: str, map_location=None) -> None:
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            self.load_state_dict(checkpoint["state_dict"])
        else:
            self.load_state_dict(checkpoint)
