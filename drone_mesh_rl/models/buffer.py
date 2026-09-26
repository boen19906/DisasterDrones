"""
Multi-agent rollout buffer with Generalized Advantage Estimation (GAE).

Stores per-timestep, per-agent transitions for MAPPO and yields flattened
mini-batches compatible with train.py's PPO update loop.
"""

from __future__ import annotations

import numpy as np
import torch


class MultiAgentRolloutBuffer:
    """Fixed-size multi-agent rollout storage with GAE advantage computation."""

    def __init__(
        self,
        buffer_size: int,
        num_agents: int,
        obs_dim: int,
        act_dim: int,
        state_dim: int,
        device: str | torch.device = "cpu",
    ):
        self.buffer_size = buffer_size
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.state_dim = state_dim
        self.device = torch.device(device)

        self.obs = np.zeros((buffer_size, num_agents, obs_dim), dtype=np.float32)
        self.global_states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, num_agents, act_dim), dtype=np.float32)
        self.log_probs = np.zeros((buffer_size, num_agents), dtype=np.float32)
        self.rewards = np.zeros((buffer_size, num_agents), dtype=np.float32)
        self.values = np.zeros((buffer_size, num_agents), dtype=np.float32)
        self.dones = np.zeros((buffer_size, num_agents), dtype=np.float32)
        self.advantages = np.zeros((buffer_size, num_agents), dtype=np.float32)
        self.returns = np.zeros((buffer_size, num_agents), dtype=np.float32)

        self.ptr = 0
        self.full = False

    def reset(self) -> None:
        self.ptr = 0
        self.full = False

    def insert(
        self,
        obs: np.ndarray,
        global_state: np.ndarray,
        actions: np.ndarray,
        log_probs: np.ndarray,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        """Insert one multi-agent timestep. Shapes: obs/actions (N, ·); state (state_dim,)."""
        if self.ptr >= self.buffer_size:
            raise IndexError("Rollout buffer is full; call reset() before inserting.")

        self.obs[self.ptr] = obs
        self.global_states[self.ptr] = global_state
        self.actions[self.ptr] = actions
        self.log_probs[self.ptr] = log_probs
        self.rewards[self.ptr] = rewards
        self.values[self.ptr] = values
        self.dones[self.ptr] = dones
        self.ptr += 1

    def compute_gae(
        self,
        next_values: np.ndarray,
        next_dones: np.ndarray,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        """
        Compute GAE-Lambda advantages and returns over the filled rollout.

        next_values / next_dones: shape (num_agents,)
        """
        T = self.ptr
        last_gae = np.zeros(self.num_agents, dtype=np.float32)
        next_val = np.asarray(next_values, dtype=np.float32)
        next_done = np.asarray(next_dones, dtype=np.float32)

        for t in reversed(range(T)):
            if t == T - 1:
                next_non_terminal = 1.0 - next_done
                next_v = next_val
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_v = self.values[t + 1]

            delta = self.rewards[t] + gamma * next_v * next_non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae

        self.returns[:T] = self.advantages[:T] + self.values[:T]

    def get_batches(self, batch_size: int):
        """
        Yield shuffled mini-batches as torch tensors on `self.device`.

        Flattens (T, N, ·) -> (T*N, ·). Each agent sample carries the shared
        global state from its timestep.
        """
        T = self.ptr
        N = self.num_agents
        total = T * N

        obs = self.obs[:T].reshape(total, self.obs_dim)
        # Repeat each timestep's global state once per agent
        states = np.repeat(self.global_states[:T], N, axis=0)
        actions = self.actions[:T].reshape(total, self.act_dim)
        log_probs = self.log_probs[:T].reshape(total)
        advantages = self.advantages[:T].reshape(total)
        returns = self.returns[:T].reshape(total)
        values = self.values[:T].reshape(total)

        # Normalize advantages for stable PPO updates
        adv_mean = advantages.mean()
        adv_std = advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        indices = np.arange(total)
        np.random.shuffle(indices)

        for start in range(0, total, batch_size):
            end = start + batch_size
            mb_idx = indices[start:end]
            yield {
                "obs": torch.as_tensor(obs[mb_idx], dtype=torch.float32, device=self.device),
                "state": torch.as_tensor(states[mb_idx], dtype=torch.float32, device=self.device),
                "actions": torch.as_tensor(actions[mb_idx], dtype=torch.float32, device=self.device),
                "log_probs": torch.as_tensor(log_probs[mb_idx], dtype=torch.float32, device=self.device),
                "advantages": torch.as_tensor(advantages[mb_idx], dtype=torch.float32, device=self.device),
                "returns": torch.as_tensor(returns[mb_idx], dtype=torch.float32, device=self.device),
                "values": torch.as_tensor(values[mb_idx], dtype=torch.float32, device=self.device),
            }
