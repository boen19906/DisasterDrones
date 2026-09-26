"""
Multi-agent rollout buffer with Generalized Advantage Estimation (GAE).
"""

import numpy as np
import torch


class MultiAgentRolloutBuffer:
    def __init__(self, buffer_size, num_agents, obs_dim, act_dim, state_dim, device="cpu"):
        self.buffer_size = buffer_size
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.state_dim = state_dim
        self.device = device
        self.ptr = 0
        self._alloc()

    def _alloc(self):
        t, n = self.buffer_size, self.num_agents
        self.obs = np.zeros((t, n, self.obs_dim), dtype=np.float32)
        self.global_state = np.zeros((t, self.state_dim), dtype=np.float32)
        self.actions = np.zeros((t, n, self.act_dim), dtype=np.float32)
        self.log_probs = np.zeros((t, n), dtype=np.float32)
        self.rewards = np.zeros((t, n), dtype=np.float32)
        self.values = np.zeros((t, n), dtype=np.float32)
        self.dones = np.zeros((t, n), dtype=np.float32)
        self.advantages = np.zeros((t, n), dtype=np.float32)
        self.returns = np.zeros((t, n), dtype=np.float32)

    def reset(self):
        self.ptr = 0

    def insert(self, obs, global_state, actions, log_probs, rewards, values, dones):
        idx = min(self.ptr, self.buffer_size - 1)
        self.obs[idx] = obs
        self.global_state[idx] = global_state
        self.actions[idx] = actions
        self.log_probs[idx] = log_probs
        self.rewards[idx] = rewards
        self.values[idx] = values
        self.dones[idx] = dones
        self.ptr = min(self.ptr + 1, self.buffer_size)

    def compute_gae(self, next_values, next_done, gamma=0.99, gae_lambda=0.95):
        t = self.ptr
        next_values = np.asarray(next_values, dtype=np.float32)
        next_done = np.asarray(next_done, dtype=np.float32)
        last_gae = np.zeros(self.num_agents, dtype=np.float32)
        for step in reversed(range(t)):
            if step == t - 1:
                next_nonterminal = 1.0 - next_done
                nxt_v = next_values
            else:
                next_nonterminal = 1.0 - self.dones[step + 1]
                nxt_v = self.values[step + 1]
            delta = self.rewards[step] + gamma * nxt_v * next_nonterminal - self.values[step]
            last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
            self.advantages[step] = last_gae
        self.returns[:t] = self.advantages[:t] + self.values[:t]

    def get_batches(self, batch_size):
        t, n = self.ptr, self.num_agents
        total = t * n
        obs = self.obs[:t].reshape(total, self.obs_dim)
        state = np.repeat(self.global_state[:t], n, axis=0)
        actions = self.actions[:t].reshape(total, self.act_dim)
        log_probs = self.log_probs[:t].reshape(total)
        advantages = self.advantages[:t].reshape(total)
        returns = self.returns[:t].reshape(total)
        values = self.values[:t].reshape(total)

        adv_std = advantages.std()
        if adv_std > 1e-8:
            advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)

        indices = np.random.permutation(total)
        for start in range(0, total, batch_size):
            mb = indices[start : start + batch_size]
            yield {
                "obs": torch.as_tensor(obs[mb], device=self.device, dtype=torch.float32),
                "state": torch.as_tensor(state[mb], device=self.device, dtype=torch.float32),
                "actions": torch.as_tensor(actions[mb], device=self.device, dtype=torch.float32),
                "log_probs": torch.as_tensor(log_probs[mb], device=self.device, dtype=torch.float32),
                "advantages": torch.as_tensor(advantages[mb], device=self.device, dtype=torch.float32),
                "returns": torch.as_tensor(returns[mb], device=self.device, dtype=torch.float32),
                "values": torch.as_tensor(values[mb], device=self.device, dtype=torch.float32),
            }
