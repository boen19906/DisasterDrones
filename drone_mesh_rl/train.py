"""
train.py — MAPPO (Multi-Agent PPO) training for Autonomous Drone Swarm Mesh Network.

Usage:
  python train.py --task search --total_timesteps 50000 --num_drones 3
  python train.py --task mesh --total_timesteps 50000 --num_drones 5
"""

import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from envs.disaster_env import DisasterMeshEnv
from envs.search_env import SurvivorSearchEnv
from models.actor_critic import MAPPOModel
from models.buffer import MultiAgentRolloutBuffer


import sys
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

def parse_args():
    parser = argparse.ArgumentParser(description="Train MAPPO on Disaster Drone Mesh")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device")
    parser.add_argument("--task", type=str, default="search", choices=["search", "mesh"], help="search = find survivors; mesh = full LEO mesh env")
    parser.add_argument("--num_drones", type=int, default=3, help="Number of drones in swarm")
    parser.add_argument("--num_clusters", type=int, default=4, help="Number of survivor clusters")
    parser.add_argument("--env_size", type=float, default=100.0, help="Terrain size in meters")
    parser.add_argument("--max_steps", type=int, default=1000, help="Max steps per episode")
    parser.add_argument("--total_timesteps", type=int, default=50000, help="Total environment steps")
    parser.add_argument("--rollout_steps", type=int, default=256, help="Steps per rollout")
    parser.add_argument("--num_epochs", type=int, default=4, help="PPO update epochs per rollout")
    parser.add_argument("--batch_size", type=int, default=128, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE lambda parameter")
    parser.add_argument("--clip_coef", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--ent_coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--vf_coef", type=float, default=0.5, help="Value loss coefficient")
    parser.add_argument("--max_grad_norm", type=float, default=0.5, help="Max gradient norm")
    parser.add_argument("--save_path", type=str, default="models/mappo_drone_mesh.pt", help="Checkpoint save path")
    parser.add_argument("--save_freq", type=int, default=10, help="Save frequency in iterations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def flatten_obs(obs_dict, agent_order):
    """Stack agent observations into (N, obs_dim) array."""
    return np.array([obs_dict[agent] for agent in agent_order], dtype=np.float32)


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)

    print("=" * 65)
    print(f" [MAPPO Training] task={args.task}")
    print(f" Swarm Size: {args.num_drones} Drones | Device: {args.device} | Total Steps: {args.total_timesteps}")
    print("=" * 65)

    if args.task == "search":
        env = SurvivorSearchEnv(
            num_drones=args.num_drones,
            num_clusters=args.num_clusters,
            env_size=args.env_size,
            max_steps=args.max_steps,
            render_mode=None,
            seed=args.seed,
        )
        if args.save_path == "models/mappo_drone_mesh.pt":
            args.save_path = "models/mappo_search.pt"
        obs_dim = env.obs_dim
        act_dim = env.act_dim
    else:
        env = DisasterMeshEnv(
            num_drones=args.num_drones,
            num_clusters=args.num_clusters,
            env_size=args.env_size,
            max_steps=args.max_steps,
            render_mode=None,
            seed=args.seed,
        )
        obs_dim = 25
        act_dim = 5
    state_dim = args.num_drones * obs_dim
    agent_names = env.possible_agents

    # Model and optimizer
    model = MAPPOModel(num_drones=args.num_drones, obs_dim=obs_dim, act_dim=act_dim).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)

    # Rollout buffer
    buffer = MultiAgentRolloutBuffer(
        buffer_size=args.rollout_steps,
        num_agents=args.num_drones,
        obs_dim=obs_dim,
        act_dim=act_dim,
        state_dim=state_dim,
        device=args.device,
    )

    obs_dict, info_dict = env.reset(seed=args.seed)
    global_step = 0
    iteration = 0
    num_iterations = args.total_timesteps // (args.rollout_steps * args.num_drones) + 1

    episode_rewards = []
    current_ep_rewards = np.zeros(args.num_drones)
    ep_connected = []
    ep_discovery = []
    start_time = time.time()

    for it in range(1, num_iterations + 1):
        iteration = it
        buffer.reset()

        for step in range(args.rollout_steps):
            global_step += args.num_drones

            # Current agent observations: shape (N, obs_dim)
            obs_array = flatten_obs(obs_dict, agent_names)
            global_state = obs_array.flatten()

            with torch.no_grad():
                obs_t = torch.tensor(obs_array, device=args.device, dtype=torch.float32)
                state_t = torch.tensor(global_state, device=args.device, dtype=torch.float32)
                actions_t, log_probs_t, _, _ = model.get_action_and_value(obs_t, state_t)
                values_t = model.get_value(state_t).repeat(args.num_drones)

            actions_np = actions_t.cpu().numpy()
            log_probs_np = log_probs_t.cpu().numpy()
            values_np = values_t.cpu().numpy()

            # Step environment
            act_dict = {agent_names[i]: actions_np[i] for i in range(args.num_drones)}
            next_obs_dict, rew_dict, term_dict, trunc_dict, infos = env.step(act_dict)

            rews_array = np.array([rew_dict[a] for a in agent_names], dtype=np.float32)
            dones_array = np.array([term_dict[a] or trunc_dict[a] for a in agent_names], dtype=np.float32)

            current_ep_rewards += rews_array
            step_connected = infos[agent_names[0]].get("connected_survivors", 0)
            step_disc = infos[agent_names[0]]["discovery_rate"]

            # Store in buffer
            buffer.insert(
                obs=obs_array,
                global_state=global_state,
                actions=actions_np,
                log_probs=log_probs_np,
                rewards=rews_array,
                values=values_np,
                dones=dones_array,
            )

            # Check if any agent done (or episode done)
            if any(term_dict.values()) or any(trunc_dict.values()):
                episode_rewards.append(current_ep_rewards.mean())
                ep_connected.append(step_connected)
                ep_discovery.append(step_disc)
                current_ep_rewards = np.zeros(args.num_drones)
                obs_dict, _ = env.reset()
            else:
                obs_dict = next_obs_dict

        # Compute next value for GAE
        with torch.no_grad():
            next_obs_array = flatten_obs(obs_dict, agent_names)
            next_state_t = torch.tensor(next_obs_array.flatten(), device=args.device, dtype=torch.float32)
            next_values = model.get_value(next_state_t).repeat(args.num_drones).cpu().numpy()
            next_done = np.zeros(args.num_drones, dtype=np.float32)

        buffer.compute_gae(next_values, next_done, gamma=args.gamma, gae_lambda=args.gae_lambda)

        # Optimize policy & value network (PPO update)
        pg_losses, v_losses, ent_losses = [], [], []

        for epoch in range(args.num_epochs):
            for batch in buffer.get_batches(args.batch_size):
                b_obs = batch["obs"]
                b_state = batch["state"]
                b_actions = batch["actions"]
                b_log_probs = batch["log_probs"]
                b_advantages = batch["advantages"]
                b_returns = batch["returns"]
                b_values = batch["values"]

                _, new_log_probs, entropy, new_values = model.get_action_and_value(
                    b_obs, b_state, action=b_actions
                )

                log_ratio = new_log_probs - b_log_probs
                ratio = torch.exp(log_ratio)

                # Policy loss (clipped surrogate)
                pg_loss1 = -b_advantages * ratio
                pg_loss2 = -b_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss (clipped)
                v_loss_unclipped = (new_values - b_returns) ** 2
                v_clipped = b_values + torch.clamp(new_values - b_values, -args.clip_coef, args.clip_coef)
                v_loss_clipped = (v_clipped - b_returns) ** 2
                v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                # Entropy loss
                entropy_loss = entropy.mean()

                loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()

                pg_losses.append(pg_loss.item())
                v_losses.append(v_loss.item())
                ent_losses.append(entropy_loss.item())

        # Progress log
        elapsed = time.time() - start_time
        fps = int(global_step / max(elapsed, 1e-3))
        mean_rew = np.mean(episode_rewards[-10:]) if episode_rewards else current_ep_rewards.mean()
        mean_conn = np.mean(ep_connected[-10:]) if ep_connected else step_connected
        mean_disc = np.mean(ep_discovery[-10:]) if ep_discovery else step_disc
        extra = ""
        if args.task == "search":
            found = infos[agent_names[0]].get("discovered_survivors", 0)
            total = infos[agent_names[0]].get("total_survivors", 0)
            extra = f" | Found: {found}/{total} | Cover: {infos[agent_names[0]].get('coverage_frac', 0)*100:4.1f}%"
            avg_battery = 100.0
        else:
            avg_battery = env.battery_levels.mean()

        print(
            f"Iter [{iteration:03d}/{num_iterations}] | Steps: {global_step:6d} | "
            f"Rew: {mean_rew:7.1f} | Conn Surv: {mean_conn:3.1f} | "
            f"Disc: {mean_disc * 100:4.1f}% | Bat: {avg_battery:4.1f}%{extra} | "
            f"Ploss: {np.mean(pg_losses):6.3f} | Vloss: {np.mean(v_losses):6.2f} | "
            f"FPS: {fps:4d}"
        )

        # Checkpoint save
        if iteration % args.save_freq == 0 or iteration == num_iterations:
            model.save(args.save_path)
            print(f"  [SAVED] Checkpoint saved to {args.save_path}")

    env.close()
    print("=" * 65)
    print(f"[COMPLETE] Training complete! Final model saved to: {args.save_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
