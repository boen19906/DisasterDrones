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
    parser.add_argument("--env_size", type=float, default=None, help="Terrain size in meters")
    parser.add_argument("--max_steps", type=int, default=800, help="Max steps per episode")
    parser.add_argument("--total_timesteps", type=int, default=200000, help="Total environment steps")
    parser.add_argument("--rollout_steps", type=int, default=256, help="Steps per rollout")
    parser.add_argument("--num_epochs", type=int, default=4, help="PPO update epochs per rollout")
    parser.add_argument("--batch_size", type=int, default=128, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE lambda parameter")
    parser.add_argument("--clip_coef", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--ent_coef", type=float, default=0.02, help="Entropy coefficient")
    parser.add_argument("--vf_coef", type=float, default=0.5, help="Value loss coefficient")
    parser.add_argument("--max_grad_norm", type=float, default=0.5, help="Max gradient norm")
    parser.add_argument("--save_path", type=str, default="models/mappo_drone_mesh.pt", help="Checkpoint save path")
    parser.add_argument("--resume", action="store_true", help="Continue from save_path instead of random weights")
    parser.add_argument("--save_freq", type=int, default=10, help="Save frequency in iterations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--bc_steps", type=int, default=0, help="Search only: imitate the coverage heading for N steps before PPO. 0 = 6000 on discrete search.")
    parser.add_argument("--bc_coef", type=float, default=0.1, help="Search only: keep imitating the unpainted-cell heading during PPO.")
    return parser.parse_args()


def flatten_obs(obs_dict, agent_order):
    """Stack agent observations into (N, obs_dim) array."""
    return np.array([obs_dict[agent] for agent in agent_order], dtype=np.float32)


def critic_state(env, obs_array):
    """Search critic sees the coarse coverage map. Mesh critic sees stacked local obs."""
    if getattr(env, "discrete_actions", False):
        return np.asarray(env.global_state(), dtype=np.float32)
    return obs_array.flatten()


def search_reset(env, base_seed, episode_idx):
    """New survivor layout each episode. Map size stays fixed."""
    return env.reset(seed=base_seed + episode_idx + 1)


def behavior_clone_search(env, model, optimizer, agent_names, device, steps=12000):
    """Imitate lawnmower locally, then PPO fine-tunes. Transfers to bigger maps."""
    print(f"[BC] Cloning coverage sweep for {steps} steps...")
    obs_dict, _ = search_reset(env, 0, 0)
    last_loss = 0.0
    for t in range(steps):
        obs = flatten_obs(obs_dict, agent_names)
        expert = env.coverage_actions()
        exp = np.array([expert[a] for a in agent_names], dtype=np.float32)
        obs_t = torch.as_tensor(obs, device=device, dtype=torch.float32)
        exp_t = torch.as_tensor(exp, device=device, dtype=torch.float32)
        mean = model.actor(obs_t)
        loss = ((mean - exp_t) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        last_loss = float(loss.item())
        obs_dict, _, terms, truncs, _ = env.step(expert)
        if any(terms.values()) or any(truncs.values()):
            obs_dict, _ = search_reset(env, 0, t + 1)
        if (t + 1) % 3000 == 0:
            print(f"  [BC] step {t+1}/{steps} mse={last_loss:.4f}")
    print(f"[BC] done mse={last_loss:.4f}")


def behavior_clone_headings(env, model, optimizer, agent_names, device, steps=6000):
    """Copy the heading toward unpainted ground. People stay hidden. Walls stay masked."""
    print(f"[BC] Cloning unpainted-cell headings for {steps} steps...")
    obs_dict, _ = search_reset(env, 0, 0)
    last_loss = 0.0
    info = None
    for t in range(steps):
        obs = flatten_obs(obs_dict, agent_names)
        mask = env.action_mask()
        teacher = env.planner_actions(mask)
        state = critic_state(env, obs)
        _, logp, _, _ = model.get_action_and_value(
            torch.as_tensor(obs, device=device, dtype=torch.float32),
            torch.as_tensor(state, device=device, dtype=torch.float32),
            action=torch.as_tensor(teacher, device=device, dtype=torch.float32),
            action_mask=torch.as_tensor(mask, device=device, dtype=torch.float32),
        )
        loss = -logp.mean()
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        last_loss = float(loss.item())
        expert = {agent_names[i]: np.array([teacher[i]], dtype=np.float32) for i in range(len(agent_names))}
        obs_dict, _, terms, truncs, infos = env.step(expert)
        info = infos[agent_names[0]]
        if any(terms.values()) or any(truncs.values()):
            obs_dict, _ = search_reset(env, 0, t + 1)
        if (t + 1) % 1000 == 0 and info is not None:
            print(
                f"  [BC] step {t+1}/{steps} ce={last_loss:.3f} "
                f"found={info['discovered_survivors']}/{info['total_survivors']} "
                f"cover={info['coverage_frac']*100:.1f}%"
            )
    print(f"[BC] done ce={last_loss:.3f}")


def eval_search(env, model, agent_names, device, episodes=5, base_seed=9000):
    """Deterministic finds/coverage on fresh map sizes."""
    found_rates, covers = [], []
    for ep in range(episodes):
        obs_dict, _ = search_reset(env, base_seed, ep)
        info = None
        while True:
            obs = flatten_obs(obs_dict, agent_names)
            with torch.no_grad():
                actions_t, _, _, _ = model.get_action_and_value(
                    torch.as_tensor(obs, device=device, dtype=torch.float32),
                    torch.as_tensor(critic_state(env, obs), device=device, dtype=torch.float32),
                    deterministic=True,
                    action_mask=torch.as_tensor(env.action_mask(), device=device, dtype=torch.float32),
                )
            actions_np = actions_t.cpu().numpy()
            act_dict = {agent_names[i]: actions_np[i] for i in range(len(agent_names))}
            obs_dict, _, terms, truncs, infos = env.step(act_dict)
            info = infos[agent_names[0]]
            if any(terms.values()) or any(truncs.values()):
                break
        total = max(int(info["total_survivors"]), 1)
        found_rates.append(info["discovered_survivors"] / total)
        covers.append(info["coverage_frac"])
    return float(np.mean(found_rates)), float(np.mean(covers))


def main():
    args = parse_args()
    if args.env_size is None:
        args.env_size = 250.0 if args.task == "search" else 100.0
    if args.task == "search" and args.max_steps == 800:
        args.max_steps = int(args.env_size * 6.0)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)

    print("=" * 65)
    print(f" [MAPPO Training] task={args.task}")
    print(f" Swarm Size: {args.num_drones} Drones | Map: {args.env_size:.0f} m | Device: {args.device} | Total Steps: {args.total_timesteps}")
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
    discrete = bool(getattr(env, "discrete_actions", False))
    state_dim = int(env.state_dim) if discrete else args.num_drones * obs_dim
    stored_act_dim = 1 if discrete else act_dim
    agent_names = env.possible_agents

    # Model and optimizer
    model = MAPPOModel(
        num_drones=args.num_drones,
        obs_dim=obs_dim,
        act_dim=act_dim,
        discrete=discrete,
        state_dim=state_dim,
    ).to(args.device)
    if args.resume and os.path.exists(args.save_path):
        model.load(args.save_path, map_location=args.device)
        print(f"[RESUME] Loaded weights from {args.save_path}")
    elif args.resume:
        print(f"[RESUME] No checkpoint at {args.save_path}, starting random.")
    optimizer = optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)

    if args.task == "search" and discrete and args.bc_steps <= 0:
        args.bc_steps = 6000
    if args.task == "search" and args.bc_steps > 0 and discrete:
        behavior_clone_headings(
            env, model, optimizer, env.possible_agents, args.device, steps=args.bc_steps
        )
        bc_path = os.path.splitext(args.save_path)[0] + "_bc.pt"
        model.save(bc_path)
        print(f"[BC] saved clone to {bc_path}")
    elif args.task == "search" and args.bc_steps > 0:
        behavior_clone_search(
            env, model, optimizer, env.possible_agents, args.device, steps=args.bc_steps
        )
        bc_path = os.path.splitext(args.save_path)[0] + "_bc.pt"
        model.save(bc_path)
        print(f"[BC] saved clone to {bc_path}")
    elif args.task == "search":
        print("[MAPPO] search policy learns from reward only (no lawnmower clone)")

    # Rollout buffer
    buffer = MultiAgentRolloutBuffer(
        buffer_size=args.rollout_steps,
        num_agents=args.num_drones,
        obs_dim=obs_dim,
        act_dim=stored_act_dim,
        state_dim=state_dim,
        device=args.device,
        action_mask_dim=(env.n_actions if discrete else 0),
        store_teacher=discrete,
    )

    obs_dict, info_dict = (
        search_reset(env, args.seed, 0)
        if args.task == "search"
        else env.reset(seed=args.seed)
    )
    global_step = 0
    iteration = 0
    num_iterations = args.total_timesteps // (args.rollout_steps * args.num_drones) + 1

    episode_rewards = []
    current_ep_rewards = np.zeros(args.num_drones)
    ep_connected = []
    ep_discovery = []
    ep_found = []
    ep_total = []
    episode_idx = 0
    last_dones = np.zeros(args.num_drones, dtype=np.float32)
    start_time = time.time()
    best_found = -1.0
    best_path = os.path.splitext(args.save_path)[0] + "_best.pt"

    for it in range(1, num_iterations + 1):
        iteration = it
        buffer.reset()

        for step in range(args.rollout_steps):
            global_step += args.num_drones

            # Current agent observations: shape (N, obs_dim)
            obs_array = flatten_obs(obs_dict, agent_names)
            global_state = critic_state(env, obs_array)

            with torch.no_grad():
                obs_t = torch.tensor(obs_array, device=args.device, dtype=torch.float32)
                state_t = torch.tensor(global_state, device=args.device, dtype=torch.float32)
                action_mask = None
                teacher_np = None
                if discrete:
                    mask_np = env.action_mask()
                    teacher_np = env.planner_actions(mask_np)
                    action_mask = torch.as_tensor(mask_np, device=args.device, dtype=torch.float32)
                actions_t, log_probs_t, _, _ = model.get_action_and_value(
                    obs_t, state_t, action_mask=action_mask
                )
                values_t = model.get_value(state_t).repeat(args.num_drones)

            actions_np = actions_t.cpu().numpy()
            log_probs_np = log_probs_t.cpu().numpy()
            values_np = values_t.cpu().numpy()

            # Step environment
            act_dict = {agent_names[i]: actions_np[i] for i in range(args.num_drones)}
            next_obs_dict, rew_dict, term_dict, trunc_dict, infos = env.step(act_dict)

            rews_array = np.clip(
                np.array([rew_dict[a] for a in agent_names], dtype=np.float32),
                -80.0,
                80.0,
            )
            dones_array = np.array([term_dict[a] or trunc_dict[a] for a in agent_names], dtype=np.float32)

            current_ep_rewards += rews_array
            last_dones = dones_array
            step_connected = infos[agent_names[0]].get("connected_survivors", 0)
            step_disc = infos[agent_names[0]]["discovery_rate"]
            step_found = infos[agent_names[0]].get("discovered_survivors", 0)
            step_total = infos[agent_names[0]].get("total_survivors", 0)

            # Store in buffer
            buffer.insert(
                obs=obs_array,
                global_state=global_state,
                actions=actions_np,
                log_probs=log_probs_np,
                rewards=rews_array,
                values=values_np,
                dones=dones_array,
                action_masks=(None if action_mask is None else action_mask.cpu().numpy()),
                teacher_actions=teacher_np,
            )

            # Check if any agent done (or episode done)
            if any(term_dict.values()) or any(trunc_dict.values()):
                episode_rewards.append(current_ep_rewards.mean())
                ep_connected.append(step_connected)
                ep_discovery.append(step_disc)
                ep_found.append(step_found)
                ep_total.append(step_total)
                current_ep_rewards = np.zeros(args.num_drones)
                episode_idx += 1
                if args.task == "search":
                    obs_dict, _ = search_reset(env, args.seed, episode_idx)
                else:
                    obs_dict, _ = env.reset(seed=args.seed + episode_idx)
            else:
                obs_dict = next_obs_dict

        # Compute next value for GAE
        with torch.no_grad():
            next_obs_array = flatten_obs(obs_dict, agent_names)
            next_state_t = torch.tensor(critic_state(env, next_obs_array), device=args.device, dtype=torch.float32)
            next_values = model.get_value(next_state_t).repeat(args.num_drones).cpu().numpy()
            next_done = last_dones

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
                    b_obs, b_state, action=b_actions, action_mask=batch.get("action_mask")
                )
                bc_loss = torch.zeros((), device=args.device)
                if args.task == "search" and args.bc_coef > 0.0 and "teacher_actions" in batch:
                    _, teacher_logp, _, _ = model.get_action_and_value(
                        b_obs,
                        b_state,
                        action=batch["teacher_actions"],
                        action_mask=batch.get("action_mask"),
                    )
                    bc_loss = -teacher_logp.mean()

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

                loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss + args.bc_coef * bc_loss

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
            found = np.mean(ep_found[-10:]) if ep_found else step_found
            total = np.mean(ep_total[-10:]) if ep_total else step_total
            extra = (
                f" | Found: {found:.1f}/{total} | Cover: "
                f"{infos[agent_names[0]].get('coverage_frac', 0)*100:4.1f}%"
            )
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

        if args.task == "search" and episode_rewards:
            if found > best_found + 0.25:
                best_found = float(found)
                model.save(best_path)
                model.save(args.save_path)
                print(f"  [BEST] found={best_found:.1f} kept at {args.save_path}")
            elif iteration % args.save_freq == 0 or iteration == num_iterations:
                model.save(args.save_path)
                print(f"  [SAVE] found={found:.1f} best={best_found:.1f} -> {args.save_path}")
        elif iteration % args.save_freq == 0 or iteration == num_iterations:
            model.save(args.save_path)
            print(f"  [SAVED] Checkpoint saved to {args.save_path}")

    if args.task == "search" and best_found >= 0 and os.path.exists(best_path):
        model.load(best_path, map_location=args.device)
        model.save(args.save_path)
        ppo_found, ppo_cover = eval_search(env, model, agent_names, args.device)
        print(
            f"[EVAL] best-train found={best_found:.1f} | "
            f"deterministic found={ppo_found*100:.1f}% cover={ppo_cover*100:.1f}%"
        )
        print(f"[KEEP] {args.save_path} is the best training checkpoint, not the final weights")

    env.close()
    print("=" * 65)
    print(f"[COMPLETE] Training complete! Final model saved to: {args.save_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
