"""
evaluate.py — Evaluate trained MAPPO policy on Disaster Drone Mesh.

Usage:
  python evaluate.py --model_path models/mappo_drone_mesh.pt --episodes 10
  python evaluate.py --random --episodes 5
"""

import os
import sys
import argparse
import numpy as np
import torch

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from envs.disaster_env import DisasterMeshEnv
from models.actor_critic import MAPPOModel


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Disaster Drone Mesh policy")
    parser.add_argument("--model_path", type=str, default="models/mappo_drone_mesh.pt", help="Path to model checkpoint")
    parser.add_argument("--num_drones", type=int, default=5, help="Number of drones")
    parser.add_argument("--num_clusters", type=int, default=4, help="Number of survivor clusters")
    parser.add_argument("--env_size", type=float, default=250.0, help="Environment size in meters")
    parser.add_argument("--max_steps", type=int, default=2400, help="Max steps per episode")
    parser.add_argument("--drone_max_speed", type=float, default=8.0, help="Max drone cruise speed m/s")
    parser.add_argument("--drone_max_altitude", type=float, default=40.0, help="Altitude cap in meters")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--random", action="store_true", help="Evaluate random baseline policy")
    parser.add_argument("--seed", type=int, default=100, help="Evaluation random seed")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device")
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("=" * 65)
    print(" [EVALUATION] Disaster Drone Mesh Autonomous Swarm")
    print(f" Episodes: {args.episodes} | Swarm: {args.num_drones} Drones | Clusters: {args.num_clusters}")
    if args.random:
        print(" Mode: Random Policy Baseline")
    else:
        print(f" Mode: Trained Model ({args.model_path})")
    print("=" * 65)

    env = DisasterMeshEnv(
        num_drones=args.num_drones,
        num_clusters=args.num_clusters,
        env_size=args.env_size,
        max_steps=args.max_steps,
        render_mode=None,
        drone_max_speed=args.drone_max_speed,
        drone_max_altitude=args.drone_max_altitude,
        seed=args.seed,
    )

    model = None
    if not args.random:
        if not os.path.exists(args.model_path):
            print(f"[ERROR] Checkpoint not found: {args.model_path}")
            print("Run with --random to evaluate random baseline or train first with: python train.py")
            sys.exit(1)

        model = MAPPOModel(num_drones=args.num_drones, obs_dim=25, act_dim=5).to(args.device)
        model.load(args.model_path, map_location=args.device)
        model.eval()
        print(f"[LOADED] Checkpoint loaded from {args.model_path}\n")

    agent_names = env.possible_agents

    ep_rewards = []
    ep_discovery_rates = []
    ep_avg_connected = []
    ep_avg_batteries = []
    ep_steps = []
    ep_success = []

    for ep in range(1, args.episodes + 1):
        obs_dict, info_dict = env.reset(seed=args.seed + ep * 13)
        total_rew = 0.0
        connected_list = []
        battery_list = []
        step = 0

        while True:
            step += 1
            if args.random:
                actions = {a: env.action_space(a).sample() for a in env.agents}
            else:
                obs_array = np.array([obs_dict[a] for a in agent_names], dtype=np.float32)
                state = obs_array.flatten()
                with torch.no_grad():
                    obs_t = torch.tensor(obs_array, device=args.device, dtype=torch.float32)
                    state_t = torch.tensor(state, device=args.device, dtype=torch.float32)
                    actions_t, _, _, _ = model.get_action_and_value(obs_t, state_t, deterministic=True)
                actions_np = actions_t.cpu().numpy()
                actions = {agent_names[i]: actions_np[i] for i in range(args.num_drones) if agent_names[i] in env.agents}

            obs_dict, rews, terms, truncs, infos = env.step(actions)
            step_rew = sum(rews.values())
            total_rew += step_rew

            first_info = infos[agent_names[0]]
            connected_list.append(first_info["connected_survivors"])
            battery_list.append(env.battery_levels.mean())

            if all(terms.values()) or all(truncs.values()):
                break

        disc_rate = first_info["discovery_rate"]
        mean_conn = np.mean(connected_list) if connected_list else 0.0
        final_bat = env.battery_levels.mean()
        is_success = disc_rate >= 1.0 and mean_conn > 0

        ep_rewards.append(total_rew)
        ep_discovery_rates.append(disc_rate)
        ep_avg_connected.append(mean_conn)
        ep_avg_batteries.append(final_bat)
        ep_steps.append(step)
        ep_success.append(is_success)

        print(
            f" Episode {ep:02d}/{args.episodes:02d} | Steps: {step:4d} | Rew: {total_rew:8.1f} | "
            f"Discovered: {disc_rate * 100:5.1f}% | Avg Conn: {mean_conn:4.1f} | Final Bat: {final_bat:5.1f}%"
        )

    env.close()

    print("\n" + "=" * 65)
    print(" SUMMARY METRICS ACROSS ALL EPISODES")
    print("=" * 65)
    print(f" Mean Episodic Return:       {np.mean(ep_rewards):10.2f} +/- {np.std(ep_rewards):.2f}")
    print(f" Mean Discovery Rate:        {np.mean(ep_discovery_rates) * 100:9.1f}% +/- {np.std(ep_discovery_rates) * 100:.1f}%")
    print(f" Mean Connected Survivors:   {np.mean(ep_avg_connected):10.2f} +/- {np.std(ep_avg_connected):.2f}")
    print(f" Mean Final Battery Level:   {np.mean(ep_avg_batteries):9.1f}% +/- {np.std(ep_avg_batteries):.1f}%")
    print(f" Mean Episode Duration:      {np.mean(ep_steps):10.1f} steps")
    print(f" Swarm Mission Success Rate: {np.mean(ep_success) * 100:9.1f}%")
    print("=" * 65)


if __name__ == "__main__":
    main()
