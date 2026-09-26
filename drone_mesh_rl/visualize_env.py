"""
visualize_env.py — 3D Interactive PyBullet Visualizer for Drone Swarm Mesh Simulation.

Features:
  - Full 3D rendering with PyBullet GUI
  - Realistic Gaussian mountain heightfield terrain
  - Color-coded drones: RED = Gateway (LEO Uplink), BLUE = Relay, GRAY = Inactive
  - Real-time RF mesh link visualization (cyan 3D lines)
  - Satellite orbital sweep indicator and uplink beam (yellow line)
  - Ground survivor cluster markers (GREEN = Discovered, RED = Undiscovered)
  - Interactive HUD: battery %, connected survivors, wind vector, satellite elevation
  - Optional model inference or manual control mode

Usage:
  python visualize_env.py                                # Exploration demo
  python visualize_env.py --model_path models/test_model.pt # Playback trained model
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import pybullet as p

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from envs.disaster_env import DisasterMeshEnv
from envs.search_env import SurvivorSearchEnv

try:
    from models.actor_critic import MAPPOModel
except ModuleNotFoundError:
    MAPPOModel = None


def parse_args():
    parser = argparse.ArgumentParser(description="3D Visualizer for Drone Swarm Mesh")
    parser.add_argument("--model_path", type=str, default="", help="Optional model checkpoint path")
    parser.add_argument("--num_drones", type=int, default=3, help="Number of drones")
    parser.add_argument("--num_clusters", type=int, default=4, help="Number of survivor clusters")
    parser.add_argument("--env_size", type=float, default=100.0, help="Terrain size in meters")
    parser.add_argument("--fps", type=float, default=30.0, help="Display FPS target")
    parser.add_argument("--seed", type=int, default=42, help="World seed")
    parser.add_argument("--task", type=str, default="search", choices=["search", "mesh"], help="search = find people; mesh = LEO demo")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 65)
    print(f" [3D VISUALIZER] task={args.task}")
    print(" Controls: Space = Pause | R = Reset Episode | Q / ESC = Exit")
    print("=" * 65)

    if args.task == "search":
        env = SurvivorSearchEnv(
            num_drones=args.num_drones,
            num_clusters=args.num_clusters,
            env_size=args.env_size,
            max_steps=5000,
            render_mode="human",
            seed=args.seed,
        )
        obs_dim, act_dim = env.obs_dim, env.act_dim
    else:
        env = DisasterMeshEnv(
            num_drones=args.num_drones,
            num_clusters=args.num_clusters,
            env_size=args.env_size,
            max_steps=5000,
            render_mode="human",
            seed=args.seed,
        )
        obs_dim, act_dim = 25, 5

    model = None
    if args.model_path and os.path.exists(args.model_path):
        if MAPPOModel is None:
            print("[WARNING] models/actor_critic.py is missing; using heuristic policy.")
        else:
            print(f"[MODEL] Loading policy checkpoint from {args.model_path}...")
            model = MAPPOModel(num_drones=args.num_drones, obs_dim=obs_dim, act_dim=act_dim)
            try:
                model.load(args.model_path, map_location="cpu")
                model.eval()
                print("[MODEL] Model loaded successfully!")
            except Exception as e:
                print(f"[WARNING] Failed to load model ({e}), using heuristic policy.")
                model = None

    obs_dict, info_dict = env.reset(seed=args.seed)
    agent_names = env.possible_agents

    dt_target = 1.0 / args.fps
    paused = False
    step_num = 0

    try:
        while True:
            loop_start = time.time()

            # Handle keyboard input
            keys = p.getKeyboardEvents(physicsClientId=env.client)
            if ord("q") in keys or ord("Q") in keys or 27 in keys:  # 27 = ESC
                print("[EXIT] User requested exit.")
                break
            if ord(" ") in keys and (keys[ord(" ")] & p.KEY_WAS_TRIGGERED):
                paused = not paused
                print(f"[{'PAUSED' if paused else 'RESUMED'}]")
            if ord("r") in keys and (keys[ord("r")] & p.KEY_WAS_TRIGGERED):
                print("[RESET] Resetting environment...")
                obs_dict, info_dict = env.reset()
                step_num = 0
                continue

            if not paused:
                step_num += 1

                # Generate actions
                if model is not None:
                    obs_array = np.array([obs_dict[a] for a in agent_names], dtype=np.float32)
                    state = obs_array.flatten()
                    with torch.no_grad():
                        obs_t = torch.tensor(obs_array, dtype=torch.float32)
                        state_t = torch.tensor(state, dtype=torch.float32)
                        actions_t, _, _, _ = model.get_action_and_value(obs_t, state_t, deterministic=True)
                    actions_np = actions_t.cpu().numpy()
                    actions = {
                        agent_names[i]: actions_np[i]
                        for i in range(args.num_drones)
                        if agent_names[i] in env.agents
                    }
                else:
                    actions = {}
                    half = args.env_size / 2.0
                    if args.task == "search":
                        if not hasattr(env, "sweep_dir"):
                            env.sweep_dir = np.ones(args.num_drones)
                        for i in range(args.num_drones):
                            agent = agent_names[i]
                            pos = env.drone_positions[i]
                            lane_x = -half + (i + 0.5) * (args.env_size / args.num_drones)
                            vx = float(np.clip((lane_x - pos[0]) / 8.0, -1.0, 1.0))
                            if pos[1] > half - 8.0:
                                env.sweep_dir[i] = -1.0
                            elif pos[1] < -half + 8.0:
                                env.sweep_dir[i] = 1.0
                            vy = float(env.sweep_dir[i])
                            hover_z = getattr(env, "hover_altitude", 8.0)
                            vz = float(np.clip((hover_z - pos[2]) / 5.0, -1.0, 1.0))
                            actions[agent] = np.array([vx, vy, vz], dtype=np.float32)
                    else:
                        t = step_num * 0.05
                        for i in range(args.num_drones):
                            if not env.drone_alive[i]:
                                continue
                            agent = agent_names[i]
                            angle = t + (2 * np.pi * i / args.num_drones)
                            vx = np.cos(angle) * 0.5
                            vy = np.sin(angle) * 0.5
                            vz = 0.05 * np.sin(t * 2 + i)
                            role_toggle = 1.0 if i == 0 else -1.0
                            actions[agent] = np.array([vx, vy, vz, role_toggle, 0.0], dtype=np.float32)

                obs_dict, rews, terms, truncs, infos = env.step(actions)

                # Update debug visuals at a smooth 10 Hz rate instead of 60 Hz to avoid strobe flickering
                if step_num % 3 == 0:
                    p.removeAllUserDebugItems(physicsClientId=env.client)

                    alive_indices = [i for i in range(args.num_drones) if env.drone_alive[i]]

                    if args.task == "mesh":
                        for i in alive_indices:
                            for j in alive_indices:
                                if i < j:
                                    pos_i = env.drone_positions[i]
                                    pos_j = env.drone_positions[j]
                                    dist = np.linalg.norm(pos_i - pos_j)
                                    if dist < 35.0 and env.terrain.check_los(pos_i, pos_j):
                                        p.addUserDebugLine(
                                            pos_i.tolist(),
                                            pos_j.tolist(),
                                            lineColorRGB=[0.0, 0.9, 1.0],
                                            lineWidth=2.0,
                                            physicsClientId=env.client,
                                        )
                        for i in alive_indices:
                            if env.gateway_roles[i]:
                                pos_gw = env.drone_positions[i]
                                if env.satellite.can_uplink(pos_gw[0], pos_gw[1], pos_gw[2]):
                                    sat_vis_pos = [env.satellite.current_x, env.satellite.current_y, 45.0]
                                    p.addUserDebugLine(
                                        pos_gw.tolist(),
                                        sat_vis_pos,
                                        lineColorRGB=[1.0, 0.9, 0.0],
                                        lineWidth=3.0,
                                        physicsClientId=env.client,
                                    )

                    for i in range(args.num_drones):
                        pos = env.drone_positions[i]
                        if args.task == "search":
                            color = [0.2, 0.5, 1.0, 1.0]
                            tag = f"D{i}"
                        else:
                            batt = env.battery_levels[i]
                            is_gw = bool(env.gateway_roles[i])
                            is_alive = bool(env.drone_alive[i])
                            if not is_alive:
                                color = [0.3, 0.3, 0.3, 1.0]
                                tag = f"D{i} [DEAD]"
                            elif is_gw:
                                color = [1.0, 0.2, 0.2, 1.0]
                                tag = f"D{i} [GATEWAY] {batt:.0f}%"
                            else:
                                color = [0.2, 0.5, 1.0, 1.0]
                                tag = f"D{i} [RELAY] {batt:.0f}%"
                        p.changeVisualShape(env.drone_ids[i], -1, rgbaColor=color, physicsClientId=env.client)
                        p.addUserDebugText(
                            tag,
                            [pos[0], pos[1], pos[2] + 1.2],
                            textColorRGB=color[:3],
                            textSize=1.0,
                            physicsClientId=env.client,
                        )

                    # 4. Survivor status markers
                    surv_pos = env.survivors.get_positions()
                    for s_idx in range(env.survivors.num_survivors):
                        is_disc = bool(env.survivors.discovered[s_idx])
                        marker = "[*]" if is_disc else "?"
                        color = [0.0, 1.0, 0.2] if is_disc else [1.0, 0.3, 0.3]
                        p.addUserDebugText(
                            marker,
                            [surv_pos[s_idx][0], surv_pos[s_idx][1], surv_pos[s_idx][2] + 0.5],
                            textColorRGB=color,
                            textSize=1.2,
                            physicsClientId=env.client,
                        )

                    # Structure labels (school, apartments, houses)
                    for box in getattr(env.terrain, "structures", []) or []:
                        if not box.label:
                            continue
                        c = box.center
                        p.addUserDebugText(
                            box.label,
                            [c[0], c[1], box.zmax + 0.8],
                            textColorRGB=[1.0, 0.85, 0.5],
                            textSize=0.9,
                            physicsClientId=env.client,
                        )

                    stats = env.survivors.get_discovery_stats()
                    if args.task == "search":
                        cover = float(getattr(env, "coverage", np.zeros(1)).mean())
                        hud_line = (
                            f"SEARCH | Step: {step_num} | Found: {stats['discovered_survivors']}/{stats['total_survivors']} "
                            f"({stats['discovery_rate']*100:.0f}%) | Cover: {cover*100:.0f}%"
                        )
                    else:
                        sat_text = (
                            f"LEO SAT: Active (x={env.satellite.current_x:.1f}m)"
                            if env.satellite.visible
                            else f"LEO SAT: Blackout ({env.satellite.blackout_timer:.0f}s)"
                        )
                        hud_line = (
                            f"Step: {step_num} | Found: {stats['discovered_survivors']}/{stats['total_survivors']} "
                            f"({stats['discovery_rate']*100:.0f}%) | Conn: {env.connected_survivors} | {sat_text}"
                        )
                    p.addUserDebugText(
                        hud_line,
                        [-args.env_size / 2 + 5, -args.env_size / 2 + 5, 25.0],
                        textColorRGB=[1.0, 1.0, 1.0],
                        textSize=1.2,
                        physicsClientId=env.client,
                    )

                # Check episode completion
                if all(terms.values()) or all(truncs.values()):
                    print(
                        f"[EPISODE END] Discovered: {stats['discovered_survivors']}/{stats['total_survivors']}"
                    )
                    obs_dict, info_dict = env.reset()
                    step_num = 0

            # Cap frame rate
            elapsed = time.time() - loop_start
            sleep_time = max(0.0, dt_target - elapsed)
            time.sleep(sleep_time)

    except (KeyboardInterrupt, p.error):
        pass
    finally:
        env.close()
        print("[VISUALIZER] Closed successfully.")


if __name__ == "__main__":
    main()
