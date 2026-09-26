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
  - Spectator (noclip) free-fly camera through the damaged town
  - Optional model inference or manual control mode

Usage:
  python visualize_env.py                                # Exploration demo
  python visualize_env.py --model_path models/test_model.pt # Playback trained model
"""

import math
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
from models.actor_critic import MAPPOModel

# Spectator fly defaults (meters / degrees per frame at target FPS)
_SPEC_MOVE_SPEED = 2.5
_SPEC_FAST_MULT = 3.0
_SPEC_TURN_SPEED = 2.0
_SPEC_FLY_DIST = 1.0  # short boom so look-around feels FPS-like
_PITCH_MIN, _PITCH_MAX = -89.0, 89.0


def parse_args():
    parser = argparse.ArgumentParser(description="3D Visualizer for Drone Swarm Mesh")
    parser.add_argument("--model_path", type=str, default="", help="Optional model checkpoint path")
    parser.add_argument("--num_drones", type=int, default=5, help="Number of drones")
    parser.add_argument("--num_clusters", type=int, default=4, help="Number of survivor clusters")
    parser.add_argument("--env_size", type=float, default=250.0, help="Terrain size in meters")
    parser.add_argument("--drone_max_speed", type=float, default=8.0, help="Max drone cruise speed m/s")
    parser.add_argument("--drone_max_altitude", type=float, default=40.0, help="Altitude cap in meters")
    parser.add_argument("--fps", type=float, default=30.0, help="Display FPS target")
    parser.add_argument("--seed", type=int, default=42, help="World seed")
    return parser.parse_args()


def camera_basis_from_yaw_pitch(yaw_deg, pitch_deg):
    """Return (forward, right, world_up) unit vectors matching PyBullet debug-camera spherical coords.

    PyBullet places the eye at:
      eye = target + dist * (sin(yaw)*cos(pitch), cos(yaw)*cos(pitch), sin(pitch))
    so forward (eye -> target) is the negation of that radial offset.
    """
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)

    forward = np.array([-sy * cp, -cy * cp, -sp], dtype=np.float64)
    # Horizontal right (viewer RHS): cross(world_up, forward); fall back near gimbal lock
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(world_up, forward)
    rn = np.linalg.norm(right)
    if rn < 1e-8:
        right = np.array([cy, -sy, 0.0], dtype=np.float64)
        rn = np.linalg.norm(right)
    right /= rn
    return forward, right, world_up


def eye_from_camera(yaw_deg, pitch_deg, dist, target):
    """Eye position for PyBullet resetDebugVisualizerCamera parameters."""
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    target = np.asarray(target, dtype=np.float64)
    offset = dist * np.array(
        [
            math.sin(yaw) * math.cos(pitch),
            math.cos(yaw) * math.cos(pitch),
            math.sin(pitch),
        ],
        dtype=np.float64,
    )
    return target + offset


def apply_spectator_look(yaw, pitch, dist, target, dyaw, dpitch):
    """Rotate view around the current eye (FPS look), keeping eye fixed."""
    eye = eye_from_camera(yaw, pitch, dist, target)
    yaw = (yaw + dyaw) % 360.0
    pitch = float(np.clip(pitch + dpitch, _PITCH_MIN, _PITCH_MAX))
    forward, _, _ = camera_basis_from_yaw_pitch(yaw, pitch)
    new_target = eye + forward * dist
    return yaw, pitch, dist, new_target


def apply_spectator_move(yaw, pitch, dist, target, move_fwd, move_right, move_up):
    """Translate target (and thus eye) along look / strafe / world-up."""
    forward, right, world_up = camera_basis_from_yaw_pitch(yaw, pitch)
    target = np.asarray(target, dtype=np.float64).copy()
    target += forward * move_fwd + right * move_right + world_up * move_up
    return target


def read_debug_camera(client):
    """Return yaw, pitch, dist, target from the active PyBullet debug camera."""
    info = p.getDebugVisualizerCamera(physicsClientId=client)
    yaw = float(info[8])
    pitch = float(info[9])
    dist = float(info[10])
    target = np.array(info[11], dtype=np.float64)
    return yaw, pitch, dist, target


def enter_spectator_fly(client, yaw, pitch, dist, target):
    """Collapse the orbit boom to a short fly distance while preserving the eye pose."""
    eye = eye_from_camera(yaw, pitch, dist, target)
    forward, _, _ = camera_basis_from_yaw_pitch(yaw, pitch)
    new_dist = _SPEC_FLY_DIST
    new_target = eye + forward * new_dist
    p.resetDebugVisualizerCamera(
        cameraDistance=new_dist,
        cameraYaw=yaw,
        cameraPitch=pitch,
        cameraTargetPosition=new_target.tolist(),
        physicsClientId=client,
    )
    return yaw, pitch, new_dist, new_target


def _key_down(keys, code):
    return bool(keys.get(code, 0) & p.KEY_IS_DOWN)


def _key_triggered(keys, code):
    return bool(keys.get(code, 0) & p.KEY_WAS_TRIGGERED)


def update_spectator_camera(client, keys, yaw, pitch, dist, target):
    """Apply WASD/EF/arrows/Shift to spectator state and push to PyBullet."""
    speed = _SPEC_MOVE_SPEED
    if _key_down(keys, p.B3G_SHIFT):
        speed *= _SPEC_FAST_MULT

    move_fwd = 0.0
    move_right = 0.0
    move_up = 0.0
    if _key_down(keys, ord("w")) or _key_down(keys, ord("W")):
        move_fwd += speed
    if _key_down(keys, ord("s")) or _key_down(keys, ord("S")):
        move_fwd -= speed
    if _key_down(keys, ord("d")) or _key_down(keys, ord("D")):
        move_right += speed
    if _key_down(keys, ord("a")) or _key_down(keys, ord("A")):
        move_right -= speed
    if _key_down(keys, ord("e")) or _key_down(keys, ord("E")):
        move_up += speed
    if _key_down(keys, ord("f")) or _key_down(keys, ord("F")):
        move_up -= speed

    dyaw = 0.0
    dpitch = 0.0
    if _key_down(keys, p.B3G_LEFT_ARROW):
        dyaw -= _SPEC_TURN_SPEED
    if _key_down(keys, p.B3G_RIGHT_ARROW):
        dyaw += _SPEC_TURN_SPEED
    if _key_down(keys, p.B3G_UP_ARROW):
        dpitch += _SPEC_TURN_SPEED
    if _key_down(keys, p.B3G_DOWN_ARROW):
        dpitch -= _SPEC_TURN_SPEED

    if dyaw != 0.0 or dpitch != 0.0:
        yaw, pitch, dist, target = apply_spectator_look(yaw, pitch, dist, target, dyaw, dpitch)
    if move_fwd != 0.0 or move_right != 0.0 or move_up != 0.0:
        target = apply_spectator_move(yaw, pitch, dist, target, move_fwd, move_right, move_up)

    p.resetDebugVisualizerCamera(
        cameraDistance=dist,
        cameraYaw=yaw,
        cameraPitch=pitch,
        cameraTargetPosition=np.asarray(target, dtype=np.float64).tolist(),
        physicsClientId=client,
    )
    return yaw, pitch, dist, target


def main():
    args = parse_args()
    print("=" * 65)
    print(" [3D VISUALIZER] Autonomous LEO-Integrated Drone Mesh Network")
    print(" Controls: Space = Pause | R = Reset Episode | Q / ESC = Exit")
    print(" Spectator (default ON, toggle C):")
    print("   W/S = Forward/Back | A/D = Strafe | E = Up | F = Down")
    print("   Arrows = Look (yaw/pitch) | Left Shift = Faster | Mouse orbit still works when OFF")
    print("=" * 65)

    env = DisasterMeshEnv(
        num_drones=args.num_drones,
        num_clusters=args.num_clusters,
        env_size=args.env_size,
        max_steps=5000,
        render_mode="human",
        drone_max_speed=args.drone_max_speed,
        drone_max_altitude=args.drone_max_altitude,
        seed=args.seed,
    )

    model = None
    if args.model_path and os.path.exists(args.model_path):
        print(f"[MODEL] Loading policy checkpoint from {args.model_path}...")
        model = MAPPOModel(num_drones=args.num_drones, obs_dim=25, act_dim=5)
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

    # Default into spectator noclip; disable env camera snap on reset so keys don't fight it.
    spectator = True
    env.auto_camera = False
    yaw, pitch, dist, target = read_debug_camera(env.client)
    yaw, pitch, dist, target = enter_spectator_fly(env.client, yaw, pitch, dist, target)
    print("[SPECTATOR] ON — click the 3D viewport, then fly with WASD / E F / arrows.")

    try:
        while True:
            loop_start = time.time()

            # Handle keyboard input
            keys = p.getKeyboardEvents(physicsClientId=env.client)
            if ord("q") in keys or ord("Q") in keys or 27 in keys:  # 27 = ESC
                print("[EXIT] User requested exit.")
                break
            if _key_triggered(keys, ord(" ")):
                paused = not paused
                print(f"[{'PAUSED' if paused else 'RESUMED'}]")
            if _key_triggered(keys, ord("r")) or _key_triggered(keys, ord("R")):
                print("[RESET] Resetting environment...")
                obs_dict, info_dict = env.reset()
                step_num = 0
                # Re-frame overview once, then restore spectator fly boom if still spectating.
                env.auto_camera = True
                # Spawn already ran with previous auto_camera=False; force overview now.
                cam_dist = max(180.0, args.env_size * 0.85)
                p.resetDebugVisualizerCamera(
                    cameraDistance=cam_dist,
                    cameraYaw=45,
                    cameraPitch=-40,
                    cameraTargetPosition=[0, 0, 8.0],
                    physicsClientId=env.client,
                )
                yaw, pitch, dist, target = read_debug_camera(env.client)
                if spectator:
                    env.auto_camera = False
                    yaw, pitch, dist, target = enter_spectator_fly(
                        env.client, yaw, pitch, dist, target
                    )
                else:
                    env.auto_camera = True
                continue
            if _key_triggered(keys, ord("c")) or _key_triggered(keys, ord("C")):
                spectator = not spectator
                env.auto_camera = not spectator
                if spectator:
                    yaw, pitch, dist, target = read_debug_camera(env.client)
                    yaw, pitch, dist, target = enter_spectator_fly(
                        env.client, yaw, pitch, dist, target
                    )
                    print("[SPECTATOR] ON")
                else:
                    print("[SPECTATOR] OFF — mouse orbit available")

            if spectator:
                yaw, pitch, dist, target = update_spectator_camera(
                    env.client, keys, yaw, pitch, dist, target
                )

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
                    # Autonomous cooperative patrol heuristic:
                    # Drones circle outwards to survey terrain and relay back
                    actions = {}
                    t = step_num * 0.05
                    for i in range(args.num_drones):
                        if not env.drone_alive[i]:
                            continue
                        agent = agent_names[i]
                        angle = t + (2 * np.pi * i / args.num_drones)
                        radius_speed = 0.5
                        vx = np.cos(angle) * radius_speed
                        vy = np.sin(angle) * radius_speed
                        vz = 0.05 * np.sin(t * 2 + i)
                        role_toggle = 1.0 if i == 0 else -1.0  # Drone 0 acts as Gateway
                        actions[agent] = np.array([vx, vy, vz, role_toggle, 0.0], dtype=np.float32)

                obs_dict, rews, terms, truncs, infos = env.step(actions)

                # Update debug visuals at a smooth 10 Hz rate instead of 60 Hz to avoid strobe flickering
                if step_num % 3 == 0:
                    p.removeAllUserDebugItems(physicsClientId=env.client)

                    alive_indices = [i for i in range(args.num_drones) if env.drone_alive[i]]

                    # 1. Mesh links between drones (Cyan)
                    for i in alive_indices:
                        for j in alive_indices:
                            if i < j:
                                pos_i = env.drone_positions[i]
                                pos_j = env.drone_positions[j]
                                dist_link = np.linalg.norm(pos_i - pos_j)
                                if dist_link < 35.0 and env.terrain.check_los(pos_i, pos_j):
                                    p.addUserDebugLine(
                                        pos_i.tolist(),
                                        pos_j.tolist(),
                                        lineColorRGB=[0.0, 0.9, 1.0],
                                        lineWidth=2.0,
                                        physicsClientId=env.client,
                                    )

                    # 2. Gateway Uplink Beam (Yellow)
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

                    # 3. Drone tags & role colors
                    for i in range(args.num_drones):
                        pos = env.drone_positions[i]
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

                    # 5. Stable HUD Overlay
                    stats = env.survivors.get_discovery_stats()
                    sat_text = (
                        f"LEO SAT: Active (x={env.satellite.current_x:.1f}m)"
                        if env.satellite.visible
                        else f"LEO SAT: Blackout ({env.satellite.blackout_timer:.0f}s)"
                    )
                    wind = env.weather.get_global_wind()
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
                        f"[EPISODE END] Discovered: {stats['discovered_survivors']}/{stats['total_survivors']} | "
                        f"Connected: {env.connected_survivors}"
                    )
                    # Keep spectator pose across auto-reset (auto_camera already False while spectating).
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
