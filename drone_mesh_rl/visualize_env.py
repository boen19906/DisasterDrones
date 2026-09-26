"""
visualize_env.py — Lightweight 3D spectator for the damaged town.

Loads the 250 m town once in PyBullet and runs a camera-only loop
(no drones, RF mesh, survivors, weather, or multi-agent stepping).

Usage:
  cd drone_mesh_rl && python3 visualize_env.py
"""

import math
import sys
import time
import argparse
import numpy as np
import pybullet as p

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from envs.terrain import Terrain
from envs.town import (
    TownLayout,
    connect_pybullet,
    spawn_town_in_pybullet,
    frame_town_camera,
)

# Spectator fly defaults (meters / degrees per key-poll at target FPS)
_SPEC_MOVE_SPEED = 3.0  # several meters per poll — usable on a 250 m map
_SPEC_FAST_MULT = 3.0  # Left Ctrl sprint only (Left Shift is descend)
_SPEC_TURN_SPEED = 2.5
_SPEC_FLY_DIST = 1.0  # short boom so look-around feels FPS-like
_PITCH_MIN, _PITCH_MAX = -89.0, 89.0


def parse_args():
    parser = argparse.ArgumentParser(description="Damaged-town spectator viewer")
    parser.add_argument("--env_size", type=float, default=250.0, help="Town size in meters")
    parser.add_argument("--fps", type=float, default=30.0, help="Display FPS target")
    parser.add_argument("--seed", type=int, default=42, help="World seed")
    return parser.parse_args()


def camera_basis_from_yaw_pitch(yaw_deg, pitch_deg):
    """Return (forward, right, world_up) unit vectors matching PyBullet debug-camera spherical coords.

    PyBullet places the eye at:
      eye = target + dist * (sin(yaw)*cos(pitch), cos(yaw)*cos(pitch), sin(pitch))
    so forward (eye -> target) is the negation of that radial offset.

    Yaw: decreasing yaw turns the look toward the viewer's right.
    Pitch: decreasing pitch tilts the look upward (forward.z > 0).
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


def apply_spectator_camera(client, eye, yaw, pitch, dist):
    """Push explicit eye + yaw/pitch to PyBullet every frame."""
    eye = np.asarray(eye, dtype=np.float64)
    forward, _, _ = camera_basis_from_yaw_pitch(yaw, pitch)
    target = eye + forward * dist
    p.resetDebugVisualizerCamera(
        cameraDistance=dist,
        cameraYaw=yaw,
        cameraPitch=pitch,
        cameraTargetPosition=target.tolist(),
        physicsClientId=client,
    )
    return target


def read_debug_camera(client):
    """Return yaw, pitch, dist, target, eye from the active PyBullet debug camera."""
    info = p.getDebugVisualizerCamera(physicsClientId=client)
    yaw = float(info[8])
    pitch = float(info[9])
    dist = float(info[10])
    target = np.array(info[11], dtype=np.float64)
    eye = eye_from_camera(yaw, pitch, dist, target)
    return yaw, pitch, dist, target, eye


def enter_spectator_fly(client, yaw, pitch, dist, target):
    """Collapse the orbit boom to a short fly distance while preserving the eye pose."""
    eye = eye_from_camera(yaw, pitch, dist, target)
    new_dist = _SPEC_FLY_DIST
    apply_spectator_camera(client, eye, yaw, pitch, new_dist)
    return yaw, pitch, new_dist, eye


def _key_down(keys, code):
    return bool(keys.get(code, 0) & p.KEY_IS_DOWN)


def _key_triggered(keys, code):
    return bool(keys.get(code, 0) & p.KEY_WAS_TRIGGERED)


def update_spectator_camera(client, keys, yaw, pitch, dist, eye):
    """Minecraft-style spectator: WASD look-relative, arrows look, Space/Shift vertical."""
    speed = _SPEC_MOVE_SPEED
    # Sprint: Left Ctrl only — Left Shift is descend, not faster
    if _key_down(keys, p.B3G_CONTROL):
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
    if _key_down(keys, p.B3G_SPACE) or _key_down(keys, ord(" ")):
        move_up += speed
    if _key_down(keys, p.B3G_SHIFT):
        move_up -= speed

    # Signs match camera_basis: -yaw = turn right, -pitch = look up
    dyaw = 0.0
    dpitch = 0.0
    if _key_down(keys, p.B3G_RIGHT_ARROW):
        dyaw -= _SPEC_TURN_SPEED
    if _key_down(keys, p.B3G_LEFT_ARROW):
        dyaw += _SPEC_TURN_SPEED
    if _key_down(keys, p.B3G_UP_ARROW):
        dpitch -= _SPEC_TURN_SPEED
    if _key_down(keys, p.B3G_DOWN_ARROW):
        dpitch += _SPEC_TURN_SPEED

    if dyaw != 0.0 or dpitch != 0.0:
        yaw = (yaw + dyaw) % 360.0
        pitch = float(np.clip(pitch + dpitch, _PITCH_MIN, _PITCH_MAX))

    if move_fwd != 0.0 or move_right != 0.0 or move_up != 0.0:
        forward, right, world_up = camera_basis_from_yaw_pitch(yaw, pitch)
        eye = np.asarray(eye, dtype=np.float64).copy()
        eye += forward * move_fwd + right * move_right + world_up * move_up

    apply_spectator_camera(client, eye, yaw, pitch, dist)
    return yaw, pitch, dist, eye


def build_world(client, env_size, seed):
    """Generate town + terrain and spawn them once in the open PyBullet client."""
    town = TownLayout(size=env_size, seed=seed, altitude_cap=40.0)
    terrain = Terrain(
        size_x=env_size,
        size_y=env_size,
        resolution=2.0,
        seed=seed,
        obstacle_boxes=town.boxes,
    )
    spawn_town_in_pybullet(client, town, terrain, env_size)
    return town, terrain


def clear_world(client):
    """Remove all bodies so the town can be rebuilt on reset."""
    n = p.getNumBodies(physicsClientId=client)
    body_ids = [p.getBodyUniqueId(i, physicsClientId=client) for i in range(n)]
    for body_id in body_ids:
        try:
            p.removeBody(body_id, physicsClientId=client)
        except Exception:
            pass


def main():
    args = parse_args()
    print("=" * 65)
    print(" [3D TOWN VIEWER] Damaged town spectator")
    print(" Click the 3D viewport first so keys reach PyBullet.")
    print(" Keys:")
    print("   W/A/S/D  = fly forward/left/back/right (look-relative)")
    print("   Arrows   = look (right/left/up/down)")
    print("   Space    = up | Left Shift = down | Left Ctrl = sprint")
    print("   C        = toggle spectator | P = pause | R = reset overview")
    print("   Q / ESC  = quit")
    print("=" * 65)

    client = connect_pybullet(gui=True, shadows=False)
    # Belt-and-suspenders: force our GUI flags again right after connect.
    p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 0, physicsClientId=client)
    p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 0, physicsClientId=client)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0, physicsClientId=client)

    town, terrain = build_world(client, args.env_size, args.seed)
    n_buildings = len(town.boxes)
    print(f"[TOWN] Loaded {n_buildings} building/rubble/overpass boxes on {args.env_size:.0f}m map.")

    frame_town_camera(client, args.env_size)
    spectator = True
    yaw, pitch, dist, target, eye = read_debug_camera(client)
    yaw, pitch, dist, eye = enter_spectator_fly(client, yaw, pitch, dist, target)
    print("[SPECTATOR] ON — click the 3D viewport, then fly with WASD / Space+Shift / arrows.")

    dt_target = 1.0 / args.fps
    paused = False
    seed = args.seed

    try:
        while True:
            loop_start = time.time()
            keys = p.getKeyboardEvents(physicsClientId=client)

            if ord("q") in keys or ord("Q") in keys or 27 in keys:
                print("[EXIT] User requested exit.")
                break
            if _key_triggered(keys, ord("p")) or _key_triggered(keys, ord("P")):
                paused = not paused
                print(f"[{'PAUSED' if paused else 'RESUMED'}]")
            if _key_triggered(keys, ord("r")) or _key_triggered(keys, ord("R")):
                print("[RESET] Rebuilding town...")
                seed = int(seed) + 1
                clear_world(client)
                town, terrain = build_world(client, args.env_size, seed)
                print(f"[TOWN] Loaded {len(town.boxes)} boxes (seed={seed}).")
                frame_town_camera(client, args.env_size)
                yaw, pitch, dist, target, eye = read_debug_camera(client)
                # Re-frame overview, then return to fly mode
                spectator = True
                yaw, pitch, dist, eye = enter_spectator_fly(
                    client, yaw, pitch, dist, target
                )
                continue
            if _key_triggered(keys, ord("c")) or _key_triggered(keys, ord("C")):
                spectator = not spectator
                if spectator:
                    yaw, pitch, dist, target, eye = read_debug_camera(client)
                    yaw, pitch, dist, eye = enter_spectator_fly(
                        client, yaw, pitch, dist, target
                    )
                    print("[SPECTATOR] ON")
                else:
                    print("[SPECTATOR] OFF — mouse orbit available")

            if not paused and spectator:
                yaw, pitch, dist, eye = update_spectator_camera(
                    client, keys, yaw, pitch, dist, eye
                )

            # Idle: no physics stepping, no drone/network updates
            elapsed = time.time() - loop_start
            sleep_time = max(0.0, dt_target - elapsed)
            time.sleep(sleep_time)

    except (KeyboardInterrupt, p.error):
        pass
    finally:
        try:
            p.disconnect(physicsClientId=client)
        except Exception:
            pass
        print("[VISUALIZER] Closed successfully.")


if __name__ == "__main__":
    main()
