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
from visualize_rubble import (
    _SPEC_MOVE_SPEED,
    enter_spectator_fly,
    update_spectator_camera,
)
from visualize_town import camera_basis_from_yaw_pitch

# Search fly is 1/3 of the shared rubble spectator base (18.0 m/poll).
_SEARCH_FLY_MOVE_SPEED = _SPEC_MOVE_SPEED / 3.0

try:
    from models.actor_critic import MAPPOModel
except ModuleNotFoundError:
    MAPPOModel = None

try:
    import tkinter as tk
except ImportError:
    tk = None


def parse_args():
    parser = argparse.ArgumentParser(description="3D Visualizer for Drone Swarm Mesh")
    parser.add_argument("--model_path", type=str, default="", help="Optional model checkpoint path")
    parser.add_argument("--num_drones", type=int, default=3, help="Number of drones")
    parser.add_argument("--num_clusters", type=int, default=4, help="Number of survivor clusters")
    parser.add_argument("--env_size", type=float, default=None, help="Terrain size in meters")
    parser.add_argument("--fps", type=float, default=30.0, help="Display FPS target")
    parser.add_argument("--seed", type=int, default=42, help="World seed")
    parser.add_argument("--task", type=str, default="search", choices=["search", "mesh"], help="search = find people; mesh = LEO demo")
    parser.add_argument(
        "--policy",
        type=str,
        default="model",
        choices=["sweep", "model"],
        help="search: sweep = scripted lawnmower. model = learned MAPPO (default)",
    )
    parser.add_argument(
        "--legacy-xy",
        action="store_true",
        help="Viewer only: 5 m XY policy (obs 136, 9 actions, critic 70). Chunks block.",
    )
    return parser.parse_args()


def _key_down(keys, code):
    return bool(keys.get(code, 0) & p.KEY_IS_DOWN)


def _key_hit(keys, code):
    return bool(keys.get(code, 0) & p.KEY_WAS_TRIGGERED)


def apply_orbit_camera(client, yaw, pitch, dist, target):
    p.resetDebugVisualizerCamera(
        cameraDistance=float(dist),
        cameraYaw=float(yaw),
        cameraPitch=float(pitch),
        cameraTargetPosition=np.asarray(target, dtype=float).tolist(),
        physicsClientId=client,
    )


def _disable_pybullet_hotkeys(client):
    if client is None:
        return
    p.configureDebugVisualizer(
        p.COV_ENABLE_KEYBOARD_SHORTCUTS, 0, physicsClientId=client
    )
    p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 0, physicsClientId=client)


class SearchCamera:
    """Keyboard camera: arrows pan, 1-9 follow a drone, 0 overview, F fly."""

    def __init__(self, client, num_drones, env_size):
        self.client = client
        self.num_drones = int(num_drones)
        self.home_yaw = 0.0
        self.home_pitch = -89.0
        self.home_dist = max(80.0, float(env_size) * 1.05)
        self.home_target = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self._applied_yaw = None
        self._applied_pitch = None
        self._applied_dist = None
        self._applied_target = None
        self.fly = False
        self.fly_eye = None
        self.reset_overview()

    def reset_overview(self):
        self.fly = False
        self.fly_eye = None
        self.yaw = self.home_yaw
        self.pitch = self.home_pitch
        self.dist = self.home_dist
        self.target = self.home_target.copy()
        self.follow = None
        self._apply_camera_if_changed()

    def enter_fly(self):
        self.follow = None
        self.yaw, self.pitch, self.dist, self.fly_eye = enter_spectator_fly(
            self.client, self.yaw, self.pitch, self.dist, self.target
        )
        self.fly = True

    def leave_fly(self):
        self.fly = False
        self.fly_eye = None

    def _apply_camera_if_changed(self):
        last = self._applied_target
        if (
            last is not None
            and self._applied_yaw == self.yaw
            and self._applied_pitch == self.pitch
            and self._applied_dist == self.dist
            and np.array_equal(last, self.target)
        ):
            return
        apply_orbit_camera(self.client, self.yaw, self.pitch, self.dist, self.target)
        self._applied_yaw = self.yaw
        self._applied_pitch = self.pitch
        self._applied_dist = self.dist
        self._applied_target = np.asarray(self.target, dtype=np.float64).copy()

    def _pan(self, dx, dy, scale):
        if self.pitch <= -70.0:
            self.target[0] += dx * scale
            self.target[1] += dy * scale
        else:
            forward, right, _up = camera_basis_from_yaw_pitch(self.yaw, self.pitch)
            right[2] = 0.0
            rn = float(np.linalg.norm(right[:2]))
            if rn > 1e-6:
                right[:2] /= rn
            fwd = np.array([forward[0], forward[1], 0.0], dtype=np.float64)
            fn = float(np.linalg.norm(fwd))
            if fn > 1e-6:
                fwd /= fn
            self.target = self.target - right * dx * scale + fwd * dy * scale
        self.target[2] = float(np.clip(self.target[2], 0.0, 40.0))

    def handle(self, keys, drone_positions, env=None):
        if _key_hit(keys, ord("f")) or _key_hit(keys, ord("F")):
            if self.fly:
                self.reset_overview()
                print("[CAM] Fly mode OFF")
            else:
                self.enter_fly()
                print("[CAM] Fly mode ON")
            return self.follow

        if _key_hit(keys, ord("0")):
            was_fly = self.fly
            self.reset_overview()
            if was_fly:
                print("[CAM] Fly mode OFF")
            print("[CAM] Overview")
            return self.follow

        left_fly = False
        if _key_hit(keys, 9):
            if self.fly:
                self.leave_fly()
                left_fly = True
                print("[CAM] Fly mode OFF")
            if self.follow is None:
                self.follow = 0
            else:
                self.follow = (int(self.follow) + 1) % self.num_drones
            print(f"[CAM] Follow D{self.follow}")
        for i in range(min(9, self.num_drones)):
            if _key_hit(keys, ord(str(i + 1))):
                if self.fly:
                    self.leave_fly()
                    left_fly = True
                    print("[CAM] Fly mode OFF")
                    if self.dist < 8.0:
                        self.dist = 20.0
                self.follow = i
                self.dist = min(self.dist, 28.0)
                self.pitch = min(self.pitch, -24.0)
                print(f"[CAM] Follow D{i}")

        if self.fly:
            return self.follow

        if _key_down(keys, ord("[")) or _key_down(keys, ord("-")):
            self.dist = min(220.0, self.dist * 1.04)
        if _key_down(keys, ord("]")) or _key_down(keys, ord("=")) or _key_down(keys, ord("+")):
            self.dist = max(8.0, self.dist * 0.96)

        step = 1.2 + 0.04 * self.dist
        dx = dy = 0.0
        if not left_fly:
            if _key_down(keys, p.B3G_UP_ARROW):
                dy += step
            if _key_down(keys, p.B3G_DOWN_ARROW):
                dy -= step
            if _key_down(keys, p.B3G_LEFT_ARROW):
                dx -= step
            if _key_down(keys, p.B3G_RIGHT_ARROW):
                dx += step
            if dx or dy:
                self.follow = None
                self._pan(dx * 12.0, dy * 12.0, 1.0)

        if self.follow is not None and drone_positions is not None:
            idx = int(np.clip(self.follow, 0, len(drone_positions) - 1))
            pos = np.asarray(drone_positions[idx], dtype=np.float64)
            self.target = pos + np.array([0.0, 0.0, 1.2], dtype=np.float64)
            self._pull_camera_out_of_solids(env)

        self._apply_camera_if_changed()
        return self.follow

    def _pull_camera_out_of_solids(self, env):
        """Slide the follow camera toward the drone until it is outside solid walls."""
        if env is None or not hasattr(env, "_hits_solid"):
            return
        yaw = np.radians(self.yaw)
        pitch = np.radians(self.pitch)
        cp = np.cos(pitch)
        direction = np.array(
            [np.sin(yaw) * cp, -np.cos(yaw) * cp, -np.sin(pitch)],
            dtype=np.float64,
        )
        dist = float(self.dist)
        for _ in range(14):
            eye = self.target + direction * dist
            if not env._hits_solid(eye, radius=0.6):
                break
            dist *= 0.7
            if dist < 6.0:
                break
        if dist < float(self.dist) - 0.4:
            self.dist = dist


def _search_drone_tag(index, alive):
    return f"D{index}" if alive else f"D{index} DEAD"


class SearchStatusPanel:
    """Dark tkinter status window beside the PyBullet search view. Created once.

    On macOS, tk.Tk() must run before p.connect(p.GUI). PyBullet's Cocoa
    backend creates an NSApplication that does not implement macOSVersion,
    and Tk 8.6 then aborts with NSInvalidArgumentException (uncaught).
    """

    BG = "#171a1f"
    FG = "#e8eaed"
    DIM = "#9aa0a6"
    CARD = "#22262e"
    LINE = "#2c313a"
    OK = "#81c995"
    DEAD = "#f28b82"
    TITLE = "#8ab4f8"

    def __init__(self, num_drones):
        self.num_drones = int(num_drones)
        # After a prior abort, macOS can modal-block Tk() on "restore windows?"
        if sys.platform == "darwin" and "-ApplePersistenceIgnoreState" not in sys.argv:
            sys.argv[1:1] = ["-ApplePersistenceIgnoreState", "YES"]
        self.root = tk.Tk()
        self.root.withdraw()
        self.root.title("Search")
        self.root.configure(bg=self.BG)
        self.root.resizable(False, False)
        try:
            sw = int(self.root.winfo_screenwidth())
            self.root.geometry(f"320x{self._window_h()}+{max(20, sw - 360)}+70")
        except tk.TclError:
            self.root.geometry(f"320x{self._window_h()}+40+70")

        mono = ("TkFixedFont", 11)
        title_font = ("TkDefaultFont", 13, "bold")
        label_font = ("TkDefaultFont", 11)
        small_font = ("TkDefaultFont", 10)

        tk.Label(
            self.root,
            text="SEARCH",
            font=title_font,
            fg=self.TITLE,
            bg=self.BG,
            anchor="w",
        ).pack(fill="x", padx=14, pady=(12, 6))

        mission = tk.Frame(self.root, bg=self.BG)
        mission.pack(fill="x", padx=10, pady=(0, 8))
        self.mission = {}
        for key, caption in (
            ("found", "People found"),
            ("left", "People left"),
            ("total", "Total people"),
            ("cover", "Map covered"),
            ("step", "Step"),
            ("flying", "Drones still flying"),
        ):
            row = tk.Frame(mission, bg=self.BG)
            row.pack(fill="x", pady=1)
            tk.Label(
                row,
                text=caption,
                font=label_font,
                fg=self.DIM,
                bg=self.BG,
                anchor="w",
            ).pack(side="left")
            val = tk.Label(
                row,
                text="—",
                font=mono,
                fg=self.FG,
                bg=self.BG,
                anchor="e",
            )
            val.pack(side="right")
            self.mission[key] = val

        tk.Frame(self.root, bg=self.LINE, height=1).pack(fill="x", padx=12, pady=(2, 8))

        self.drones = []
        for i in range(self.num_drones):
            card = tk.Frame(self.root, bg=self.CARD)
            card.pack(fill="x", padx=10, pady=3)
            top = tk.Frame(card, bg=self.CARD)
            top.pack(fill="x", padx=8, pady=(6, 0))
            name = tk.Label(
                top,
                text=f"D{i}",
                font=("TkDefaultFont", 12, "bold"),
                fg=self.FG,
                bg=self.CARD,
                anchor="w",
            )
            name.pack(side="left")
            status = tk.Label(
                top,
                text="Flying",
                font=label_font,
                fg=self.OK,
                bg=self.CARD,
                anchor="e",
            )
            status.pack(side="right")
            pos = tk.Label(
                card,
                text="x 0.0  y 0.0  h 0.0",
                font=mono,
                fg=self.FG,
                bg=self.CARD,
                anchor="w",
            )
            pos.pack(fill="x", padx=8, pady=(2, 0))
            extra = tk.Label(
                card,
                text="0.0 m/s    found 0",
                font=small_font,
                fg=self.DIM,
                bg=self.CARD,
                anchor="w",
            )
            extra.pack(fill="x", padx=8, pady=(0, 6))
            self.drones.append(
                {"name": name, "status": status, "pos": pos, "extra": extra}
            )

        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def show(self):
        if self.root is None:
            return
        try:
            self.root.deiconify()
            self.root.lift()
            self.pump()
        except tk.TclError:
            self.root = None

    def _window_h(self):
        return 250 + 78 * max(1, self.num_drones)

    def refresh(self, env):
        if self.root is None:
            return
        stats = env.survivors.get_discovery_stats()
        info = env._infos()[env.possible_agents[0]]
        found = int(info.get("discovered_survivors", stats["discovered_survivors"]))
        total = int(info.get("total_survivors", stats["total_survivors"]))
        left = max(0, total - found)
        cover = float(info.get("coverage_frac", 0.0)) * 100.0
        step = int(info.get("step", getattr(env, "step_count", 0)))
        flying = int(info.get("alive_drones", int(np.asarray(env.drone_alive).sum())))
        self.mission["found"].config(text=str(found))
        self.mission["left"].config(text=str(left))
        self.mission["total"].config(text=str(total))
        self.mission["cover"].config(text=f"{cover:.0f}%")
        self.mission["step"].config(text=str(step))
        self.mission["flying"].config(text=str(flying))

        finds = np.asarray(getattr(env, "drone_finds", np.zeros(self.num_drones)), dtype=int)
        for i in range(self.num_drones):
            alive = bool(env.drone_alive[i])
            pos = np.asarray(env.drone_positions[i], dtype=float)
            vel = np.asarray(env.drone_velocities[i], dtype=float)
            speed = float(np.hypot(vel[0], vel[1]))
            found_n = int(finds[i]) if i < len(finds) else 0
            row = self.drones[i]
            row["name"].config(text=f"D{i}")
            row["status"].config(
                text="Flying" if alive else "Dead",
                fg=self.OK if alive else self.DEAD,
            )
            row["pos"].config(
                text=f"x {pos[0]:.1f}  y {pos[1]:.1f}  h {pos[2]:.1f}"
            )
            row["extra"].config(text=f"{speed:.1f} m/s    found {found_n}")
        self.pump()

    def pump(self):
        if self.root is None:
            return
        try:
            self.root.update_idletasks()
            self.root.update()
        except tk.TclError:
            self.root = None

    def close(self):
        if self.root is None:
            return
        try:
            self.root.destroy()
        except tk.TclError:
            pass
        self.root = None


class SearchDroneLabels:
    """D0/D1/... text above each search drone, replaced in place (no wipe)."""

    def __init__(self, client, num_drones):
        self.client = client
        self.num_drones = int(num_drones)
        self.item_ids = [-1] * self.num_drones
        self._last = [None] * self.num_drones

    def sync(self, drone_positions, drone_alive, follow=None):
        for i in range(self.num_drones):
            alive = bool(drone_alive[i])
            tag = _search_drone_tag(i, alive)
            color = [0.2, 0.5, 1.0] if alive else [1.0, 0.05, 0.05]
            size = 1.4 if follow == i else 1.0
            pos = np.asarray(drone_positions[i], dtype=np.float64)
            xyz = [float(pos[0]), float(pos[1]), float(pos[2]) + 1.2]
            key = (tag, tuple(color), size, round(xyz[0], 3), round(xyz[1], 3), round(xyz[2], 3))
            if self.item_ids[i] >= 0 and self._last[i] == key:
                continue
            kwargs = {
                "textColorRGB": color,
                "textSize": size,
                "physicsClientId": self.client,
            }
            if self.item_ids[i] >= 0:
                kwargs["replaceItemUniqueId"] = self.item_ids[i]
            item_id = int(p.addUserDebugText(tag, xyz, **kwargs))
            self.item_ids[i] = item_id
            self._last[i] = None if item_id < 0 else key


class SearchHud:
    """One status line (found / alive / cover), replaced in place."""

    def __init__(self, client):
        self.client = client
        self.item_id = -1
        self._last = None

    def sync(self, text, position):
        xyz = [float(position[0]), float(position[1]), float(position[2])]
        key = (text, round(xyz[0], 2), round(xyz[1], 2), round(xyz[2], 2))
        if self.item_id >= 0 and self._last == key:
            return
        kwargs = {
            "textColorRGB": [1.0, 1.0, 1.0],
            "textSize": 1.4,
            "physicsClientId": self.client,
        }
        if self.item_id >= 0:
            kwargs["replaceItemUniqueId"] = self.item_id
        item_id = int(p.addUserDebugText(text, xyz, **kwargs))
        self.item_id = item_id
        self._last = None if item_id < 0 else key


def main():
    args = parse_args()
    if args.env_size is None:
        args.env_size = 250.0 if args.task == "search" else 100.0
    print("=" * 65)
    print(f" [3D VISUALIZER] task={args.task}")
    if args.task == "search":
        print(" Search: 240 m rubble city, 5 m AGL, character meshes, MAPPO XY")
        print(" Camera: starts top-down | arrows pan | 1/2/3 follow a drone | 0 overview | F fly")
        print("         [ ] zoom | Space pause | R reset | Q quit")
        print(" Click the 3D view first or keys do nothing.")
    print(" Controls: Space = Pause | R = Reset Episode | Q / ESC = Exit")
    print("=" * 65)

    # Own NSApplication before SurvivorSearchEnv.reset -> p.connect(p.GUI).
    status_panel = None
    last_status = 0.0
    if args.task == "search":
        if tk is None:
            print("[WARNING] tkinter unavailable; continuing 3D view.")
        else:
            try:
                status_panel = SearchStatusPanel(args.num_drones)
                print("[STATUS] Tk root created before PyBullet GUI")
            except Exception as exc:
                print(f"[WARNING] tkinter status window failed ({exc}); continuing 3D view.")
                status_panel = None

    if args.task == "search":
        env = SurvivorSearchEnv(
            num_drones=args.num_drones,
            num_clusters=args.num_clusters,
            env_size=args.env_size,
            max_steps=max(5000, int(args.env_size * 8)),
            render_mode="human",
            seed=args.seed,
            legacy_xy=args.legacy_xy,
        )
        obs_dim, act_dim = env.obs_dim, env.act_dim
        if args.legacy_xy:
            print(
                f" Legacy XY: obs={obs_dim} act={act_dim} state={env.state_dim} "
                "(hover 5 m, chunks solid, action 8 = stay)"
            )
    else:
        env = DisasterMeshEnv(
            num_drones=args.num_drones,
            num_clusters=args.num_clusters,
            env_size=args.env_size,
            max_steps=max(5000, int(args.env_size * 8)),
            render_mode="human",
            seed=args.seed,
        )
        obs_dim, act_dim = 25, 5

    model = None
    if args.task == "search" and args.policy == "model" and not args.model_path:
        args.model_path = "models/mappo_search.pt"
    load_nn = bool(args.model_path) and os.path.exists(args.model_path)
    if args.task == "search" and args.policy == "sweep":
        load_nn = False
    if load_nn:
        if MAPPOModel is None:
            print("[WARNING] models/actor_critic.py is missing; using heuristic policy.")
        else:
            print(f"[MODEL] Loading policy checkpoint from {args.model_path}...")
            discrete = bool(getattr(env, "discrete_actions", False))
            state_dim = int(getattr(env, "state_dim", args.num_drones * obs_dim))
            model = MAPPOModel(
                num_drones=args.num_drones,
                obs_dim=obs_dim,
                act_dim=act_dim,
                discrete=discrete,
                state_dim=state_dim,
            )
            try:
                model.load(args.model_path, map_location="cpu")
                model.eval()
                print("[MODEL] Model loaded successfully!")
            except Exception as e:
                print(f"[WARNING] Failed to load model ({e}), using heuristic policy.")
                model = None

    obs_dict, info_dict = env.reset(seed=args.seed)
    agent_names = env.possible_agents
    last_infos = info_dict
    _disable_pybullet_hotkeys(env.client)

    dt_target = 1.0 / args.fps
    paused = False
    step_num = 0
    cam = SearchCamera(env.client, args.num_drones, args.env_size)
    prev_alive = np.ones(args.num_drones, dtype=bool)
    search_labels = (
        SearchDroneLabels(env.client, args.num_drones) if args.task == "search" else None
    )
    search_hud = SearchHud(env.client) if args.task == "search" else None
    if status_panel is not None:
        status_panel.show()
        status_panel.refresh(env)
        last_status = time.time()
        print("[STATUS] Search window ready")

    try:
        while True:
            loop_start = time.time()

            # Handle keyboard input
            keys = p.getKeyboardEvents(physicsClientId=env.client)
            if ord("q") in keys or ord("Q") in keys or 27 in keys:  # 27 = ESC
                print("[EXIT] User requested exit.")
                break
            if ord("r") in keys and (keys[ord("r")] & p.KEY_WAS_TRIGGERED):
                if cam.fly:
                    print("[CAM] Fly mode OFF")
                print("[RESET] Resetting environment...")
                obs_dict, info_dict = env.reset()
                last_infos = info_dict
                step_num = 0
                prev_alive[:] = True
                _disable_pybullet_hotkeys(env.client)
                cam = SearchCamera(env.client, args.num_drones, args.env_size)
                if args.task == "search":
                    search_labels = SearchDroneLabels(env.client, args.num_drones)
                    search_hud = SearchHud(env.client)
                if status_panel is not None:
                    status_panel.refresh(env)
                    last_status = time.time()
                continue

            cam.handle(keys, env.drone_positions, env)
            if (
                not cam.fly
                and ord(" ") in keys
                and (keys[ord(" ")] & p.KEY_WAS_TRIGGERED)
            ):
                paused = not paused
                print(f"[{'PAUSED' if paused else 'RESUMED'}]")
            if cam.fly:
                cam.yaw, cam.pitch, cam.dist, cam.fly_eye = update_spectator_camera(
                    env.client,
                    keys,
                    cam.yaw,
                    cam.pitch,
                    cam.dist,
                    cam.fly_eye,
                    move_speed=_SEARCH_FLY_MOVE_SPEED,
                )

            if not paused:
                # One physics/policy step per displayed frame. HUD below is ~2 Hz.
                playback = 1
                for _ in range(playback):
                    step_num += 1

                    use_sweep = args.task == "search" and (
                        args.policy == "sweep" or model is None
                    )
                    if use_sweep:
                        actions = env.coverage_actions()
                    elif model is not None:
                        obs_array = np.array([obs_dict[a] for a in agent_names], dtype=np.float32)
                        if getattr(env, "discrete_actions", False):
                            state = np.asarray(env.global_state(), dtype=np.float32)
                        else:
                            state = obs_array.flatten()
                        with torch.no_grad():
                            obs_t = torch.tensor(obs_array, dtype=torch.float32)
                            state_t = torch.tensor(state, dtype=torch.float32)
                            action_mask = None
                            if getattr(env, "discrete_actions", False) and hasattr(env, "action_mask"):
                                action_mask = torch.tensor(env.action_mask(), dtype=torch.float32)
                            actions_t, _, _, _ = model.get_action_and_value(
                                obs_t, state_t, deterministic=True, action_mask=action_mask
                            )
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
                                hover_z = getattr(env, "hover_altitude", 5.0)
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
                    last_infos = infos
                    if args.task == "search":
                        for i in range(args.num_drones):
                            if prev_alive[i] and not env.drone_alive[i]:
                                print(f"[CRASH] D{i} hit rubble and is dead. Reward {rews[agent_names[i]]:.1f}")
                                wreck = [0.45, 0.06, 0.06, 1.0]
                                n_links = p.getNumJoints(env.drone_ids[i], physicsClientId=env.client)
                                for link in range(-1, n_links):
                                    p.changeVisualShape(
                                        env.drone_ids[i],
                                        link,
                                        rgbaColor=wreck,
                                        physicsClientId=env.client,
                                    )
                            prev_alive[i] = bool(env.drone_alive[i])
                    if all(terms.values()) or all(truncs.values()):
                        stats = env.survivors.get_discovery_stats()
                        print(
                            f"[EPISODE END] Discovered: {stats['discovered_survivors']}/{stats['total_survivors']}"
                        )
                        obs_dict, info_dict = env.reset()
                        step_num = 0
                        prev_alive[:] = True
                        if args.task == "search":
                            search_labels = SearchDroneLabels(env.client, args.num_drones)
                        if status_panel is not None:
                            status_panel.refresh(env)
                            last_status = time.time()
                        break

                # Mesh still refreshes its overlay. Search labels are replaced in place above.
                if args.task != "search" and step_num % 15 == 0:
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
                            if not env.drone_alive[i]:
                                color = [1.0, 0.05, 0.05, 1.0]
                                tag = f"D{i} DEAD"
                            else:
                                color = [0.2, 0.5, 1.0, 1.0]
                                tag = f"D{i}"
                            n_links = p.getNumJoints(env.drone_ids[i], physicsClientId=env.client)
                            for link in range(-1, n_links):
                                p.changeVisualShape(
                                    env.drone_ids[i],
                                    link,
                                    rgbaColor=color,
                                    physicsClientId=env.client,
                                )
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
                            textSize=1.4 if cam.follow == i else 1.0,
                            physicsClientId=env.client,
                        )

                    if args.task != "search":
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
                        cover = float(infos[env.possible_agents[0]].get("coverage_frac", 0.0))
                        alive_n = int(env.drone_alive.sum())
                        cam_txt = f"FOLLOW D{cam.follow}" if cam.follow is not None else "FREE CAM"
                        hud_line = (
                            f"SEARCH | {cam_txt} | Alive: {alive_n}/{args.num_drones} | "
                            f"Found: {stats['discovered_survivors']}/{stats['total_survivors']} "
                            f"| Cover: {cover*100:.0f}%"
                        )
                        hud_pos = cam.target + np.array([0.0, 0.0, 10.0])
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
                        (hud_pos.tolist() if args.task == "search" else [-args.env_size / 2 + 5, -args.env_size / 2 + 5, 25.0]),
                        textColorRGB=[1.0, 1.0, 1.0],
                        textSize=1.2,
                        physicsClientId=env.client,
                    )

            if search_hud is not None:
                stats = env.survivors.get_discovery_stats()
                cover = 0.0
                if last_infos:
                    cover = float(last_infos[env.possible_agents[0]].get("coverage_frac", 0.0))
                alive_n = int(np.sum(env.drone_alive))
                cam_txt = f"FOLLOW D{cam.follow}" if cam.follow is not None else "OVERVIEW"
                hud_line = (
                    f"SEARCH | {cam_txt} | Alive: {alive_n}/{args.num_drones} | "
                    f"Found: {stats['discovered_survivors']}/{stats['total_survivors']} "
                    f"| Cover: {cover * 100:.0f}%"
                )
                search_hud.sync(hud_line, cam.target + np.array([0.0, 0.0, 10.0]))

            if search_labels is not None:
                search_labels.sync(
                    env.drone_positions,
                    env.drone_alive,
                    follow=cam.follow,
                )
            if status_panel is not None and (time.time() - last_status) >= 0.25:
                status_panel.refresh(env)
                last_status = time.time()

            # Cap frame rate
            elapsed = time.time() - loop_start
            sleep_time = max(0.0, dt_target - elapsed)
            time.sleep(sleep_time)

    except (KeyboardInterrupt, p.error):
        pass
    finally:
        if status_panel is not None:
            status_panel.close()
        env.close()
        print("[VISUALIZER] Closed successfully.")


if __name__ == "__main__":
    main()
