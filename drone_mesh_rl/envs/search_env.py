"""
search_env.py — V1 multi-agent search: find as many survivors as possible.

Fog-of-war coverage. A survivor is found when a drone flies over them
(XY distance <= 5 m). No mesh, satellite, battery, or buildings.

Rewards (shared unless noted):
  +40.0  each newly found survivor (primary)
  +150.0 all survivors found
  +~5.7  one full new 5 m sensor disk (~1/7 of a find) so unmapped ground stays worth painting
  -0.05  per step × remaining unexplored interior
  +25.0  interior coverage ≥ 99%
  -3.0   stalling (slow) while the interior is not fully covered
  -5.0   camping a corner
  Drones use next-best-view search: each flies to unmapped ground it is closest
  to, preferring the patch that would paint the most new cells per meter.
  -0.01  per step
  -0.002 per already-covered cell still under a sensor (revisit / overlap)
  -0.5   hugging the map rim
  -8.0   ramped penalty if another drone is closer than 12 m
  -40.0  crash if another drone is closer than 3.5 m
"""

import functools
import importlib.util
import os
import time

import numpy as np
try:
    import pybullet as p
    import pybullet_data
    _HAS_PYBULLET = True
except ImportError:
    p = None
    pybullet_data = None
    _HAS_PYBULLET = False
from gymnasium import spaces
from pettingzoo import ParallelEnv

from .survivors import SurvivorCluster
from .terrain import Terrain


COVER_CROP = 7  # local visited-map window (odd)
# Hard stay-on-map rules. Do not weaken these — drones were hugging/leaving the rim.
MAP_EDGE_MARGIN = 8.0
DRONE_VISUAL_SCALE = 2.5
SEPARATION_RANGE = 12.0  # start paying to stay apart (meters, XY)
CRASH_RANGE = 3.5
BOUNCE_INSET = 3.0  # meters to pop off a wall so they do not stick
BOUNCE_CMD = 0.95  # forced inward command while on a wall
WALL_COOLDOWN_STEPS = 20
TURN_MARGIN = 3.0  # reverse only when truly about to leave the pad
GOAL_HOLD_STEPS = 150
COVER_CELL_M = 5.0
HUNT_COVER_FRAC = 0.55  # after this much map is painted, finish leftover people


class SurvivorSearchEnv(ParallelEnv):
    """N drones maximize survivor discoveries in a fixed-horizon episode."""

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "survivor_search_v1",
        "is_parallelizable": True,
    }

    def __init__(
        self,
        num_drones=3,
        num_clusters=9,
        survivors_per_cluster=(3, 7),
        num_loners=12,
        env_size=None,
        env_size_x=250.0,
        env_size_y=160.0,
        max_steps=2500,
        render_mode=None,
        drone_max_speed=40.0,
        drone_max_altitude=20.0,
        hover_altitude=8.0,
        sensor_radius=5.0,
        cover_cells=None,
        pyb_freq=30,
        seed=None,
    ):
        super().__init__()
        self.num_drones = num_drones
        self.num_clusters = num_clusters
        self.survivors_per_cluster = survivors_per_cluster
        self.num_loners = num_loners
        if env_size is not None:
            self.size_x = float(env_size)
            self.size_y = float(env_size)
        else:
            self.size_x = float(env_size_x)
            self.size_y = float(env_size_y)
        self.env_size = max(self.size_x, self.size_y)
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.drone_max_speed = drone_max_speed
        self.drone_max_altitude = drone_max_altitude
        self.hover_altitude = hover_altitude
        self.sensor_radius = sensor_radius
        self.cover_nx = max(8, int(round(self.size_x / COVER_CELL_M)))
        self.cover_ny = max(8, int(round(self.size_y / COVER_CELL_M)))
        if cover_cells is not None:
            self.cover_nx = int(cover_cells)
            self.cover_ny = max(8, int(round(cover_cells * self.size_y / self.size_x)))
        self.cover_cells = self.cover_nx
        self.pyb_freq = pyb_freq
        self._seed = seed

        self.possible_agents = [f"drone_{i}" for i in range(num_drones)]
        self.agents = list(self.possible_agents)

        self.obs_dim = 6 + 3 * (num_drones - 1) + 5 + COVER_CROP * COVER_CROP
        self.act_dim = 3

        self.terrain = None
        self.survivors = None
        self.coverage = np.zeros((self.cover_ny, self.cover_nx), dtype=np.float32)
        self.drone_positions = np.zeros((num_drones, 3))
        self.drone_velocities = np.zeros((num_drones, 3))
        self.drone_alive = np.ones(num_drones, dtype=bool)
        self.step_count = 0
        self.newly_discovered = 0
        self.new_cells = 0
        self.overlap_cells = 0
        self.wall_cooldown = np.zeros(num_drones, dtype=np.int32)
        self.nav_goal = np.zeros((num_drones, 2), dtype=np.float64)
        self.has_nav_goal = np.zeros(num_drones, dtype=bool)
        self.goal_hold = np.zeros(num_drones, dtype=np.int32)
        self.goal_is_person = np.zeros(num_drones, dtype=bool)

        self.client = None
        self.drone_ids = []
        self.terrain_body = None

        self._observation_spaces = {
            agent: spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
            for agent in self.possible_agents
        }
        self._action_spaces = {
            agent: spaces.Box(low=-1.0, high=1.0, shape=(self.act_dim,), dtype=np.float32)
            for agent in self.possible_agents
        }

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self._observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self._action_spaces[agent]

    def _world_to_cell(self, x, y):
        gx = int((x + self.size_x / 2.0) / self.size_x * self.cover_nx)
        gy = int((y + self.size_y / 2.0) / self.size_y * self.cover_ny)
        gx = int(np.clip(gx, 0, self.cover_nx - 1))
        gy = int(np.clip(gy, 0, self.cover_ny - 1))
        return gx, gy

    def _lim_x(self):
        return max(5.0, self.size_x / 2.0 - MAP_EDGE_MARGIN)

    def _lim_y(self):
        return max(5.0, self.size_y / 2.0 - MAP_EDGE_MARGIN)

    def _xy_limit(self):
        """Smaller playable half-extent (kept for callers that assume a square)."""
        return min(self._lim_x(), self._lim_y())

    def _confine(self, pos, vel):
        """Keep XY inside the pad, lock hover, and bounce inward off walls."""
        pos = np.array(pos, dtype=np.float64, copy=True)
        vel = np.array(vel, dtype=np.float64, copy=True)
        lx, ly = self._lim_x(), self._lim_y()
        if pos[0] >= lx:
            pos[0] = lx - BOUNCE_INSET
            vel[0] = -self.drone_max_speed
        elif pos[0] <= -lx:
            pos[0] = -lx + BOUNCE_INSET
            vel[0] = self.drone_max_speed
        if pos[1] >= ly:
            pos[1] = ly - BOUNCE_INSET
            vel[1] = -self.drone_max_speed
        elif pos[1] <= -ly:
            pos[1] = -ly + BOUNCE_INSET
            vel[1] = self.drone_max_speed
        pos[0] = float(np.clip(pos[0], -lx, lx))
        pos[1] = float(np.clip(pos[1], -ly, ly))
        pos[2] = float(self.hover_altitude)
        vel[2] = 0.0
        return pos, vel

    def _confine_drone(self, drone_index, pos, vel):
        incoming = np.asarray(pos, dtype=np.float64)
        lx, ly = self._lim_x(), self._lim_y()
        hit = (
            incoming[0] >= lx
            or incoming[0] <= -lx
            or incoming[1] >= ly
            or incoming[1] <= -ly
        )
        pos, vel = self._confine(pos, vel)
        pos[0] = float(np.clip(pos[0], -lx, lx))
        pos[1] = float(np.clip(pos[1], -ly, ly))
        if hit:
            self.wall_cooldown[drone_index] = WALL_COOLDOWN_STEPS
            self.has_nav_goal[drone_index] = False
            if incoming[0] >= lx:
                vel[0] = -self.drone_max_speed
            elif incoming[0] <= -lx:
                vel[0] = self.drone_max_speed
            if incoming[1] >= ly:
                vel[1] = -self.drone_max_speed
            elif incoming[1] <= -ly:
                vel[1] = self.drone_max_speed
        return pos, vel

    def _seal_map_border(self):
        """Outer coverage ring is already 'searched' so policies are not paid to leave."""
        self.coverage[0, :] = 1.0
        self.coverage[-1, :] = 1.0
        self.coverage[:, 0] = 1.0
        self.coverage[:, -1] = 1.0

    def _cell_to_world(self, gx, gy):
        wx = (gx + 0.5) / self.cover_nx * self.size_x - self.size_x / 2.0
        wy = (gy + 0.5) / self.cover_ny * self.size_y - self.size_y / 2.0
        return np.array([wx, wy], dtype=np.float64)

    def _interior_coverage(self):
        if self.cover_nx <= 2 or self.cover_ny <= 2:
            return float(self.coverage.mean())
        return float(self.coverage[1:-1, 1:-1].mean())

    def _playable_cell(self, w):
        return abs(w[0]) <= self._lim_x() - 2.0 and abs(w[1]) <= self._lim_y() - 2.0

    def _undiscovered_people(self):
        if self.survivors is None:
            return []
        out = []
        for i in range(self.survivors.num_survivors):
            if self.survivors.discovered[i]:
                continue
            out.append(np.asarray(self.survivors.positions[i][:2], dtype=np.float64))
        return out

    def _person_was_missed(self, xy):
        gx, gy = self._world_to_cell(xy[0], xy[1])
        return self.coverage[gy, gx] >= 0.5

    def _hunt_survivor_xy(self, drone_index):
        """Closest remaining person this drone owns so leftover finds are not abandoned at the rim."""
        people = self._undiscovered_people()
        if not people:
            return None
        my_xy = np.asarray(self.drone_positions[drone_index][:2], dtype=np.float64)
        others = [
            np.asarray(self.drone_positions[j][:2], dtype=np.float64)
            for j in range(self.num_drones)
            if j != drone_index
        ]
        claimed = [
            self.nav_goal[j]
            for j in range(self.num_drones)
            if j != drone_index and self.has_nav_goal[j] and self.goal_is_person[j]
        ]
        best = None
        best_d = 1e18
        nearest = None
        nearest_d = 1e18
        for p in people:
            d = float(np.linalg.norm(p - my_xy))
            if d < nearest_d:
                nearest_d = d
                nearest = p
            stolen = False
            for o in others:
                if float(np.linalg.norm(p - o)) + 1.0 < d:
                    stolen = True
                    break
            if stolen:
                continue
            taken = False
            for c in claimed:
                if float(np.linalg.norm(p - c)) < 6.0:
                    taken = True
                    break
            if taken:
                continue
            if d < best_d:
                best_d = d
                best = p
        return best if best is not None else nearest

    def _new_cells_if_visit(self, gx, gy):
        """How many unmapped cells a 5 m sensor would paint from this grid cell."""
        cell_mx = self.size_x / self.cover_nx
        cell_my = self.size_y / self.cover_ny
        rad_x = max(1, int(np.ceil(self.sensor_radius / cell_mx)))
        rad_y = max(1, int(np.ceil(self.sensor_radius / cell_my)))
        origin = self._cell_to_world(gx, gy)
        painted = 0
        for dy in range(-rad_y, rad_y + 1):
            ny = gy + dy
            if ny < 0 or ny >= self.cover_ny:
                continue
            for dx in range(-rad_x, rad_x + 1):
                nx = gx + dx
                if nx < 0 or nx >= self.cover_nx:
                    continue
                if self.coverage[ny, nx] >= 0.5:
                    continue
                w = self._cell_to_world(nx, ny)
                if np.hypot(w[0] - origin[0], w[1] - origin[1]) <= self.sensor_radius:
                    painted += 1
        return painted

    def _search_goal_xy(self, drone_index):
        """Hunt leftover people once mapping is mostly done; otherwise next-best-view coverage."""
        people = self._undiscovered_people()
        missed = any(self._person_was_missed(p) for p in people)
        if people and (missed or self._interior_coverage() >= HUNT_COVER_FRAC or len(people) <= self.num_drones * 3):
            hunt = self._hunt_survivor_xy(drone_index)
            if hunt is not None:
                self.goal_is_person[drone_index] = True
                return hunt

        self.goal_is_person[drone_index] = False
        my_xy = np.asarray(self.drone_positions[drone_index][:2], dtype=np.float64)
        others = [
            np.asarray(self.drone_positions[j][:2], dtype=np.float64)
            for j in range(self.num_drones)
            if j != drone_index
        ]
        claimed = [
            self.nav_goal[j]
            for j in range(self.num_drones)
            if j != drone_index and self.has_nav_goal[j]
        ]
        heading = self.drone_velocities[drone_index][:2]
        hnorm = float(np.linalg.norm(heading))
        if hnorm > 1e-3:
            heading = heading / hnorm
        else:
            heading = None

        candidates = []
        for gy in range(1, self.cover_ny - 1):
            for gx in range(1, self.cover_nx - 1):
                if self.coverage[gy, gx] >= 0.5:
                    continue
                w = self._cell_to_world(gx, gy)
                if not self._playable_cell(w):
                    continue
                my_d = float(np.linalg.norm(w - my_xy))
                stolen = False
                for o in others:
                    if float(np.linalg.norm(w - o)) + 2.0 < my_d:
                        stolen = True
                        break
                if stolen:
                    continue
                skip_claim = False
                for c in claimed:
                    if float(np.linalg.norm(w - c)) < 14.0:
                        skip_claim = True
                        break
                if skip_claim:
                    continue
                candidates.append((w, gx, gy, my_d))
        if not candidates:
            return None

        best = None
        best_score = -1e18
        nearest = None
        nearest_d = 1e18
        for w, gx, gy, my_d in candidates:
            if my_d < nearest_d:
                nearest_d = my_d
                nearest = w
            if my_d < 8.0:
                continue
            gain = self._new_cells_if_visit(gx, gy)
            score = gain / (1.0 + 0.035 * my_d)
            if heading is not None and my_d > 1e-3:
                align = float(np.dot(heading, (w - my_xy) / my_d))
                score += 0.15 * max(0.0, align)
            if score > best_score:
                best_score = score
                best = w
        return best if best is not None else nearest

    def _full_speed_xy(self, vec):
        n = float(np.linalg.norm(vec[:2]))
        if n < 1e-6:
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)
        out = np.zeros(3, dtype=np.float32)
        out[0] = float(vec[0] / n)
        out[1] = float(vec[1] / n)
        return out

    def _unexplored_dir(self, pos_xy, drone_index=0):
        """Unit XY vector toward this drone's current search goal."""
        if self.has_nav_goal[drone_index]:
            goal = self.nav_goal[drone_index]
        else:
            goal = self._search_goal_xy(drone_index)
        if goal is None:
            return np.zeros(2, dtype=np.float32)
        delta = goal - np.asarray(pos_xy, dtype=np.float64)
        n = float(np.linalg.norm(delta))
        if n < 1e-6:
            return np.zeros(2, dtype=np.float32)
        return (delta / n).astype(np.float32)

    def _goal_still_valid(self, drone_index):
        if not self.has_nav_goal[drone_index]:
            return False
        if self.goal_hold[drone_index] <= 0:
            return False
        goal = self.nav_goal[drone_index]
        if self.goal_is_person[drone_index]:
            for p in self._undiscovered_people():
                if float(np.linalg.norm(p - goal)) < 8.0:
                    return True
            return False
        gx, gy = self._world_to_cell(goal[0], goal[1])
        if self.coverage[gy, gx] >= 0.5:
            return False
        return True

    def _separation_push(self, drone_index):
        pos = self.drone_positions[drone_index][:2]
        push = np.zeros(2, dtype=np.float64)
        for j in range(self.num_drones):
            if j == drone_index:
                continue
            delta = pos - self.drone_positions[j][:2]
            dist = float(np.linalg.norm(delta))
            if dist < 1e-4:
                delta = np.array([1.0 if drone_index > j else -1.0, 0.0], dtype=np.float64)
                dist = 1e-4
            if dist < SEPARATION_RANGE:
                push += (delta / dist) * ((SEPARATION_RANGE - dist) / SEPARATION_RANGE)
        return push

    def _mark_coverage(self, positions):
        """Paint sensor footprints; return newly covered cells and overlap count."""
        new_cells = 0
        overlap = 0
        cell_mx = self.size_x / self.cover_nx
        cell_my = self.size_y / self.cover_ny
        radius_x = max(1, int(np.ceil(self.sensor_radius / cell_mx)))
        radius_y = max(1, int(np.ceil(self.sensor_radius / cell_my)))
        for pos in positions:
            cx, cy = self._world_to_cell(pos[0], pos[1])
            for dx in range(-radius_x, radius_x + 1):
                for dy in range(-radius_y, radius_y + 1):
                    gx, gy = cx + dx, cy + dy
                    if gx < 0 or gy < 0 or gx >= self.cover_nx or gy >= self.cover_ny:
                        continue
                    w = self._cell_to_world(gx, gy)
                    if np.hypot(w[0] - pos[0], w[1] - pos[1]) > self.sensor_radius:
                        continue
                    if self.coverage[gy, gx] >= 1.0:
                        overlap += 1
                    else:
                        new_cells += 1
                    self.coverage[gy, gx] = 1.0
        return new_cells, overlap

    def _coverage_crop(self, x, y):
        """Local visited window. Off-map cells are 1 (not searchable), not 0 (unexplored)."""
        cx, cy = self._world_to_cell(x, y)
        half = COVER_CROP // 2
        crop = np.ones((COVER_CROP, COVER_CROP), dtype=np.float32)
        for i, dy in enumerate(range(cy - half, cy + half + 1)):
            for j, dx in enumerate(range(cx - half, cx + half + 1)):
                if 0 <= dy < self.cover_ny and 0 <= dx < self.cover_nx:
                    crop[i, j] = self.coverage[dy, dx]
        return crop.ravel()

    def _init_pybullet(self):
        if self.render_mode is None or not _HAS_PYBULLET:
            self.client = None
            return
        if self.client is not None:
            try:
                p.disconnect(physicsClientId=self.client)
            except Exception:
                pass
        if self.render_mode == "human":
            self.client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.client)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0, physicsClientId=self.client)
        else:
            self.client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, 0, physicsClientId=self.client)
        p.setTimeStep(1.0 / self.pyb_freq, physicsClientId=self.client)

    def _build_ground(self):
        if self.client is None:
            return
        hx, hy = self.size_x / 2.0, self.size_y / 2.0
        col = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[hx, hy, 0.2], physicsClientId=self.client
        )
        vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[hx, hy, 0.2],
            rgbaColor=[0.25, 0.55, 0.28, 1],
            physicsClientId=self.client,
        )
        self.terrain_body = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=[0, 0, -0.2],
            physicsClientId=self.client,
        )

    def _spawn_drones(self):
        self.drone_ids = []
        urdf_path = None
        try:
            gp = importlib.util.find_spec("gym_pybullet_drones")
            if gp is not None:
                assets_dir = os.path.join(os.path.dirname(gp.origin), "assets")
                candidate = os.path.join(assets_dir, "cf2x.urdf")
                if os.path.exists(candidate):
                    urdf_path = candidate
        except Exception:
            urdf_path = None

        z = self.hover_altitude
        lx, ly = self._lim_x(), self._lim_y()
        for i in range(self.num_drones):
            frac = (i + 0.5) / max(self.num_drones, 1)
            x = -lx * 0.7 + frac * (1.4 * lx)
            y = -ly * 0.35 + (i % 2) * (0.25 * ly)
            self.drone_positions[i] = [x, y, z]
            if self.client is None:
                self.drone_ids.append(-1)
                continue
            if urdf_path is not None:
                original_cwd = os.getcwd()
                os.chdir(os.path.dirname(urdf_path))
                try:
                    did = p.loadURDF(
                        os.path.basename(urdf_path),
                        [x, y, z],
                        globalScaling=DRONE_VISUAL_SCALE,
                        physicsClientId=self.client,
                    )
                except Exception:
                    did = self._sphere_drone([x, y, z], i)
                finally:
                    os.chdir(original_cwd)
            else:
                did = self._sphere_drone([x, y, z], i)
            self.drone_ids.append(did)

        if self.render_mode == "human" and self.client is not None:
            p.resetDebugVisualizerCamera(
                cameraDistance=max(self.size_x, self.size_y) * 0.85,
                cameraYaw=45,
                cameraPitch=-40,
                cameraTargetPosition=[0, 0, 2.0],
                physicsClientId=self.client,
            )

    def _sphere_drone(self, pos, idx):
        colors = [
            [1, 0.2, 0.2, 1],
            [0.2, 0.5, 1, 1],
            [1, 0.8, 0, 1],
            [0.2, 1, 0.4, 1],
            [1, 0.4, 1, 1],
        ]
        color = colors[idx % len(colors)]
        col = p.createCollisionShape(p.GEOM_SPHERE, radius=0.5, physicsClientId=self.client)
        vis = p.createVisualShape(
            p.GEOM_SPHERE, radius=0.5, rgbaColor=color, physicsClientId=self.client
        )
        return p.createMultiBody(
            baseMass=0.027,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=pos,
            physicsClientId=self.client,
        )

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._seed = seed
        self.agents = list(self.possible_agents)
        self.step_count = 0
        self.coverage = np.zeros((self.cover_ny, self.cover_nx), dtype=np.float32)
        self._seal_map_border()
        self.drone_velocities = np.zeros((self.num_drones, 3))
        self.drone_alive = np.ones(self.num_drones, dtype=bool)
        self.newly_discovered = 0
        self.new_cells = 0
        self.overlap_cells = 0
        self.wall_cooldown = np.zeros(self.num_drones, dtype=np.int32)
        self.nav_goal = np.zeros((self.num_drones, 2), dtype=np.float64)
        self.has_nav_goal = np.zeros(self.num_drones, dtype=bool)
        self.goal_hold = np.zeros(self.num_drones, dtype=np.int32)
        self.goal_is_person = np.zeros(self.num_drones, dtype=bool)

        surv_seed = (int(time.time() * 1e6) ^ (os.getpid() << 8) ^ id(self)) & 0x7FFFFFFF
        if seed is not None:
            surv_seed ^= int(seed) & 0x7FFFFFFF

        self.terrain = Terrain(
            size_x=self.size_x,
            size_y=self.size_y,
            resolution=1.0,
            seed=self._seed if self._seed is not None else 42,
            flat=True,
            with_structures=False,
        )
        self.survivors = SurvivorCluster(
            num_clusters=self.num_clusters,
            survivors_per_cluster=self.survivors_per_cluster,
            num_loners=self.num_loners,
            env_size_x=self.size_x,
            env_size_y=self.size_y,
            terrain_obj=self.terrain,
            discovery_range=self.sensor_radius,
            cluster_spread=12.0,
            spawn_frac=0.40,
            clip_xy=(self._lim_x() - 8.0, self._lim_y() - 8.0),
            require_los=False,
            frozen=True,
            seed=surv_seed,
        )

        self._init_pybullet()
        self._build_ground()
        self._spawn_drones()
        self._mark_coverage(self.drone_positions)

        return self._get_observations(), self._infos()

    def step(self, actions):
        self.step_count += 1
        dt = 1.0 / self.pyb_freq
        lx, ly = self._lim_x(), self._lim_y()

        for i, agent in enumerate(self.possible_agents):
            if agent not in actions:
                continue
            act = np.clip(np.array(actions[agent], dtype=np.float32), -1.0, 1.0)
            act[2] = 0.0
            pos = self.drone_positions[i]

            if not self._goal_still_valid(i):
                goal = self._search_goal_xy(i)
                if goal is None:
                    self.has_nav_goal[i] = False
                else:
                    self.nav_goal[i] = goal
                    self.has_nav_goal[i] = True
                    self.goal_hold[i] = GOAL_HOLD_STEPS
                    # goal_is_person is set inside _search_goal_xy / hunt
            if self.has_nav_goal[i]:
                self.goal_hold[i] -= 1
                heading = self._full_speed_xy(self.nav_goal[i] - pos[:2])
                act[0] = heading[0]
                act[1] = heading[1]

            push = self._separation_push(i)
            steered = np.array([act[0] + 0.35 * push[0], act[1] + 0.35 * push[1]], dtype=np.float64)
            nrm = float(np.linalg.norm(steered))
            if nrm > 1e-4:
                act[0] = float(np.clip(steered[0] / nrm, -1.0, 1.0))
                act[1] = float(np.clip(steered[1] / nrm, -1.0, 1.0))

            goal_xy = self.nav_goal[i] if self.has_nav_goal[i] else None
            toward_goal_x = goal_xy is not None and goal_xy[0] > pos[0] + 0.5
            toward_goal_x_neg = goal_xy is not None and goal_xy[0] < pos[0] - 0.5
            toward_goal_y = goal_xy is not None and goal_xy[1] > pos[1] + 0.5
            toward_goal_y_neg = goal_xy is not None and goal_xy[1] < pos[1] - 0.5
            hunting = bool(self.goal_is_person[i])

            if pos[0] > lx - TURN_MARGIN and act[0] > 0 and not (hunting and toward_goal_x):
                act[0] = -BOUNCE_CMD
            elif pos[0] < -lx + TURN_MARGIN and act[0] < 0 and not (hunting and toward_goal_x_neg):
                act[0] = BOUNCE_CMD
            if pos[1] > ly - TURN_MARGIN and act[1] > 0 and not (hunting and toward_goal_y):
                act[1] = -BOUNCE_CMD
            elif pos[1] < -ly + TURN_MARGIN and act[1] < 0 and not (hunting and toward_goal_y_neg):
                act[1] = BOUNCE_CMD

            if self.wall_cooldown[i] > 0:
                self.wall_cooldown[i] -= 1
                if not hunting:
                    if pos[0] > lx - TURN_MARGIN:
                        act[0] = -BOUNCE_CMD
                    elif pos[0] < -lx + TURN_MARGIN:
                        act[0] = BOUNCE_CMD
                    if pos[1] > ly - TURN_MARGIN:
                        act[1] = -BOUNCE_CMD
                    elif pos[1] < -ly + TURN_MARGIN:
                        act[1] = BOUNCE_CMD

            self.drone_velocities[i] = act * self.drone_max_speed

        for i in range(self.num_drones):
            vel = self.drone_velocities[i].copy()
            pos = self.drone_positions[i] + vel * dt
            pos, vel = self._confine_drone(i, pos, vel)
            self.drone_positions[i] = pos
            self.drone_velocities[i] = vel

        collisions = np.zeros(self.num_drones, dtype=bool)
        for i in range(self.num_drones):
            for j in range(i + 1, self.num_drones):
                delta = self.drone_positions[i][:2] - self.drone_positions[j][:2]
                dist = float(np.linalg.norm(delta))
                if dist < 1e-4:
                    delta = np.array([1.0, 0.0])
                    dist = 1e-4
                if dist < SEPARATION_RANGE:
                    nrm = delta / dist
                    soft = 0.04 * (SEPARATION_RANGE - dist)
                    self.drone_positions[i][:2] = self.drone_positions[i][:2] + nrm * soft
                    self.drone_positions[j][:2] = self.drone_positions[j][:2] - nrm * soft
                if dist < CRASH_RANGE:
                    collisions[i] = True
                    collisions[j] = True
                    nrm = delta / dist
                    push = max(1.2, CRASH_RANGE - dist)
                    self.drone_positions[i][:2] = self.drone_positions[i][:2] + nrm * push
                    self.drone_positions[j][:2] = self.drone_positions[j][:2] - nrm * push
                    self.drone_positions[i], self.drone_velocities[i] = self._confine_drone(
                        i, self.drone_positions[i], self.drone_velocities[i]
                    )
                    self.drone_positions[j], self.drone_velocities[j] = self._confine_drone(
                        j, self.drone_positions[j], self.drone_velocities[j]
                    )

        for i in range(self.num_drones):
            self.drone_positions[i], self.drone_velocities[i] = self._confine_drone(
                i, self.drone_positions[i], self.drone_velocities[i]
            )

        for i in range(self.num_drones):
            if self.client is None:
                break
            pos = self.drone_positions[i]
            orn = p.getQuaternionFromEuler([0, 0, 0])
            p.resetBasePositionAndOrientation(
                self.drone_ids[i], pos.tolist(), orn, physicsClientId=self.client
            )
            p.resetBaseVelocity(
                self.drone_ids[i], [0, 0, 0], [0, 0, 0], physicsClientId=self.client
            )

        self.new_cells, self.overlap_cells = self._mark_coverage(self.drone_positions)

        self.newly_discovered, _ = self.survivors.update_discovery(
            list(self.drone_positions), self.terrain
        )

        rewards = self._compute_rewards(collisions)
        all_found = int(self.survivors.discovered.sum()) == self.survivors.num_survivors
        maxed = self.step_count >= self.max_steps
        truncations = {a: (all_found or maxed) for a in self.possible_agents}
        terminations = {a: False for a in self.possible_agents}

        if self.render_mode == "human":
            pass

        return self._get_observations(), rewards, terminations, truncations, self._infos()

    def _compute_rewards(self, collisions):
        """People-first search: find survivors, then expand into unpainted ground."""
        stats = self.survivors.get_discovery_stats()
        all_found = stats["discovered_survivors"] == stats["total_survivors"] and stats["total_survivors"] > 0

        find_bonus = 40.0
        cell_m = COVER_CELL_M
        footprint_cells = max(1.0, np.pi * (self.sensor_radius ** 2) / (cell_m ** 2))
        cover_per_cell = (find_bonus / 7.0) / footprint_cells

        interior = self._interior_coverage()
        shared = find_bonus * float(self.newly_discovered)
        shared += cover_per_cell * float(self.new_cells)
        shared -= 0.05 * (1.0 - interior)
        shared -= 0.01
        shared -= 0.002 * float(self.overlap_cells)
        if all_found:
            shared += 150.0
        if interior >= 0.99:
            shared += 25.0

        rewards = {}
        lx, ly = self._lim_x(), self._lim_y()
        for i, agent in enumerate(self.possible_agents):
            rew = shared
            pos = self.drone_positions[i]
            vel_xy = self.drone_velocities[i][:2]
            udir = self._unexplored_dir(pos[:2], i)
            if float(np.linalg.norm(udir)) > 0.05:
                vnorm = float(np.linalg.norm(vel_xy)) + 1e-6
                rew += 0.8 * float(np.dot(vel_xy / vnorm, udir))
            speed = float(np.linalg.norm(vel_xy))
            if interior < 0.99 and speed < 0.35 * self.drone_max_speed:
                rew -= 3.0
            if abs(pos[0]) > lx * 0.72 and abs(pos[1]) > ly * 0.72:
                rew -= 5.0
            if abs(pos[0]) >= lx - 0.75 or abs(pos[1]) >= ly - 0.75:
                rew -= 0.5
            closest = SEPARATION_RANGE
            for j in range(self.num_drones):
                if j == i:
                    continue
                d = float(np.linalg.norm(pos[:2] - self.drone_positions[j][:2]))
                closest = min(closest, d)
            if closest < SEPARATION_RANGE:
                rew -= 8.0 * (1.0 - closest / SEPARATION_RANGE)
            if collisions[i]:
                rew -= 40.0
            rewards[agent] = rew
        return rewards

    def _get_observations(self):
        stats = self.survivors.get_discovery_stats()
        remaining = 1.0 - stats["discovery_rate"]
        observations = {}
        for i, agent in enumerate(self.possible_agents):
            obs = np.zeros(self.obs_dim, dtype=np.float32)
            pos = self.drone_positions[i]
            vel = self.drone_velocities[i]
            obs[0] = pos[0] / (self.size_x / 2.0)
            obs[1] = pos[1] / (self.size_y / 2.0)
            obs[2] = pos[2] / self.drone_max_altitude
            obs[3:6] = vel / self.drone_max_speed
            idx = 6
            for j in range(self.num_drones):
                if j == i:
                    continue
                rel = (self.drone_positions[j] - pos) / np.array(
                    [self.size_x, self.size_y, self.drone_max_altitude], dtype=np.float32
                )
                obs[idx : idx + 3] = rel
                idx += 3
            obs[idx] = stats["discovery_rate"]
            obs[idx + 1] = remaining
            udir = self._unexplored_dir(pos[:2], i)
            obs[idx + 2] = udir[0]
            obs[idx + 3] = udir[1]
            obs[idx + 4] = self._interior_coverage()
            crop = self._coverage_crop(pos[0], pos[1])
            obs[idx + 5 : idx + 5 + crop.size] = crop
            observations[agent] = obs
        return observations

    def _infos(self):
        stats = self.survivors.get_discovery_stats()
        info = {
            "step": self.step_count,
            "connected_survivors": 0,
            "discovered_survivors": stats["discovered_survivors"],
            "total_survivors": stats["total_survivors"],
            "discovery_rate": stats["discovery_rate"],
            "coverage_frac": self._interior_coverage(),
            "new_cells": int(self.new_cells),
            "overlap_cells": int(self.overlap_cells),
        }
        return {agent: dict(info) for agent in self.possible_agents}

    def render(self):
        if self.render_mode == "human":
            time.sleep(1.0 / self.pyb_freq)

    def close(self):
        if self.client is not None:
            try:
                p.disconnect(physicsClientId=self.client)
            except Exception:
                pass
            self.client = None
