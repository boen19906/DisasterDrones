"""
search_env.py — V1 multi-agent search: find as many survivors as possible.

Fog-of-war coverage. A survivor is found when a drone flies over them
(XY distance <= sensor_radius). No mesh, satellite, battery, or buildings.

Each action is one of 9 grid moves (8 neighbors or stay). A move into the
wall is replaced by a legal heading toward unfinished ground.
The chosen heading is flown at full speed for one kinematic step. Altitude
is locked to hover. Every drone reads the same coarse coverage map and a
vector toward the middle of the remaining unpainted region. That vector is
not a chase reward.

Rewards:
  +1.0   each newly painted cell, paid only to the drone that painted it
  +10.0  each newly found survivor (shared)
  +50.0  all survivors found (shared)
  -0.25  per step a drone paints nothing and is only re-covering searched cells
  -0.01  per step (shared)
  -1.0   drone-drone collision (that drone only)
  -1.0   not moving while cells or people remain (that drone only)
  -8.0   per survivor still missing, on the timeout step only
"""

import functools
import importlib.util
import os
import time

import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces
from pettingzoo import ParallelEnv

from .survivors import SurvivorCluster
from .terrain import Terrain


COVER_CROP = 7  # local visited-map window (odd)
GLOBAL_BINS = 8  # coarse coverage map for the training critic
N_ACTIONS = 9

# Eight neighbor headings, plus stay. Stay on the wall is turned back inward.
_HEADING_XY = np.array(
    [
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
        [-1.0, 1.0],
        [-1.0, 0.0],
        [-1.0, -1.0],
        [0.0, -1.0],
        [1.0, -1.0],
        [0.0, 0.0],
    ],
    dtype=np.float64,
)
_HEADING_NORM = np.linalg.norm(_HEADING_XY, axis=1)
_HEADING_NORM[_HEADING_NORM == 0.0] = 1.0
_HEADING_XY = _HEADING_XY / _HEADING_NORM[:, None]


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
        num_clusters=4,
        survivors_per_cluster=(2, 5),
        env_size=100.0,
        max_steps=800,
        render_mode=None,
        drone_max_speed=12.0,
        drone_max_altitude=20.0,
        hover_altitude=8.0,
        sensor_radius=8.0,
        cover_cells=20,
        pyb_freq=10,
        seed=None,
    ):
        super().__init__()
        self.num_drones = num_drones
        self.num_clusters = num_clusters
        self.survivors_per_cluster = survivors_per_cluster
        self.env_size = env_size
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.drone_max_speed = drone_max_speed
        self.drone_max_altitude = drone_max_altitude
        self.hover_altitude = hover_altitude
        self.sensor_radius = sensor_radius
        self.cover_cells = max(int(cover_cells), int(round(float(env_size) / 5.0)))
        self.pyb_freq = pyb_freq
        self._seed = seed

        self.possible_agents = [f"drone_{i}" for i in range(num_drones)]
        self.agents = list(self.possible_agents)

        # Vel, shared-map heading, walls, teammates, progress, local crop, shared map
        self.obs_dim = (
            2 + 2 + 4 + 2 * (num_drones - 1) + 2 + COVER_CROP * COVER_CROP + GLOBAL_BINS * GLOBAL_BINS
        )
        self.discrete_actions = True
        self.n_actions = N_ACTIONS
        self.act_dim = N_ACTIONS
        self.state_dim = GLOBAL_BINS * GLOBAL_BINS + 2 * num_drones

        self.terrain = None
        self.survivors = None
        self.coverage = np.zeros((self.cover_cells, self.cover_cells), dtype=np.float32)
        self.visit_age = np.zeros((self.cover_cells, self.cover_cells), dtype=np.float32)
        self.drone_positions = np.zeros((num_drones, 3))
        self.drone_velocities = np.zeros((num_drones, 3))
        self.drone_alive = np.ones(num_drones, dtype=bool)
        self.step_count = 0
        self.newly_discovered = 0
        self.new_cells = 0
        self.overlap_cells = 0
        self._prev_uncovered_dist = np.zeros(num_drones, dtype=np.float32)
        self._prev_frontier_dist = np.zeros(num_drones, dtype=np.float32)

        self.client = None
        self.drone_ids = []
        self.terrain_body = None

        self._observation_spaces = {
            agent: spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
            for agent in self.possible_agents
        }
        self._action_spaces = {
            agent: spaces.Discrete(self.n_actions) for agent in self.possible_agents
        }

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self._observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self._action_spaces[agent]

    def _world_to_cell(self, x, y):
        half = self.env_size / 2.0
        gx = int((x + half) / self.env_size * self.cover_cells)
        gy = int((y + half) / self.env_size * self.cover_cells)
        gx = int(np.clip(gx, 0, self.cover_cells - 1))
        gy = int(np.clip(gy, 0, self.cover_cells - 1))
        return gx, gy

    def _mark_coverage(self, positions):
        """Paint a cell only when the whole cell was inside the sensor.

        Otherwise a survivor near the cell edge is marked 'searched' without
        ever being found, and the policy never comes back.
        """
        new_cells = 0
        overlap = 0
        per_drone = np.zeros(len(positions), dtype=np.int32)
        per_overlap = np.zeros(len(positions), dtype=np.int32)
        # Snapshot so a cell painted earlier this step is not also overlap.
        already = self.coverage >= 1.0
        cell_m = self.env_size / self.cover_cells
        half_diag = cell_m * np.sqrt(2.0) / 2.0
        paint_radius = max(cell_m * 0.5, self.sensor_radius - half_diag)
        radius_cells = max(1, int(np.ceil(paint_radius / cell_m)))
        for i, pos in enumerate(positions):
            cx, cy = self._world_to_cell(pos[0], pos[1])
            for dx in range(-radius_cells, radius_cells + 1):
                for dy in range(-radius_cells, radius_cells + 1):
                    gx, gy = cx + dx, cy + dy
                    if gx < 0 or gy < 0 or gx >= self.cover_cells or gy >= self.cover_cells:
                        continue
                    wx = (gx + 0.5) / self.cover_cells * self.env_size - self.env_size / 2.0
                    wy = (gy + 0.5) / self.cover_cells * self.env_size - self.env_size / 2.0
                    if np.hypot(wx - pos[0], wy - pos[1]) > paint_radius:
                        continue
                    if already[gy, gx]:
                        overlap += 1
                        per_overlap[i] += 1
                    elif self.coverage[gy, gx] < 1.0:
                        new_cells += 1
                        per_drone[i] += 1
                    self.coverage[gy, gx] = 1.0
                    self.visit_age[gy, gx] = 0.0
        self._drone_new_cells = per_drone
        self._drone_overlap = per_overlap
        return new_cells, overlap

    def _coverage_crop(self, x, y):
        cx, cy = self._world_to_cell(x, y)
        half = COVER_CROP // 2
        crop = np.zeros((COVER_CROP, COVER_CROP), dtype=np.float32)
        for i, dy in enumerate(range(cy - half, cy + half + 1)):
            for j, dx in enumerate(range(cx - half, cx + half + 1)):
                if 0 <= dy < self.cover_cells and 0 <= dx < self.cover_cells:
                    crop[i, j] = self.coverage[dy, dx]
        return crop.ravel()

    def _uncovered_targets(self, sector=None):
        uncovered = np.argwhere(self.coverage < 0.5)
        if uncovered.shape[0] == 0:
            return None
        half = self.env_size / 2.0
        cells = uncovered.astype(np.float32)
        wx = (cells[:, 1] + 0.5) / self.cover_cells * self.env_size - half
        wy = (cells[:, 0] + 0.5) / self.cover_cells * self.env_size - half
        if sector is not None:
            width = self.env_size / self.num_drones
            x0 = -half + sector * width
            x1 = x0 + width
            keep = (wx >= x0) & (wx < x1)
            if not np.any(keep):
                return None
            wx, wy = wx[keep], wy[keep]
        return wx, wy

    def _frontier_xy(self, drone_idx, x, y):
        """Middle of the remaining unpainted patch in this drone's strip.

        Aiming at the nearest edge cell makes a deterministic policy orbit
        that patch and miss whoever is inside it.
        """
        pts = self._uncovered_targets(sector=drone_idx)
        if pts is None:
            pts = self._uncovered_targets()
        if pts is None:
            return 0.0, 0.0
        wx, wy = pts
        return float((np.mean(wx) - x) / self.env_size), float((np.mean(wy) - y) / self.env_size)

    def _search_heading(self, x, y):
        """Direction to the middle of the ground the team has not searched yet.

        The nearest unpainted cell makes a deterministic policy orbit that
        edge. The centroid pulls the drone across the remaining region.
        """
        pts = self._uncovered_targets()
        if pts is not None:
            wx, wy = pts
            return float((np.mean(wx) - x) / self.env_size), float((np.mean(wy) - y) / self.env_size)
        if self.survivors is not None and not bool(np.all(self.survivors.discovered)):
            gy, gx = np.unravel_index(int(np.argmax(self.visit_age)), self.visit_age.shape)
            half = self.env_size / 2.0
            wx = (gx + 0.5) / self.cover_cells * self.env_size - half
            wy = (gy + 0.5) / self.cover_cells * self.env_size - half
            return float((wx - x) / self.env_size), float((wy - y) / self.env_size)
        return 0.0, 0.0

    def _shared_map(self):
        """Coarse coverage grid the whole team paints and every drone can read."""
        bins = GLOBAL_BINS
        g = self.cover_cells
        small = np.zeros((bins, bins), dtype=np.float32)
        for iy in range(bins):
            y0 = iy * g // bins
            y1 = max(y0 + 1, (iy + 1) * g // bins)
            for ix in range(bins):
                x0 = ix * g // bins
                x1 = max(x0 + 1, (ix + 1) * g // bins)
                small[iy, ix] = float(self.coverage[y0:y1, x0:x1].mean())
        return small.ravel()

    def global_state(self):
        """Shared coverage map plus normalized drone XY, for the critic."""
        half = max(self.env_size / 2.0, 1.0)
        pos = (self.drone_positions[:, :2] / half).astype(np.float32).ravel()
        return np.concatenate([self._shared_map(), pos])

    def _nearest_uncovered_xy(self, x, y):
        pts = self._uncovered_targets()
        if pts is None:
            return 0.0, 0.0
        wx, wy = pts
        dist = np.hypot(wx - x, wy - y)
        k = int(np.argmin(dist))
        return float((wx[k] - x) / self.env_size), float((wy[k] - y) / self.env_size)

    def _uncovered_dist(self, x, y):
        pts = self._uncovered_targets()
        if pts is None:
            return 0.0
        wx, wy = pts
        return float(np.min(np.hypot(wx - x, wy - y)))

    def _assigned_survivor_xy(self, drone_idx):
        """Each drone gets a distinct unfound survivor (greedy by id)."""
        mask = ~self.survivors.discovered
        if not np.any(mask):
            return 0.0, 0.0
        pts = self.survivors.positions[mask, :2].copy()
        taken = np.zeros(len(pts), dtype=bool)
        target = np.zeros(2, dtype=np.float64)
        for i in range(self.num_drones):
            pos = self.drone_positions[i, :2]
            dist = np.linalg.norm(pts - pos, axis=1)
            dist[taken] = np.inf
            k = int(np.argmin(dist))
            if not np.isfinite(dist[k]):
                break
            taken[k] = True
            if i == drone_idx:
                target = pts[k] - pos
                break
        return float(target[0] / self.env_size), float(target[1] / self.env_size)

    def _on_border(self, pos, margin=0.4):
        limit = self.env_size / 2.0 - 1.0
        return abs(pos[0]) >= limit - margin or abs(pos[1]) >= limit - margin

    def _heading_blocked(self, pos, heading):
        """True when this heading only pushes into a wall the drone is already on."""
        limit = self.env_size / 2.0 - 1.0
        x, y = float(pos[0]), float(pos[1])
        hx, hy = float(heading[0]), float(heading[1])
        if x >= limit - 0.4 and hx > 0.1:
            return True
        if x <= -limit + 0.4 and hx < -0.1:
            return True
        if y >= limit - 0.4 and hy > 0.1:
            return True
        if y <= -limit + 0.4 and hy < -0.1:
            return True
        return False

    def _best_legal_heading(self, pos, ux, uy):
        """Neighbor heading toward (ux, uy) that does not point out of the map."""
        best = None
        best_dot = -np.inf
        for heading in _HEADING_XY:
            if self._heading_blocked(pos, heading):
                continue
            dot = float(heading[0]) * float(ux) + float(heading[1]) * float(uy)
            if dot > best_dot:
                best_dot = dot
                best = heading
        if best is None:
            inward = np.array([-np.sign(pos[0]), -np.sign(pos[1])], dtype=np.float64)
            norm = float(np.linalg.norm(inward))
            best = np.array([1.0, 0.0]) if norm < 1e-6 else inward / norm
        return best

    def _move_with_soft_walls(self, pos, vel, dt, half):
        """Kinematic motion. A move into the wall turns back toward unfinished cells."""
        pos = np.asarray(pos, dtype=np.float64).copy()
        vel = np.asarray(vel, dtype=np.float64).copy()
        heading = vel[:2]
        speed = float(np.hypot(heading[0], heading[1]))
        if speed > 0.5 and self._heading_blocked(pos, heading):
            ux, uy = self._search_heading(pos[0], pos[1])
            if abs(ux) + abs(uy) < 1e-6:
                ux, uy = -float(np.sign(pos[0])), -float(np.sign(pos[1]))
            legal = self._best_legal_heading(pos, ux, uy)
            vel[0] = legal[0] * self.drone_max_speed
            vel[1] = legal[1] * self.drone_max_speed
        pos = pos + vel * dt
        limit = half - 1.0
        for axis in (0, 1):
            if pos[axis] >= limit:
                pos[axis] = limit
                if vel[axis] > 0.0:
                    vel[axis] = 0.0
            elif pos[axis] <= -limit:
                pos[axis] = -limit
                if vel[axis] < 0.0:
                    vel[axis] = 0.0
        if float(np.hypot(vel[0], vel[1])) < 0.5 and self._on_border(pos):
            ux, uy = -float(np.sign(pos[0])) or 0.0, -float(np.sign(pos[1])) or 0.0
            legal = self._best_legal_heading(pos, ux, uy)
            vel[0] = legal[0] * self.drone_max_speed
            vel[1] = legal[1] * self.drone_max_speed
            pos[0] = float(np.clip(pos[0] + vel[0] * dt, -limit, limit))
            pos[1] = float(np.clip(pos[1] + vel[1] * dt, -limit, limit))
        pos[2] = self.hover_altitude
        vel[2] = 0.0
        return pos, vel

    def _init_coverage_sweep(self):
        """Split the map into north-south lanes, one subset per drone."""
        half = self.env_size / 2.0 - 4.0
        spacing = max(self.sensor_radius * 1.5, 8.0)
        xs = np.arange(-half, half + 0.01, spacing)
        if xs.size == 0:
            xs = np.array([0.0])
        self._lanes = [xs[i :: self.num_drones] for i in range(self.num_drones)]
        self._lane_idx = np.zeros(self.num_drones, dtype=int)
        self._sweep_dir = np.ones(self.num_drones, dtype=np.float64)

    def coverage_actions(self):
        """Boustrophedon (lawnmower) covering the whole square."""
        if not hasattr(self, "_lanes"):
            self._init_coverage_sweep()
        half = self.env_size / 2.0 - 4.0
        actions = {}
        for i, agent in enumerate(self.possible_agents):
            lanes = np.asarray(self._lanes[i], dtype=np.float64)
            if lanes.size == 0:
                actions[agent] = np.zeros(3, dtype=np.float32)
                continue
            idx = int(np.clip(self._lane_idx[i], 0, lanes.size - 1))
            target_x = float(lanes[idx])
            pos = self.drone_positions[i]
            vx = float(np.clip((target_x - pos[0]) / 3.0, -1.0, 1.0))
            on_lane = abs(pos[0] - target_x) < 2.5
            if on_lane:
                if pos[1] >= half and self._sweep_dir[i] > 0:
                    self._sweep_dir[i] = -1.0
                    self._lane_idx[i] = min(idx + 1, lanes.size - 1)
                elif pos[1] <= -half and self._sweep_dir[i] < 0:
                    self._sweep_dir[i] = 1.0
                    self._lane_idx[i] = min(idx + 1, lanes.size - 1)
            vy = float(self._sweep_dir[i])
            if not on_lane:
                vy *= 0.15
            actions[agent] = np.array([vx, vy, 0.0], dtype=np.float32)
        return actions

    def _init_pybullet(self):
        if self.client is not None and p.isConnected(self.client):
            p.resetSimulation(physicsClientId=self.client)
            p.setGravity(0, 0, 0, physicsClientId=self.client)
            p.setTimeStep(1.0 / self.pyb_freq, physicsClientId=self.client)
            if self.render_mode == "human":
                p.setRealTimeSimulation(0, physicsClientId=self.client)
            return
        if self.render_mode == "human":
            self.client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.client)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0, physicsClientId=self.client)
            p.setRealTimeSimulation(0, physicsClientId=self.client)
        else:
            self.client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, 0, physicsClientId=self.client)
        p.setTimeStep(1.0 / self.pyb_freq, physicsClientId=self.client)

    def _build_ground(self):
        half = self.env_size / 2.0
        col = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[half, half, 0.2], physicsClientId=self.client
        )
        vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[half, half, 0.2],
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
        self._build_fence()

    def _build_fence(self):
        """Solid rim so a drone body cannot leave the green square."""
        half = self.env_size / 2.0
        thick = 1.0
        height = 4.0
        walls = (
            ([0.0, half + thick, height / 2], [half + thick, thick, height / 2]),
            ([0.0, -(half + thick), height / 2], [half + thick, thick, height / 2]),
            ([half + thick, 0.0, height / 2], [thick, half + thick, height / 2]),
            ([-(half + thick), 0.0, height / 2], [thick, half + thick, height / 2]),
        )
        for pos, ext in walls:
            col = p.createCollisionShape(
                p.GEOM_BOX, halfExtents=ext, physicsClientId=self.client
            )
            vis = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=ext,
                rgbaColor=[0.25, 0.25, 0.28, 1],
                physicsClientId=self.client,
            )
            p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=col,
                baseVisualShapeIndex=vis,
                basePosition=pos,
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
        self._init_coverage_sweep()
        rng = np.random.default_rng((self._seed or 0) + 17)
        margin = self.env_size * 0.25
        for i in range(self.num_drones):
            x = float(rng.uniform(-margin, margin))
            y = float(rng.uniform(-margin, margin))
            self.drone_positions[i] = [x, y, z]
            if urdf_path is not None:
                original_cwd = os.getcwd()
                os.chdir(os.path.dirname(urdf_path))
                try:
                    did = p.loadURDF(
                        os.path.basename(urdf_path),
                        [x, y, z],
                        globalScaling=8.0,
                        physicsClientId=self.client,
                    )
                except Exception:
                    did = self._sphere_drone([x, y, z], i)
                finally:
                    os.chdir(original_cwd)
            else:
                did = self._sphere_drone([x, y, z], i)
            p.changeDynamics(did, -1, mass=0, physicsClientId=self.client)
            p.resetBaseVelocity(
                did, [0, 0, 0], [0, 0, 0], physicsClientId=self.client
            )
            self.drone_ids.append(did)

        if self.render_mode == "human":
            p.resetDebugVisualizerCamera(
                cameraDistance=max(90.0, self.env_size * 0.95),
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
            self._seed = int(seed)
        else:
            self._seed = int(np.random.randint(0, 1_000_000_000))
        if options and options.get("env_size") is not None:
            self.env_size = float(options["env_size"])
            self.cover_cells = max(16, int(round(self.env_size / 5.0)))
            self.max_steps = max(500, int(self.env_size * 8.0))
        self.agents = list(self.possible_agents)
        self.step_count = 0
        self.coverage = np.zeros((self.cover_cells, self.cover_cells), dtype=np.float32)
        self.visit_age = np.zeros((self.cover_cells, self.cover_cells), dtype=np.float32)
        self.drone_velocities = np.zeros((self.num_drones, 3))
        self.drone_alive = np.ones(self.num_drones, dtype=bool)
        self.newly_discovered = 0
        self.new_cells = 0
        self.overlap_cells = 0

        self.terrain = Terrain(
            size_x=self.env_size,
            size_y=self.env_size,
            resolution=1.0,
            seed=self._seed if self._seed is not None else 42,
            flat=True,
            with_structures=False,
        )
        self.survivors = SurvivorCluster(
            num_clusters=self.num_clusters,
            survivors_per_cluster=self.survivors_per_cluster,
            env_size_x=self.env_size,
            env_size_y=self.env_size,
            terrain_obj=self.terrain,
            discovery_range=self.sensor_radius,
            require_los=False,
            frozen=True,
            seed=self._seed,
        )

        self._init_pybullet()
        self._build_ground()
        self._spawn_drones()
        self._init_coverage_sweep()
        self._mark_coverage(self.drone_positions)

        return self._get_observations(), self._infos()

    def step(self, actions):
        self.step_count += 1
        self.visit_age += 1.0
        dt = 1.0 / self.pyb_freq
        half = self.env_size / 2.0

        for i, agent in enumerate(self.possible_agents):
            if agent not in actions:
                continue
            raw = np.asarray(actions[agent], dtype=np.float32).reshape(-1)
            if raw.size >= 3:
                act = np.clip(raw[:3], -1.0, 1.0)
                self.drone_velocities[i] = act * self.drone_max_speed
            else:
                idx = int(np.clip(np.rint(float(raw[0])), 0, self.n_actions - 1))
                heading = _HEADING_XY[idx]
                self.drone_velocities[i, 0] = heading[0] * self.drone_max_speed
                self.drone_velocities[i, 1] = heading[1] * self.drone_max_speed
                self.drone_velocities[i, 2] = 0.0

        collisions = np.zeros(self.num_drones, dtype=bool)
        for i in range(self.num_drones):
            pos, vel = self._move_with_soft_walls(
                self.drone_positions[i], self.drone_velocities[i], dt, half
            )
            self.drone_velocities[i] = vel
            self.drone_positions[i] = pos
            orn = p.getQuaternionFromEuler([0, 0, 0])
            p.resetBasePositionAndOrientation(
                self.drone_ids[i], pos.tolist(), orn, physicsClientId=self.client
            )
            p.resetBaseVelocity(
                self.drone_ids[i], [0, 0, 0], [0, 0, 0], physicsClientId=self.client
            )
        if self.render_mode == "human":
            p.setRealTimeSimulation(0, physicsClientId=self.client)

        for i in range(self.num_drones):
            for j in range(i + 1, self.num_drones):
                if np.linalg.norm(self.drone_positions[i] - self.drone_positions[j]) < 1.2:
                    collisions[i] = True
                    collisions[j] = True

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
        """New cells and new people. Sitting still is costly while work remains."""
        stats = self.survivors.get_discovery_stats()
        all_found = stats["discovered_survivors"] == stats["total_survivors"] and stats["total_survivors"] > 0
        uncovered_left = float(self.coverage.mean()) < 0.995

        shared = 10.0 * float(self.newly_discovered)
        shared -= 0.01
        if all_found:
            shared += 50.0
        elif self.step_count >= self.max_steps:
            remaining = stats["total_survivors"] - stats["discovered_survivors"]
            shared -= 8.0 * float(remaining)

        rewards = {}
        for i, agent in enumerate(self.possible_agents):
            rew = shared + float(self._drone_new_cells[i])
            # The sensor always overlaps cells painted a moment ago. Charge only
            # when this step is pure re-coverage, so crossing old ground to reach
            # a new region is not punished on every step.
            if self._drone_new_cells[i] == 0 and self._drone_overlap[i] > 0:
                rew -= 0.25
            speed = float(np.hypot(self.drone_velocities[i][0], self.drone_velocities[i][1]))
            if speed < 0.5 and (uncovered_left or not all_found):
                rew -= 1.0
            if collisions[i]:
                rew -= 1.0
            x, y = self.drone_positions[i][0], self.drone_positions[i][1]
            limit = self.env_size / 2.0 - 1.0
            if abs(x) >= limit - 0.05 or abs(y) >= limit - 0.05:
                rew -= 0.15
                if not all_found:
                    rew -= 0.5
            rewards[agent] = rew
        return rewards

    def _get_observations(self):
        """Scale-invariant local obs so the policy can transfer to larger maps."""
        half = self.env_size / 2.0
        sr = max(self.sensor_radius, 1.0)
        stats = self.survivors.get_discovery_stats()
        uncovered_frac = 1.0 - float(self.coverage.mean())
        observations = {}
        for i, agent in enumerate(self.possible_agents):
            obs = np.zeros(self.obs_dim, dtype=np.float32)
            pos = self.drone_positions[i]
            vel = self.drone_velocities[i]
            obs[0] = vel[0] / self.drone_max_speed
            obs[1] = vel[1] / self.drone_max_speed
            ux, uy = self._search_heading(pos[0], pos[1])
            obs[2] = float(np.clip(ux * self.env_size / sr, -4.0, 4.0))
            obs[3] = float(np.clip(uy * self.env_size / sr, -4.0, 4.0))
            obs[4] = float(np.clip((half - pos[0]) / sr, 0.0, 8.0))
            obs[5] = float(np.clip((pos[0] + half) / sr, 0.0, 8.0))
            obs[6] = float(np.clip((half - pos[1]) / sr, 0.0, 8.0))
            obs[7] = float(np.clip((pos[1] + half) / sr, 0.0, 8.0))
            idx = 8
            for j in range(self.num_drones):
                if j == i:
                    continue
                rel = (self.drone_positions[j][:2] - pos[:2]) / sr
                obs[idx] = float(np.clip(rel[0], -8.0, 8.0))
                obs[idx + 1] = float(np.clip(rel[1], -8.0, 8.0))
                idx += 2
            obs[idx] = stats["discovery_rate"]
            obs[idx + 1] = uncovered_frac
            crop = self._coverage_crop(pos[0], pos[1])
            obs[idx + 2 : idx + 2 + crop.size] = crop
            shared = self._shared_map()
            obs[idx + 2 + crop.size : idx + 2 + crop.size + shared.size] = shared
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
            "coverage_frac": float(self.coverage.mean()),
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
