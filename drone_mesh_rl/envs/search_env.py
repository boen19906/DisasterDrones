"""
search_env.py — V1 multi-agent search: find as many survivors as possible.

Fog-of-war coverage. A survivor is found when a drone flies over them
(XY distance <= sensor_radius). No mesh, satellite, battery, or buildings.
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
        drone_max_speed=4.0,
        drone_max_altitude=20.0,
        hover_altitude=8.0,
        sensor_radius=8.0,
        cover_cells=20,
        pyb_freq=30,
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
        self.cover_cells = cover_cells
        self.pyb_freq = pyb_freq
        self._seed = seed

        self.possible_agents = [f"drone_{i}" for i in range(num_drones)]
        self.agents = list(self.possible_agents)

        self.obs_dim = 6 + 3 * (num_drones - 1) + 2 + COVER_CROP * COVER_CROP
        self.act_dim = 3

        self.terrain = None
        self.survivors = None
        self.coverage = np.zeros((cover_cells, cover_cells), dtype=np.float32)
        self.drone_positions = np.zeros((num_drones, 3))
        self.drone_velocities = np.zeros((num_drones, 3))
        self.drone_alive = np.ones(num_drones, dtype=bool)
        self.step_count = 0
        self.newly_discovered = 0
        self.overlap_cells = 0

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
        half = self.env_size / 2.0
        gx = int((x + half) / self.env_size * self.cover_cells)
        gy = int((y + half) / self.env_size * self.cover_cells)
        gx = int(np.clip(gx, 0, self.cover_cells - 1))
        gy = int(np.clip(gy, 0, self.cover_cells - 1))
        return gx, gy

    def _mark_coverage(self, positions):
        """Paint sensor footprints; return newly covered cells and overlap count."""
        new_cells = 0
        overlap = 0
        cell_m = self.env_size / self.cover_cells
        radius_cells = max(1, int(np.ceil(self.sensor_radius / cell_m)))
        for pos in positions:
            cx, cy = self._world_to_cell(pos[0], pos[1])
            for dx in range(-radius_cells, radius_cells + 1):
                for dy in range(-radius_cells, radius_cells + 1):
                    gx, gy = cx + dx, cy + dy
                    if gx < 0 or gy < 0 or gx >= self.cover_cells or gy >= self.cover_cells:
                        continue
                    wx = (gx + 0.5) / self.cover_cells * self.env_size - self.env_size / 2.0
                    wy = (gy + 0.5) / self.cover_cells * self.env_size - self.env_size / 2.0
                    if np.hypot(wx - pos[0], wy - pos[1]) > self.sensor_radius:
                        continue
                    if self.coverage[gy, gx] >= 1.0:
                        overlap += 1
                    else:
                        new_cells += 1
                    self.coverage[gy, gx] = 1.0
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

    def _init_pybullet(self):
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

        spread = min(self.num_drones * 4.0, self.env_size * 0.25)
        z = self.hover_altitude
        for i in range(self.num_drones):
            x = (i - (self.num_drones - 1) / 2.0) * (spread / max(self.num_drones - 1, 1))
            y = -self.env_size * 0.35
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
            self.drone_ids.append(did)

        if self.render_mode == "human":
            p.resetDebugVisualizerCamera(
                cameraDistance=90,
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
        self.coverage = np.zeros((self.cover_cells, self.cover_cells), dtype=np.float32)
        self.drone_velocities = np.zeros((self.num_drones, 3))
        self.drone_alive = np.ones(self.num_drones, dtype=bool)
        self.newly_discovered = 0
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
        self._mark_coverage(self.drone_positions)

        return self._get_observations(), self._infos()

    def step(self, actions):
        self.step_count += 1
        dt = 1.0 / self.pyb_freq
        half = self.env_size / 2.0

        for i, agent in enumerate(self.possible_agents):
            if agent not in actions:
                continue
            act = np.clip(np.array(actions[agent], dtype=np.float32), -1.0, 1.0)
            self.drone_velocities[i] = act * self.drone_max_speed

        collisions = np.zeros(self.num_drones, dtype=bool)
        for i in range(self.num_drones):
            pos = self.drone_positions[i] + self.drone_velocities[i] * dt
            pos[0] = np.clip(pos[0], -half, half)
            pos[1] = np.clip(pos[1], -half, half)
            pos[2] = np.clip(pos[2], 1.0, self.drone_max_altitude)
            self.drone_positions[i] = pos
            orn = p.getQuaternionFromEuler([0, 0, 0])
            p.resetBasePositionAndOrientation(
                self.drone_ids[i], pos.tolist(), orn, physicsClientId=self.client
            )

        for i in range(self.num_drones):
            for j in range(i + 1, self.num_drones):
                if np.linalg.norm(self.drone_positions[i] - self.drone_positions[j]) < 1.2:
                    collisions[i] = True
                    collisions[j] = True

        _, overlap = self._mark_coverage(self.drone_positions)
        self.overlap_cells = overlap

        self.newly_discovered, _ = self.survivors.update_discovery(
            list(self.drone_positions), self.terrain
        )

        rewards = self._compute_rewards(collisions)
        all_found = int(self.survivors.discovered.sum()) == self.survivors.num_survivors
        maxed = self.step_count >= self.max_steps
        truncations = {a: (all_found or maxed) for a in self.possible_agents}
        terminations = {a: False for a in self.possible_agents}

        if self.render_mode == "human":
            time.sleep(max(0.0, dt * 0.25))

        return self._get_observations(), rewards, terminations, truncations, self._infos()

    def _compute_rewards(self, collisions):
        stats = self.survivors.get_discovery_stats()
        all_found = stats["discovered_survivors"] == stats["total_survivors"] and stats["total_survivors"] > 0
        shared = 10.0 * float(self.newly_discovered)
        shared -= 0.01
        shared -= 0.002 * float(self.overlap_cells)
        if all_found:
            shared += 50.0
        rewards = {}
        for i, agent in enumerate(self.possible_agents):
            rew = shared
            if collisions[i]:
                rew -= 5.0
            rewards[agent] = rew
        return rewards

    def _get_observations(self):
        half = self.env_size / 2.0
        stats = self.survivors.get_discovery_stats()
        remaining = 1.0 - stats["discovery_rate"]
        observations = {}
        for i, agent in enumerate(self.possible_agents):
            obs = np.zeros(self.obs_dim, dtype=np.float32)
            pos = self.drone_positions[i]
            vel = self.drone_velocities[i]
            obs[0] = pos[0] / half
            obs[1] = pos[1] / half
            obs[2] = pos[2] / self.drone_max_altitude
            obs[3:6] = vel / self.drone_max_speed
            idx = 6
            for j in range(self.num_drones):
                if j == i:
                    continue
                rel = (self.drone_positions[j] - pos) / self.env_size
                obs[idx : idx + 3] = rel
                idx += 3
            obs[idx] = stats["discovery_rate"]
            obs[idx + 1] = remaining
            crop = self._coverage_crop(pos[0], pos[1])
            obs[idx + 2 : idx + 2 + crop.size] = crop
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
