"""
disaster_env.py — PettingZoo ParallelEnv for the drone swarm mesh network.

This is the main RL environment. It uses PyBullet directly (with Crazyflie
URDF for visuals) and implements the PettingZoo ParallelEnv API for
multi-agent reinforcement learning.

Observation Space (per drone, 25-dim):
  [0:3]   - Position (x, y, z)
  [3:6]   - Velocity (vx, vy, vz)
  [6:9]   - Orientation (roll, pitch, yaw)
  [9:12]  - Wind vector (wx, wy, wz)
  [12:15] - Satellite relative position (dx, dy, dz)
  [15]    - Satellite visible (0/1)
  [16]    - Battery level (0-1)
  [17]    - Current role (0=Relay, 1=Gateway)
  [18:21] - Nearest undiscovered survivor relative pos (dx, dy, dz)
  [21]    - Survivor discovery rate (0-1)
  [22]    - Nearest neighbor drone distance (normalized)
  [23]    - Nearest neighbor RSSI (normalized)
  [24]    - Terrain height below drone (normalized)

Action Space (per drone, 5-dim):
  [0:3] - Velocity command (vx, vy, vz) in [-1, 1]
  [3]   - Role toggle (continuous, >0.5 → Gateway)
  [4]   - Communication power level (0-1)
"""

import os
import functools
import importlib.util
import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces
from pettingzoo import ParallelEnv

from .terrain import Terrain
from .network import Satellite, calculate_throughput, calculate_link
from .survivors import SurvivorCluster
from .weather import WeatherSystem
from .structures import point_hits_structure


class DisasterMeshEnv(ParallelEnv):
    """
    Multi-agent drone swarm environment for mesh network search-and-rescue.

    PettingZoo ParallelEnv — all agents act simultaneously each step.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "disaster_mesh_v1",
        "is_parallelizable": True,
    }

    def __init__(
        self,
        num_drones=5,
        num_clusters=4,
        survivors_per_cluster=(2, 5),
        env_size=100.0,
        max_steps=2000,
        render_mode=None,
        drone_max_speed=2.0,
        drone_max_altitude=30.0,
        pyb_freq=60,
        seed=None,
    ):
        """
        Parameters
        ----------
        num_drones : int
            Number of drone agents in the swarm.
        num_clusters : int
            Number of survivor clusters.
        survivors_per_cluster : tuple(int, int)
            (min, max) survivors per cluster.
        env_size : float
            Environment size in meters (square).
        max_steps : int
            Maximum episode length.
        render_mode : str or None
            "human" for PyBullet GUI, None for headless.
        drone_max_speed : float
            Maximum drone speed in m/s.
        drone_max_altitude : float
            Maximum altitude cap.
        pyb_freq : int
            PyBullet simulation frequency in Hz.
        seed : int or None
            Random seed.
        """
        super().__init__()

        self.num_drones = num_drones
        self.env_size = env_size
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.drone_max_speed = drone_max_speed
        self.drone_max_altitude = drone_max_altitude
        self.pyb_freq = pyb_freq
        self._seed = seed

        # Agent naming
        self.possible_agents = [f"drone_{i}" for i in range(num_drones)]
        self.agents = list(self.possible_agents)

        # Subsystems (initialized in reset)
        self.terrain = Terrain(size_x=env_size, size_y=env_size, resolution=1.0, seed=seed)
        self.satellite = Satellite(sim_boundary_x=env_size)
        self.survivors = SurvivorCluster(
            num_clusters=num_clusters,
            survivors_per_cluster=survivors_per_cluster,
            env_size_x=env_size,
            env_size_y=env_size,
            terrain_obj=self.terrain,
            seed=seed,
        )
        self.weather = WeatherSystem(terrain=self.terrain, seed=seed)

        # State arrays
        self.drone_positions = np.zeros((num_drones, 3))
        self.drone_velocities = np.zeros((num_drones, 3))
        self.drone_orientations = np.zeros((num_drones, 3))  # roll, pitch, yaw
        self.battery_levels = np.ones(num_drones) * 100.0
        self.gateway_roles = np.zeros(num_drones, dtype=int)
        self.drone_alive = np.ones(num_drones, dtype=bool)

        # Tracking
        self.step_count = 0
        self.connected_survivors = 0
        self.prev_connected = 0
        self.newly_discovered = 0
        self.newly_discovered_clusters = []
        self.rssi_values = {}
        self.mesh_graph = None
        self.prev_discovered_count = 0

        # PyBullet
        self.client = None
        self.drone_ids = []
        self.terrain_body = None
        self.structure_ids = []

        # Observation & action spaces
        obs_dim = 25
        act_dim = 5

        self._observation_spaces = {
            agent: spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
            for agent in self.possible_agents
        }
        self._action_spaces = {
            agent: spaces.Box(low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32)
            for agent in self.possible_agents
        }

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self._observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self._action_spaces[agent]

    def _init_pybullet(self):
        """Initialize PyBullet physics server."""
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
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        p.setTimeStep(1.0 / self.pyb_freq, physicsClientId=self.client)

    def _build_terrain(self):
        """Build the terrain heightfield in PyBullet."""
        grid = self.terrain.grid_x
        size = self.env_size

        terrain_shape = p.createCollisionShape(
            p.GEOM_HEIGHTFIELD,
            meshScale=[size / grid, size / grid, 1.0],
            heightfieldData=self.terrain.heightmap.flatten().tolist(),
            numHeightfieldRows=grid,
            numHeightfieldColumns=grid,
            physicsClientId=self.client,
        )
        self.terrain_body = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=terrain_shape,
            physicsClientId=self.client,
        )
        p.changeVisualShape(
            self.terrain_body, -1,
            rgbaColor=[0.2, 0.65, 0.2, 1],
            physicsClientId=self.client,
        )
        self._spawn_structures()

    def _spawn_structures(self):
        """Draw the same AABBs the trainer uses as static PyBullet boxes."""
        self.structure_ids = []
        for box in self.terrain.structures:
            half = box.half_extents
            center = box.center
            col = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=half.tolist(),
                physicsClientId=self.client,
            )
            vis = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=half.tolist(),
                rgbaColor=list(box.rgba),
                physicsClientId=self.client,
            )
            body = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=col,
                baseVisualShapeIndex=vis,
                basePosition=center.tolist(),
                physicsClientId=self.client,
            )
            self.structure_ids.append(body)

    def _spawn_drones(self):
        """Spawn drone bodies in PyBullet."""
        self.drone_ids = []
        max_z = float(self.terrain.heightmap.max())
        if self.terrain.structures:
            max_z = max(max_z, max(box.zmax for box in self.terrain.structures))
        spawn_z = max_z + 5.0

        # Try to load Crazyflie URDF
        urdf_path = None
        try:
            gp = importlib.util.find_spec("gym_pybullet_drones")
            if gp is not None:
                assets_dir = os.path.join(os.path.dirname(gp.origin), "assets")
                urdf_path = os.path.join(assets_dir, "cf2x.urdf")
                if not os.path.exists(urdf_path):
                    urdf_path = None
        except Exception:
            urdf_path = None

        # Spread drones in a line formation above terrain
        spread = min(self.num_drones * 3.0, self.env_size * 0.3)
        for i in range(self.num_drones):
            x = (i - self.num_drones / 2) * (spread / self.num_drones)
            y = 0.0
            z = spawn_z
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
                    did = self._create_sphere_drone([x, y, z], i)
                finally:
                    os.chdir(original_cwd)
            else:
                did = self._create_sphere_drone([x, y, z], i)

            self.drone_ids.append(did)

        # Set initial wide camera for GUI to view entire 100m mountain region
        if self.render_mode == "human":
            p.resetDebugVisualizerCamera(
                cameraDistance=85,
                cameraYaw=45,
                cameraPitch=-35,
                cameraTargetPosition=[0, 0, 4.0],
                physicsClientId=self.client,
            )

    def _create_sphere_drone(self, pos, idx):
        """Fallback: create a sphere drone body."""
        colors = [
            [1, 0.2, 0.2, 1],
            [0.2, 0.5, 1, 1],
            [1, 0.8, 0, 1],
            [0.2, 1, 0.4, 1],
            [1, 0.4, 1, 1],
            [0.5, 0.5, 1, 1],
            [1, 0.6, 0.2, 1],
            [0.3, 1, 1, 1],
        ]
        color = colors[idx % len(colors)]
        col = p.createCollisionShape(p.GEOM_SPHERE, radius=0.4, physicsClientId=self.client)
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.4, rgbaColor=color, physicsClientId=self.client)
        did = p.createMultiBody(
            baseMass=0.027,  # 27g Crazyflie
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=pos,
            physicsClientId=self.client,
        )
        return did

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        if seed is not None:
            self._seed = seed

        self.agents = list(self.possible_agents)
        self.step_count = 0

        # Reset subsystems
        self.terrain = Terrain(
            size_x=self.env_size,
            size_y=self.env_size,
            resolution=1.0,
            seed=self._seed if self._seed is not None else 42,
        )
        self.satellite.reset()
        self.survivors.reset(seed=self._seed)
        self.survivors.place_cluster_near(0, [-18.0, 16.0], spread=3.0)
        self.survivors.keep_clear_of_structures(self.terrain.structures)
        self.weather.reset(seed=self._seed)

        # Reset drone state
        self.battery_levels = np.ones(self.num_drones) * 100.0
        self.gateway_roles = np.zeros(self.num_drones, dtype=int)
        # First drone starts as gateway
        if self.num_drones > 0:
            self.gateway_roles[0] = 1
        self.drone_alive = np.ones(self.num_drones, dtype=bool)
        self.drone_velocities = np.zeros((self.num_drones, 3))
        self.drone_orientations = np.zeros((self.num_drones, 3))

        self.connected_survivors = 0
        self.prev_connected = 0
        self.newly_discovered = 0
        self.newly_discovered_clusters = []
        self.rssi_values = {}
        self.mesh_graph = None
        self.prev_discovered_count = 0

        # Initialize PyBullet
        self._init_pybullet()
        self._build_terrain()
        self._spawn_drones()

        observations = self._get_observations()
        infos = {agent: {} for agent in self.agents}

        return observations, infos

    def step(self, actions):
        """
        Execute one environment step.

        Parameters
        ----------
        actions : dict
            {agent_name: action_array} for each alive agent.

        Returns
        -------
        observations, rewards, terminations, truncations, infos
        """
        self.step_count += 1
        dt = 1.0 / self.pyb_freq

        # --- 1. Process actions ---
        for i, agent in enumerate(self.possible_agents):
            if not self.drone_alive[i]:
                continue
            if agent not in actions:
                continue

            act = np.array(actions[agent], dtype=np.float32)
            act = np.clip(act, -1.0, 1.0)

            # Velocity command
            vel_cmd = act[0:3] * self.drone_max_speed

            # Role toggle
            self.gateway_roles[i] = 1 if act[3] > 0.0 else 0

            # Communication power (affects range — future use)
            # comm_power = (act[4] + 1.0) / 2.0  # Map [-1,1] to [0,1]

            # Apply velocity
            self.drone_velocities[i] = vel_cmd

        # --- 2. Update weather ---
        self.weather.step(dt)

        # --- 3. Physics update ---
        for i in range(self.num_drones):
            if not self.drone_alive[i]:
                continue

            pos = self.drone_positions[i]

            # Get wind at drone position
            wind = self.weather.get_wind_at(pos[0], pos[1], pos[2])

            # Apply wind as velocity perturbation (scaled for tiny drone)
            wind_effect = wind * 10.0  # Scale up for effect on velocity

            # Update position with velocity + wind
            new_pos = pos + (self.drone_velocities[i] + wind_effect) * dt

            # Altitude clamping
            terrain_z = self._get_terrain_height(new_pos[0], new_pos[1])
            new_pos[2] = np.clip(new_pos[2], terrain_z + 0.5, self.drone_max_altitude)

            # Stay inset on the map (do not allow sitting on / past the rim)
            xy_lim = max(5.0, self.env_size / 2.0 - 8.0)
            vel = self.drone_velocities[i].copy()
            if new_pos[0] > xy_lim:
                new_pos[0] = xy_lim
                vel[0] = -abs(self.drone_max_speed) * 0.4
            elif new_pos[0] < -xy_lim:
                new_pos[0] = -xy_lim
                vel[0] = abs(self.drone_max_speed) * 0.4
            if new_pos[1] > xy_lim:
                new_pos[1] = xy_lim
                vel[1] = -abs(self.drone_max_speed) * 0.4
            elif new_pos[1] < -xy_lim:
                new_pos[1] = -xy_lim
                vel[1] = abs(self.drone_max_speed) * 0.4
            self.drone_velocities[i] = vel
            self.drone_positions[i] = new_pos

            # Update PyBullet body
            orn = p.getQuaternionFromEuler(self.drone_orientations[i].tolist())
            p.resetBasePositionAndOrientation(
                self.drone_ids[i], new_pos.tolist(), orn,
                physicsClientId=self.client,
            )
            p.resetBaseVelocity(
                self.drone_ids[i], [0, 0, 0], [0, 0, 0], physicsClientId=self.client
            )

            # Battery drain
            drain = self.weather.compute_battery_drain(
                wind, pos[2], bool(self.gateway_roles[i]), dt
            )
            self.battery_levels[i] -= drain

            if self.battery_levels[i] <= 0:
                self.battery_levels[i] = 0
                self.drone_alive[i] = False

        # Step PyBullet simulation
        p.stepSimulation(physicsClientId=self.client)

        # --- 4. Update satellite ---
        self.satellite.update_position(dt)

        # --- 5. Update survivors ---
        self.survivors.step(dt)

        # --- 6. Discovery check ---
        alive_positions = [
            self.drone_positions[i] for i in range(self.num_drones) if self.drone_alive[i]
        ]
        self.prev_discovered_count = int(self.survivors.discovered.sum())
        self.newly_discovered, self.newly_discovered_clusters = (
            self.survivors.update_discovery(alive_positions, self.terrain)
        )

        # --- 7. Network connectivity ---
        self.prev_connected = self.connected_survivors
        self._compute_network()

        # --- 8. Collision detection ---
        terrain_collisions = self._check_terrain_collisions()
        drone_collisions = self._check_drone_collisions()

        # --- 9. Compute outputs ---
        rewards = self._compute_rewards(terrain_collisions, drone_collisions)
        terminations = self._compute_terminations()
        truncations = self._compute_truncations()
        observations = self._get_observations()
        infos = self._compute_infos()

        # Remove dead agents
        self.agents = [
            agent for i, agent in enumerate(self.possible_agents)
            if self.drone_alive[i]
        ]

        return observations, rewards, terminations, truncations, infos

    def _get_terrain_height(self, x, y):
        """Get terrain height at world coordinates (x, y)."""
        gx = int((x + self.env_size / 2) / self.terrain.resolution)
        gy = int((y + self.env_size / 2) / self.terrain.resolution)
        gx = np.clip(gx, 0, self.terrain.grid_x - 1)
        gy = np.clip(gy, 0, self.terrain.grid_y - 1)
        return self.terrain.heightmap[gy, gx]

    def _compute_network(self):
        """Compute mesh network connectivity using network.py."""
        alive_positions = []
        alive_roles = []
        alive_indices = []
        for i in range(self.num_drones):
            if self.drone_alive[i]:
                alive_positions.append(self.drone_positions[i])
                alive_roles.append(self.gateway_roles[i])
                alive_indices.append(i)

        if len(alive_positions) == 0:
            self.connected_survivors = 0
            self.rssi_values = {}
            self.mesh_graph = None
            return

        N = len(alive_positions)
        los_matrix = np.zeros((N, N), dtype=bool)
        for i in range(N):
            for j in range(i + 1, N):
                los = self.terrain.check_los(alive_positions[i], alive_positions[j])
                los_matrix[i, j] = los
                los_matrix[j, i] = los

        surv_positions = self.survivors.get_positions()

        self.connected_survivors, self.rssi_values, self.mesh_graph = calculate_throughput(
            alive_positions,
            surv_positions,
            np.array(alive_roles),
            los_matrix,
            self.satellite,
            terrain=self.terrain,
        )

    def _check_terrain_collisions(self):
        """Check which drones have collided with terrain."""
        collisions = np.zeros(self.num_drones, dtype=bool)
        for i in range(self.num_drones):
            if not self.drone_alive[i]:
                continue
            pos = self.drone_positions[i]
            terrain_z = self._get_terrain_height(pos[0], pos[1])
            if pos[2] <= terrain_z + 0.2:
                collisions[i] = True
            elif point_hits_structure(pos, self.terrain.structures, margin=0.15):
                collisions[i] = True
        return collisions

    def _check_drone_collisions(self):
        """Check which drones have collided with each other."""
        collisions = np.zeros(self.num_drones, dtype=bool)
        collision_dist = 1.0  # meters
        for i in range(self.num_drones):
            if not self.drone_alive[i]:
                continue
            for j in range(i + 1, self.num_drones):
                if not self.drone_alive[j]:
                    continue
                dist = np.linalg.norm(self.drone_positions[i] - self.drone_positions[j])
                if dist < collision_dist:
                    collisions[i] = True
                    collisions[j] = True
        return collisions

    def _compute_rewards(self, terrain_collisions, drone_collisions):
        """
        Compute rewards for all agents.

        Reward Table:
          +20  first-time cluster discovery
          +5   each survivor connected to SAT per step
          +100 all survivors found (episode bonus)
          -50  terrain collision
          -30  drone-drone collision
          -40  battery depletion (drone dies)
          -2/step  low battery warning (< 20%)
          -5   broken mesh link (was connected, now isn't)
          -3/step  no satellite uplink when one could exist
          +1   energy efficiency (battery > 50%)
          -1/step  altitude penalty (too high)
        """
        rewards = {}
        total_survivors = self.survivors.num_survivors
        current_discovered = int(self.survivors.discovered.sum())
        all_found = current_discovered == total_survivors and total_survivors > 0

        for i, agent in enumerate(self.possible_agents):
            rew = 0.0

            if not self.drone_alive[i]:
                # Dead drones get the death penalty only on the step they die
                if self.battery_levels[i] <= 0:
                    rew -= 40.0
                rewards[agent] = rew
                continue

            # --- Positive rewards ---

            # Cluster discovery bonus (shared across all agents)
            rew += 20.0 * len(self.newly_discovered_clusters)

            # Per-step connectivity reward
            rew += 5.0 * self.connected_survivors

            # All survivors found (one-time episode bonus)
            if all_found:
                rew += 100.0

            # Energy efficiency bonus
            if self.battery_levels[i] > 50.0:
                rew += 1.0

            # --- Negative rewards ---

            # Terrain collision
            if terrain_collisions[i]:
                rew -= 50.0

            # Drone-drone collision
            if drone_collisions[i]:
                rew -= 30.0

            # Low battery warning
            if self.battery_levels[i] < 20.0:
                rew -= 2.0

            # Broken mesh links (connectivity dropped)
            if self.connected_survivors < self.prev_connected:
                rew -= 5.0 * (self.prev_connected - self.connected_survivors)

            # No satellite uplink penalty
            has_any_gateway_uplink = any(
                self.gateway_roles[j] and self.satellite.can_uplink(
                    self.drone_positions[j][0],
                    self.drone_positions[j][1],
                    self.drone_positions[j][2],
                )
                for j in range(self.num_drones)
                if self.drone_alive[j]
            )
            if not has_any_gateway_uplink and self.satellite.visible:
                rew -= 3.0

            # Altitude penalty (too high wastes energy)
            terrain_z = self._get_terrain_height(
                self.drone_positions[i][0], self.drone_positions[i][1]
            )
            altitude_agl = self.drone_positions[i][2] - terrain_z
            if altitude_agl > 20.0:
                rew -= 1.0

            rewards[agent] = rew

        return rewards

    def _compute_terminations(self):
        """
        Compute termination conditions.
        - Individual: drone dies when battery = 0
        - Global: all drones dead
        """
        terminations = {}
        all_dead = not np.any(self.drone_alive)

        for i, agent in enumerate(self.possible_agents):
            if all_dead:
                terminations[agent] = True
            elif not self.drone_alive[i]:
                terminations[agent] = True
            else:
                terminations[agent] = False

        return terminations

    def _compute_truncations(self):
        """
        Compute truncation conditions.
        - Max steps exceeded
        - All survivors found and connected
        """
        truncations = {}
        total_survivors = self.survivors.num_survivors
        current_discovered = int(self.survivors.discovered.sum())
        all_found_and_connected = (
            current_discovered == total_survivors
            and self.connected_survivors == total_survivors
            and total_survivors > 0
        )
        max_steps_reached = self.step_count >= self.max_steps

        for agent in self.possible_agents:
            truncations[agent] = all_found_and_connected or max_steps_reached

        return truncations

    def _get_observations(self):
        """Compute observation for each agent."""
        observations = {}
        surv_positions = self.survivors.get_positions()
        undiscovered_mask = ~self.survivors.discovered

        for i, agent in enumerate(self.possible_agents):
            obs = np.zeros(25, dtype=np.float32)

            if not self.drone_alive[i]:
                observations[agent] = obs
                continue

            pos = self.drone_positions[i]
            vel = self.drone_velocities[i]
            orn = self.drone_orientations[i]

            # [0:3] Position (normalized by env size)
            obs[0:3] = pos / (self.env_size / 2)

            # [3:6] Velocity (normalized by max speed)
            obs[3:6] = vel / self.drone_max_speed

            # [6:9] Orientation
            obs[6:9] = orn / np.pi

            # [9:12] Wind vector
            wind = self.weather.get_wind_at(pos[0], pos[1], pos[2])
            obs[9:12] = wind * 10.0  # Scale up for visibility in obs

            # [12:15] Satellite relative position (normalized)
            sat_rel = self.satellite.get_relative_position(pos[0], pos[1], pos[2])
            obs[12:14] = sat_rel[:2] / (self.env_size / 2)
            obs[14] = np.clip(sat_rel[2] / 550_000.0, -1, 1)

            # [15] Satellite visible
            obs[15] = 1.0 if self.satellite.visible else 0.0

            # [16] Battery level (0-1)
            obs[16] = self.battery_levels[i] / 100.0

            # [17] Role
            obs[17] = float(self.gateway_roles[i])

            # [18:21] Nearest undiscovered survivor relative pos
            if undiscovered_mask.any():
                undiscovered_pos = surv_positions[undiscovered_mask]
                diffs = undiscovered_pos - pos
                dists = np.linalg.norm(diffs, axis=1)
                nearest_idx = np.argmin(dists)
                obs[18:21] = diffs[nearest_idx] / (self.env_size / 2)
            else:
                obs[18:21] = 0.0

            # [21] Discovery rate
            obs[21] = self.survivors.get_discovery_stats()["discovery_rate"]

            # [22] Nearest neighbor drone distance
            min_dist = float("inf")
            min_rssi = -100.0
            for j in range(self.num_drones):
                if j == i or not self.drone_alive[j]:
                    continue
                d = np.linalg.norm(self.drone_positions[i] - self.drone_positions[j])
                if d < min_dist:
                    min_dist = d
                    # Estimate RSSI
                    _, rssi = calculate_link(
                        self.drone_positions[i], self.drone_positions[j], True
                    )
                    min_rssi = rssi

            obs[22] = np.clip(min_dist / self.env_size, 0, 1) if min_dist < float("inf") else 1.0

            # [23] Nearest neighbor RSSI (normalized: -100 → 0, 0 → 1)
            obs[23] = np.clip((min_rssi + 100) / 100, 0, 1)

            # [24] Terrain height below drone (normalized)
            terrain_z = self._get_terrain_height(pos[0], pos[1])
            obs[24] = terrain_z / float(self.terrain.heightmap.max() + 1e-6)

            observations[agent] = obs

        return observations

    def _compute_infos(self):
        """Compute info dict for each agent."""
        stats = self.survivors.get_discovery_stats()
        info = {
            "step": self.step_count,
            "connected_survivors": self.connected_survivors,
            "satellite_visible": self.satellite.visible,
            "storm_active": self.weather.is_storm_active(),
            **stats,
        }
        return {agent: dict(info) for agent in self.possible_agents}

    def _render_overlays(self):
        """Draw debug overlays in PyBullet GUI."""
        # Remove old debug items
        p.removeAllUserDebugItems(physicsClientId=self.client)

        # Draw mesh links
        if self.mesh_graph is not None:
            for u, v in self.mesh_graph.edges():
                if u == "SAT" or v == "SAT":
                    continue
                u_data = self.mesh_graph.nodes[u]
                v_data = self.mesh_graph.nodes[v]
                if "pos" in u_data and "pos" in v_data:
                    p.addUserDebugLine(
                        u_data["pos"].tolist(),
                        v_data["pos"].tolist(),
                        lineColorRGB=[0, 1, 1],
                        lineWidth=1.5,
                        physicsClientId=self.client,
                    )

        # Color drones by role
        for i in range(self.num_drones):
            if not self.drone_alive[i]:
                color = [0.3, 0.3, 0.3, 1]  # Gray = dead
            elif self.gateway_roles[i]:
                color = [1, 0.2, 0.2, 1]  # Red = gateway
            else:
                color = [0.2, 0.5, 1, 1]  # Blue = relay
            p.changeVisualShape(
                self.drone_ids[i], -1,
                rgbaColor=color,
                physicsClientId=self.client,
            )

            # Battery bar
            pos = self.drone_positions[i]
            batt = self.battery_levels[i]
            label = f"D{i} B:{batt:.0f}% {'GW' if self.gateway_roles[i] else 'RL'}"
            p.addUserDebugText(
                label,
                [pos[0], pos[1], pos[2] + 1.5],
                textColorRGB=[1, 1, 1],
                textSize=1.0,
                physicsClientId=self.client,
            )

        # Draw survivor positions
        surv_pos = self.survivors.get_positions()
        for i in range(self.survivors.num_survivors):
            color = [0, 1, 0] if self.survivors.discovered[i] else [1, 0, 0]
            p.addUserDebugText(
                "★" if self.survivors.discovered[i] else "?",
                surv_pos[i].tolist(),
                textColorRGB=color,
                textSize=1.5,
                physicsClientId=self.client,
            )

        # Satellite indicator
        if self.satellite.visible:
            p.addUserDebugText(
                f"🛰 SAT x={self.satellite.current_x:.0f}",
                [0, 0, self.drone_max_altitude + 3],
                textColorRGB=[1, 1, 0],
                textSize=1.2,
                physicsClientId=self.client,
            )
        else:
            p.addUserDebugText(
                f"SAT BLACKOUT {self.satellite.blackout_timer:.0f}s",
                [0, 0, self.drone_max_altitude + 3],
                textColorRGB=[1, 0.3, 0.3],
                textSize=1.2,
                physicsClientId=self.client,
            )

    def render(self):
        """Render the environment (handled by PyBullet in 'human' mode)."""
        if self.render_mode == "human":
            import time
            time.sleep(1.0 / self.pyb_freq)

    def close(self):
        """Clean up PyBullet."""
        if self.client is not None:
            try:
                p.disconnect(physicsClientId=self.client)
            except Exception:
                pass
            self.client = None
