"""
survivors.py — Clustered ground survivors with drift models and discovery state.

Survivors spawn in clusters on walkable streets/lots (not inside building
footprints) and drift while staying on the walkable mask.
"""

import numpy as np


class SurvivorCluster:
    """Manages multiple clusters of ground survivors in the simulation."""

    def __init__(
        self,
        num_clusters=4,
        survivors_per_cluster=(2, 6),
        env_size_x=250,
        env_size_y=250,
        terrain_obj=None,
        discovery_range=25.0,
        cluster_spread=8.0,
        drift_strength=0.3,
        seed=None,
        town=None,
    ):
        """
        Parameters
        ----------
        num_clusters : int
            Number of survivor clusters to spawn.
        survivors_per_cluster : tuple(int, int)
            (min, max) survivors per cluster.
        env_size_x, env_size_y : float
            Environment dimensions in meters.
        terrain_obj : Terrain or None
            Terrain object for snapping z-coords and drift bias.
        discovery_range : float
            Maximum distance (meters) for a drone to discover a survivor.
        cluster_spread : float
            Standard deviation (meters) for Gaussian cluster spawning.
        drift_strength : float
            Strength of the downhill drift bias.
        seed : int or None
            Random seed for reproducibility.
        town : TownLayout or None
            Town layout providing walkable streets/lots.
        """
        self.num_clusters = num_clusters
        self.survivors_per_cluster = survivors_per_cluster
        self.env_size_x = env_size_x
        self.env_size_y = env_size_y
        self.terrain = terrain_obj
        self.discovery_range = discovery_range
        self.cluster_spread = cluster_spread
        self.drift_strength = drift_strength
        self.town = town

        self.rng = np.random.default_rng(seed)

        self.cluster_centers = []
        self.cluster_ids = []
        self.positions = np.zeros((0, 3), dtype=np.float64)
        self.num_survivors = 0
        self.discovered = np.zeros(0, dtype=bool)
        self.discovered_clusters = np.zeros(num_clusters, dtype=bool)

        self.patrol_waypoints = {}
        self.patrol_indices = {}
        self.patrol_speed = 0.5

        self._spawn_clusters()

    def set_town(self, town):
        """Attach / replace town walkable mask (call before reset preferred)."""
        self.town = town

    def _is_walkable(self, x, y):
        if self.town is None:
            return True
        return self.town.is_walkable(x, y)

    def _sample_walkable_point(self, prefer_street=True, near=None, spread=8.0, max_tries=80):
        """Sample one walkable (x, y), optionally near a center."""
        half_x = self.env_size_x / 2 * 0.9
        half_y = self.env_size_y / 2 * 0.9

        if near is not None:
            for _ in range(max_tries):
                sx = near[0] + self.rng.normal(0, min(spread, 4.0))
                sy = near[1] + self.rng.normal(0, min(spread, 4.0))
                sx = float(np.clip(sx, -half_x, half_x))
                sy = float(np.clip(sy, -half_y, half_y))
                if self._is_walkable(sx, sy):
                    return sx, sy

        if self.town is not None:
            pts = self.town.sample_walkable(self.rng, n=1, prefer_street=prefer_street)
            return float(pts[0, 0]), float(pts[0, 1])

        return float(self.rng.uniform(-half_x, half_x)), float(self.rng.uniform(-half_y, half_y))

    def _spawn_clusters(self):
        """Spawn clusters on walkable streets/lots only."""
        self.cluster_centers = []
        self.cluster_ids = []
        positions_list = []

        for c in range(self.num_clusters):
            cx, cy = self._sample_walkable_point(prefer_street=True)
            self.cluster_centers.append(np.array([cx, cy]))

            n_surv = self.rng.integers(
                self.survivors_per_cluster[0], self.survivors_per_cluster[1] + 1
            )
            for _ in range(n_surv):
                sx, sy = self._sample_walkable_point(
                    prefer_street=True, near=(cx, cy), spread=self.cluster_spread
                )
                positions_list.append([sx, sy, 0.0])
                self.cluster_ids.append(c)

        self.num_survivors = len(positions_list)
        self.positions = np.array(positions_list, dtype=np.float64) if positions_list else np.zeros((0, 3))
        self.cluster_ids = np.array(self.cluster_ids, dtype=int)
        self.discovered = np.zeros(self.num_survivors, dtype=bool)
        self.discovered_clusters = np.zeros(self.num_clusters, dtype=bool)
        self._snap_to_ground()

    def _snap_to_ground(self):
        """Set z coordinate based on terrain heightmap."""
        if self.terrain is None:
            self.positions[:, 2] = 0.0
            return

        for i in range(self.num_survivors):
            x, y = self.positions[i, 0], self.positions[i, 1]
            grid_x = int((x + self.env_size_x / 2) / self.terrain.resolution)
            grid_y = int((y + self.env_size_y / 2) / self.terrain.resolution)

            grid_x = np.clip(grid_x, 0, self.terrain.grid_x - 1)
            grid_y = np.clip(grid_y, 0, self.terrain.grid_y - 1)

            self.positions[i, 2] = self.terrain.heightmap[grid_y, grid_x] + 0.3

    def _get_terrain_gradient(self, x, y):
        """Downhill direction from heightmap (weak on flat dusty ground)."""
        if self.terrain is None:
            return np.array([0.0, 0.0])

        gx = int((x + self.env_size_x / 2) / self.terrain.resolution)
        gy = int((y + self.env_size_y / 2) / self.terrain.resolution)
        gx = np.clip(gx, 1, self.terrain.grid_x - 2)
        gy = np.clip(gy, 1, self.terrain.grid_y - 2)

        dzdx = (
            self.terrain.heightmap[gy, gx + 1] - self.terrain.heightmap[gy, gx - 1]
        ) / (2 * self.terrain.resolution)
        dzdy = (
            self.terrain.heightmap[gy + 1, gx] - self.terrain.heightmap[gy - 1, gx]
        ) / (2 * self.terrain.resolution)

        return -np.array([dzdx, dzdy])

    def set_patrol_waypoints(self, cluster_id, waypoints):
        """Set a hidden waypoint patrol path for a cluster."""
        self.patrol_waypoints[cluster_id] = [np.array(wp) for wp in waypoints]
        self.patrol_indices[cluster_id] = 0

    def step(self, dt):
        """
        Advance survivor positions. Drift stays on the walkable mask so
        survivors do not slide through building footprints.
        """
        for i in range(self.num_survivors):
            cluster_id = self.cluster_ids[i]
            x, y = self.positions[i, 0], self.positions[i, 1]

            noise = self.rng.normal(0, 0.5 * dt, size=2)
            grad = self._get_terrain_gradient(x, y)
            drift = self.drift_strength * grad * dt

            patrol_pull = np.zeros(2)
            if cluster_id in self.patrol_waypoints:
                waypoints = self.patrol_waypoints[cluster_id]
                idx = self.patrol_indices[cluster_id]
                target = waypoints[idx]
                direction = target - np.array([x, y])
                dist = np.linalg.norm(direction)
                if dist < 2.0:
                    self.patrol_indices[cluster_id] = (idx + 1) % len(waypoints)
                elif dist > 0:
                    patrol_pull = (direction / dist) * self.patrol_speed * dt

            nx = x + noise[0] + drift[0] + patrol_pull[0]
            ny = y + noise[1] + drift[1] + patrol_pull[1]
            nx = float(np.clip(nx, -self.env_size_x / 2, self.env_size_x / 2))
            ny = float(np.clip(ny, -self.env_size_y / 2, self.env_size_y / 2))

            # Stay on walkable streets/lots
            if self._is_walkable(nx, ny):
                self.positions[i, 0] = nx
                self.positions[i, 1] = ny
            else:
                # Try axis-separated slides along streets
                if self._is_walkable(nx, y):
                    self.positions[i, 0] = nx
                if self._is_walkable(self.positions[i, 0], ny):
                    self.positions[i, 1] = ny

        self._snap_to_ground()

    def update_discovery(self, drone_positions, terrain):
        """Discover survivors within range with LoS."""
        newly_discovered = 0
        newly_discovered_clusters = []

        for i in range(self.num_survivors):
            if self.discovered[i]:
                continue

            surv_pos = self.positions[i]
            for drone_pos in drone_positions:
                dist = np.linalg.norm(surv_pos[:2] - np.array(drone_pos[:2]))
                if dist <= self.discovery_range:
                    if terrain is not None and terrain.check_los(drone_pos, surv_pos):
                        self.discovered[i] = True
                        newly_discovered += 1

                        cid = self.cluster_ids[i]
                        if not self.discovered_clusters[cid]:
                            self.discovered_clusters[cid] = True
                            newly_discovered_clusters.append(cid)
                        break

        return newly_discovered, newly_discovered_clusters

    def get_positions(self):
        return self.positions.copy()

    def get_discovered_positions(self):
        return self.positions[self.discovered].copy()

    def get_discovery_stats(self):
        return {
            "total_survivors": self.num_survivors,
            "discovered_survivors": int(self.discovered.sum()),
            "total_clusters": self.num_clusters,
            "discovered_clusters": int(self.discovered_clusters.sum()),
            "discovery_rate": float(self.discovered.sum()) / max(self.num_survivors, 1),
        }

    def reset(self, seed=None, town=None):
        """Reset all survivors to new random clustered walkable positions."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if town is not None:
            self.town = town

        for cid in self.patrol_indices:
            self.patrol_indices[cid] = 0

        self._spawn_clusters()
