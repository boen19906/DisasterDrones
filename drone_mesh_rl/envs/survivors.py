"""
survivors.py — Clustered ground survivors with drift models and discovery state.

Survivors spawn in clusters (Gaussian blobs) and drift over time via:
  - Biased random walk toward lower terrain (valleys)
  - Optional SAR patrol mode (hidden waypoint path)

Each survivor tracks a `discovered` flag. Mesh mode requires range AND LoS;
search v1 uses fly-over range only.
"""

import numpy as np

from .structures import push_xy_out_of_structures


class SurvivorCluster:
    """Manages multiple clusters of ground survivors in the simulation."""

    def __init__(
        self,
        num_clusters=4,
        survivors_per_cluster=(2, 6),
        env_size_x=100,
        env_size_y=100,
        terrain_obj=None,
        discovery_range=15.0,
        cluster_spread=8.0,
        drift_strength=0.3,
        require_los=True,
        frozen=False,
        seed=None,
        num_loners=0,
        spawn_frac=0.42,
        clip_xy=None,
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
        """
        self.num_clusters = num_clusters
        self.survivors_per_cluster = survivors_per_cluster
        self.env_size_x = env_size_x
        self.env_size_y = env_size_y
        self.terrain = terrain_obj
        self.discovery_range = discovery_range
        self.cluster_spread = cluster_spread
        self.drift_strength = drift_strength
        self.require_los = require_los
        self.frozen = frozen
        self.num_loners = int(max(0, num_loners))
        self.spawn_frac = float(spawn_frac)

        self.rng = np.random.default_rng(seed)

        # Generate clusters
        self.cluster_centers = []
        self.cluster_ids = []  # which cluster each survivor belongs to
        positions_list = []

        spawn_bound_x = env_size_x * self.spawn_frac
        spawn_bound_y = env_size_y * self.spawn_frac
        max_reach_x = env_size_x * 0.45
        max_reach_y = env_size_y * 0.45
        if clip_xy is not None:
            max_reach_x = min(max_reach_x, float(clip_xy[0]))
            max_reach_y = min(max_reach_y, float(clip_xy[1]))
            spawn_bound_x = min(spawn_bound_x, max_reach_x)
            spawn_bound_y = min(spawn_bound_y, max_reach_y)
        spread = max(2.0, float(cluster_spread))

        for c in range(num_clusters):
            cx = self.rng.uniform(-spawn_bound_x, spawn_bound_x)
            cy = self.rng.uniform(-spawn_bound_y, spawn_bound_y)
            self.cluster_centers.append(np.array([cx, cy]))

            n_surv = self.rng.integers(
                survivors_per_cluster[0], survivors_per_cluster[1] + 1
            )
            for _ in range(n_surv):
                sx = cx + self.rng.normal(0, spread)
                sy = cy + self.rng.normal(0, spread)
                sx = np.clip(sx, -max_reach_x, max_reach_x)
                sy = np.clip(sy, -max_reach_y, max_reach_y)
                positions_list.append([sx, sy, 0.0])
                self.cluster_ids.append(c)

        loner_id0 = num_clusters
        for k in range(self.num_loners):
            sx = self.rng.uniform(-max_reach_x, max_reach_x)
            sy = self.rng.uniform(-max_reach_y, max_reach_y)
            positions_list.append([sx, sy, 0.0])
            self.cluster_ids.append(loner_id0 + k)
        self.num_clusters = num_clusters + self.num_loners

        self.num_survivors = len(positions_list)
        self.positions = np.array(positions_list, dtype=np.float64)
        self.cluster_ids = np.array(self.cluster_ids, dtype=int)

        # Discovery state
        self.discovered = np.zeros(self.num_survivors, dtype=bool)
        self.discovered_clusters = np.zeros(self.num_clusters, dtype=bool)

        # SAR patrol waypoints (optional, per-cluster)
        self.patrol_waypoints = {}  # cluster_id -> list of [x, y] waypoints
        self.patrol_indices = {}    # cluster_id -> current waypoint index
        self.patrol_speed = 0.5     # m/s for patrol movement

        self._snap_to_ground()
        if self.terrain is not None and getattr(self.terrain, "structures", None):
            self.keep_clear_of_structures(self.terrain.structures)

    def keep_clear_of_structures(self, boxes):
        """Push survivors out of solid building / rubble footprints."""
        if not boxes:
            return
        for i in range(self.num_survivors):
            x, y = push_xy_out_of_structures(
                self.positions[i, 0], self.positions[i, 1], boxes
            )
            self.positions[i, 0] = x
            self.positions[i, 1] = y
        self._snap_to_ground()

    def place_cluster_near(self, cluster_id, xy, spread=3.0):
        """Move one cluster (e.g. into the school courtyard after a quake)."""
        cx, cy = xy
        mask = self.cluster_ids == cluster_id
        n = int(mask.sum())
        if n == 0:
            return
        offsets = self.rng.normal(0.0, spread, size=(n, 2))
        self.positions[mask, 0] = cx + offsets[:, 0]
        self.positions[mask, 1] = cy + offsets[:, 1]
        self.cluster_centers[cluster_id] = np.array([cx, cy], dtype=float)
        self._snap_to_ground()

    def _snap_to_ground(self):
        """Set z coordinate based on terrain heightmap (floating slightly above terrain mesh)."""
        if self.terrain is None:
            self.positions[:, 2] = 0.0
            return

        for i in range(self.num_survivors):
            x, y = self.positions[i, 0], self.positions[i, 1]
            grid_x = int((x + self.env_size_x / 2) / self.terrain.resolution)
            grid_y = int((y + self.env_size_y / 2) / self.terrain.resolution)

            grid_x = np.clip(grid_x, 0, self.terrain.grid_x - 1)
            grid_y = np.clip(grid_y, 0, self.terrain.grid_y - 1)

            # Float 0.3m above the terrain surface so the marker is clearly visible
            self.positions[i, 2] = self.terrain.heightmap[grid_y, grid_x] + 0.3

    def _get_terrain_gradient(self, x, y):
        """
        Compute the terrain gradient at (x, y) — points downhill.
        Returns a 2D vector pointing in the direction of steepest descent.
        """
        if self.terrain is None:
            return np.array([0.0, 0.0])

        gx = int((x + self.env_size_x / 2) / self.terrain.resolution)
        gy = int((y + self.env_size_y / 2) / self.terrain.resolution)
        gx = np.clip(gx, 1, self.terrain.grid_x - 2)
        gy = np.clip(gy, 1, self.terrain.grid_y - 2)

        # Central difference gradient
        dzdx = (
            self.terrain.heightmap[gy, gx + 1] - self.terrain.heightmap[gy, gx - 1]
        ) / (2 * self.terrain.resolution)
        dzdy = (
            self.terrain.heightmap[gy + 1, gx] - self.terrain.heightmap[gy - 1, gx]
        ) / (2 * self.terrain.resolution)

        # Negative gradient = downhill direction
        return -np.array([dzdx, dzdy])

    def set_patrol_waypoints(self, cluster_id, waypoints):
        """
        Set a hidden waypoint patrol path for a cluster.

        Parameters
        ----------
        cluster_id : int
            Which cluster follows this path.
        waypoints : list of [x, y]
            Ordered waypoint positions.
        """
        self.patrol_waypoints[cluster_id] = [np.array(wp) for wp in waypoints]
        self.patrol_indices[cluster_id] = 0

    def step(self, dt):
        """
        Advance survivor positions by one timestep.

        Movement model:
          1. Brownian noise (random walk)
          2. Downhill drift bias (survivors tend toward valleys)
          3. SAR patrol following (if waypoints set for the cluster)
        """
        if self.frozen:
            self._snap_to_ground()
            return

        for i in range(self.num_survivors):
            cluster_id = self.cluster_ids[i]
            x, y = self.positions[i, 0], self.positions[i, 1]

            # 1. Brownian noise
            noise = self.rng.normal(0, 0.5 * dt, size=2)

            # 2. Downhill drift
            grad = self._get_terrain_gradient(x, y)
            drift = self.drift_strength * grad * dt

            # 3. SAR patrol (if active for this cluster)
            patrol_pull = np.zeros(2)
            if cluster_id in self.patrol_waypoints:
                waypoints = self.patrol_waypoints[cluster_id]
                idx = self.patrol_indices[cluster_id]
                target = waypoints[idx]
                direction = target - np.array([x, y])
                dist = np.linalg.norm(direction)
                if dist < 2.0:
                    # Advance to next waypoint (loop)
                    self.patrol_indices[cluster_id] = (idx + 1) % len(waypoints)
                elif dist > 0:
                    patrol_pull = (direction / dist) * self.patrol_speed * dt

            # Combine movement
            self.positions[i, 0] += noise[0] + drift[0] + patrol_pull[0]
            self.positions[i, 1] += noise[1] + drift[1] + patrol_pull[1]

        # Clamp to bounds
        self.positions[:, 0] = np.clip(
            self.positions[:, 0], -self.env_size_x / 2, self.env_size_x / 2
        )
        self.positions[:, 1] = np.clip(
            self.positions[:, 1], -self.env_size_y / 2, self.env_size_y / 2
        )

        self._snap_to_ground()
        if self.terrain is not None and getattr(self.terrain, "structures", None):
            self.keep_clear_of_structures(self.terrain.structures)

    def update_discovery(self, drone_positions, terrain):
        """
        Check which survivors are within discovery range of any drone
        and have line-of-sight. Returns count of newly discovered survivors.

        Parameters
        ----------
        drone_positions : list of array-like [x, y, z]
        terrain : Terrain object with check_los()

        Returns
        -------
        newly_discovered : int
            Number of survivors discovered this step.
        newly_discovered_clusters : list of int
            Cluster IDs discovered for the first time this step.
        """
        newly_discovered = 0
        newly_discovered_clusters = []

        for i in range(self.num_survivors):
            if self.discovered[i]:
                continue

            surv_pos = self.positions[i]
            for drone_pos in drone_positions:
                dist = np.linalg.norm(surv_pos[:2] - np.array(drone_pos[:2]))
                if dist <= self.discovery_range:
                    los_ok = True
                    if self.require_los and terrain is not None:
                        los_ok = terrain.check_los(drone_pos, surv_pos)
                    if los_ok:
                        self.discovered[i] = True
                        newly_discovered += 1

                        cid = self.cluster_ids[i]
                        if not self.discovered_clusters[cid]:
                            self.discovered_clusters[cid] = True
                            newly_discovered_clusters.append(cid)
                        break

        return newly_discovered, newly_discovered_clusters

    def get_positions(self):
        """Return numpy array of [x, y, z] coordinates."""
        return self.positions.copy()

    def get_discovered_positions(self):
        """Return positions of only discovered survivors."""
        return self.positions[self.discovered].copy()

    def get_discovery_stats(self):
        """Return discovery statistics."""
        return {
            "total_survivors": self.num_survivors,
            "discovered_survivors": int(self.discovered.sum()),
            "total_clusters": self.num_clusters,
            "discovered_clusters": int(self.discovered_clusters.sum()),
            "discovery_rate": float(self.discovered.sum()) / max(self.num_survivors, 1),
        }

    def reset(self, seed=None):
        """Reset all survivors to new random clustered positions."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.cluster_centers = []
        self.cluster_ids = []
        positions_list = []

        spawn_bound_x = self.env_size_x * 0.35
        spawn_bound_y = self.env_size_y * 0.35

        for c in range(self.num_clusters):
            cx = self.rng.uniform(-spawn_bound_x, spawn_bound_x)
            cy = self.rng.uniform(-spawn_bound_y, spawn_bound_y)
            self.cluster_centers.append(np.array([cx, cy]))

            n_surv = self.rng.integers(
                self.survivors_per_cluster[0], self.survivors_per_cluster[1] + 1
            )
            for _ in range(n_surv):
                sx = cx + self.rng.normal(0, min(self.cluster_spread, 4.0))
                sy = cy + self.rng.normal(0, min(self.cluster_spread, 4.0))
                max_reach_x = self.env_size_x * 0.42
                max_reach_y = self.env_size_y * 0.42
                sx = np.clip(sx, -max_reach_x, max_reach_x)
                sy = np.clip(sy, -max_reach_y, max_reach_y)
                positions_list.append([sx, sy, 0.0])
                self.cluster_ids.append(c)

        self.num_survivors = len(positions_list)
        self.positions = np.array(positions_list, dtype=np.float64)
        self.cluster_ids = np.array(self.cluster_ids, dtype=int)

        self.discovered = np.zeros(self.num_survivors, dtype=bool)
        self.discovered_clusters = np.zeros(self.num_clusters, dtype=bool)

        # Reset patrol indices
        for cid in self.patrol_indices:
            self.patrol_indices[cid] = 0

        self._snap_to_ground()
