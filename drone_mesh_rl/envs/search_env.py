"""
search_env.py - Multi-agent search on a rectangular map.

The world is a 250 m downtown pad: next-best-view coverage
and a leftover-survivor hunt choose where unfinished work is. The learning
interface is Boen's. Each drone picks one of 9 headings (8 neighbors or
stay), reads a shared coarse 8x8 coverage map, and is turned back when a
heading would pin it on a wall. Drones hold 5 m AGL over the heightfield. The heading in the observation points at
that search goal (a high-gain unpainted patch, or a leftover person), not
at the nearest uncovered cell. Every drone reads the same painted map and
teammate XY; overlapping headings are turned toward cells that drone owns.

Rewards:
  +1.0   each newly painted cell, paid only to the drone that painted it
  +10.0  each newly found survivor (shared)
  +50.0  all survivors found (shared)
  -0.25  a drone paints nothing and is only re-covering
  -0.01  per step (shared)
  -1.0   drone-drone collision (that drone only)
  -1.0   not moving while cells or people remain (that drone only)
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
from .town import TownLayout, spawn_town_in_pybullet


COVER_CROP = 7  # local visited-map window (odd)
GLOBAL_BINS = 8  # coarse coverage map shared by every drone and the critic
N_ACTIONS = 9
MAP_EDGE_MARGIN = 8.0
DRONE_VISUAL_SCALE = 2.5
CRASH_RANGE = 3.5
GOAL_HOLD_STEPS = 150
COVER_CELL_M = 5.0
HUNT_COVER_FRAC = 0.55  # after this much map is painted, finish leftover people
TEAM_KEEP_M = 12.0  # do not claim cells inside another drone's sensor disk
CLAIM_KEEP_M = 22.0  # keep assigned unmapped goals apart
LOOKAHEAD_M = 8.0  # if this far ahead is already painted, peel off to new ground
TURN_MARGIN = 8.0  # do not command into a wall inside this band
BOUNCE_INSET = 10.0  # pop off the rim so they cannot sit on it
WALL_COOLDOWN_STEPS = 90  # ignore outbound MAPPO after a wall hit

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
_MOVE_HEADINGS = np.array(
    [h / np.linalg.norm(h) for h in _HEADING_XY if np.hypot(h[0], h[1]) > 0.5],
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
        num_clusters=9,
        survivors_per_cluster=(3, 7),
        num_loners=12,
        env_size=None,
        env_size_x=250.0,
        env_size_y=250.0,
        max_steps=2500,
        render_mode=None,
        drone_max_speed=40.0,
        drone_max_altitude=20.0,
        hover_altitude=5.0,
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

        # Vel, search heading, walls, teammates, progress, local crop, shared map.
        self.obs_dim = (
            2
            + 2
            + 4
            + 2 * (num_drones - 1)
            + 2
            + COVER_CROP * COVER_CROP
            + GLOBAL_BINS * GLOBAL_BINS
        )
        self.discrete_actions = True
        self.n_actions = N_ACTIONS
        self.act_dim = N_ACTIONS
        self.state_dim = GLOBAL_BINS * GLOBAL_BINS + 2 * num_drones

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
        self._drone_new_cells = np.zeros(num_drones, dtype=np.int32)
        self._drone_overlap = np.zeros(num_drones, dtype=np.int32)
        self.nav_goal = np.zeros((num_drones, 2), dtype=np.float64)
        self.has_nav_goal = np.zeros(num_drones, dtype=bool)
        self.goal_hold = np.zeros(num_drones, dtype=np.int32)
        self.goal_is_person = np.zeros(num_drones, dtype=bool)
        self.wall_cooldown = np.zeros(num_drones, dtype=np.int32)

        self.client = None
        self.drone_ids = []
        self.terrain_body = None
        self.town = None
        self.person_torso_ids = []
        self.person_head_ids = []

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
        gx = int((x + self.size_x / 2.0) / self.size_x * self.cover_nx)
        gy = int((y + self.size_y / 2.0) / self.size_y * self.cover_ny)
        gx = int(np.clip(gx, 0, self.cover_nx - 1))
        gy = int(np.clip(gy, 0, self.cover_ny - 1))
        return gx, gy

    def _lim_x(self):
        return max(5.0, self.size_x / 2.0 - MAP_EDGE_MARGIN)

    def _lim_y(self):
        return max(5.0, self.size_y / 2.0 - MAP_EDGE_MARGIN)

    def _ground_z(self, x, y):
        if self.terrain is None:
            return 0.0
        if hasattr(self.terrain, "get_height"):
            return float(self.terrain.get_height(x, y))
        return 0.0

    def _agl_z(self, x, y):
        return self._ground_z(x, y) + float(self.hover_altitude)

    def _cell_to_world(self, gx, gy):
        wx = (gx + 0.5) / self.cover_nx * self.size_x - self.size_x / 2.0
        wy = (gy + 0.5) / self.cover_ny * self.size_y - self.size_y / 2.0
        return np.array([wx, wy], dtype=np.float64)

    def _interior_coverage(self):
        if self.cover_nx <= 2 or self.cover_ny <= 2:
            return float(self.coverage.mean())
        return float(self.coverage[1:-1, 1:-1].mean())

    def _playable_cell(self, w):
        return abs(w[0]) <= self._lim_x() - TURN_MARGIN and abs(w[1]) <= self._lim_y() - TURN_MARGIN

    def _seal_map_border(self):
        """Outer coverage ring is already searched so policies are not paid to leave."""
        self.coverage[0, :] = 1.0
        self.coverage[-1, :] = 1.0
        self.coverage[:, 0] = 1.0
        self.coverage[:, -1] = 1.0

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
        """Closest remaining person this drone owns so leftover finds are not abandoned."""
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
        for pxy in people:
            dist = float(np.linalg.norm(pxy - my_xy))
            if dist < nearest_d:
                nearest_d = dist
                nearest = pxy
            stolen = False
            for other in others:
                if float(np.linalg.norm(pxy - other)) + 1.0 < dist:
                    stolen = True
                    break
            if stolen:
                continue
            taken = False
            for claim in claimed:
                if float(np.linalg.norm(pxy - claim)) < 6.0:
                    taken = True
                    break
            if taken:
                continue
            if dist < best_d:
                best_d = dist
                best = pxy
        return best if best is not None else nearest

    def _new_cells_if_visit(self, gx, gy):
        """How many unmapped cells a sensor disk would paint from this grid cell."""
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
                world = self._cell_to_world(nx, ny)
                if np.hypot(world[0] - origin[0], world[1] - origin[1]) <= self.sensor_radius:
                    painted += 1
        return painted

    def _cell_is_searchable(self, world):
        if not self._playable_cell(world):
            return False
        if self.town is not None and not self.town.is_walkable(float(world[0]), float(world[1])):
            return False
        return True

    def _has_walkable_unmapped(self):
        for gy in range(1, self.cover_ny - 1):
            for gx in range(1, self.cover_nx - 1):
                if self.coverage[gy, gx] >= 0.5:
                    continue
                if self._cell_is_searchable(self._cell_to_world(gx, gy)):
                    return True
        return False

    def _search_goal_xy(self, drone_index):
        """Hunt leftover people once mapping is mostly done; otherwise next-best-view."""
        people = self._undiscovered_people()
        streets_left = self._has_walkable_unmapped()
        if people and not streets_left:
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
                world = self._cell_to_world(gx, gy)
                if not self._cell_is_searchable(world):
                    continue
                my_d = float(np.linalg.norm(world - my_xy))
                stolen = False
                for other in others:
                    other_d = float(np.linalg.norm(world - other))
                    if other_d + 1.0 < my_d:
                        stolen = True
                        break
                if stolen:
                    continue
                skip_claim = False
                for claim in claimed:
                    if float(np.linalg.norm(world - claim)) < CLAIM_KEEP_M:
                        skip_claim = True
                        break
                if skip_claim:
                    continue
                candidates.append((world, gx, gy, my_d))
        if not candidates:
            fallback = None
            fallback_d = 1e18
            for gy in range(1, self.cover_ny - 1):
                for gx in range(1, self.cover_nx - 1):
                    if self.coverage[gy, gx] >= 0.5:
                        continue
                    world = self._cell_to_world(gx, gy)
                    if not self._cell_is_searchable(world):
                        continue
                    d = float(np.linalg.norm(world - my_xy))
                    if d < fallback_d:
                        fallback_d = d
                        fallback = world
            return fallback

        best = None
        best_score = -1e18
        nearest = None
        nearest_d = 1e18
        for world, gx, gy, my_d in candidates:
            if my_d < nearest_d:
                nearest_d = my_d
                nearest = world
            if my_d < 8.0:
                continue
            gain = self._new_cells_if_visit(gx, gy)
            score = gain / (1.0 + 0.035 * my_d)
            if heading is not None and my_d > 1e-3:
                align = float(np.dot(heading, (world - my_xy) / my_d))
                score += 0.15 * max(0.0, align)
            if score > best_score:
                best_score = score
                best = world
        return best if best is not None else nearest

    def _owned_unfinished_centroid(self, drone_index):
        """Mean of unpainted cells this drone is closest to (team Voronoi)."""
        my_xy = np.asarray(self.drone_positions[drone_index][:2], dtype=np.float64)
        others = [
            np.asarray(self.drone_positions[j][:2], dtype=np.float64)
            for j in range(self.num_drones)
            if j != drone_index
        ]
        if self.cover_nx <= 2 or self.cover_ny <= 2:
            mask = self.coverage < 0.5
        else:
            mask = np.zeros_like(self.coverage, dtype=bool)
            mask[1:-1, 1:-1] = self.coverage[1:-1, 1:-1] < 0.5
        ys, xs = np.nonzero(mask)
        if xs.size == 0:
            return None
        wx = (xs.astype(np.float64) + 0.5) / self.cover_nx * self.size_x - self.size_x / 2.0
        wy = (ys.astype(np.float64) + 0.5) / self.cover_ny * self.size_y - self.size_y / 2.0
        mine_x = []
        mine_y = []
        for x, y in zip(wx, wy):
            cell = np.array([x, y], dtype=np.float64)
            if not self._cell_is_searchable(cell):
                continue
            my_d = float(np.linalg.norm(cell - my_xy))
            if any(float(np.linalg.norm(cell - other)) + 1.0 < my_d for other in others):
                continue
            mine_x.append(x)
            mine_y.append(y)
        if not mine_x:
            return None
        return np.array([float(np.mean(mine_x)), float(np.mean(mine_y))], dtype=np.float64)

    def _unfinished_centroid(self):
        """Middle of the remaining unpainted ground, not the nearest cell.

        Aiming at the nearest uncovered cell makes a policy orbit that edge.
        """
        if self.cover_nx <= 2 or self.cover_ny <= 2:
            mask = self.coverage < 0.5
        else:
            mask = np.zeros_like(self.coverage, dtype=bool)
            mask[1:-1, 1:-1] = self.coverage[1:-1, 1:-1] < 0.5
        ys, xs = np.nonzero(mask)
        if xs.size == 0:
            return None
        wx = (xs.astype(np.float64) + 0.5) / self.cover_nx * self.size_x - self.size_x / 2.0
        wy = (ys.astype(np.float64) + 0.5) / self.cover_ny * self.size_y - self.size_y / 2.0
        keep_x = []
        keep_y = []
        for x, y in zip(wx, wy):
            if self._cell_is_searchable(np.array([x, y], dtype=np.float64)):
                keep_x.append(x)
                keep_y.append(y)
        if not keep_x:
            return None
        return np.array([float(np.mean(keep_x)), float(np.mean(keep_y))], dtype=np.float64)

    def _goal_still_valid(self, drone_index):
        if not self.has_nav_goal[drone_index]:
            return False
        if self.goal_hold[drone_index] <= 0:
            return False
        goal = self.nav_goal[drone_index]
        if self.goal_is_person[drone_index]:
            for person in self._undiscovered_people():
                if float(np.linalg.norm(person - goal)) < 8.0:
                    return True
            return False
        gx, gy = self._world_to_cell(goal[0], goal[1])
        if self.coverage[gy, gx] >= 0.5:
            return False
        here = self.drone_positions[drone_index][:2]
        if float(np.linalg.norm(here - goal)) < self.sensor_radius:
            return False
        return True

    def _refresh_nav_goals(self):
        """Point each drone at next-best-view ground or a leftover survivor."""
        for i in range(self.num_drones):
            if not self._goal_still_valid(i):
                goal = self._search_goal_xy(i)
                if goal is None:
                    self.has_nav_goal[i] = False
                else:
                    self.nav_goal[i] = goal
                    self.has_nav_goal[i] = True
                    self.goal_hold[i] = GOAL_HOLD_STEPS
            if self.has_nav_goal[i]:
                self.goal_hold[i] -= 1

    def _search_target_xy(self, drone_index):
        if self.has_nav_goal[drone_index]:
            return np.asarray(self.nav_goal[drone_index], dtype=np.float64)
        owned = self._owned_unfinished_centroid(drone_index)
        if owned is not None:
            return owned
        return self._unfinished_centroid()

    def _metric_search_dir(self, drone_index, x, y):
        target = self._search_target_xy(drone_index)
        if target is None:
            return 0.0, 0.0
        return float(target[0] - x), float(target[1] - y)

    def _inland_search_dir(self, drone_index, pos):
        """Direction toward unfinished work, with an inland component if a wall is close.

        Leftover-person goals stay on-target so a rim survivor can still be found.
        """
        ux, uy = self._metric_search_dir(drone_index, pos[0], pos[1])
        if self.goal_is_person[drone_index]:
            if abs(ux) + abs(uy) < 1.0:
                ang = 0.35 * self.step_count + drone_index
                ux, uy = float(np.cos(ang)), float(np.sin(ang))
            return ux, uy
        lx, ly = self._lim_x(), self._lim_y()
        if pos[0] >= lx - TURN_MARGIN:
            ux = min(ux, -1.0)
        elif pos[0] <= -lx + TURN_MARGIN:
            ux = max(ux, 1.0)
        if pos[1] >= ly - TURN_MARGIN:
            uy = min(uy, -1.0)
        elif pos[1] <= -ly + TURN_MARGIN:
            uy = max(uy, 1.0)
        if abs(ux) + abs(uy) < 1e-6:
            ux = -float(np.sign(pos[0])) or 0.0
            uy = -float(np.sign(pos[1])) or 0.0
        return ux, uy

    def _velocity_toward_unmapped(self, drone_index, pos):
        ux, uy = self._inland_search_dir(drone_index, pos)
        if not self.goal_is_person[drone_index]:
            for j in range(self.num_drones):
                if j == drone_index:
                    continue
                rel = pos[:2] - self.drone_positions[j][:2]
                dist = float(np.hypot(rel[0], rel[1]))
                if 1e-3 < dist < TEAM_KEEP_M:
                    push = (TEAM_KEEP_M - dist) / TEAM_KEEP_M
                    ux += float(rel[0]) / dist * push
                    uy += float(rel[1]) / dist * push
        if abs(ux) + abs(uy) < 1e-6:
            ang = 2.0 * np.pi * drone_index / max(self.num_drones, 1) + 0.15 * self.step_count
            ux = float(np.cos(ang))
            uy = float(np.sin(ang))
        legal = self._best_legal_heading(pos, ux, uy)
        n = float(np.hypot(legal[0], legal[1]))
        if n < 1e-6:
            legal = np.array([1.0, 0.0], dtype=np.float64)
            n = 1.0
        vel = np.zeros(3, dtype=np.float64)
        vel[0] = legal[0] / n * self.drone_max_speed
        vel[1] = legal[1] / n * self.drone_max_speed
        return vel

    def _heading_retraces(self, pos, vel):
        """True when this heading is about to fly over ground the team already painted."""
        if self._interior_coverage() >= 0.995:
            return False
        speed = float(np.hypot(vel[0], vel[1]))
        if speed < 0.5:
            gx, gy = self._world_to_cell(pos[0], pos[1])
            return self.coverage[gy, gx] >= 0.5
        look = np.asarray(pos[:2], dtype=np.float64) + np.asarray(vel[:2], dtype=np.float64) / speed * LOOKAHEAD_M
        gx, gy = self._world_to_cell(look[0], look[1])
        return self.coverage[gy, gx] >= 0.5

    def _heading_at_teammate(self, drone_index, pos, vel):
        heading = np.asarray(vel[:2], dtype=np.float64)
        speed = float(np.hypot(heading[0], heading[1]))
        for j in range(self.num_drones):
            if j == drone_index:
                continue
            rel = self.drone_positions[j][:2] - pos[:2]
            dist = float(np.hypot(rel[0], rel[1]))
            if dist < 1e-3:
                return True
            if dist < TEAM_KEEP_M and speed < 0.5:
                return True
            if dist < TEAM_KEEP_M and speed >= 0.5 and float(np.dot(heading, rel)) > 0.0:
                return True
        return False

    def _work_remains(self):
        if self.survivors is not None:
            left = int(self.survivors.num_survivors - self.survivors.discovered.sum())
            if left > 0:
                return True
        return self._interior_coverage() < 0.995

    def _keep_moving(self, drone_index):
        if not self._work_remains():
            return
        self.drone_velocities[drone_index] = self._velocity_toward_unmapped(
            drone_index, self.drone_positions[drone_index]
        )

    def _search_heading(self, drone_index, x, y):
        """Map-scaled direction toward unfinished ground or a leftover person."""
        dx, dy = self._metric_search_dir(drone_index, x, y)
        return dx / max(self.size_x, 1.0), dy / max(self.size_y, 1.0)

    def _on_border(self, pos, margin=None):
        if margin is None:
            margin = TURN_MARGIN
        return abs(pos[0]) >= self._lim_x() - margin or abs(pos[1]) >= self._lim_y() - margin

    def _heading_blocked(self, pos, heading):
        """True when this heading would drive into a wall the drone is near."""
        lx, ly = self._lim_x(), self._lim_y()
        x, y = float(pos[0]), float(pos[1])
        hx, hy = float(heading[0]), float(heading[1])
        if x >= lx - TURN_MARGIN and hx > 0.05:
            return True
        if x <= -lx + TURN_MARGIN and hx < -0.05:
            return True
        if y >= ly - TURN_MARGIN and hy > 0.05:
            return True
        if y <= -ly + TURN_MARGIN and hy < -0.05:
            return True
        return False

    def _best_legal_heading(self, pos, ux, uy):
        """Neighbor heading toward (ux, uy) that does not point out of the map."""
        best = None
        best_dot = -np.inf
        for heading in _MOVE_HEADINGS:
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

    def _move_with_soft_walls(self, drone_index, pos, vel, dt):
        """Fly the chosen heading. Hitting a wall pops inland and turns toward unfinished ground."""
        pos = np.asarray(pos, dtype=np.float64).copy()
        vel = np.asarray(vel, dtype=np.float64).copy()
        lx, ly = self._lim_x(), self._lim_y()
        heading = vel[:2]
        speed = float(np.hypot(heading[0], heading[1]))
        blocked = self._heading_blocked(pos, heading) if speed > 0.5 else False
        stuck_on_rim = speed <= 0.5 and self._on_border(pos)
        if blocked or stuck_on_rim:
            self.wall_cooldown[drone_index] = max(
                int(self.wall_cooldown[drone_index]), WALL_COOLDOWN_STEPS
            )
        if self.wall_cooldown[drone_index] > 0 or blocked or stuck_on_rim:
            vel = self._velocity_toward_unmapped(drone_index, pos)
        pos = pos + vel * dt
        bounced = False
        if pos[0] >= lx:
            pos[0] = lx - BOUNCE_INSET
            bounced = True
        elif pos[0] <= -lx:
            pos[0] = -lx + BOUNCE_INSET
            bounced = True
        if pos[1] >= ly:
            pos[1] = ly - BOUNCE_INSET
            bounced = True
        elif pos[1] <= -ly:
            pos[1] = -ly + BOUNCE_INSET
            bounced = True
        if bounced:
            self.wall_cooldown[drone_index] = WALL_COOLDOWN_STEPS
            vel = self._velocity_toward_unmapped(drone_index, pos)
        pos[0] = float(np.clip(pos[0], -lx, lx))
        pos[1] = float(np.clip(pos[1], -ly, ly))
        pos[2] = float(self._agl_z(pos[0], pos[1]))
        vel[2] = 0.0
        return pos, vel

    def _mark_coverage(self, positions):
        """Paint sensor footprints. Credit a new cell only to the drone that painted it."""
        new_cells = 0
        overlap = 0
        per_drone = np.zeros(len(positions), dtype=np.int32)
        per_overlap = np.zeros(len(positions), dtype=np.int32)
        already = self.coverage >= 1.0
        for i, pos in enumerate(positions):
            cx, cy = self._world_to_cell(pos[0], pos[1])
            cell_mx = self.size_x / self.cover_nx
            cell_my = self.size_y / self.cover_ny
            radius_x = max(1, int(np.ceil(self.sensor_radius / cell_mx)))
            radius_y = max(1, int(np.ceil(self.sensor_radius / cell_my)))
            for dx in range(-radius_x, radius_x + 1):
                for dy in range(-radius_y, radius_y + 1):
                    gx, gy = cx + dx, cy + dy
                    if gx < 0 or gy < 0 or gx >= self.cover_nx or gy >= self.cover_ny:
                        continue
                    world = self._cell_to_world(gx, gy)
                    if np.hypot(world[0] - pos[0], world[1] - pos[1]) > self.sensor_radius:
                        continue
                    if already[gy, gx]:
                        overlap += 1
                        per_overlap[i] += 1
                    elif self.coverage[gy, gx] < 1.0:
                        new_cells += 1
                        per_drone[i] += 1
                    self.coverage[gy, gx] = 1.0
        self._drone_new_cells = per_drone
        self._drone_overlap = per_overlap
        return new_cells, overlap

    def _coverage_crop(self, x, y):
        """Local visited window. Off-map cells are 1 (not searchable), not 0."""
        cx, cy = self._world_to_cell(x, y)
        half = COVER_CROP // 2
        crop = np.ones((COVER_CROP, COVER_CROP), dtype=np.float32)
        for i, dy in enumerate(range(cy - half, cy + half + 1)):
            for j, dx in enumerate(range(cx - half, cx + half + 1)):
                if 0 <= dy < self.cover_ny and 0 <= dx < self.cover_nx:
                    crop[i, j] = self.coverage[dy, dx]
        return crop.ravel()

    def _shared_map(self):
        """Coarse 8x8 coverage grid the whole team paints and every drone can read."""
        bins = GLOBAL_BINS
        small = np.zeros((bins, bins), dtype=np.float32)
        ny, nx = self.cover_ny, self.cover_nx
        for iy in range(bins):
            y0 = iy * ny // bins
            y1 = max(y0 + 1, (iy + 1) * ny // bins)
            for ix in range(bins):
                x0 = ix * nx // bins
                x1 = max(x0 + 1, (ix + 1) * nx // bins)
                small[iy, ix] = float(self.coverage[y0:y1, x0:x1].mean())
        return small.ravel()

    def global_state(self):
        """Shared coverage map plus normalized drone XY, for the critic."""
        hx = max(self.size_x / 2.0, 1.0)
        hy = max(self.size_y / 2.0, 1.0)
        pos = np.empty((self.num_drones, 2), dtype=np.float32)
        pos[:, 0] = self.drone_positions[:, 0] / hx
        pos[:, 1] = self.drone_positions[:, 1] / hy
        return np.concatenate([self._shared_map(), pos.ravel()])

    def _init_coverage_sweep(self):
        """Split the rectangle into north-south lanes, one subset per drone."""
        lx = max(1.0, self._lim_x() - 4.0)
        ly = max(1.0, self._lim_y() - 4.0)
        spacing = max(self.sensor_radius * 1.5, 8.0)
        xs = np.arange(-lx, lx + 0.01, spacing)
        if xs.size == 0:
            xs = np.array([0.0])
        self._lanes = [xs[i :: self.num_drones] for i in range(self.num_drones)]
        self._lane_idx = np.zeros(self.num_drones, dtype=int)
        self._sweep_dir = np.ones(self.num_drones, dtype=np.float64)
        self._sweep_limit_y = ly

    def coverage_actions(self):
        """Boustrophedon lanes. The viewer can still request this; training does not."""
        if not hasattr(self, "_lanes"):
            self._init_coverage_sweep()
        half_y = float(self._sweep_limit_y)
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
                if pos[1] >= half_y and self._sweep_dir[i] > 0:
                    self._sweep_dir[i] = -1.0
                    self._lane_idx[i] = min(idx + 1, lanes.size - 1)
                elif pos[1] <= -half_y and self._sweep_dir[i] < 0:
                    self._sweep_dir[i] = 1.0
                    self._lane_idx[i] = min(idx + 1, lanes.size - 1)
            vy = float(self._sweep_dir[i])
            if not on_lane:
                vy *= 0.15
            actions[agent] = np.array([vx, vy, 0.0], dtype=np.float32)
        return actions

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
            p.setRealTimeSimulation(0, physicsClientId=self.client)
        else:
            self.client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, 0, physicsClientId=self.client)
        p.setTimeStep(1.0 / self.pyb_freq, physicsClientId=self.client)

    def _build_ground(self):
        if self.client is None:
            return
        if self.town is not None:
            terrain_body, _, _ = spawn_town_in_pybullet(
                self.client, self.town, self.terrain, self.size_x
            )
            self.terrain_body = terrain_body
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

        z = self._agl_z(0.0, 0.0)
        lx, ly = self._lim_x(), self._lim_y()
        for i in range(self.num_drones):
            frac = (i + 0.5) / max(self.num_drones, 1)
            x = -lx * 0.7 + frac * (1.4 * lx)
            y = -ly * 0.35 + (i % 2) * (0.25 * ly)
            z = self._agl_z(x, y)
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
                cameraDistance=max(self.size_x, self.size_y) * 1.05,
                cameraYaw=0,
                cameraPitch=-89,
                cameraTargetPosition=[0.0, 0.0, 0.0],
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

    def _person_rgba(self, discovered):
        if discovered:
            return [0.15, 0.82, 0.28, 1.0]
        return [0.88, 0.22, 0.18, 1.0]

    def _spawn_person_capsules(self):
        """Two-capsule figures (torso + head) snapped to ground."""
        self.person_torso_ids = []
        self.person_head_ids = []
        if self.client is None or self.survivors is None:
            return
        positions = self.survivors.get_positions()
        for i in range(self.survivors.num_survivors):
            rgba = self._person_rgba(bool(self.survivors.discovered[i]))
            x, y, z = float(positions[i][0]), float(positions[i][1]), float(positions[i][2])
            ground = z - 0.3
            torso_z = ground + 0.95
            head_z = ground + 1.85
            torso_col = p.createCollisionShape(
                p.GEOM_CAPSULE, radius=0.32, height=0.95, physicsClientId=self.client
            )
            try:
                torso_vis = p.createVisualShape(
                    p.GEOM_CAPSULE,
                    radius=0.32,
                    length=0.95,
                    rgbaColor=rgba,
                    physicsClientId=self.client,
                )
            except Exception:
                torso_vis = p.createVisualShape(
                    p.GEOM_CAPSULE,
                    radius=0.32,
                    height=0.95,
                    rgbaColor=rgba,
                    physicsClientId=self.client,
                )
            torso = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=torso_col,
                baseVisualShapeIndex=torso_vis,
                basePosition=[x, y, torso_z],
                physicsClientId=self.client,
            )
            head_col = p.createCollisionShape(
                p.GEOM_SPHERE, radius=0.28, physicsClientId=self.client
            )
            head_vis = p.createVisualShape(
                p.GEOM_SPHERE, radius=0.28, rgbaColor=rgba, physicsClientId=self.client
            )
            head = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=head_col,
                baseVisualShapeIndex=head_vis,
                basePosition=[x, y, head_z],
                physicsClientId=self.client,
            )
            self.person_torso_ids.append(torso)
            self.person_head_ids.append(head)

    def _sync_person_capsules(self):
        if self.client is None or not self.person_torso_ids:
            return
        positions = self.survivors.get_positions()
        orn = [0, 0, 0, 1]
        n = min(len(self.person_torso_ids), self.survivors.num_survivors)
        for i in range(n):
            x, y, z = float(positions[i][0]), float(positions[i][1]), float(positions[i][2])
            ground = z - 0.3
            rgba = self._person_rgba(bool(self.survivors.discovered[i]))
            p.resetBasePositionAndOrientation(
                self.person_torso_ids[i],
                [x, y, ground + 0.95],
                orn,
                physicsClientId=self.client,
            )
            p.resetBasePositionAndOrientation(
                self.person_head_ids[i],
                [x, y, ground + 1.85],
                orn,
                physicsClientId=self.client,
            )
            p.changeVisualShape(
                self.person_torso_ids[i], -1, rgbaColor=rgba, physicsClientId=self.client
            )
            p.changeVisualShape(
                self.person_head_ids[i], -1, rgbaColor=rgba, physicsClientId=self.client
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
        self._drone_new_cells = np.zeros(self.num_drones, dtype=np.int32)
        self._drone_overlap = np.zeros(self.num_drones, dtype=np.int32)
        self.nav_goal = np.zeros((self.num_drones, 2), dtype=np.float64)
        self.has_nav_goal = np.zeros(self.num_drones, dtype=bool)
        self.goal_hold = np.zeros(self.num_drones, dtype=np.int32)
        self.goal_is_person = np.zeros(self.num_drones, dtype=bool)
        self.wall_cooldown = np.zeros(self.num_drones, dtype=np.int32)
        self.town = None
        self.person_torso_ids = []
        self.person_head_ids = []
        if hasattr(self, "_lanes"):
            del self._lanes

        surv_seed = (int(time.time() * 1e6) ^ (os.getpid() << 8) ^ id(self)) & 0x7FFFFFFF
        if seed is not None:
            surv_seed ^= int(seed) & 0x7FFFFFFF

        pad = max(self.size_x, self.size_y)
        self.terrain = Terrain(
            size_x=self.size_x,
            size_y=self.size_y,
            resolution=5.0,
            seed=self._seed if self._seed is not None else 42,
            flat=False,
            with_structures=False,
            city_size=pad,
            city_centers=[(0.0, 0.0)],
            connector_segment=None,
            rim_width=0.0,
            n_rim_ranges=(0, 0),
            n_tall_hills=(0, 0),
            n_hill_clusters=(0, 0),
            meadow_amp=0.12,
            grass_amp=0.35,
        )
        self.town = TownLayout(
            size=pad,
            seed=surv_seed,
            altitude_cap=40.0,
            origin=(0.0, 0.0),
        )
        self.town.apply_ground_heights(self.terrain)
        if hasattr(self.terrain, "set_obstacle_boxes"):
            self.terrain.set_obstacle_boxes(self.town.boxes)

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
        self.survivors.place_on_walkable(self.town)

        self._init_pybullet()
        self._build_ground()
        self._spawn_drones()
        self._spawn_person_capsules()
        self._mark_coverage(self.drone_positions)
        self._refresh_nav_goals()

        return self._get_observations(), self._infos()

    def step(self, actions):
        self.step_count += 1
        dt = 1.0 / self.pyb_freq
        self._refresh_nav_goals()

        for i, agent in enumerate(self.possible_agents):
            if agent not in actions:
                continue
            raw = np.asarray(actions[agent], dtype=np.float32).reshape(-1)
            if raw.size >= 3:
                act = np.clip(raw[:3], -1.0, 1.0)
                self.drone_velocities[i] = act * self.drone_max_speed
            else:
                idx = int(np.clip(np.rint(float(raw[0])) if raw.size else 0, 0, self.n_actions - 1))
                heading = _HEADING_XY[idx]
                if heading[0] == 0.0 and heading[1] == 0.0:
                    self.drone_velocities[i] = self._velocity_toward_unmapped(
                        i, self.drone_positions[i]
                    )
                else:
                    self.drone_velocities[i, 0] = heading[0] * self.drone_max_speed
                    self.drone_velocities[i, 1] = heading[1] * self.drone_max_speed
                    self.drone_velocities[i, 2] = 0.0

        for i in range(self.num_drones):
            if self.wall_cooldown[i] > 0:
                self.wall_cooldown[i] -= 1
            if self._work_remains():
                self._keep_moving(i)
            pos, vel = self._move_with_soft_walls(
                i, self.drone_positions[i], self.drone_velocities[i], dt
            )
            if self._work_remains() and float(np.hypot(vel[0], vel[1])) < 0.5 * self.drone_max_speed:
                vel = self._velocity_toward_unmapped(i, pos)
            self.drone_positions[i] = pos
            self.drone_velocities[i] = vel

        collisions = np.zeros(self.num_drones, dtype=bool)
        for i in range(self.num_drones):
            for j in range(i + 1, self.num_drones):
                dist = float(np.linalg.norm(self.drone_positions[i][:2] - self.drone_positions[j][:2]))
                if dist < CRASH_RANGE:
                    collisions[i] = True
                    collisions[j] = True

        if self.client is not None:
            for i in range(self.num_drones):
                if self.drone_ids[i] < 0:
                    continue
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
        self._sync_person_capsules()

        rewards = self._compute_rewards(collisions)
        all_found = int(self.survivors.discovered.sum()) == self.survivors.num_survivors
        maxed = self.step_count >= self.max_steps
        truncations = {a: (all_found or maxed) for a in self.possible_agents}
        terminations = {a: False for a in self.possible_agents}
        return self._get_observations(), rewards, terminations, truncations, self._infos()

    def _compute_rewards(self, collisions):
        """New cells and new people. Sitting still is costly while work remains."""
        stats = self.survivors.get_discovery_stats()
        all_found = (
            stats["discovered_survivors"] == stats["total_survivors"] and stats["total_survivors"] > 0
        )
        work_remains = self._interior_coverage() < 0.995 or not all_found

        shared = 10.0 * float(self.newly_discovered)
        shared -= 0.01
        if all_found:
            shared += 50.0

        rewards = {}
        for i, agent in enumerate(self.possible_agents):
            rew = shared + float(self._drone_new_cells[i])
            if self._drone_new_cells[i] == 0 and self._drone_overlap[i] > 0:
                rew -= 0.25
            speed = float(np.hypot(self.drone_velocities[i][0], self.drone_velocities[i][1]))
            if speed < 0.5 and work_remains:
                rew -= 1.0
            if collisions[i]:
                rew -= 1.0
            rewards[agent] = rew
        return rewards

    def _get_observations(self):
        """Local motion and walls, plus the shared coarse coverage map."""
        sr = max(self.sensor_radius, 1.0)
        lx, ly = self._lim_x(), self._lim_y()
        stats = self.survivors.get_discovery_stats()
        uncovered_frac = 1.0 - self._interior_coverage()
        observations = {}
        for i, agent in enumerate(self.possible_agents):
            obs = np.zeros(self.obs_dim, dtype=np.float32)
            pos = self.drone_positions[i]
            vel = self.drone_velocities[i]
            obs[0] = vel[0] / self.drone_max_speed
            obs[1] = vel[1] / self.drone_max_speed
            ux, uy = self._search_heading(i, pos[0], pos[1])
            obs[2] = float(np.clip(ux * self.size_x / sr, -4.0, 4.0))
            obs[3] = float(np.clip(uy * self.size_y / sr, -4.0, 4.0))
            obs[4] = float(np.clip((lx - pos[0]) / sr, 0.0, 8.0))
            obs[5] = float(np.clip((pos[0] + lx) / sr, 0.0, 8.0))
            obs[6] = float(np.clip((ly - pos[1]) / sr, 0.0, 8.0))
            obs[7] = float(np.clip((pos[1] + ly) / sr, 0.0, 8.0))
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
