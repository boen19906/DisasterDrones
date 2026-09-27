"""
search_env.py - Multi-agent search on a rectangular map.

The playable world is a 240 m rubble city on a 250 m pad: next-best-view
coverage and a leftover-survivor hunt choose where unfinished work is.
GUI / search-viewer only also dress a ~1.2 km heightfield around that pad
(forest, then mountains). Drones, coverage, and rewards stay on the 250 m
square. The learning interface is Boen's. Each drone picks one of 9 headings
(8 neighbors or stay), reads a shared coarse 8x8 coverage map, and is turned
back when a heading would pin it on a wall. Drones hold 5 m AGL over the
heightfield. The heading in the observation points at that search goal
(a high-gain unpainted patch, or a leftover person), not at the nearest
uncovered cell. Every drone reads the same painted map and teammate XY;
overlapping headings are turned toward cells that drone owns.

Rewards:
  +1.0   each newly painted cell, paid only to the drone that painted it
  +10.0  each newly found survivor (shared)
  +50.0  all survivors found (shared)
  -0.25  a drone paints nothing and is only re-covering
  -0.01  per step (shared)
  -1.0   drone-drone collision (that drone only)
  -2.0   the alive drones are not one radio mesh (that drone only)
  -50.0  building crash, that drone is dead
  -50.0  battery empty, that drone is dead
  -1.0   not moving while cells or people remain (that drone only)
"""

import functools
import math
import os
import tempfile
import time
from heapq import heappop, heappush

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

from .district import (
    RUBBLE_TOWN_SIZE,
    RubbleTownLayout,
    spawn_rubble_town_dressing,
    spawn_rubble_town_in_pybullet,
)
from .network import calculate_link


def _farthest_link_m(has_los):
    """Largest separation where calculate_link still closes. FSPL is monotonic."""
    lo = 0.0
    hi = 4000.0
    origin = (0.0, 0.0, 0.0)
    for _ in range(48):
        mid = 0.5 * (lo + hi)
        ok, _rssi = calculate_link(origin, (mid, 0.0, 0.0), has_los)
        if ok:
            lo = mid
        else:
            hi = mid
    return lo


# A blocked hop dies near 56 m. A clear hop still closes across this whole city.
_NLOS_LINK_M = _farthest_link_m(False)
_LOS_LINK_M = _farthest_link_m(True)
from .survivors import SurvivorCluster
from .terrain import Terrain, spawn_terrain_in_pybullet
from .town import TownLayout, apply_spectator_sun


COVER_CROP = 7  # local visited-map window (odd)
GLOBAL_BINS = 8  # coarse coverage map shared by every drone and the critic
N_ACTIONS = 27  # 8 headings x (level, climb, descend) + stay, climb, descend
MIN_AGL = 1.5
SEARCH_AGL_MAX = 8.0  # paint and find only at or below this height
MAP_EDGE_MARGIN = 0.75  # fly almost to the pad wall so a 5 m sensor covers the rim
PERSON_TARGET_H = 1.7
DRONE_TARGET_LONG = 3.6
_PERSON_LETTERS = tuple(chr(c) for c in range(ord("a"), ord("r") + 1))
_MESH_RGBA = [1.0, 1.0, 1.0, 1.0]
_FOUND_RGBA = [0.15, 1.0, 0.25, 1.0]
_CRASH_RGBA = [0.45, 0.06, 0.06, 1.0]
CRASH_RANGE = 3.5
BUILDING_RADIUS = 0.55  # point used for a hard crash test
BODY_KEEP_M = 1.7  # drone mesh reaches about this far; closer and the body is in the ruin
BODY_EXIT_M = 6.0  # one-step pull to here so the mesh is out of the ruin immediately
BUILDING_HARD_M = 1.25  # refuse a heading that would clip this soon
BUILDING_STANDOFF_M = 2.5  # prefer street center; not a hard stop
BUILDING_LOOKAHEAD_M = 3.5  # about one fast step; longer looks freeze them in town
BUILDING_COMMIT_M = 6.0  # this close, a heading into rubble is a crash, not a hover
BUILDING_CRASH_PENALTY = 50.0  # lethal building hit, that drone only
BATTERY_DEATH_PENALTY = 50.0  # empty battery, that drone only
BATTERY_LEVEL_DRAIN = 1.0 / 1100.0  # continuous cruise dies late in a 1500-step episode
BATTERY_HOVER_DRAIN = 1.0 / 2500.0
BATTERY_CLIMB_MULT = 2.0  # climbing spends the pack faster than level flight
WIND_MAX_MS = 8.0  # prevailing wind is sampled inside this cap
WIND_GUST_MS = 2.5  # slow gust added on top, then the sum is capped
WIND_GUST_PERIOD = 600  # steps for one gust cycle (~20 s at 30 Hz)
LETHAL_RUBBLE = False  # False: block the step, stay alive, small penalty. True: die and pay the crash penalty.
RUBBLE_BLOCK_PENALTY = 5.0
MESH_BREAK_PENALTY = 2.0  # alive drones are not one radio mesh
GOAL_HOLD_STEPS = 150
COVER_CELL_M = 5.0
TEAM_KEEP_M = 12.0  # do not claim cells inside another drone's sensor disk
CLAIM_KEEP_M = 22.0  # keep assigned unmapped goals apart
LOOKAHEAD_M = 8.0  # if this far ahead is already painted, peel off to new ground
TURN_MARGIN = 1.25  # only slide when actually on the wall, not 8 m inland
GOAL_EDGE_M = 0.25  # claim streets on the rim
LEFTOVER_CELLS = 80  # drop keep-out so last alleys are not abandoned
WALL_COOLDOWN_STEPS = 12  # brief peel if a corner pins them
CLIMB_STALL_STEPS = 8  # climb commands with no altitude gain, then go around
CAMERA_M = 12.0  # forward sensor: a solid on this ray, not a preloaded obstacle map
LEVEL_WALL_M = CAMERA_M  # do not cruise into a wall the sensor already sees
TALL_AVOID_M = 14.0  # a tower this close is not a climb; go around
DETOUR_LOCK_STEPS = 16  # keep the around-heading so a corner cannot flip it
STUCK_STEPS = 60  # net movement inside STUCK_RADIUS this long means the drone is stuck
STUCK_RADIUS = 3.5
EXIT_LOCK_STEPS = 12  # after a rubble bump, keep the away-heading this many steps
PACE_STEPS = 36  # still inside PACE_RADIUS on painted ground: stop the shuttle
PACE_RADIUS = 14.0
PACE_LOCK_STEPS = 100  # hold the heading toward leftover work across empty streets
STREET_LOOKAHEAD_CELLS = 6  # follow streets this far around a block
# Visual land around the 250 m pad. PyBullet GUI fades ~1000 m from camera.
SEARCH_WORLD_M = 1200.0
SEARCH_TERRAIN_RES = 8.0

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
# Action layout: 0-7 level, 8-15 climb, 16-23 descend, 24 stay, 25 climb, 26 descend.
_VZ_BY_BAND = (0.0, 1.0, -1.0)


def _action_parts(idx):
    """Return (heading index into _HEADING_XY, vertical sign)."""
    idx = int(idx)
    if idx >= 24:
        return 8, _VZ_BY_BAND[min(max(idx - 24, 0), 2)]
    return idx % 8, _VZ_BY_BAND[idx // 8]


def _obj_bounds(path):
    xs, ys, zs = [], [], []
    with open(path) as handle:
        for line in handle:
            if not line.startswith("v "):
                continue
            parts = line.split()
            xs.append(float(parts[1]))
            ys.append(float(parts[2]))
            zs.append(float(parts[3]))
    return {
        "xmin": min(xs),
        "xmax": max(xs),
        "ymin": min(ys),
        "ymax": max(ys),
        "zmin": min(zs),
        "zmax": max(zs),
    }


def _search_prefabs_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    for rel in (("..", "Prefabs"), ("..", "..", "Prefabs")):
        cand = os.path.normpath(os.path.join(here, *rel))
        if os.path.isdir(cand):
            return cand
    return os.path.normpath(os.path.join(here, "..", "Prefabs"))


def _upright_fix_euler(bounds):
    """Rotate the thin file axis onto world Z (Kenney Y-up uses +90 deg about X)."""
    dx = bounds["xmax"] - bounds["xmin"]
    dy = bounds["ymax"] - bounds["ymin"]
    dz = bounds["zmax"] - bounds["zmin"]
    thin = min((dx, "x"), (dy, "y"), (dz, "z"))[1]
    if thin == "y":
        return [math.pi / 2.0, 0.0, 0.0]
    if thin == "x":
        return [0.0, -math.pi / 2.0, 0.0]
    return [0.0, 0.0, 0.0]


def _mesh_yaw_orn(yaw):
    """Y-up → Z-up, then world yaw. Same compose as scenery._q_yaw."""
    q_fix = p.getQuaternionFromEuler([math.pi / 2.0, 0.0, 0.0])
    q_yaw = p.getQuaternionFromEuler([0.0, 0.0, float(yaw)])
    return p.multiplyTransforms([0, 0, 0], q_yaw, [0, 0, 0], q_fix)[1]


def _write_textureless_obj(src_path):
    """Copy an OBJ without mtllib/usemtl so rgbaColor is a solid mesh tint."""
    fd, dst = tempfile.mkstemp(prefix="dd_person_green_", suffix=".obj")
    with os.fdopen(fd, "w") as out, open(src_path) as src:
        for line in src:
            if line.startswith("mtllib ") or line.startswith("usemtl "):
                continue
            out.write(line)
    return dst


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
        legacy_xy=False,
    ):
        super().__init__()
        self.legacy_xy = bool(legacy_xy)
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

        # Vel, search heading, walls, teammates, progress, battery, body-frame wind,
        # rubble sense, altitude, climb-helps, local crop, shared map.
        # legacy_xy drops altitude, vz, and the 8 climb-helps bits.
        self.obs_dim = (
            2
            + 2
            + 4
            + 2 * (num_drones - 1)
            + 2
            + 1
            + 2
            + 1
            + 8
            + (0 if self.legacy_xy else 1 + 1 + 8)
            + COVER_CROP * COVER_CROP
            + GLOBAL_BINS * GLOBAL_BINS
        )
        self.discrete_actions = True
        if self.legacy_xy:
            self.n_actions = 9  # 8 headings + stay. Action 8 is hover.
            self.state_dim = GLOBAL_BINS * GLOBAL_BINS + 2 * num_drones + num_drones + 2
        else:
            self.n_actions = N_ACTIONS
            self.state_dim = GLOBAL_BINS * GLOBAL_BINS + 3 * num_drones + num_drones + 2
        self.act_dim = self.n_actions

        self.terrain = None
        self.forest_trees = 0
        self.survivors = None
        self.coverage = np.zeros((self.cover_ny, self.cover_nx), dtype=np.float32)
        self.drone_positions = np.zeros((num_drones, 3))
        self.drone_velocities = np.zeros((num_drones, 3))
        self.drone_alive = np.ones(num_drones, dtype=bool)
        self.battery = np.ones(num_drones, dtype=np.float64)
        self._cmd_vel = np.zeros((num_drones, 3), dtype=np.float64)
        self._sample_wind(seed)
        self.drone_finds = np.zeros(num_drones, dtype=np.int32)
        self.step_count = 0
        self.newly_discovered = 0
        self.new_cells = 0
        self.overlap_cells = 0
        self._drone_new_cells = np.zeros(num_drones, dtype=np.int32)
        self._drone_overlap = np.zeros(num_drones, dtype=np.int32)
        self.nav_goal = np.zeros((num_drones, 2), dtype=np.float64)
        self.has_nav_goal = np.zeros(num_drones, dtype=bool)
        self.goal_hold = np.zeros(num_drones, dtype=np.int32)
        self.abandoned_goals = [[] for _ in range(num_drones)]
        self._plan_action = np.full(num_drones, 24, dtype=np.int32)
        self._stall_z = np.zeros(num_drones, dtype=np.float64)
        self._stall_n = np.zeros(num_drones, dtype=np.int32)
        self._climb_intent = np.zeros(num_drones, dtype=bool)
        self._stall_xy = np.zeros((num_drones, 2), dtype=np.float64)
        self._xy_stall = np.zeros(num_drones, dtype=np.int32)
        self._lock_h = np.full(num_drones, -1, dtype=np.int32)
        self._lock_n = np.zeros(num_drones, dtype=np.int32)
        self._stuck_cool = np.zeros(num_drones, dtype=np.int32)
        self._free_xy = np.zeros((num_drones, 2), dtype=np.float64)
        self._free_n = np.zeros(num_drones, dtype=np.int32)
        self._free_left = np.zeros(num_drones, dtype=np.int32)
        self._free_h = np.full(num_drones, -1, dtype=np.int32)
        self._free_up = np.zeros(num_drones, dtype=bool)
        self._free_dir = np.zeros((num_drones, 2), dtype=np.float64)
        self._pace_xy = np.zeros((num_drones, 2), dtype=np.float64)
        self._pace_n = np.zeros(num_drones, dtype=np.int32)
        self._pace_left = np.zeros(num_drones, dtype=np.int32)
        self._pace_h = np.zeros(num_drones, dtype=np.int32)
        self._pace_tick = np.full(num_drones, -1, dtype=np.int32)
        self._pace_cached = np.full(num_drones, -1, dtype=np.int32)
        self.goal_is_person = np.zeros(num_drones, dtype=bool)
        self.wall_cooldown = np.zeros(num_drones, dtype=np.int32)
        self._exit_action = np.full(num_drones, 24, dtype=np.int32)
        self._exit_left = np.zeros(num_drones, dtype=np.int32)
        self.street_mask = np.ones((self.cover_ny, self.cover_nx), dtype=bool)
        self.street_clearance = np.full((self.cover_ny, self.cover_nx), 99.0)

        self.client = None
        self.drone_ids = []
        self.terrain_body = None
        self.town = None
        self.person_ids = []
        self._person_vis = []
        self._person_ymin = []
        self._person_scale = []
        self._person_paths = []
        self._person_models = []
        self._person_tinted = []
        self._person_green_vis = []
        self._person_green_obj_files = []
        self._drone_vis = -1
        self._drone_yaw = np.zeros(num_drones, dtype=np.float64)
        self._drone_crash_tinted = np.zeros(num_drones, dtype=bool)
        self._drone_fix_euler = [math.pi / 2.0, 0.0, 0.0]
        self._drone_yaw_offset = 0.5 * math.pi

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
        mask = getattr(self, "street_mask", None)
        if mask is not None and mask.any():
            return float(self.coverage[mask].mean())
        if self.cover_nx <= 2 or self.cover_ny <= 2:
            return float(self.coverage.mean())
        return float(self.coverage[1:-1, 1:-1].mean())

    def _playable_cell(self, w):
        return abs(w[0]) <= self._lim_x() - GOAL_EDGE_M and abs(w[1]) <= self._lim_y() - GOAL_EDGE_M

    def _seal_unsearchable(self):
        """Ruin and tower footprints are not search ground. Lots and streets are."""
        self.street_mask = np.zeros((self.cover_ny, self.cover_nx), dtype=bool)
        self.street_clearance = np.full((self.cover_ny, self.cover_nx), 99.0)
        if self.town is None:
            self.street_mask[:] = True
            return
        lx, ly = self._lim_x(), self._lim_y()
        boxes = self._town_boxes()
        for gy in range(self.cover_ny):
            for gx in range(self.cover_nx):
                world = self._cell_to_world(gx, gy)
                if abs(world[0]) > lx or abs(world[1]) > ly:
                    self.coverage[gy, gx] = 1.0
                    continue
                if self.town.is_walkable(float(world[0]), float(world[1])):
                    self.street_mask[gy, gx] = True
                    z = self._agl_z(world[0], world[1])
                    if boxes is not None:
                        self.street_clearance[gy, gx] = TownLayout.sphere_clearance(
                            [world[0], world[1], z], boxes
                        )
                else:
                    self.coverage[gy, gx] = 1.0

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

    def _nearest_street_cell(self, gx, gy):
        gx = int(np.clip(gx, 0, self.cover_nx - 1))
        gy = int(np.clip(gy, 0, self.cover_ny - 1))
        mask = getattr(self, "street_mask", None)
        if mask is None or mask[gy, gx]:
            return gx, gy
        ys, xs = np.nonzero(mask)
        if xs.size == 0:
            return gx, gy
        d2 = (xs - gx) ** 2 + (ys - gy) ** 2
        i = int(np.argmin(d2))
        return int(xs[i]), int(ys[i])

    def _line_clear_xy(self, a, b):
        ax, ay = float(a[0]), float(a[1])
        bx, by = float(b[0]), float(b[1])
        dist = float(np.hypot(bx - ax, by - ay))
        if dist < 1e-3:
            return True
        steps = max(2, int(np.ceil(dist / 2.0)))
        mask = getattr(self, "street_mask", None)
        for i in range(steps + 1):
            t = i / steps
            x = ax + t * (bx - ax)
            y = ay + t * (by - ay)
            gx, gy = self._world_to_cell(x, y)
            if mask is not None and not mask[gy, gx]:
                return False
            if self._hits_building([x, y, self._agl_z(x, y)], radius=BUILDING_HARD_M):
                return False
        return True

    def _street_waypoint(self, start_xy, goal_xy):
        """Next street point toward the goal, going around buildings."""
        start = np.asarray(start_xy, dtype=np.float64).reshape(-1)[:2]
        goal = np.asarray(goal_xy, dtype=np.float64).reshape(-1)[:2]
        if self._line_clear_xy(start, goal):
            return goal
        mask = getattr(self, "street_mask", None)
        if mask is None or not mask.any():
            return goal
        sgx, sgy = self._nearest_street_cell(*self._world_to_cell(start[0], start[1]))
        ggx, ggy = self._nearest_street_cell(*self._world_to_cell(goal[0], goal[1]))
        if sgx == ggx and sgy == ggy:
            return self._cell_to_world(ggx, ggy)
        neighbors = (
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        )
        dist = np.full(mask.shape, np.inf, dtype=np.float64)
        dist[sgy, sgx] = 0.0
        parent = {}
        heap = [(0.0, sgx, sgy)]
        found = False
        while heap:
            cost, cx, cy = heappop(heap)
            if cost > dist[cy, cx]:
                continue
            if cx == ggx and cy == ggy:
                found = True
                break
            for dx, dy in neighbors:
                nx, ny = cx + dx, cy + dy
                if nx < 0 or ny < 0 or nx >= self.cover_nx or ny >= self.cover_ny:
                    continue
                if not mask[ny, nx]:
                    continue
                step = 1.414 if dx != 0 and dy != 0 else 1.0
                clear = float(self.street_clearance[ny, nx])
                if clear < BUILDING_STANDOFF_M:
                    step += 3.0 * (BUILDING_STANDOFF_M - clear) / BUILDING_STANDOFF_M
                ncost = cost + step
                if ncost < dist[ny, nx]:
                    dist[ny, nx] = ncost
                    parent[(nx, ny)] = (cx, cy)
                    heappush(heap, (ncost, nx, ny))
        if not found:
            return goal
        path = [(ggx, ggy)]
        cur = (ggx, ggy)
        while cur != (sgx, sgy):
            cur = parent[cur]
            path.append(cur)
        path.reverse()
        idx = min(len(path) - 1, STREET_LOOKAHEAD_CELLS)
        return self._cell_to_world(path[idx][0], path[idx][1])

    def _unmapped_searchable_count(self):
        mask = getattr(self, "street_mask", None)
        if mask is not None:
            return int(np.sum((self.coverage < 0.5) & mask))
        n = 0
        for gy in range(self.cover_ny):
            for gx in range(self.cover_nx):
                if self.coverage[gy, gx] >= 0.5:
                    continue
                if self._cell_is_searchable(self._cell_to_world(gx, gy)):
                    n += 1
        return n

    def _has_walkable_unmapped(self):
        return self._unmapped_searchable_count() > 0

    def _cell_center_axes(self):
        """World x of each coverage column and y of each row. Built once."""
        key = (self.cover_nx, self.cover_ny, float(self.size_x), float(self.size_y))
        cached = getattr(self, "_cell_center_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1], cached[2]
        gx = np.arange(self.cover_nx, dtype=np.float64)
        gy = np.arange(self.cover_ny, dtype=np.float64)
        wx = (gx + 0.5) / self.cover_nx * self.size_x - self.size_x / 2.0
        wy = (gy + 0.5) / self.cover_ny * self.size_y - self.size_y / 2.0
        self._cell_center_cache = (key, wx, wy)
        return wx, wy

    def _visit_gain_grid(self):
        """Unmapped cells a sensor disk would paint, for every coverage cell."""
        unmapped = self.coverage < 0.5
        ny, nx = unmapped.shape
        cell_mx = self.size_x / self.cover_nx
        cell_my = self.size_y / self.cover_ny
        rad_x = max(1, int(np.ceil(self.sensor_radius / cell_mx)))
        rad_y = max(1, int(np.ceil(self.sensor_radius / cell_my)))
        gain = np.zeros((ny, nx), dtype=np.int32)
        radius = float(self.sensor_radius)
        for dy in range(-rad_y, rad_y + 1):
            for dx in range(-rad_x, rad_x + 1):
                if np.hypot(dx * cell_mx, dy * cell_my) > radius:
                    continue
                src_y = slice(max(dy, 0), ny + min(dy, 0))
                dst_y = slice(max(-dy, 0), ny + min(-dy, 0))
                src_x = slice(max(dx, 0), nx + min(dx, 0))
                dst_x = slice(max(-dx, 0), nx + min(-dx, 0))
                gain[dst_y, dst_x] += unmapped[src_y, src_x]
        return gain

    def _search_goal_xy(self, drone_index):
        """Hunt leftover people once mapping is mostly done; otherwise next-best-view."""
        people = self._undiscovered_people()
        leftover = self._unmapped_searchable_count()
        if people and leftover == 0:
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
        share = leftover <= LEFTOVER_CELLS

        wx_axis, wy_axis = self._cell_center_axes()
        playable = (
            (np.abs(wx_axis) <= self._lim_x() - GOAL_EDGE_M)[None, :]
            & (np.abs(wy_axis) <= self._lim_y() - GOAL_EDGE_M)[:, None]
        )
        open_cells = (self.coverage < 0.5) & self.street_mask & playable
        ys, xs = np.nonzero(open_cells)
        if xs.size == 0:
            if people:
                hunt = self._hunt_survivor_xy(drone_index)
                if hunt is not None:
                    self.goal_is_person[drone_index] = True
                    return hunt
            return None

        wx = wx_axis[xs]
        wy = wy_axis[ys]
        my_d = np.hypot(wx - my_xy[0], wy - my_xy[1])
        keep = np.ones(xs.shape[0], dtype=bool)
        if not share:
            for other in others:
                other_d = np.hypot(wx - other[0], wy - other[1])
                keep &= ~(other_d + 1.0 < my_d)
            for claim in claimed:
                claim_xy = np.asarray(claim, dtype=np.float64).reshape(-1)[:2]
                keep &= ~(np.hypot(wx - claim_xy[0], wy - claim_xy[1]) < CLAIM_KEEP_M)
        if not np.any(keep):
            i = int(np.argmin(my_d))
            return np.array([wx[i], wy[i]], dtype=np.float64)

        xs, ys, wx, wy, my_d = xs[keep], ys[keep], wx[keep], wy[keep], my_d[keep]
        gain = self._visit_gain_grid()[ys, xs].astype(np.float64)
        score = gain / (1.0 + 0.035 * my_d)
        eligible = my_d >= 8.0
        if heading is not None:
            align = np.zeros(my_d.shape, dtype=np.float64)
            far = my_d > 1e-3
            align[far] = (
                heading[0] * (wx[far] - my_xy[0]) + heading[1] * (wy[far] - my_xy[1])
            ) / my_d[far]
            score = score + 0.15 * np.maximum(0.0, align)
        score = np.where(eligible, score, -np.inf)
        if np.any(eligible):
            i = int(np.argmax(score))
            return np.array([wx[i], wy[i]], dtype=np.float64)
        i = int(np.argmin(my_d))
        return np.array([wx[i], wy[i]], dtype=np.float64)

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
        if self._goal_abandoned(drone_index, goal):
            return False
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

    def _ahead_street_xy(self, x, y):
        """A street cell far enough ahead to leave the cell the drone is already on."""
        mask = getattr(self, "street_mask", None)
        if mask is None or not mask.any():
            return None
        sx, sy = self._nearest_street_cell(*self._world_to_cell(x, y))
        neighbors = (
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        )
        best = None
        best_key = None
        for dx, dy in neighbors:
            cx, cy = sx, sy
            last = None
            unpainted = 0
            clear_sum = 0.0
            for _ in range(STREET_LOOKAHEAD_CELLS):
                nx, ny = cx + dx, cy + dy
                if nx < 0 or ny < 0 or nx >= self.cover_nx or ny >= self.cover_ny:
                    break
                if not mask[ny, nx]:
                    break
                cx, cy = nx, ny
                last = (cx, cy)
                if self.coverage[cy, cx] < 0.5:
                    unpainted += 1
                clear_sum += float(self.street_clearance[cy, cx])
            if last is None:
                continue
            world = self._cell_to_world(last[0], last[1])
            dist = float(np.hypot(world[0] - x, world[1] - y))
            if dist < 8.0:
                continue
            key = (unpainted, dist, clear_sum)
            if best_key is None or key > best_key:
                best_key = key
                best = world
        return best

    def _metric_search_dir(self, drone_index, x, y):
        target = self._search_target_xy(drone_index)
        if target is None:
            return 0.0, 0.0
        waypoint = self._street_waypoint((x, y), target)
        dx = float(waypoint[0] - x)
        dy = float(waypoint[1] - y)
        # On the near face of a pile the street point collapses under the drone
        # and the old code spun in place. Drop that goal and leave down the street.
        if abs(dx) + abs(dy) < 8.0 and not self._line_clear_xy((x, y), target):
            self._abandon_goal(drone_index, target)
            ahead = self._ahead_street_xy(x, y)
            if ahead is not None:
                dx = float(ahead[0] - x)
                dy = float(ahead[1] - y)
        return dx, dy

    def _abandon_goal(self, drone_index, target):
        if not hasattr(self, "abandoned_goals"):
            self.abandoned_goals = [[] for _ in range(self.num_drones)]
        bucket = self.abandoned_goals[drone_index]
        point = np.asarray(target, dtype=np.float64).reshape(2).copy()
        for prev in bucket:
            if float(np.hypot(prev[0] - point[0], prev[1] - point[1])) < 12.0:
                return
        bucket.append(point)

    def _goal_abandoned(self, drone_index, world):
        bucket = getattr(self, "abandoned_goals", None)
        if not bucket:
            return False
        point = np.asarray(world, dtype=np.float64).reshape(2)
        for prev in bucket[drone_index]:
            if float(np.hypot(prev[0] - point[0], prev[1] - point[1])) < 12.0:
                return True
        return False

    def _inland_search_dir(self, drone_index, pos):
        """Fly at the search goal. On a wall, drop the outbound component so they slide."""
        ux, uy = self._metric_search_dir(drone_index, pos[0], pos[1])
        lx, ly = self._lim_x(), self._lim_y()
        if pos[0] >= lx - TURN_MARGIN:
            ux = min(ux, 0.0)
        elif pos[0] <= -lx + TURN_MARGIN:
            ux = max(ux, 0.0)
        if pos[1] >= ly - TURN_MARGIN:
            uy = min(uy, 0.0)
        elif pos[1] <= -ly + TURN_MARGIN:
            uy = max(uy, 0.0)
        if abs(ux) + abs(uy) < 1e-6:
            if abs(pos[0]) >= lx - TURN_MARGIN:
                uy = 1.0 if pos[1] <= 0.0 else -1.0
            elif abs(pos[1]) >= ly - TURN_MARGIN:
                ux = 1.0 if pos[0] <= 0.0 else -1.0
            else:
                ux = -float(np.sign(pos[0])) or 0.0
                uy = -float(np.sign(pos[1])) or 0.0
        return ux, uy


    def _velocity_toward_unmapped(self, drone_index, pos):
        ux, uy = self._inland_search_dir(drone_index, pos)
        for j in range(self.num_drones):
            if j == drone_index or not self.drone_alive[j]:
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
        here = np.array([float(pos[0]), float(pos[1]), float(self._agl_z(pos[0], pos[1]))])
        # Parked on a facade used to pick a sideways heading and hover forever.
        # Inside the commit range, a goal that clips rubble is flown into the rubble.
        if (
            self._building_clearance(here) < BUILDING_COMMIT_M
            and self._heading_into_building(pos, np.array([ux, uy]))
        ):
            n = float(np.hypot(ux, uy)) or 1.0
            vel = np.zeros(3, dtype=np.float64)
            vel[0] = ux / n * self.drone_max_speed
            vel[1] = uy / n * self.drone_max_speed
            return vel
        legal = self._best_legal_heading(pos, ux, uy, drone_index=drone_index)
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

    def _agl_of(self, pos):
        return float(pos[2]) - self._ground_z(float(pos[0]), float(pos[1]))

    def _solid_boxes(self):
        town = self.town
        cached = getattr(self, "_solid_boxes_cache", None)
        if cached is not None and cached[0] is town:
            return cached[1]
        boxes = self._town_boxes()
        result = None
        if boxes is not None:
            if self.legacy_xy:
                result = boxes
            else:
                solid = getattr(town, "box_solid", None) if town is not None else None
                if solid is None or len(solid) != len(boxes):
                    result = boxes
                elif np.any(solid):
                    result = np.ascontiguousarray(boxes[np.asarray(solid, dtype=bool)])
        self._solid_boxes_cache = (town, result)
        return result

    def _boxes_near_points(self, boxes, points, margin):
        """Boxes whose AABB can touch the points expanded by margin."""
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        pad = float(margin)
        xmin = float(pts[:, 0].min()) - pad
        xmax = float(pts[:, 0].max()) + pad
        ymin = float(pts[:, 1].min()) - pad
        ymax = float(pts[:, 1].max()) + pad
        zmin = float(pts[:, 2].min()) - pad
        zmax = float(pts[:, 2].max()) + pad
        keep = (
            (boxes[:, 3] >= xmin)
            & (boxes[:, 0] <= xmax)
            & (boxes[:, 4] >= ymin)
            & (boxes[:, 1] <= ymax)
            & (boxes[:, 5] >= zmin)
            & (boxes[:, 2] <= zmax)
        )
        return boxes[keep]

    def _hits_solid(self, pos, radius=None):
        boxes = self._solid_boxes()
        if boxes is None:
            return False
        r = BUILDING_RADIUS if radius is None else float(radius)
        p = np.asarray(pos, dtype=np.float64).reshape(3)
        near = self._boxes_near_points(boxes, p, r)
        if len(near) == 0:
            return False
        return TownLayout.sphere_hits_boxes(p, r, near)

    def _solid_clearance(self, pos):
        boxes = self._solid_boxes()
        if boxes is None:
            return 1e6
        p = np.asarray(pos, dtype=np.float64).reshape(3)
        return TownLayout.sphere_clearance(p, boxes)

    def _path_hits_solid(self, pos_a, pos_b):
        boxes = self._solid_boxes()
        if boxes is None:
            return False
        a = np.asarray(pos_a, dtype=np.float64).reshape(3)
        b = np.asarray(pos_b, dtype=np.float64).reshape(3)
        near = self._boxes_near_points(boxes, np.stack((a, b)), BUILDING_RADIUS)
        if len(near) == 0:
            return False
        if TownLayout.sphere_hits_boxes(b, BUILDING_RADIUS, near):
            return True
        if TownLayout.segment_hits_boxes(a, b, near):
            return True
        mid = 0.5 * (a + b)
        return TownLayout.sphere_hits_boxes(mid, BUILDING_RADIUS, near)

    def _pair_paths_hit(self, origins, targets):
        """Same blocked test as _path_hits_solid, for many segments at once."""
        origins = np.asarray(origins, dtype=np.float64).reshape(-1, 3)
        targets = np.asarray(targets, dtype=np.float64).reshape(-1, 3)
        count = len(origins)
        boxes = self._solid_boxes()
        if boxes is None or count == 0:
            return np.zeros(count, dtype=bool)
        end_hit = TownLayout.sphere_hit_mask(targets, BUILDING_RADIUS, boxes)
        seg_hit = TownLayout.segments_hit_boxes(origins, targets, boxes)
        mid_hit = TownLayout.sphere_hit_mask(0.5 * (origins + targets), BUILDING_RADIUS, boxes)
        return end_hit | seg_hit | mid_hit

    def _motion_end(self, pos, heading_idx, vz_sign):
        """Where one discrete step would land, including the altitude clamp."""
        dt = 1.0 / self.pyb_freq
        step = self.drone_max_speed * dt
        heading = _HEADING_XY[int(heading_idx)]
        end = np.asarray(pos, dtype=np.float64).reshape(3).copy()
        end[0] += float(heading[0]) * step
        end[1] += float(heading[1]) * step
        lx, ly = self._lim_x(), self._lim_y()
        end[0] = float(np.clip(end[0], -lx, lx))
        end[1] = float(np.clip(end[1], -ly, ly))
        if self.legacy_xy:
            end[2] = self._agl_z(end[0], end[1])
            return end
        gz = self._ground_z(end[0], end[1])
        z = float(pos[2]) + float(vz_sign) * step
        end[2] = float(np.clip(z, gz + MIN_AGL, gz + self.drone_max_altitude))
        return end

    def _motion_blocked(self, pos, heading_idx, vz_sign):
        """True when this heading and vertical sign would hit a wall or the altitude limit."""
        heading = _HEADING_XY[int(heading_idx)]
        if abs(float(heading[0])) + abs(float(heading[1])) > 0.1 and self._map_edge_blocked(pos, heading):
            return True
        end = self._motion_end(pos, heading_idx, vz_sign)
        if float(vz_sign) > 0.0 and end[2] <= float(pos[2]) + 0.05:
            return True
        if float(vz_sign) < 0.0 and end[2] >= float(pos[2]) - 0.05:
            return True
        start = np.asarray(pos, dtype=np.float64).reshape(3).copy()
        if self.legacy_xy:
            start[2] = self._agl_z(float(start[0]), float(start[1]))
        if self._path_hits_solid(start, end):
            return True
        end_fly = np.asarray(end, dtype=np.float64).reshape(3).copy()
        end_fly[2] = self._agl_z(float(end_fly[0]), float(end_fly[1]))
        return self._building_clearance(end_fly) < BODY_KEEP_M


    def _rubble_sense(self, pos):
        """Clearance at this height, level hits, and which headings open if the drone climbs."""
        if self.legacy_xy:
            pos = np.asarray(pos, dtype=np.float64).reshape(3).copy()
            pos[2] = self._agl_z(float(pos[0]), float(pos[1]))
        here = [float(pos[0]), float(pos[1]), float(pos[2])]
        clear = float(np.clip(self._solid_clearance(here) / 20.0, 0.0, 1.0))
        hits = np.zeros(8, dtype=np.float32)
        helps = np.zeros(8, dtype=np.float32)
        for k in range(8):
            level_hit = self._motion_blocked(pos, k, 0.0)
            climb_hit = self._motion_blocked(pos, k, 1.0)
            # Camera ray from the drone. A hit means a solid within CAMERA_M, not one step.
            ahead = self._solid_blocking(
                pos, _HEADING_XY[k, 0], _HEADING_XY[k, 1], reach=CAMERA_M
            )
            hits[k] = 1.0 if ahead is not None else 0.0
            # Climb helps only for a shell. A tower on this heading is not a climb.
            too_tall = ahead is not None and self._too_tall_to_clear(pos, ahead[1])
            if too_tall:
                helps[k] = 0.0
            elif ahead is not None and ahead[0] <= TALL_AVOID_M:
                helps[k] = 1.0
            else:
                helps[k] = 1.0 if level_hit and not climb_hit else 0.0
        return clear, hits, helps

    def _radio_pos(self, pos):
        """Where the drone actually flies: hover height, which is what the radio uses."""
        p = np.asarray(pos, dtype=np.float64).reshape(3).copy()
        p[2] = self._agl_z(float(p[0]), float(p[1]))
        return p

    def _radio_link_pos(self, a, b):
        """True when the mesh link budget still closes between two hover positions."""
        a = np.asarray(a, dtype=np.float64).reshape(3)
        b = np.asarray(b, dtype=np.float64).reshape(3)
        delta = a - b
        dist = float(np.sqrt(delta @ delta))
        # Inside the blocked-hop range a wall cannot drop the link, so skip the ray.
        if dist <= _NLOS_LINK_M:
            return True
        if dist > _LOS_LINK_M:
            return False
        return not self._path_hits_solid(a, b)

    def _end_for_action(self, drone_index, action):
        pos = self.drone_positions[drone_index]
        action = int(action)
        if self.legacy_xy:
            heading = 8 if action >= 8 else action
            return self._radio_pos(self._motion_end(pos, heading, 0.0))
        if action >= 24:
            return self._radio_pos(pos)
        heading, _vz = _action_parts(action)
        return self._radio_pos(self._motion_end(pos, heading, 0.0))

    def _links_connected(self, alive, links):
        if len(alive) <= 1:
            return True
        parent = {i: i for i in alive}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for (i, j), ok in links.items():
            if not ok:
                continue
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[rj] = ri
        root = find(alive[0])
        return all(find(i) == root for i in alive)

    def mesh_connected(self):
        """Alive drones form one mesh. One drone is linked by itself."""
        alive = [i for i in range(self.num_drones) if self.drone_alive[i]]
        if len(alive) <= 1:
            return True
        pos = [self._radio_pos(self.drone_positions[i]) for i in range(self.num_drones)]
        links = {}
        for a, i in enumerate(alive):
            for j in alive[a + 1 :]:
                links[(i, j)] = self._radio_link_pos(pos[i], pos[j])
        return self._links_connected(alive, links)

    def _nearest_teammate_dist(self, origin, positions, others):
        best = 1e9
        for j in others:
            best = min(best, float(np.hypot(origin[0] - positions[j][0], origin[1] - positions[j][1])))
        return best

    def _gap_to_others(self, end, drone_index):
        best = 1e9
        for j in range(self.num_drones):
            if j == drone_index or not self.drone_alive[j]:
                continue
            best = min(
                best,
                float(np.hypot(end[0] - self.drone_positions[j, 0], end[1] - self.drone_positions[j, 1])),
            )
        return best

    def _apply_separation_mask(self, mask):
        """Drop a heading that would land inside another drone."""
        alive = [i for i in range(self.num_drones) if self.drone_alive[i]]
        if len(alive) <= 1:
            return mask
        n_actions = mask.shape[1]
        for i in alive:
            legal = [a for a in range(n_actions) if mask[i, a] > 0.5]
            if not legal:
                continue
            kept = []
            best_a = legal[0]
            best_gap = -1.0
            for a in legal:
                gap = self._gap_to_others(self._end_for_action(i, a), i)
                if gap > best_gap:
                    best_gap = gap
                    best_a = a
                if gap >= CRASH_RANGE:
                    kept.append(a)
            if not kept:
                kept = [best_a]
            allowed = set(kept)
            for a in range(n_actions):
                if a not in allowed:
                    mask[i, a] = 0.0
        return mask

    def _nudge_apart(self, drone_index, ux, uy, dist):
        cand = np.asarray(self.drone_positions[drone_index], dtype=np.float64).copy()
        cand[0] = float(np.clip(cand[0] + ux * dist, -self._lim_x(), self._lim_x()))
        cand[1] = float(np.clip(cand[1] + uy * dist, -self._lim_y(), self._lim_y()))
        cand[2] = self._agl_z(cand[0], cand[1])
        if self._hits_building(cand, radius=BUILDING_HARD_M):
            return False
        self.drone_positions[drone_index] = cand
        return True

    def _separate_drones(self):
        """Slide overlapping drones out to CRASH_RANGE. A wall keeps the one that has room."""
        alive = [i for i in range(self.num_drones) if self.drone_alive[i]]
        if len(alive) <= 1:
            return
        for _ in range(3):
            moved = False
            for a, i in enumerate(alive):
                for j in alive[a + 1 :]:
                    dx = float(self.drone_positions[i, 0] - self.drone_positions[j, 0])
                    dy = float(self.drone_positions[i, 1] - self.drone_positions[j, 1])
                    dist = float(np.hypot(dx, dy))
                    if dist >= CRASH_RANGE:
                        continue
                    if dist < 1e-4:
                        dx, dy, dist = 1.0, 0.0, 1.0
                    need = (CRASH_RANGE - dist) * 0.5 + 0.05
                    ux, uy = dx / dist, dy / dist
                    self._nudge_apart(i, ux, uy, need)
                    self._nudge_apart(j, -ux, -uy, need)
                    moved = True
            if not moved:
                return

    def _mesh_end_states(self, drone_index, ends, others, pos, alive, base):
        """For each landing, whether the mesh still holds and the nearest teammate gap."""
        ends = [np.asarray(end, dtype=np.float64).reshape(3) for end in ends]
        count = len(ends)
        if count == 0:
            return []
        others = list(others)
        link_ok = np.ones((count, len(others)), dtype=bool)
        origins = []
        targets = []
        slots = []
        for s, end in enumerate(ends):
            for k, j in enumerate(others):
                other = np.asarray(pos[j], dtype=np.float64).reshape(3)
                delta = end - other
                dist = float(np.sqrt(delta @ delta))
                if dist <= _NLOS_LINK_M:
                    continue
                if dist > _LOS_LINK_M:
                    link_ok[s, k] = False
                    continue
                origins.append(end)
                targets.append(other)
                slots.append((s, k))
        if slots:
            hits = self._pair_paths_hit(origins, targets)
            for (s, k), blocked in zip(slots, hits):
                link_ok[s, k] = not bool(blocked)
        states = []
        for s, end in enumerate(ends):
            links = dict(base)
            for k, j in enumerate(others):
                key = (drone_index, j) if drone_index < j else (j, drone_index)
                links[key] = bool(link_ok[s, k])
            gap = self._nearest_teammate_dist(end, pos, others)
            states.append((self._links_connected(alive, links), gap))
        return states

    def _apply_mesh_mask(self, mask):
        """Drop headings that would split the radio mesh. A broken mesh may only step closer."""
        mask = self._apply_separation_mask(mask)
        alive = [i for i in range(self.num_drones) if self.drone_alive[i]]
        if len(alive) <= 1:
            return mask
        pos = [self._radio_pos(self.drone_positions[i]) for i in range(self.num_drones)]
        base = {}
        for a, i in enumerate(alive):
            for j in alive[a + 1 :]:
                base[(i, j)] = self._radio_link_pos(pos[i], pos[j])
        connected_now = self._links_connected(alive, base)
        n_actions = mask.shape[1]
        stay = 8 if self.legacy_xy else 24
        for i in alive:
            others = [j for j in alive if j != i]
            legal = [a for a in range(n_actions) if mask[i, a] > 0.5]
            if not legal:
                continue
            here = pos[i]
            here_gap = self._nearest_teammate_dist(here, pos, others)
            kept = []
            closest = None
            end_state = {}
            pending = []
            action_ends = []
            for a in legal:
                end = self._end_for_action(i, a)
                key = end.tobytes()
                action_ends.append((a, key))
                if key not in end_state:
                    end_state[key] = None
                    pending.append((key, end))
            for key, state in zip(
                (item[0] for item in pending),
                self._mesh_end_states(i, [item[1] for item in pending], others, pos, alive, base),
            ):
                end_state[key] = state
            for a, key in action_ends:
                holds, gap = end_state[key]
                if connected_now:
                    if holds:
                        kept.append(a)
                elif holds or gap <= here_gap + 1.4:
                    kept.append(a)
                if closest is None or gap < closest[0]:
                    closest = (gap, a)
            if connected_now and not kept:
                kept = [stay] if stay in legal else ([closest[1]] if closest else legal[:1])
            elif not kept and closest is not None:
                kept = [closest[1]]
            if not kept:
                continue
            allowed = set(kept)
            for a in range(n_actions):
                if a not in allowed:
                    mask[i, a] = 0.0
        return mask

    def _mesh_keep_action(self, drone_index, idx):
        """If a later override picked a heading that drops the mesh, take a legal one."""
        mask = self.action_mask()[drone_index]
        idx = int(idx)
        if 0 <= idx < len(mask) and mask[idx] > 0.5:
            return idx
        stay = 8 if self.legacy_xy else 24
        if mask[stay] > 0.5:
            return stay
        for k in range(len(mask)):
            if mask[k] > 0.5:
                return k
        return idx

    def action_mask(self):
        """1 where the move is legal. Walls and towers are removed. Chunks are not. Stay stays."""
        sig = self.drone_positions.tobytes()
        cached = getattr(self, "_mask_cache", None)
        if cached is not None and cached[0] == sig:
            return cached[1]
        mask = self._build_action_mask()
        self._mask_cache = (sig, mask)
        return mask

    def _build_action_mask(self):
        mask = np.ones((self.num_drones, self.n_actions), dtype=np.float32)
        if self.legacy_xy:
            for i in range(self.num_drones):
                pos = self.drone_positions[i]
                for h in range(8):
                    if self._motion_blocked(pos, h, 0.0):
                        mask[i, h] = 0.0
            return self._apply_mesh_mask(mask)
        for i in range(self.num_drones):
            pos = self.drone_positions[i]
            tower_close = False
            for h in range(8):
                ahead = self._solid_blocking(
                    pos, _HEADING_XY[h, 0], _HEADING_XY[h, 1], reach=TALL_AVOID_M
                )
                too_tall = (
                    ahead is not None
                    and ahead[0] <= TALL_AVOID_M
                    and self._too_tall_to_clear(pos, ahead[1])
                )
                level_wall = ahead is not None and ahead[0] <= LEVEL_WALL_M
                if too_tall and ahead[0] <= 8.0:
                    tower_close = True
                for band, vz in enumerate(_VZ_BY_BAND):
                    blocked = self._motion_blocked(pos, h, vz)
                    # Never climb or cruise into a tower. Level flight also stops
                    # short of a shell so the drone can turn down the street.
                    if blocked or too_tall or (level_wall and vz == 0.0):
                        mask[i, band * 8 + h] = 0.0
            if self._motion_blocked(pos, 8, 1.0) or tower_close:
                mask[i, 25] = 0.0
            if self._motion_blocked(pos, 8, -1.0):
                mask[i, 26] = 0.0
        return self._apply_mesh_mask(mask)

    def _first_solid_ahead(self, pos, ux, uy, reach=45.0):
        """Nearest wall or tower the ray hits at this altitude. (distance, top_z) or None."""
        boxes = self._solid_boxes()
        if boxes is None or len(boxes) == 0:
            return None
        dx, dy = float(ux), float(uy)
        nrm = float(np.hypot(dx, dy))
        if nrm < 1e-6:
            return None
        dx, dy = dx / nrm, dy / nrm
        ox, oy, oz = float(pos[0]), float(pos[1]), float(pos[2])
        xmin, ymin, zmin = boxes[:, 0], boxes[:, 1], boxes[:, 2]
        xmax, ymax, zmax = boxes[:, 3], boxes[:, 4], boxes[:, 5]
        big = 1e9
        if abs(dx) < 1e-8:
            inside = (ox >= xmin) & (ox <= xmax)
            tx0 = np.where(inside, -big, big)
            tx1 = np.where(inside, big, -big)
        else:
            tx0 = np.minimum((xmin - ox) / dx, (xmax - ox) / dx)
            tx1 = np.maximum((xmin - ox) / dx, (xmax - ox) / dx)
        if abs(dy) < 1e-8:
            inside = (oy >= ymin) & (oy <= ymax)
            ty0 = np.where(inside, -big, big)
            ty1 = np.where(inside, big, -big)
        else:
            ty0 = np.minimum((ymin - oy) / dy, (ymax - oy) / dy)
            ty1 = np.maximum((ymin - oy) / dy, (ymax - oy) / dy)
        t_enter = np.maximum(tx0, ty0)
        t_exit = np.minimum(tx1, ty1)
        valid = (t_exit >= np.maximum(t_enter, 0.0)) & (t_enter <= reach) & (t_exit >= 0.0)
        valid &= (oz + 0.4 >= zmin) & (oz <= zmax + 0.4)
        if not np.any(valid):
            return None
        dist = np.where(valid, np.maximum(t_enter, 0.0), big)
        i = int(np.argmin(dist))
        if not np.isfinite(dist[i]) or dist[i] >= big / 2:
            return None
        return float(dist[i]), float(zmax[i])

    def _clearable_wall_ahead(self, pos, ux, uy, reach=28.0):
        """A wall on this ray the drone can fly over. (distance, top_z) or None.

        Towers above the altitude cap are ignored. A wall already below the drone is ignored.
        """
        info = self._first_solid_ahead(pos, ux, uy, reach=reach)
        if info is None:
            return None
        dist, top = info
        gz = self._ground_z(float(pos[0]), float(pos[1]))
        if top >= gz + self.drone_max_altitude - 1.0:
            return None
        if float(pos[2]) >= top + 1.0:
            return None
        return dist, top

    def _too_tall_to_clear(self, pos, top):
        gz = self._ground_z(float(pos[0]), float(pos[1]))
        return float(top) >= gz + self.drone_max_altitude - 1.0

    def _solid_blocking(self, pos, ux, uy, reach=22.0):
        """Solid on this ray the drone has not already flown over. (distance, top_z) or None."""
        info = self._first_solid_ahead(pos, ux, uy, reach=reach)
        if info is None:
            return None
        if float(pos[2]) >= info[1] + 1.0:
            return None
        return info

    def _note_climb_progress(self, drone_index, pos):
        """Count climb commands that do not raise the drone. A real climb resets it."""
        z = float(pos[2])
        if self._climb_intent[drone_index] and z <= float(self._stall_z[drone_index]) + 0.25:
            self._stall_n[drone_index] += 1
        else:
            self._stall_n[drone_index] = 0
        self._stall_z[drone_index] = z

    def _note_stuck(self, drone_index, pos):
        """True when the drone is still near where it was a while ago.

        A one-step wiggle does not clear this. A climb that is actually rising does.
        """
        if int(self._stuck_cool[drone_index]) > 0:
            self._stuck_cool[drone_index] -= 1
            self._xy_stall[drone_index] = 0
            self._stall_xy[drone_index, 0] = float(pos[0])
            self._stall_xy[drone_index, 1] = float(pos[1])
            return False
        z = float(pos[2])
        rising = self._climb_intent[drone_index] and z > float(self._stall_z[drone_index]) + 0.25
        dist = float(np.hypot(
            float(pos[0]) - float(self._stall_xy[drone_index, 0]),
            float(pos[1]) - float(self._stall_xy[drone_index, 1]),
        ))
        if rising or dist > STUCK_RADIUS:
            self._xy_stall[drone_index] = 0
            self._stall_xy[drone_index, 0] = float(pos[0])
            self._stall_xy[drone_index, 1] = float(pos[1])
            return False
        self._xy_stall[drone_index] += 1
        return int(self._xy_stall[drone_index]) >= STUCK_STEPS

    def _escape_aim(self, drone_index, pos, blocked, target):
        """Leave a wall that cannot be crossed. Street first, then back away from it."""
        if target is not None:
            self._abandon_goal(drone_index, target)
        self._stall_n[drone_index] = 0
        ahead = self._ahead_street_xy(float(pos[0]), float(pos[1]))
        if ahead is not None:
            delta = np.asarray(ahead, dtype=np.float64) - np.asarray(pos[:2], dtype=np.float64)
            norm = float(np.hypot(delta[0], delta[1]))
            if norm >= 1.0:
                aim = delta / norm
                if float(np.dot(aim, blocked)) < 0.35:
                    return aim
        return -np.asarray(blocked, dtype=np.float64)

    def _heading_index(self, ux, uy):
        best_k = 0
        best_dot = -1e9
        for k in range(8):
            dot = float(_HEADING_XY[k, 0]) * float(ux) + float(_HEADING_XY[k, 1]) * float(uy)
            if dot > best_dot:
                best_dot = dot
                best_k = k
        return best_k

    def _on_painted(self, pos):
        gx, gy = self._world_to_cell(float(pos[0]), float(pos[1]))
        return bool(self.coverage[gy, gx] >= 0.5)

    def _legal_toward(self, mask_row, ux, uy):
        best_h = None
        best_dot = -1e9
        for h in range(8):
            if mask_row[h] < 0.5:
                continue
            dot = float(_HEADING_XY[h, 0]) * float(ux) + float(_HEADING_XY[h, 1]) * float(uy)
            if dot > best_dot:
                best_dot = dot
                best_h = h
        return best_h

    def _pace_goal_heading(self, drone_index, pos):
        """Level heading toward leftover work, or None when the drone is already there."""
        if not self._work_remains():
            return None
        target = self._search_target_xy(drone_index)
        if target is None:
            return None
        delta = np.asarray(target, dtype=np.float64) - np.asarray(pos[:2], dtype=np.float64)
        dist = float(np.hypot(delta[0], delta[1]))
        if dist < self.sensor_radius:
            return None
        return self._heading_index(float(delta[0]), float(delta[1])), delta / dist

    def _pace_action(self, pos, heading, mask_row):
        """Descend to search height on the way; otherwise fly level toward the goal."""
        if self.legacy_xy:
            return heading if mask_row[heading] > 0.5 else None
        if self._agl_of(pos) > SEARCH_AGL_MAX and mask_row[16 + heading] > 0.5:
            return 16 + heading
        if mask_row[heading] > 0.5:
            return heading
        return None

    def _pace_decide(self, drone_index, pos, idx, mask_row=None):
        """Hold a heading toward leftover work after a shuttle on already painted ground."""
        if int(self._lock_n[drone_index]) > 0 or int(self._free_left[drone_index]) > 0:
            return None
        if mask_row is None:
            mask_row = self.action_mask()[drone_index]
        if int(self._pace_left[drone_index]) > 0:
            if not self._on_painted(pos):
                self._pace_left[drone_index] = 0
            else:
                goal = self._pace_goal_heading(drone_index, pos)
                if goal is None:
                    self._pace_left[drone_index] = 0
                else:
                    heading, aim = goal
                    held = int(self._pace_h[drone_index])
                    action = self._pace_action(pos, held, mask_row)
                    if action is None:
                        heading = self._legal_toward(mask_row, float(aim[0]), float(aim[1]))
                        if heading is None:
                            self._pace_left[drone_index] = 0
                            return None
                        self._pace_h[drone_index] = heading
                        action = self._pace_action(pos, heading, mask_row)
                    if action is None:
                        self._pace_left[drone_index] = 0
                        return None
                    self._pace_left[drone_index] -= 1
                    return int(action)
        if not self.legacy_xy and (8 <= idx <= 15 or idx == 25):
            return None
        if not self._on_painted(pos) or not self._work_remains():
            self._pace_n[drone_index] = 0
            self._pace_xy[drone_index, 0] = float(pos[0])
            self._pace_xy[drone_index, 1] = float(pos[1])
            return None
        dist = float(np.hypot(
            float(pos[0]) - float(self._pace_xy[drone_index, 0]),
            float(pos[1]) - float(self._pace_xy[drone_index, 1]),
        ))
        if dist > PACE_RADIUS:
            self._pace_n[drone_index] = 0
            self._pace_xy[drone_index, 0] = float(pos[0])
            self._pace_xy[drone_index, 1] = float(pos[1])
            return None
        self._pace_n[drone_index] += 1
        if int(self._pace_n[drone_index]) < PACE_STEPS:
            return None
        goal = self._pace_goal_heading(drone_index, pos)
        if goal is None:
            return None
        heading, aim = goal
        if mask_row[heading] < 0.5:
            heading = self._legal_toward(mask_row, float(aim[0]), float(aim[1]))
        if heading is None:
            return None
        action = self._pace_action(pos, heading, mask_row)
        if action is None:
            return None
        self._pace_h[drone_index] = heading
        self._pace_left[drone_index] = PACE_LOCK_STEPS
        self._pace_n[drone_index] = 0
        self._pace_xy[drone_index, 0] = float(pos[0])
        self._pace_xy[drone_index, 1] = float(pos[1])
        return int(action)

    def _pace_override(self, drone_index, pos, idx, tick, mask_row=None):
        """Same decision if the lesson and the step both ask during one env step."""
        if int(self._pace_tick[drone_index]) == int(tick):
            cached = int(self._pace_cached[drone_index])
            return idx if cached < 0 else cached
        self._pace_tick[drone_index] = int(tick)
        choice = self._pace_decide(drone_index, pos, idx, mask_row)
        self._pace_cached[drone_index] = -1 if choice is None else int(choice)
        return idx if choice is None else int(choice)

    def planner_actions(self, mask=None):
        """Head toward unpainted ground. Climb a clearable shell, then drop to search."""
        if mask is None:
            mask = self.action_mask()
        stay = 8.0 if self.legacy_xy else 24.0
        actions = np.full(self.num_drones, stay, dtype=np.float32)
        for i in range(self.num_drones):
            if not self.drone_alive[i]:
                continue
            pos = self.drone_positions[i]
            if not self.legacy_xy and int(self._lock_n[i]) > 0:
                locked = int(self._lock_h[i])
                if 0 <= locked < 8 and mask[i, locked] > 0.5:
                    self._lock_n[i] -= 1
                    actions[i] = float(locked)
                    self._plan_action[i] = locked
                    self._climb_intent[i] = False
                    self._note_stuck(i, pos)
                    continue
                self._lock_n[i] = 0
            ux, uy = self._inland_search_dir(i, pos)
            street = np.array([ux, uy], dtype=np.float64)
            sn = float(np.hypot(street[0], street[1])) or 1.0
            street = street / sn
            direct = None
            target = self._search_target_xy(i)
            if target is not None:
                direct = np.asarray(target, dtype=np.float64) - pos[:2]
                dn = float(np.hypot(direct[0], direct[1]))
                direct = direct / dn if dn > 1.0 else None
            stuck = self._note_stuck(i, pos)
            self._note_climb_progress(i, pos)
            wall = None
            aim = street
            if not self.legacy_xy:
                # Climb a shell early enough to clear it. A tower, or a climb that
                # stops gaining height, is not a climb: drop that cell and leave
                # down the street, or back away from the face.
                block = None
                blocked_dir = street
                if direct is not None:
                    dist_goal = float(np.linalg.norm(np.asarray(target, dtype=np.float64) - pos[:2]))
                    hit = self._solid_blocking(
                        pos, direct[0], direct[1], reach=max(dist_goal, 12.0)
                    )
                    if hit is not None and hit[0] < dist_goal - 1.5:
                        block = hit
                        blocked_dir = direct
                if block is None:
                    hit = self._solid_blocking(pos, street[0], street[1], reach=28.0)
                    if hit is not None:
                        block = hit
                        blocked_dir = street
                stalled = int(self._stall_n[i]) >= CLIMB_STALL_STEPS
                too_tall = block is not None and self._too_tall_to_clear(pos, block[1])
                # Held in place: climb a shell, or leave a tower. A one-step wiggle
                # does not count as progress.
                if stuck and block is not None and not too_tall:
                    wall = block
                    aim = blocked_dir
                    self._stuck_cool[i] = STUCK_STEPS
                    self._xy_stall[i] = 0
                elif stuck or (block is not None and (too_tall or stalled)):
                    aim = self._escape_aim(i, pos, blocked_dir, target)
                    locked = self._heading_index(float(aim[0]), float(aim[1]))
                    if mask[i, locked] > 0.5:
                        self._lock_h[i] = locked
                        self._lock_n[i] = DETOUR_LOCK_STEPS
                    self._stuck_cool[i] = STUCK_STEPS
                    self._xy_stall[i] = 0
                    wall = None
                elif block is not None:
                    wall = block
                    aim = blocked_dir
            desired = self._heading_index(aim[0], aim[1])
            agl = self._agl_of(pos)
            climbing = wall is not None and float(pos[2]) < wall[1] + 1.2
            choice = stay
            if self.legacy_xy:
                if mask[i, desired] > 0.5:
                    choice = float(desired)
            elif climbing:
                # Far enough that one climb step still clears the face: climb forward.
                # Against the wall, climb in place. Never fall through to a level ram.
                if wall[0] > 4.0 and mask[i, 8 + desired] > 0.5:
                    choice = float(8 + desired)
                elif mask[i, 25] > 0.5:
                    choice = 25.0
                elif mask[i, 8 + desired] > 0.5:
                    choice = float(8 + desired)
            else:
                crossing = False
                if agl > SEARCH_AGL_MAX:
                    over = self._first_solid_ahead(pos, aim[0], aim[1], reach=16.0)
                    if over is not None and float(pos[2]) > over[1] + 0.3 and over[0] < 14.0:
                        crossing = True
                    elif mask[i, 16 + desired] > 0.5:
                        choice = float(16 + desired)
                    elif mask[i, 26] > 0.5:
                        choice = 26.0
                if choice == stay and mask[i, desired] > 0.5:
                    choice = float(desired)
                elif choice == stay and not crossing and mask[i, 8 + desired] > 0.5:
                    choice = float(8 + desired)
                elif choice == stay and mask[i, 25] > 0.5 and self._clearable_wall_ahead(
                    pos, aim[0], aim[1], reach=10.0
                ):
                    choice = 25.0
                elif choice == stay:
                    best_k = 24
                    best_dot = -1e9
                    for k in range(8):
                        if mask[i, k] < 0.5:
                            continue
                        dot = (
                            float(_HEADING_XY[k, 0]) * float(aim[0])
                            + float(_HEADING_XY[k, 1]) * float(aim[1])
                        )
                        if dot > best_dot:
                            best_dot = dot
                            best_k = k
                    choice = float(best_k)
            choice = float(self._pace_override(i, pos, int(choice), self.step_count, mask[i]))
            choice = float(self._mesh_keep_action(i, int(choice)))
            actions[i] = choice
            self._plan_action[i] = int(choice)
            final = int(choice)
            climbing_now = (not self.legacy_xy) and (8 <= final <= 15 or final == 25)
            self._climb_intent[i] = bool(climbing_now and climbing)
        return actions

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

    def _town_boxes(self):
        boxes = getattr(self.town, "boxes", None) if self.town is not None else None
        if boxes is None or len(boxes) == 0:
            return None
        return np.asarray(boxes, dtype=np.float64)

    def _hits_building(self, pos, radius=None):
        boxes = self._town_boxes()
        if boxes is None:
            return False
        r = BUILDING_RADIUS if radius is None else float(radius)
        p = np.asarray(pos, dtype=np.float64).reshape(3)
        return TownLayout.sphere_hits_boxes(p, r, boxes)

    def _building_clearance(self, pos):
        boxes = self._town_boxes()
        if boxes is None:
            return 1e6
        p = np.asarray(pos, dtype=np.float64).reshape(3)
        return TownLayout.sphere_clearance(p, boxes)

    def _path_hits_building(self, pos_a, pos_b):
        if self._hits_building(pos_b):
            return True
        boxes = self._town_boxes()
        if boxes is None:
            return False
        a = np.asarray(pos_a, dtype=np.float64).reshape(3)
        b = np.asarray(pos_b, dtype=np.float64).reshape(3)
        if TownLayout.segment_hits_boxes(a, b, boxes):
            return True
        mid = 0.5 * (a + b)
        mid[2] = self._agl_z(mid[0], mid[1])
        return self._hits_building(mid)

    def _heading_into_building(self, pos, heading):
        """True only if this heading would actually clip a building soon."""
        hx, hy = float(heading[0]), float(heading[1])
        n = float(np.hypot(hx, hy))
        if n < 1e-6:
            return self._hits_building(pos, radius=BUILDING_HARD_M)
        hx, hy = hx / n, hy / n
        start = np.array(
            [float(pos[0]), float(pos[1]), float(self._agl_z(pos[0], pos[1]))],
            dtype=np.float64,
        )
        look = start.copy()
        look[0] += hx * BUILDING_LOOKAHEAD_M
        look[1] += hy * BUILDING_LOOKAHEAD_M
        look[2] = self._agl_z(look[0], look[1])
        if self._hits_building(look, radius=BUILDING_HARD_M):
            return True
        return self._path_hits_building(start, look)

    def _push_out_of_buildings(self, pos, radius=None):
        r = BUILDING_HARD_M if radius is None else float(radius)
        p = np.asarray(pos, dtype=np.float64).reshape(3).copy()
        p[2] = self._agl_z(p[0], p[1])
        if not self._hits_building(p, radius=r):
            return p
        boxes = self._town_boxes()
        best = None
        best_dist = 1e9
        if boxes is not None:
            pad = r + 0.4
            for box in boxes:
                if p[2] < box[2] - r or p[2] > box[5] + r:
                    continue
                if p[0] < box[0] - r or p[0] > box[3] + r or p[1] < box[1] - r or p[1] > box[4] + r:
                    continue
                faces = (
                    (box[0] - pad, p[1], p[0] - (box[0] - r)),
                    (box[3] + pad, p[1], (box[3] + r) - p[0]),
                    (p[0], box[1] - pad, p[1] - (box[1] - r)),
                    (p[0], box[4] + pad, (box[4] + r) - p[1]),
                )
                for x, y, dist in faces:
                    if dist >= best_dist:
                        continue
                    cand = p.copy()
                    cand[0] = float(np.clip(x, -self._lim_x(), self._lim_x()))
                    cand[1] = float(np.clip(y, -self._lim_y(), self._lim_y()))
                    cand[2] = self._agl_z(cand[0], cand[1])
                    if self._hits_building(cand, radius=r):
                        continue
                    best = cand
                    best_dist = float(dist)
        if best is not None:
            return best
        for dist in (r, r + 1.0, r + 2.5, r + 4.0, 8.0, 16.0, 28.0):
            for heading in _MOVE_HEADINGS:
                cand = p.copy()
                cand[0] += heading[0] * dist
                cand[1] += heading[1] * dist
                cand[0] = float(np.clip(cand[0], -self._lim_x(), self._lim_x()))
                cand[1] = float(np.clip(cand[1], -self._lim_y(), self._lim_y()))
                cand[2] = self._agl_z(cand[0], cand[1])
                if not self._hits_building(cand, radius=r):
                    return cand
        return p

    def _nearest_building_away(self, pos):
        """Unit XY pointing out of the nearest building, and the gap to it."""
        boxes = self._town_boxes()
        p = np.asarray(pos, dtype=np.float64).reshape(3)
        if boxes is None or len(boxes) == 0:
            return np.array([1.0, 0.0], dtype=np.float64), 1e9
        cx = np.clip(p[0], boxes[:, 0], boxes[:, 3])
        cy = np.clip(p[1], boxes[:, 1], boxes[:, 4])
        cz = np.clip(p[2], boxes[:, 2], boxes[:, 5])
        delta = np.stack((p[0] - cx, p[1] - cy, p[2] - cz), axis=1)
        dist = np.linalg.norm(delta, axis=1)
        i = int(np.argmin(dist))
        if dist[i] < 1e-3:
            left = float(p[0] - boxes[i, 0])
            right = float(boxes[i, 3] - p[0])
            down = float(p[1] - boxes[i, 1])
            up = float(boxes[i, 4] - p[1])
            if min(left, right) <= min(down, up):
                return np.array([-1.0 if left < right else 1.0, 0.0]), 0.0
            return np.array([0.0, -1.0 if down < up else 1.0]), 0.0
        away = delta[i, :2]
        norm = float(np.hypot(away[0], away[1])) or 1.0
        return away / norm, float(dist[i])

    def _best_exit_action(self, pos):
        """Level heading straight out from the nearest face. None if every step goes deeper."""
        pos = np.asarray(pos, dtype=np.float64).reshape(3)
        away, _gap = self._nearest_building_away(pos)
        here = self._building_clearance(pos)
        best_h = None
        best_dot = 0.15
        for h in range(8):
            end = self._motion_end(pos, h, 0.0)
            end[2] = self._agl_z(float(end[0]), float(end[1]))
            clear = self._building_clearance(end)
            if clear < here - 0.05:
                continue
            if self._path_hits_building(pos, end) and clear <= here + 0.05:
                continue
            dot = float(_HEADING_XY[h, 0]) * float(away[0]) + float(_HEADING_XY[h, 1]) * float(away[1])
            if dot > best_dot:
                best_dot = dot
                best_h = h
        return best_h

    def _keep_body_out_of_rubble(self, drone_index):
        """If the mesh overlaps a ruin, place it clear in this step."""
        clear = self._building_clearance(self.drone_positions[drone_index])
        if clear >= BODY_KEEP_M:
            if clear >= BODY_EXIT_M:
                self._exit_left[drone_index] = 0
            return
        pushed = self._push_out_of_buildings(self.drone_positions[drone_index], radius=BODY_EXIT_M)
        if self._building_clearance(pushed) > clear + 0.05:
            self.drone_positions[drone_index] = pushed
            self._mask_cache = None
            clear = self._building_clearance(pushed)
        if clear < BODY_EXIT_M:
            away, _gap = self._nearest_building_away(self.drone_positions[drone_index])
            nudged = np.asarray(self.drone_positions[drone_index], dtype=np.float64).copy()
            step = max(BODY_EXIT_M, BODY_EXIT_M - clear + 0.5)
            nudged[0] = float(np.clip(nudged[0] + float(away[0]) * step, -self._lim_x(), self._lim_x()))
            nudged[1] = float(np.clip(nudged[1] + float(away[1]) * step, -self._lim_y(), self._lim_y()))
            nudged[2] = self._agl_z(nudged[0], nudged[1])
            if self._building_clearance(nudged) >= clear + 1.0:
                self.drone_positions[drone_index] = nudged
                self._mask_cache = None
                clear = self._building_clearance(nudged)
        if clear >= BODY_KEEP_M:
            self._exit_left[drone_index] = 0
            return
        opened = self._best_exit_action(self.drone_positions[drone_index])
        if opened is None:
            return
        self._exit_action[drone_index] = int(opened)
        self._exit_left[drone_index] = 2

    def _on_border(self, pos, margin=None):
        if margin is None:
            margin = TURN_MARGIN
        return abs(pos[0]) >= self._lim_x() - margin or abs(pos[1]) >= self._lim_y() - margin

    def _map_edge_blocked(self, pos, heading):
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

    def _heading_blocked(self, pos, heading, drone_index=None):
        """True when this heading would leave the pad or fly into a building."""
        if self._map_edge_blocked(pos, heading):
            return True
        if self._heading_into_building(pos, heading):
            return True
        return False

    def _best_legal_heading(self, pos, ux, uy, drone_index=None):
        """Move toward (ux, uy). Refuse a clip; prefer a bit of facade clearance."""
        best = None
        best_key = None
        for heading in _MOVE_HEADINGS:
            map_ok = not self._map_edge_blocked(pos, heading)
            crash = self._heading_into_building(pos, heading)
            n = float(np.hypot(heading[0], heading[1]))
            look_x = float(pos[0]) + heading[0] / max(n, 1e-6) * 3.0
            look_y = float(pos[1]) + heading[1] / max(n, 1e-6) * 3.0
            clear = self._building_clearance([look_x, look_y, self._agl_z(look_x, look_y)])
            prefer = min(clear, BUILDING_STANDOFF_M)
            dot = float(heading[0]) * float(ux) + float(heading[1]) * float(uy)
            key = (map_ok, not crash, prefer, dot)
            if best_key is None or key > best_key:
                best_key = key
                best = heading
        if best is None:
            inward = np.array([-np.sign(pos[0]), -np.sign(pos[1])], dtype=np.float64)
            norm = float(np.linalg.norm(inward))
            best = np.array([1.0, 0.0]) if norm < 1e-6 else inward / norm
        return best

    def _sample_wind(self, seed):
        """One prevailing wind for the episode, plus a slow gust around it."""
        rng = np.random.default_rng(None if seed is None else int(seed) + 104729)
        speed = float(rng.uniform(0.0, WIND_MAX_MS))
        ang = float(rng.uniform(0.0, 2.0 * math.pi))
        self._wind_base = np.array(
            [math.cos(ang) * speed, math.sin(ang) * speed], dtype=np.float64
        )
        self._wind_gust_amp = float(rng.uniform(0.0, WIND_GUST_MS))
        self._wind_gust_heading = float(rng.uniform(0.0, 2.0 * math.pi))
        self._wind_phase = float(rng.uniform(0.0, 2.0 * math.pi))

    def _wind_xy(self):
        """Horizontal wind this step, in m/s, capped at WIND_MAX_MS."""
        phase = self._wind_phase + 2.0 * math.pi * self.step_count / WIND_GUST_PERIOD
        gust = self._wind_gust_amp * math.sin(phase)
        wx = float(self._wind_base[0] + math.cos(self._wind_gust_heading) * gust)
        wy = float(self._wind_base[1] + math.sin(self._wind_gust_heading) * gust)
        mag = math.hypot(wx, wy)
        if mag > WIND_MAX_MS:
            scale = WIND_MAX_MS / mag
            wx *= scale
            wy *= scale
        return wx, wy

    def wind_speed(self):
        wx, wy = self._wind_xy()
        return math.hypot(wx, wy)

    def _body_wind(self, vel):
        """Wind in the drone's air-velocity frame. Hover uses world +x as forward."""
        wx, wy = self._wind_xy()
        speed = float(np.hypot(vel[0], vel[1]))
        if speed > 0.5:
            fx, fy = float(vel[0]) / speed, float(vel[1]) / speed
        else:
            fx, fy = 1.0, 0.0
        forward = wx * fx + wy * fy
        right = wx * fy - wy * fx
        return forward, right

    def _move_with_soft_walls(self, drone_index, pos, vel, dt):
        """Fly the chosen heading. Map edges slide; rubble contact is a crash."""
        pos = np.asarray(pos, dtype=np.float64).copy()
        vel = np.asarray(vel, dtype=np.float64).copy()
        old = pos.copy()
        lx, ly = self._lim_x(), self._lim_y()
        # The chosen heading is air velocity. Wind is added on the ground track.
        # The map edge only drops the outward component so a drone can still slide.
        if pos[0] >= lx - TURN_MARGIN and vel[0] > 0.0:
            vel[0] = 0.0
        elif pos[0] <= -lx + TURN_MARGIN and vel[0] < 0.0:
            vel[0] = 0.0
        if pos[1] >= ly - TURN_MARGIN and vel[1] > 0.0:
            vel[1] = 0.0
        elif pos[1] <= -ly + TURN_MARGIN and vel[1] < 0.0:
            vel[1] = 0.0
        wx, wy = self._wind_xy()
        ground = vel.copy()
        ground[0] += wx
        ground[1] += wy
        if pos[0] >= lx - TURN_MARGIN and ground[0] > 0.0:
            ground[0] = 0.0
        elif pos[0] <= -lx + TURN_MARGIN and ground[0] < 0.0:
            ground[0] = 0.0
        if pos[1] >= ly - TURN_MARGIN and ground[1] > 0.0:
            ground[1] = 0.0
        elif pos[1] <= -ly + TURN_MARGIN and ground[1] < 0.0:
            ground[1] = 0.0
        pos = pos + ground * dt
        if pos[0] > lx:
            pos[0] = lx
            if vel[0] > 0.0:
                vel[0] = 0.0
        elif pos[0] < -lx:
            pos[0] = -lx
            if vel[0] < 0.0:
                vel[0] = 0.0
        if pos[1] > ly:
            pos[1] = ly
            if vel[1] > 0.0:
                vel[1] = 0.0
        elif pos[1] < -ly:
            pos[1] = -ly
            if vel[1] < 0.0:
                vel[1] = 0.0
        pos[0] = float(np.clip(pos[0], -lx, lx))
        pos[1] = float(np.clip(pos[1], -ly, ly))
        pos[2] = float(self._agl_z(pos[0], pos[1]))
        vel[2] = 0.0
        building_hit = False
        # Map edges slide. A step that would enter rubble stops short of it.
        new_clear = self._building_clearance(pos)
        blocked = (
            new_clear < BODY_KEEP_M
            or self._path_hits_building(old, pos)
            or self._hits_building(pos)
        )
        if blocked:
            # Stay put. If the body is already in the ruin, the yank below
            # places it clear in this same step instead of creeping out.
            building_hit = True
            pos = old.copy()
            pos[2] = float(self._agl_z(pos[0], pos[1]))
            vel[:] = 0.0
        pos[2] = float(self._agl_z(pos[0], pos[1]))
        vel[2] = 0.0
        return pos, vel, building_hit

    def _mark_coverage(self, positions):
        """Paint sensor footprints. Credit a new cell only to the drone that painted it."""
        new_cells = 0
        overlap = 0
        per_drone = np.zeros(len(positions), dtype=np.int32)
        per_overlap = np.zeros(len(positions), dtype=np.int32)
        already = self.coverage >= 1.0
        for i, pos in enumerate(positions):
            if i < len(self.drone_alive) and not self.drone_alive[i]:
                continue
            if self._agl_of(pos) > SEARCH_AGL_MAX:
                continue
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
        """Shared coverage map plus normalized drone XYZ, for the critic."""
        hx = max(self.size_x / 2.0, 1.0)
        hy = max(self.size_y / 2.0, 1.0)
        pos = np.empty((self.num_drones, 2), dtype=np.float32)
        pos[:, 0] = self.drone_positions[:, 0] / hx
        pos[:, 1] = self.drone_positions[:, 1] / hy
        alt = np.array(
            [self._agl_of(self.drone_positions[i]) / max(self.drone_max_altitude, 1.0) for i in range(self.num_drones)],
            dtype=np.float32,
        )
        bat = np.asarray(self.battery, dtype=np.float32)
        wx, wy = self._wind_xy()
        wind = np.array([wx / WIND_MAX_MS, wy / WIND_MAX_MS], dtype=np.float32)
        if self.legacy_xy:
            return np.concatenate([self._shared_map(), pos.ravel(), bat, wind])
        return np.concatenate([self._shared_map(), pos.ravel(), alt, bat, wind])

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
            p.configureDebugVisualizer(
                p.COV_ENABLE_KEYBOARD_SHORTCUTS, 0, physicsClientId=self.client
            )
            p.configureDebugVisualizer(
                p.COV_ENABLE_WIREFRAME, 0, physicsClientId=self.client
            )
            apply_spectator_sun(self.client, shadows=True, spawn_disc=True)
            p.removeAllUserDebugItems(physicsClientId=self.client)
            p.setRealTimeSimulation(0, physicsClientId=self.client)
        else:
            self.client = p.connect(p.DIRECT)
            apply_spectator_sun(self.client, shadows=False, spawn_disc=False)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, 0, physicsClientId=self.client)
        p.setTimeStep(1.0 / self.pyb_freq, physicsClientId=self.client)

    def _build_ground(self):
        if self.client is None:
            return
        if self.town is not None:
            self.terrain_body = spawn_terrain_in_pybullet(self.client, self.terrain)
            if getattr(self.town, "ruins", None):
                spawn_rubble_town_in_pybullet(self.client, self.town)
            if self._uses_search_meshes():
                people = None
                if self.survivors is not None:
                    people = self.survivors.get_positions()[:, :2]
                spawn_rubble_town_dressing(
                    self.client,
                    self.town,
                    self.terrain,
                    people_xy=people,
                    map_hx=0.5 * self.size_x,
                    map_hy=0.5 * self.size_y,
                )
                from .scenery import spawn_search_forest

                self.forest_trees = spawn_search_forest(
                    self.client,
                    self.terrain,
                    self.size_x,
                    self.size_y,
                    seed=self._seed if self._seed is not None else 42,
                )
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

    def _uses_search_meshes(self):
        return self.render_mode == "human" and self.client is not None

    def _clear_person_green_cache(self):
        """Drop cached found-person visuals and their temp mesh files."""
        self._person_green_vis = []
        files = getattr(self, "_person_green_obj_files", None) or []
        self._person_green_obj_files = []
        for path in files:
            try:
                os.unlink(path)
            except OSError:
                pass

    def _load_search_visuals(self):
        """Load character and drone meshes once per GUI client. Headless skips files."""
        self._person_vis = []
        self._person_ymin = []
        self._person_scale = []
        self._person_paths = []
        self._drone_vis = -1
        self._clear_person_green_cache()
        if not self._uses_search_meshes():
            return
        prefabs = _search_prefabs_dir()
        char_dir = os.path.join(prefabs, "CharacterModels")
        for letter in _PERSON_LETTERS:
            path = os.path.join(char_dir, f"character-{letter}.pb.obj")
            bounds = _obj_bounds(path)
            height = bounds["ymax"] - bounds["ymin"]
            scale = PERSON_TARGET_H / height if height > 1e-6 else 1.0
            vis = p.createVisualShape(
                p.GEOM_MESH,
                fileName=path,
                meshScale=[scale, scale, scale],
                rgbaColor=_MESH_RGBA,
                physicsClientId=self.client,
            )
            self._person_vis.append(vis)
            self._person_ymin.append(bounds["ymin"])
            self._person_scale.append(scale)
            self._person_paths.append(path)

        drone_path = os.path.join(prefabs, "Drone.obj")
        bounds = _obj_bounds(drone_path)
        longest = max(
            bounds["xmax"] - bounds["xmin"],
            bounds["ymax"] - bounds["ymin"],
            bounds["zmax"] - bounds["zmin"],
        )
        dscale = DRONE_TARGET_LONG / longest if longest > 1e-6 else 1.0
        # The file origin is not the body center. Shift the mesh so the label sits on it.
        cx = 0.5 * (bounds["xmin"] + bounds["xmax"])
        cy = 0.5 * (bounds["ymin"] + bounds["ymax"])
        cz = 0.5 * (bounds["zmin"] + bounds["zmax"])
        self._drone_vis = p.createVisualShape(
            p.GEOM_MESH,
            fileName=drone_path,
            meshScale=[dscale, dscale, dscale],
            visualFramePosition=[-cx * dscale, -cy * dscale, -cz * dscale],
            rgbaColor=_MESH_RGBA,
            physicsClientId=self.client,
        )
        self._drone_fix_euler = _upright_fix_euler(bounds)
        # File +Z is the nose; after +90 X that axis is world -Y, so +90 yaw faces +X.
        self._drone_yaw_offset = 0.5 * math.pi

    def _drone_orn(self, yaw):
        q_fix = p.getQuaternionFromEuler(list(self._drone_fix_euler))
        q_yaw = p.getQuaternionFromEuler(
            [0.0, 0.0, float(yaw) + float(self._drone_yaw_offset)]
        )
        return p.multiplyTransforms([0, 0, 0], q_yaw, [0, 0, 0], q_fix)[1]

    def _spawn_drones(self):
        self.drone_ids = []
        self._drone_yaw = np.zeros(self.num_drones, dtype=np.float64)
        self._drone_crash_tinted = np.zeros(self.num_drones, dtype=bool)
        use_mesh = self._uses_search_meshes() and self._drone_vis >= 0
        # Start inside the blocked-link range so the radio mesh is already one piece.
        spacing = 20.0
        cols = int(np.ceil(np.sqrt(self.num_drones)))
        for i in range(self.num_drones):
            col, row = i % cols, i // cols
            x = (col - (cols - 1) / 2.0) * spacing
            y = (row - (cols - 1) / 2.0) * spacing
            z = self._agl_z(x, y)
            cleared = self._push_out_of_buildings([x, y, z], radius=BUILDING_HARD_M)
            x, y, z = float(cleared[0]), float(cleared[1]), float(cleared[2])
            self.drone_positions[i] = [x, y, z]
        self._pull_spawn_into_mesh()
        for i in range(self.num_drones):
            if not use_mesh:
                self.drone_ids.append(-1)
                continue
            x, y, z = (float(v) for v in self.drone_positions[i])
            did = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=self._drone_vis,
                basePosition=[x, y, z],
                baseOrientation=self._drone_orn(self._drone_yaw[i]),
                physicsClientId=self.client,
            )
            self.drone_ids.append(did)
        if self.render_mode == "human" and self.client is not None:
            p.resetDebugVisualizerCamera(
                cameraDistance=max(self.size_x, self.size_y) * 1.05,
                cameraYaw=0,
                cameraPitch=-89,
                cameraTargetPosition=[0.0, 0.0, 0.0],
                physicsClientId=self.client,
            )

    def _pull_spawn_into_mesh(self):
        """Nudge any drone a ruin pushed out of radio range back toward the group."""
        for _ in range(40):
            if self.mesh_connected():
                return
            alive = [i for i in range(self.num_drones) if self.drone_alive[i]]
            pos = [self._radio_pos(self.drone_positions[i]) for i in range(self.num_drones)]
            parent = {i: i for i in alive}

            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            for a, i in enumerate(alive):
                for j in alive[a + 1 :]:
                    if self._radio_link_pos(pos[i], pos[j]):
                        ri, rj = find(i), find(j)
                        if ri != rj:
                            parent[rj] = ri
            groups = {}
            for i in alive:
                groups.setdefault(find(i), []).append(i)
            largest = max(groups.values(), key=len)
            for i in alive:
                if i in largest:
                    continue
                target = min(
                    largest,
                    key=lambda j: float(np.hypot(pos[i][0] - pos[j][0], pos[i][1] - pos[j][1])),
                )
                step = pos[target][:2] - pos[i][:2]
                norm = float(np.hypot(step[0], step[1]))
                if norm < 1e-3:
                    continue
                moved = pos[i].copy()
                moved[0] += float(step[0]) / norm * min(12.0, norm)
                moved[1] += float(step[1]) / norm * min(12.0, norm)
                cleared = self._push_out_of_buildings(self._radio_pos(moved), radius=BUILDING_HARD_M)
                self.drone_positions[i] = [
                    float(cleared[0]),
                    float(cleared[1]),
                    float(cleared[2]),
                ]

    def _spawn_person_meshes(self, rng):
        """One textured character per survivor. Pose is set once; no per-step updates."""
        self.person_ids = []
        self._person_models = []
        self._person_tinted = []
        if not self._uses_search_meshes() or self.survivors is None or not self._person_vis:
            return
        positions = self.survivors.get_positions()
        n_models = len(self._person_vis)
        for i in range(self.survivors.num_survivors):
            model = int(rng.integers(0, n_models))
            yaw = float(rng.uniform(0.0, 2.0 * math.pi))
            x, y = float(positions[i][0]), float(positions[i][1])
            gz = self._ground_z(x, y)
            scale = self._person_scale[model]
            z = gz - self._person_ymin[model] * scale
            bid = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=self._person_vis[model],
                basePosition=[x, y, z],
                baseOrientation=_mesh_yaw_orn(yaw),
                physicsClientId=self.client,
            )
            self.person_ids.append(bid)
            self._person_models.append(model)
            self._person_tinted.append(False)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._seed = seed
        self.agents = list(self.possible_agents)
        self.step_count = 0
        self.coverage = np.zeros((self.cover_ny, self.cover_nx), dtype=np.float32)
        self.drone_velocities = np.zeros((self.num_drones, 3))
        self.drone_alive = np.ones(self.num_drones, dtype=bool)
        self.battery = np.ones(self.num_drones, dtype=np.float64)
        self._cmd_vel = np.zeros((self.num_drones, 3), dtype=np.float64)
        self._sample_wind(self._seed)
        self.drone_finds = np.zeros(self.num_drones, dtype=np.int32)
        self.newly_discovered = 0
        self.new_cells = 0
        self.overlap_cells = 0
        self._drone_new_cells = np.zeros(self.num_drones, dtype=np.int32)
        self._drone_overlap = np.zeros(self.num_drones, dtype=np.int32)
        self.nav_goal = np.zeros((self.num_drones, 2), dtype=np.float64)
        self.has_nav_goal = np.zeros(self.num_drones, dtype=bool)
        self.goal_hold = np.zeros(self.num_drones, dtype=np.int32)
        self.abandoned_goals = [[] for _ in range(self.num_drones)]
        self._plan_action = np.full(self.num_drones, 24, dtype=np.int32)
        self._stall_z = np.zeros(self.num_drones, dtype=np.float64)
        self._stall_n = np.zeros(self.num_drones, dtype=np.int32)
        self._climb_intent = np.zeros(self.num_drones, dtype=bool)
        self._stall_xy = np.zeros((self.num_drones, 2), dtype=np.float64)
        self._xy_stall = np.zeros(self.num_drones, dtype=np.int32)
        self._lock_h = np.full(self.num_drones, -1, dtype=np.int32)
        self._lock_n = np.zeros(self.num_drones, dtype=np.int32)
        self._stuck_cool = np.zeros(self.num_drones, dtype=np.int32)
        self._free_xy = np.zeros((self.num_drones, 2), dtype=np.float64)
        self._free_n = np.zeros(self.num_drones, dtype=np.int32)
        self._free_left = np.zeros(self.num_drones, dtype=np.int32)
        self._free_h = np.full(self.num_drones, -1, dtype=np.int32)
        self._free_up = np.zeros(self.num_drones, dtype=bool)
        self._free_dir = np.zeros((self.num_drones, 2), dtype=np.float64)
        self._pace_xy = np.zeros((self.num_drones, 2), dtype=np.float64)
        self._pace_n = np.zeros(self.num_drones, dtype=np.int32)
        self._pace_left = np.zeros(self.num_drones, dtype=np.int32)
        self._pace_h = np.zeros(self.num_drones, dtype=np.int32)
        self._pace_tick = np.full(self.num_drones, -1, dtype=np.int32)
        self._pace_cached = np.full(self.num_drones, -1, dtype=np.int32)
        self.goal_is_person = np.zeros(self.num_drones, dtype=bool)
        self.wall_cooldown = np.zeros(self.num_drones, dtype=np.int32)
        self._exit_action = np.full(self.num_drones, 24, dtype=np.int32)
        self._exit_left = np.zeros(self.num_drones, dtype=np.int32)
        self.town = None
        self.forest_trees = 0
        self.person_ids = []
        self._person_vis = []
        self._person_paths = []
        self._person_models = []
        self._person_tinted = []
        self._drone_vis = -1
        if hasattr(self, "_lanes"):
            del self._lanes

        surv_seed = (int(time.time() * 1e6) ^ (os.getpid() << 8) ^ id(self)) & 0x7FFFFFFF
        if seed is not None:
            surv_seed ^= int(seed) & 0x7FFFFFFF

        pad = max(self.size_x, self.size_y)
        world = float(SEARCH_WORLD_M)
        hf_side = int(round(world / SEARCH_TERRAIN_RES))
        assert hf_side <= 512, f"heightfield side {hf_side} exceeds 512"
        assert hf_side * hf_side * 4 < 1_000_000, (
            f"heightfield upload {hf_side * hf_side * 4} bytes exceeds 1 MiB"
        )
        self.terrain = Terrain(
            size_x=world,
            size_y=world,
            resolution=SEARCH_TERRAIN_RES,
            seed=self._seed if self._seed is not None else 42,
            flat=False,
            with_structures=False,
            city_size=pad,
            city_centers=[(0.0, 0.0)],
            blend_width=40.0,
            color_blend_width=55.0,
            connector_segment=None,
            rim_width=220.0,
            n_rim_ranges=(5, 7),
            rim_peak_amp=(42.0, 78.0),
            rim_tall_amp=(80.0, 98.0),
            rim_tall_prob=0.18,
            rim_max_slope=0.36,
            rim_min_city_dist=180.0,
            n_tall_hills=(0, 0),
            n_hill_clusters=(3, 6),
            hill_amp_min=4.0,
            hill_amp_max=9.0,
            meadow_amp=0.25,
            grass_amp=1.4,
            edge_grass_width=25.0,
            edge_rise_amp=2.8,
            outer_fade_width=100.0,
        )
        hf_bytes = int(self.terrain.grid_x) * int(self.terrain.grid_y) * 4
        peak_lo, peak_hi = getattr(self.terrain, "peak_height_range", (0.0, 0.0))
        print(
            f"[SEARCH-WORLD] playable={self.size_x:.0f}x{self.size_y:.0f} m  "
            f"terrain={self.terrain.size_x:.0f}x{self.terrain.size_y:.0f} m  "
            f"grid={self.terrain.grid_x}x{self.terrain.grid_y}  "
            f"hf_bytes={hf_bytes}  peaks={peak_lo:.1f}–{peak_hi:.1f} m AGL"
        )
        town_size = min(float(RUBBLE_TOWN_SIZE), pad)
        self.town = RubbleTownLayout(
            self.terrain,
            size=town_size,
            seed=surv_seed,
            center=(0.0, 0.0),
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
        hx, hy = self._lim_x(), self._lim_y()
        self.survivors.positions[:, 0] = np.clip(self.survivors.positions[:, 0], -hx, hx)
        self.survivors.positions[:, 1] = np.clip(self.survivors.positions[:, 1], -hy, hy)
        self.survivors._snap_to_ground()
        self._seal_unsearchable()

        self._init_pybullet()
        self._load_search_visuals()
        self._build_ground()
        self._spawn_drones()
        self._spawn_person_meshes(np.random.default_rng(surv_seed))
        self._mark_coverage(self.drone_positions)
        self._refresh_nav_goals()

        return self._get_observations(), self._infos()

    def _nearest_blocking(self, pos):
        """Closest solid the forward rays see. (distance, top, heading) or None."""
        best = None
        for h in range(8):
            info = self._solid_blocking(
                pos, _HEADING_XY[h, 0], _HEADING_XY[h, 1], reach=CAMERA_M
            )
            if info is None:
                continue
            if best is None or info[0] < best[0]:
                best = (info[0], info[1], h)
        return best

    def _open_escape_heading(self, drone_index, pos, into):
        """Legal level heading that leaves the obstacle, or None."""
        mask = self.action_mask()[drone_index]
        best_h = None
        best_dot = 1e9
        for h in range(8):
            if mask[h] < 0.5:
                continue
            info = self._solid_blocking(
                pos, _HEADING_XY[h, 0], _HEADING_XY[h, 1], reach=6.0
            )
            if info is not None and info[0] < 4.0:
                continue
            dot = float(np.dot(_HEADING_XY[h], into))
            if dot < best_dot:
                best_dot = dot
                best_h = h
        return best_h

    def _held_too_long(self, drone_index, pos):
        """True after the drone has stayed inside STUCK_RADIUS for STUCK_STEPS."""
        if int(self._free_left[drone_index]) > 0:
            return False
        dist = float(np.hypot(
            float(pos[0]) - float(self._free_xy[drone_index, 0]),
            float(pos[1]) - float(self._free_xy[drone_index, 1]),
        ))
        if dist > STUCK_RADIUS:
            self._free_n[drone_index] = 0
            self._free_xy[drone_index, 0] = float(pos[0])
            self._free_xy[drone_index, 1] = float(pos[1])
            return False
        self._free_n[drone_index] += 1
        return int(self._free_n[drone_index]) >= STUCK_STEPS

    def _unstick_action(self, drone_index, idx, pos):
        """Replace a stuck drone's action. Shells get a climb. Towers get a held detour."""
        if self.legacy_xy:
            return idx
        mask = self.action_mask()[drone_index]
        if int(self._free_left[drone_index]) > 0:
            if self._free_up[drone_index]:
                opened = self._open_escape_heading(
                    drone_index, pos, self._free_dir[drone_index]
                )
                if opened is not None:
                    self._free_up[drone_index] = False
                    self._free_h[drone_index] = opened
                    self._free_left[drone_index] = DETOUR_LOCK_STEPS
                    return opened
                if mask[25] > 0.5:
                    self._free_left[drone_index] -= 1
                    return 25
            else:
                held = int(self._free_h[drone_index])
                if 0 <= held < 8 and mask[held] > 0.5:
                    self._free_left[drone_index] -= 1
                    return held
            self._free_left[drone_index] = 0
        if not self._held_too_long(drone_index, pos):
            return idx
        block = self._nearest_blocking(pos)
        if block is None:
            into = -_HEADING_XY[idx % 8] if idx < 24 else np.array([1.0, 0.0])
            top_tall = True
        else:
            into = _HEADING_XY[block[2]]
            top_tall = self._too_tall_to_clear(pos, block[1])
        self._free_dir[drone_index] = into
        opened = self._open_escape_heading(drone_index, pos, into)
        if top_tall or opened is not None:
            if opened is None:
                opened = self._heading_index(float(-into[0]), float(-into[1]))
                if mask[opened] < 0.5:
                    return idx
            self._free_n[drone_index] = 0
            self._free_xy[drone_index, 0] = float(pos[0])
            self._free_xy[drone_index, 1] = float(pos[1])
            self._free_up[drone_index] = False
            self._free_h[drone_index] = opened
            self._free_left[drone_index] = DETOUR_LOCK_STEPS
            return int(opened)
        if mask[25] > 0.5:
            self._free_up[drone_index] = True
            self._free_left[drone_index] = DETOUR_LOCK_STEPS
            return 25
        return idx

    def step(self, actions):
        self.step_count += 1
        dt = 1.0 / self.pyb_freq
        self._refresh_nav_goals()

        for i, agent in enumerate(self.possible_agents):
            if not self.drone_alive[i]:
                self.drone_velocities[i][:] = 0.0
                continue
            if agent not in actions:
                continue
            raw = np.asarray(actions[agent], dtype=np.float32).reshape(-1)
            if raw.size >= 3:
                act = np.clip(raw[:3], -1.0, 1.0)
                self.drone_velocities[i] = act * self.drone_max_speed
            else:
                idx = int(np.clip(np.rint(float(raw[0])) if raw.size else 0, 0, self.n_actions - 1))
                if int(self._exit_left[i]) > 0:
                    idx = int(self._exit_action[i])
                    self._exit_left[i] -= 1
                else:
                    idx = self._unstick_action(i, idx, self.drone_positions[i])
                    idx = self._pace_override(i, self.drone_positions[i], idx, self.step_count - 1)
                    idx = self._mesh_keep_action(i, idx)
                if self.legacy_xy:
                    heading_idx, vz_sign = (8, 0.0) if idx >= 8 else (idx, 0.0)
                else:
                    heading_idx, vz_sign = _action_parts(idx)
                heading = _HEADING_XY[heading_idx]
                self.drone_velocities[i, 0] = heading[0] * self.drone_max_speed
                self.drone_velocities[i, 1] = heading[1] * self.drone_max_speed
                self.drone_velocities[i, 2] = float(vz_sign) * self.drone_max_speed

        self._cmd_vel = self.drone_velocities.copy()
        collisions = np.zeros(self.num_drones, dtype=bool)
        building_kills = np.zeros(self.num_drones, dtype=bool)
        rubble_blocks = np.zeros(self.num_drones, dtype=bool)
        for i in range(self.num_drones):
            if not self.drone_alive[i]:
                self.drone_velocities[i][:] = 0.0
                continue
            if self.wall_cooldown[i] > 0:
                self.wall_cooldown[i] -= 1
            pos, vel, building_hit = self._move_with_soft_walls(
                i, self.drone_positions[i], self.drone_velocities[i], dt
            )
            self.drone_positions[i] = pos
            self.drone_velocities[i] = vel
            if building_hit and LETHAL_RUBBLE:
                self.drone_alive[i] = False
                building_kills[i] = True
                self.drone_velocities[i][:] = 0.0
                self.drone_positions[i, 2] = self._ground_z(pos[0], pos[1]) + 0.35
            elif building_hit:
                rubble_blocks[i] = True
                self.drone_velocities[i][:] = 0.0

        for i in range(self.num_drones):
            if not self.drone_alive[i]:
                continue
            self._keep_body_out_of_rubble(i)

        self._separate_drones()
        for i in range(self.num_drones):
            if not self.drone_alive[i]:
                continue
            for j in range(i + 1, self.num_drones):
                if not self.drone_alive[j]:
                    continue
                dist = float(np.linalg.norm(self.drone_positions[i][:2] - self.drone_positions[j][:2]))
                if dist < CRASH_RANGE:
                    collisions[i] = True
                    collisions[j] = True

        if self.client is not None:
            for i in range(self.num_drones):
                if self.drone_ids[i] < 0:
                    continue
                pos = self.drone_positions[i]
                vx = float(self.drone_velocities[i][0])
                vy = float(self.drone_velocities[i][1])
                if math.hypot(vx, vy) > 0.15:
                    self._drone_yaw[i] = math.atan2(vy, vx)
                p.resetBasePositionAndOrientation(
                    self.drone_ids[i],
                    pos.tolist(),
                    self._drone_orn(self._drone_yaw[i]),
                    physicsClientId=self.client,
                )
                p.resetBaseVelocity(
                    self.drone_ids[i], [0, 0, 0], [0, 0, 0], physicsClientId=self.client
                )
                if (not self.drone_alive[i]) and (not self._drone_crash_tinted[i]):
                    p.changeVisualShape(
                        self.drone_ids[i],
                        -1,
                        rgbaColor=_CRASH_RGBA,
                        physicsClientId=self.client,
                    )
                    self._drone_crash_tinted[i] = True

        self.new_cells, self.overlap_cells = self._mark_coverage(self.drone_positions)
        alive_pos = [
            self.drone_positions[i]
            for i in range(self.num_drones)
            if self.drone_alive[i] and self._agl_of(self.drone_positions[i]) <= SEARCH_AGL_MAX
        ]
        prev_discovered = np.asarray(self.survivors.discovered, dtype=bool).copy()
        self.newly_discovered, _ = self.survivors.update_discovery(alive_pos, self.terrain)
        self._tint_found_people(prev_discovered)
        self._credit_drone_finds(prev_discovered)
        battery_deaths = self._drain_batteries()

        rewards = self._compute_rewards(collisions, building_kills, rubble_blocks, battery_deaths)
        all_found = int(self.survivors.discovered.sum()) == self.survivors.num_survivors
        mapped = self._interior_coverage() >= 0.995
        maxed = self.step_count >= self.max_steps
        all_dead = not bool(self.drone_alive.any())
        truncations = {a: ((all_found and mapped) or maxed or all_dead) for a in self.possible_agents}
        terminations = {a: False for a in self.possible_agents}
        return self._get_observations(), rewards, terminations, truncations, self._infos()

    def _ensure_found_person_vis(self, model):
        """One shared solid-green visual per character model. Found people reuse it."""
        while len(self._person_green_vis) <= model:
            self._person_green_vis.append(-1)
        if self._person_green_vis[model] >= 0:
            return self._person_green_vis[model]
        path = self._person_paths[model]
        scale = self._person_scale[model]
        stripped = _write_textureless_obj(path)
        self._person_green_obj_files.append(stripped)
        vis = p.createVisualShape(
            p.GEOM_MESH,
            fileName=stripped,
            meshScale=[scale, scale, scale],
            rgbaColor=_FOUND_RGBA,
            physicsClientId=self.client,
        )
        self._person_green_vis[model] = vis
        return vis

    def _replace_person_with_green(self, i):
        """Give this body its own green visual so siblings keep their texture."""
        bid = int(self.person_ids[i])
        if bid < 0:
            return
        pos, orn = p.getBasePositionAndOrientation(bid, physicsClientId=self.client)
        model = int(self._person_models[i]) if i < len(self._person_models) else 0
        green_vis = self._ensure_found_person_vis(model)
        p.removeBody(bid, physicsClientId=self.client)
        new_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=green_vis,
            basePosition=pos,
            baseOrientation=orn,
            physicsClientId=self.client,
        )
        p.changeVisualShape(
            new_id,
            -1,
            rgbaColor=_FOUND_RGBA,
            textureUniqueId=-1,
            physicsClientId=self.client,
        )
        self.person_ids[i] = new_id

    def _tint_found_people(self, prev_discovered):
        """On false→true discovery, recolor that one person. GUI meshes only."""
        if self.client is None or not self.person_ids:
            return
        if self.survivors is None:
            return
        discovered = np.asarray(self.survivors.discovered, dtype=bool)
        prev = np.asarray(prev_discovered, dtype=bool)
        if discovered.shape != prev.shape:
            return
        for si in np.flatnonzero(discovered & ~prev):
            i = int(si)
            if i >= len(self.person_ids) or i >= len(self._person_tinted):
                continue
            if self._person_tinted[i]:
                continue
            self._replace_person_with_green(i)
            self._person_tinted[i] = True

    def _credit_drone_finds(self, prev_discovered):
        """When a discovered flag flips, credit the closest alive drone in sensor range."""
        discovered = np.asarray(self.survivors.discovered, dtype=bool)
        prev = np.asarray(prev_discovered, dtype=bool)
        if discovered.shape != prev.shape:
            return
        new_ids = np.flatnonzero(discovered & ~prev)
        if new_ids.size == 0:
            return
        radius = float(self.sensor_radius)
        for si in new_ids:
            surv_xy = self.survivors.positions[int(si)][:2]
            best_i = -1
            best_d = radius + 1.0
            for i in range(self.num_drones):
                if not self.drone_alive[i]:
                    continue
                dist = float(
                    np.hypot(
                        self.drone_positions[i][0] - surv_xy[0],
                        self.drone_positions[i][1] - surv_xy[1],
                    )
                )
                if dist <= radius and dist < best_d:
                    best_d = dist
                    best_i = i
            if best_i >= 0:
                self.drone_finds[best_i] += 1

    def _battery_cost(self, drone_index):
        """Charge for the commanded air move. Headwind costs more. Tailwind cannot go below hover."""
        vel = self._cmd_vel[drone_index]
        speed = float(np.hypot(vel[0], vel[1]))
        if speed < 0.5 and abs(float(vel[2])) < 0.5:
            return BATTERY_HOVER_DRAIN
        wx, wy = self._wind_xy()
        along = (float(vel[0]) * wx + float(vel[1]) * wy) / max(speed, 1e-6)
        factor = 1.0 - along / WIND_MAX_MS
        floor = BATTERY_HOVER_DRAIN / BATTERY_LEVEL_DRAIN
        cost = BATTERY_LEVEL_DRAIN * max(floor, factor)
        if float(vel[2]) > 0.5:
            cost *= BATTERY_CLIMB_MULT
        return cost

    def _drain_batteries(self):
        """Spend charge. An empty pack kills that drone for the rest of the episode."""
        deaths = np.zeros(self.num_drones, dtype=bool)
        for i in range(self.num_drones):
            if not self.drone_alive[i]:
                self.battery[i] = 0.0
                continue
            self.battery[i] = max(0.0, float(self.battery[i]) - self._battery_cost(i))
            if self.battery[i] <= 0.0:
                self.battery[i] = 0.0
                self.drone_alive[i] = False
                self.drone_velocities[i][:] = 0.0
                deaths[i] = True
        return deaths

    def _compute_rewards(self, collisions, building_kills=None, rubble_blocks=None, battery_deaths=None):
        """New cells and new people. Sitting still is costly while work remains."""
        if building_kills is None:
            building_kills = np.zeros(self.num_drones, dtype=bool)
        if rubble_blocks is None:
            rubble_blocks = np.zeros(self.num_drones, dtype=bool)
        if battery_deaths is None:
            battery_deaths = np.zeros(self.num_drones, dtype=bool)
        stats = self.survivors.get_discovery_stats()
        all_found = (
            stats["discovered_survivors"] == stats["total_survivors"] and stats["total_survivors"] > 0
        )
        work_remains = self._interior_coverage() < 0.995 or not all_found
        mesh_broken = not self.mesh_connected()

        shared = 10.0 * float(self.newly_discovered)
        shared -= 0.01
        if all_found:
            shared += 50.0

        rewards = {}
        for i, agent in enumerate(self.possible_agents):
            rew = shared + float(self._drone_new_cells[i])
            if self._drone_new_cells[i] == 0 and self._drone_overlap[i] > 0:
                rew -= 0.25
            if self.drone_alive[i] and not rubble_blocks[i]:
                speed = float(np.hypot(self.drone_velocities[i][0], self.drone_velocities[i][1]))
                if speed < 0.5 and work_remains:
                    rew -= 1.0
            if collisions[i]:
                rew -= 1.0
            if mesh_broken and self.drone_alive[i]:
                rew -= MESH_BREAK_PENALTY
            if rubble_blocks[i]:
                rew -= RUBBLE_BLOCK_PENALTY
            if building_kills[i]:
                rew -= BUILDING_CRASH_PENALTY
            if battery_deaths[i]:
                rew -= BATTERY_DEATH_PENALTY
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
            obs[idx + 2] = float(self.battery[i])
            wind_fwd, wind_right = self._body_wind(self._cmd_vel[i])
            obs[idx + 3] = float(np.clip(wind_fwd / WIND_MAX_MS, -1.5, 1.5))
            obs[idx + 4] = float(np.clip(wind_right / WIND_MAX_MS, -1.5, 1.5))
            clear, hits, helps = self._rubble_sense(pos)
            obs[idx + 5] = clear
            obs[idx + 6 : idx + 14] = hits
            if self.legacy_xy:
                tail = idx + 14
            else:
                obs[idx + 14] = float(np.clip(self._agl_of(pos) / max(self.drone_max_altitude, 1.0), 0.0, 1.5))
                obs[idx + 15] = vel[2] / self.drone_max_speed
                obs[idx + 16 : idx + 24] = helps
                tail = idx + 24
            crop = self._coverage_crop(pos[0], pos[1])
            obs[tail : tail + crop.size] = crop
            shared = self._shared_map()
            obs[tail + crop.size : tail + crop.size + shared.size] = shared
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
            "alive_drones": int(self.drone_alive.sum()),
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
        self._clear_person_green_cache()
