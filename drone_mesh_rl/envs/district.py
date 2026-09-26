"""
district.py — One 300 m block district on the USGS DEM.

Seeded axis-aligned boxes on the existing heightmap. Walls share one
window texture (light wall, dark window grid, darker ground-floor band).
A thin untinted roof lid covers the top face so roofs have no windows.
Streets are long pavement slabs through leftover space in the same 300 m
square (buildings are not moved). Four box-built ruined shells sit in
street-edge leftover lots near the center. Sidewalk trees, street cars,
and one water tower sit in that square. copy_to can shift the same
relative layout elsewhere, but the viewer only spawns this one district.

A second ~240 m rubble town (RubbleTownLayout) sits ≥200 m outside that
square: twelve reused ruin shells and a larger broken-street grid only.
"""

from __future__ import annotations

import copy
import math
import os

import numpy as np
import pybullet as p

from .terrain import _write_png_rgb
from .town import _sample_heightmap

_WINDOW_TEX_NAME = "district_windows.png"
_ROAD_TEX_NAME = "district_road.png"
_ROOF_OVERHANG = 0.28
_ROOF_THICK = 0.32
_ROOF_TINT = 0.70
_ROAD_WIDTH = 8.0
_ROAD_CLEAR = 0.8
_ROAD_MIN_LEN = 40.0
_ROAD_LIFT = 0.10
_ROAD_HALF_THICK = 0.04
_ROAD_REPEAT_M = 8.0
_ROAD_MAX_BODIES = 36
_N_TREES = 20
_N_CARS = 12
_CAR_COLORS = (
    [0.90, 0.18, 0.16, 1.0],
    [0.16, 0.38, 0.90, 1.0],
    [0.95, 0.82, 0.16, 1.0],
    [0.94, 0.94, 0.96, 1.0],
)
_TRUNK_RGB = [0.40, 0.26, 0.14, 1.0]
_CANOPY_RGB = [0.22, 0.48, 0.22, 1.0]
_TOWER_STEM_RGB = [0.62, 0.62, 0.64, 1.0]
_TOWER_TANK_RGB = [0.55, 0.62, 0.70, 1.0]

DISTRICT_SIZE = 300.0
DISTRICT_SEED = 42
DISTRICT_GAP = 4.0
DISTRICT_MARGIN = 5.0

# Flat colors: shops warm white, mid-rises brick red, towers light blue
DISTRICT_COLORS = {
    "shop": [0.93, 0.90, 0.84, 1.0],
    "midrise": [0.62, 0.28, 0.24, 1.0],
    "tower": [0.62, 0.76, 0.90, 1.0],
}

# (kind, count, sx, sy, height_lo, height_hi)
_BUILDING_SPEC = (
    ("tower", 3, 18.0, 18.0, 45.0, 45.0),
    ("midrise", 15, 14.0, 14.0, 20.0, 28.0),
    ("shop", 30, 16.0, 12.0, 8.0, 12.0),
)

_N_RUINS = 4
_RUIN_SX = 16.0
_RUIN_SY = 14.0
_RUIN_H = (13.5, 15.2, 16.8, 17.6)
_RUIN_MAX_R = 70.0
_RUIN_WALL_T = (1.05, 0.90, 1.15, 0.85)
_RUIN_TIP_DEG = (34.0, 42.0, 48.0, 38.0)
# Charred gray / broken concrete — not shop white, brick red, or tower blue
_RUIN_WALL_RGB = (
    [0.34, 0.31, 0.28, 1.0],
    [0.30, 0.28, 0.25, 1.0],
    [0.32, 0.29, 0.26, 1.0],
    [0.28, 0.26, 0.23, 1.0],
)
_RUIN_SLAB_RGB = [0.31, 0.29, 0.26, 1.0]
_RUIN_CHUNK_RGB = (
    [0.27, 0.25, 0.22, 1.0],
    [0.36, 0.33, 0.29, 1.0],
    [0.24, 0.22, 0.20, 1.0],
    [0.33, 0.30, 0.26, 1.0],
)

RUBBLE_TOWN_SIZE = 240.0
RUBBLE_TOWN_GAP = 200.0
RUBBLE_TOWN_SEED = 71
_N_RUBBLE_SHELLS = 36
_RUBBLE_PITCH = 28.0
_RUBBLE_LINES = tuple(i * _RUBBLE_PITCH for i in range(-3, 4))
_RUBBLE_H = (
    13.5,
    14.6,
    15.2,
    16.1,
    16.8,
    17.6,
    14.1,
    15.7,
    16.4,
    13.8,
    17.2,
    15.0,
)
_BROKEN_ROAD_TEX_NAME = "rubble_town_road.png"
_BROKEN_ROAD_WIDTH = 6.4
_BROKEN_ROAD_LIFT = 0.08
_BROKEN_ROAD_HALF_THICK = 0.045
def _rubble_lots():
    """One shell in every block of the 28 m street grid."""
    lines = _RUBBLE_LINES
    sides = ("e", "n", "w", "s")
    lots = []
    for iy in range(len(lines) - 1):
        for ix in range(len(lines) - 1):
            dx = 0.5 * (lines[ix] + lines[ix + 1])
            dy = 0.5 * (lines[iy] + lines[iy + 1])
            lots.append((dx, dy, 15.0, 13.0, sides[(ix + iy) % 4]))
    return tuple(lots)


_RUBBLE_LOTS = _rubble_lots()


def find_flattest_square(terrain, size=DISTRICT_SIZE):
    """
    Sliding-window search for the flattest `size` x `size` square fully
    inside the terrain AABB. Score is mean |gradient|, then relief (max-min),
    then distance to the origin so the search prefers a central site.
    """
    Z = np.asarray(terrain.heightmap, dtype=np.float64)
    rx = float(getattr(terrain, "resolution_x", terrain.resolution))
    ry = float(getattr(terrain, "resolution_y", terrain.resolution))
    gy, gx = Z.shape
    nx = max(2, int(round(size / rx)))
    ny = max(2, int(round(size / ry)))
    if nx > gx or ny > gy:
        raise ValueError(
            f"district {size:.0f} m does not fit the {gx}x{gy} heightmap"
        )

    dzdy, dzdx = np.gradient(Z, ry, rx)
    slope = np.hypot(dzdx, dzdy)

    half_x = float(terrain.size_x) * 0.5
    half_y = float(terrain.size_y) * 0.5
    half = size * 0.5

    best = None
    for i in range(0, gy - ny + 1):
        for j in range(0, gx - nx + 1):
            cx = -half_x + (j + 0.5 * nx) * rx
            cy = -half_y + (i + 0.5 * ny) * ry
            if (
                cx - half < -half_x
                or cx + half > half_x
                or cy - half < -half_y
                or cy + half > half_y
            ):
                continue
            patch_s = slope[i : i + ny, j : j + nx]
            patch_z = Z[i : i + ny, j : j + nx]
            mean_slope = float(np.mean(patch_s))
            relief = float(np.max(patch_z) - np.min(patch_z))
            key = (mean_slope, relief, abs(cx) + abs(cy))
            if best is None or key < best[0]:
                best = (key, float(cx), float(cy))

    if best is None:
        raise RuntimeError("no in-bounds 300 m square on this heightmap")
    return best[1], best[2]


def _rects_overlap(a, b, gap):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return not (
        ax1 + gap <= bx0
        or ax0 - gap >= bx1
        or ay1 + gap <= by0
        or ay0 - gap >= by1
    )


def _place_buildings(terrain, center, size, seed):
    """Seeded non-overlapping AABBs fully inside the district square."""
    rng = np.random.default_rng(seed)
    cx, cy = float(center[0]), float(center[1])
    half = size * 0.5
    x_lo, x_hi = cx - half + DISTRICT_MARGIN, cx + half - DISTRICT_MARGIN
    y_lo, y_hi = cy - half + DISTRICT_MARGIN, cy + half - DISTRICT_MARGIN

    placed_xy = []
    buildings = []
    for kind, count, sx, sy, h_lo, h_hi in _BUILDING_SPEC:
        for _ in range(count):
            hx, hy = 0.5 * sx, 0.5 * sy
            ok = False
            for _try in range(4000):
                px = float(rng.uniform(x_lo + hx, x_hi - hx))
                py = float(rng.uniform(y_lo + hy, y_hi - hy))
                rect = (px - hx, py - hy, px + hx, py + hy)
                if any(_rects_overlap(rect, other, DISTRICT_GAP) for other in placed_xy):
                    continue
                gz = _sample_heightmap(terrain, px, py)
                height = float(h_lo if h_lo == h_hi else rng.uniform(h_lo, h_hi))
                buildings.append(
                    {
                        "kind": kind,
                        "cx": px,
                        "cy": py,
                        "xmin": rect[0],
                        "ymin": rect[1],
                        "xmax": rect[2],
                        "ymax": rect[3],
                        "zmin": gz,
                        "zmax": gz + height,
                        "rgba": list(DISTRICT_COLORS[kind]),
                    }
                )
                placed_xy.append(rect)
                ok = True
                break
            if not ok:
                raise RuntimeError(f"could not place {kind} without overlap")
    return buildings


def _merge_intervals(intervals):
    if not intervals:
        return []
    intervals = sorted(intervals)
    out = [list(intervals[0])]
    for a, b in intervals[1:]:
        if a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def _free_runs(lo, hi, blocked, min_len):
    """Open intervals in [lo, hi] after subtracting merged blocked ranges."""
    runs = []
    cursor = lo
    for a, b in _merge_intervals(blocked):
        a = max(lo, a)
        b = min(hi, b)
        if b <= a:
            continue
        if a - cursor >= min_len:
            runs.append((cursor, a))
        cursor = max(cursor, b)
    if hi - cursor >= min_len:
        runs.append((cursor, hi))
    return runs


def _ns_runs(x, buildings, y_lo, y_hi, half_w, pad, min_len):
    blocked = []
    for b in buildings:
        if b["xmax"] + pad < x - half_w or b["xmin"] - pad > x + half_w:
            continue
        blocked.append((b["ymin"] - pad, b["ymax"] + pad))
    return _free_runs(y_lo, y_hi, blocked, min_len)


def _ew_runs(y, buildings, x_lo, x_hi, half_w, pad, min_len):
    blocked = []
    for b in buildings:
        if b["ymax"] + pad < y - half_w or b["ymin"] - pad > y + half_w:
            continue
        blocked.append((b["xmin"] - pad, b["xmax"] + pad))
    return _free_runs(x_lo, x_hi, blocked, min_len)


def _pick_street_lines(candidates, min_gap, max_n):
    """Greedy pick of well-spaced (coord, score, runs) triples."""
    picked = []
    for coord, score, runs in sorted(candidates, key=lambda t: -t[1]):
        if all(abs(coord - p[0]) >= min_gap for p in picked):
            picked.append((coord, score, runs))
        if len(picked) >= max_n:
            break
    picked.sort(key=lambda t: t[0])
    return picked


def _layout_streets(buildings, center, size):
    """
    Axis-aligned pavement slabs through leftover space in the 300 m square.

    Buildings are not moved. Each slab is ≥40 m, clipped to the district,
    and kept clear of footprints.
    """
    cx, cy = float(center[0]), float(center[1])
    half = size * 0.5
    hw = 0.5 * _ROAD_WIDTH
    x_lo, x_hi = cx - half + hw, cx + half - hw
    y_lo, y_hi = cy - half + hw, cy + half - hw

    ns_cands = []
    for x in np.linspace(x_lo, x_hi, 49):
        runs = _ns_runs(float(x), buildings, y_lo, y_hi, hw, _ROAD_CLEAR, _ROAD_MIN_LEN)
        score = sum(b - a for a, b in runs)
        if score >= _ROAD_MIN_LEN:
            ns_cands.append((float(x), score, runs))
    ew_cands = []
    for y in np.linspace(y_lo, y_hi, 49):
        runs = _ew_runs(float(y), buildings, x_lo, x_hi, hw, _ROAD_CLEAR, _ROAD_MIN_LEN)
        score = sum(b - a for a, b in runs)
        if score >= _ROAD_MIN_LEN:
            ew_cands.append((float(y), score, runs))

    ns_lines = _pick_street_lines(ns_cands, min_gap=55.0, max_n=4)
    ew_lines = _pick_street_lines(ew_cands, min_gap=55.0, max_n=4)

    slabs = []
    for x, _, runs in ns_lines:
        for y0, y1 in runs:
            slabs.append(
                {
                    "along": "y",
                    "xmin": x - hw,
                    "xmax": x + hw,
                    "ymin": y0,
                    "ymax": y1,
                    "cx": x,
                    "cy": 0.5 * (y0 + y1),
                    "hx": hw,
                    "hy": 0.5 * (y1 - y0),
                }
            )
    for y, _, runs in ew_lines:
        for x0, x1 in runs:
            slabs.append(
                {
                    "along": "x",
                    "xmin": x0,
                    "xmax": x1,
                    "ymin": y - hw,
                    "ymax": y + hw,
                    "cx": 0.5 * (x0 + x1),
                    "cy": y,
                    "hx": 0.5 * (x1 - x0),
                    "hy": hw,
                }
            )
    if len(slabs) > _ROAD_MAX_BODIES:
        slabs.sort(key=lambda s: -(s["xmax"] - s["xmin"] + s["ymax"] - s["ymin"]))
        slabs = slabs[:_ROAD_MAX_BODIES]
    return slabs


def _inside_square(x, y, center, size, margin=2.0):
    half = 0.5 * size - margin
    return abs(x - center[0]) <= half and abs(y - center[1]) <= half


def _hits_building(x, y, buildings, pad):
    for b in buildings:
        if (
            b["xmin"] - pad <= x <= b["xmax"] + pad
            and b["ymin"] - pad <= y <= b["ymax"] + pad
        ):
            return True
    return False


def _spaced_pick(rng, candidates, n, min_sep):
    rng.shuffle(candidates)
    picked = []
    for x, y, extra in candidates:
        if all(math.hypot(x - px, y - py) >= min_sep for px, py, _ in picked):
            picked.append((x, y, extra))
        if len(picked) >= n:
            break
    return picked


def _layout_props(terrain, buildings, roads, center, size, seed):
    """20 sidewalk trees, 12 street cars, one water tower. Buildings/roads unchanged."""
    rng = np.random.default_rng(int(seed) + 17)
    sidewalk = []
    lanes = []
    for slab in roads:
        along = slab["along"]
        if along == "y":
            x_left = slab["xmin"] + 0.7
            x_right = slab["xmax"] - 0.7
            y0, y1 = slab["ymin"] + 3.0, slab["ymax"] - 3.0
            if y1 <= y0:
                continue
            for y in np.linspace(y0, y1, max(2, int((y1 - y0) / 8.0))):
                sidewalk.append((x_left, float(y), slab))
                sidewalk.append((x_right, float(y), slab))
                lanes.append((slab["cx"] - 1.3, float(y), slab))
                lanes.append((slab["cx"] + 1.3, float(y), slab))
        else:
            y_south = slab["ymin"] + 0.7
            y_north = slab["ymax"] - 0.7
            x0, x1 = slab["xmin"] + 3.0, slab["xmax"] - 3.0
            if x1 <= x0:
                continue
            for x in np.linspace(x0, x1, max(2, int((x1 - x0) / 8.0))):
                sidewalk.append((float(x), y_south, slab))
                sidewalk.append((float(x), y_north, slab))
                lanes.append((float(x), slab["cy"] - 1.3, slab))
                lanes.append((float(x), slab["cy"] + 1.3, slab))

    tree_cands = [
        (x, y, slab)
        for x, y, slab in sidewalk
        if _inside_square(x, y, center, size) and not _hits_building(x, y, buildings, 1.6)
    ]
    car_cands = [
        (x, y, slab)
        for x, y, slab in lanes
        if _inside_square(x, y, center, size) and not _hits_building(x, y, buildings, 1.2)
    ]
    tree_pts = _spaced_pick(rng, tree_cands, _N_TREES, 11.0)
    if len(tree_pts) < _N_TREES:
        tree_pts = _spaced_pick(rng, tree_cands, _N_TREES, 7.0)
    car_pts = _spaced_pick(rng, car_cands, _N_CARS, 16.0)
    if len(car_pts) < _N_CARS:
        car_pts = _spaced_pick(rng, car_cands, _N_CARS, 10.0)

    trees = []
    for x, y, slab in tree_pts:
        pavement = float(slab["z"]) + _ROAD_HALF_THICK
        gz = _sample_heightmap(terrain, x, y)
        z0 = max(gz, pavement)
        trees.append({"cx": x, "cy": y, "z0": z0})

    cars = []
    car_h = 1.35
    for i, (x, y, slab) in enumerate(car_pts):
        along = slab["along"]
        # Long axis follows the street
        if along == "y":
            hx, hy = 0.85, 2.05
        else:
            hx, hy = 2.05, 0.85
        z0 = _sample_heightmap(terrain, x, y) + _ROAD_LIFT
        cars.append(
            {
                "cx": x,
                "cy": y,
                "hx": hx,
                "hy": hy,
                "hz": 0.5 * car_h,
                "z": z0 + 0.5 * car_h,
                "rgba": list(_CAR_COLORS[i % len(_CAR_COLORS)]),
            }
        )

    # Water tower near center: offset until the tank footprint is clear
    tank_hxy = 3.1
    tw_x, tw_y = float(center[0]), float(center[1])
    found = False
    for r in np.linspace(0.0, 70.0, 15):
        n_ang = 1 if r < 1e-6 else 10
        for k in range(n_ang):
            ang = 2.0 * math.pi * k / n_ang + 0.4
            px = float(center[0] + r * math.cos(ang))
            py = float(center[1] + r * math.sin(ang))
            if not _inside_square(px, py, center, size, margin=8.0):
                continue
            if _hits_building(px, py, buildings, tank_hxy + 1.0):
                continue
            tw_x, tw_y = px, py
            found = True
            break
        if found:
            break
    gz = _sample_heightmap(terrain, tw_x, tw_y)
    stem_h, tank_h, collar_h = 14.0, 4.2, 0.55
    landmark = {
        "cx": tw_x,
        "cy": tw_y,
        "z0": gz,
        "boxes": [
            {
                "hx": 0.85,
                "hy": 0.85,
                "hz": 0.5 * stem_h,
                "z": gz + 0.5 * stem_h,
                "rgba": list(_TOWER_STEM_RGB),
            },
            {
                "hx": 2.4,
                "hy": 2.4,
                "hz": 0.5 * collar_h,
                "z": gz + stem_h + 0.5 * collar_h,
                "rgba": list(_TOWER_STEM_RGB),
            },
            {
                "hx": tank_hxy,
                "hy": tank_hxy,
                "hz": 0.5 * tank_h,
                "z": gz + stem_h + collar_h + 0.5 * tank_h,
                "rgba": list(_TOWER_TANK_RGB),
            },
        ],
    }
    return trees, cars, landmark


def _quat_axis_angle(ax, ay, az, angle):
    s = math.sin(0.5 * angle)
    return [ax * s, ay * s, az * s, math.cos(0.5 * angle)]


def _quat_rotate(q, v):
    qx, qy, qz, qw = q
    x, y, z = v
    tx = 2.0 * (qy * z - qz * y)
    ty = 2.0 * (qz * x - qx * z)
    tz = 2.0 * (qx * y - qy * x)
    return (
        x + qw * tx + (qy * tz - qz * ty),
        y + qw * ty + (qz * tx - qx * tz),
        z + qw * tz + (qx * ty - qy * tx),
    )


def _support_z(terrain, points):
    return max(_sample_heightmap(terrain, float(x), float(y)) for x, y in points)


def _sit_oriented(terrain, cx, cy, hx, hy, hz, orn):
    corners = [
        _quat_rotate(orn, (sx * hx, sy * hy, sz * hz))
        for sx in (-1.0, 1.0)
        for sy in (-1.0, 1.0)
        for sz in (-1.0, 1.0)
    ]
    samples = [(cx + c[0], cy + c[1]) for c in corners]
    samples.append((cx, cy))
    gz = _support_z(terrain, samples)
    return gz - min(c[2] for c in corners)


def _aabb_overlap(a, b, gap=0.0):
    return _rects_overlap(a, b, gap)


def _car_rect(car):
    return (
        car["cx"] - car["hx"],
        car["cy"] - car["hy"],
        car["cx"] + car["hx"],
        car["cy"] + car["hy"],
    )


def _center_on_road(cx, cy, roads, inset=1.2):
    for slab in roads:
        if (
            slab["xmin"] + inset <= cx <= slab["xmax"] - inset
            and slab["ymin"] + inset <= cy <= slab["ymax"] - inset
        ):
            return True
    return False


def _nearest_street(cx, cy, roads):
    best = None
    best_d = 1e9
    for slab in roads:
        dx = max(slab["xmin"] - cx, 0.0, cx - slab["xmax"])
        dy = max(slab["ymin"] - cy, 0.0, cy - slab["ymax"])
        d = math.hypot(dx, dy)
        if d < best_d:
            best_d = d
            best = slab
    return best, best_d


def _beside_street(rect, cx, cy, roads, max_gap=8.0):
    if _center_on_road(cx, cy, roads):
        return False, None
    slab, dist = _nearest_street(cx, cy, roads)
    if slab is None:
        return False, None
    road_rect = (slab["xmin"], slab["ymin"], slab["xmax"], slab["ymax"])
    if _aabb_overlap(rect, road_rect, max_gap) or dist <= max_gap + 2.0:
        return True, slab
    return False, None


def _footprint_clear(rect, buildings, cars, landmark, placed, gap_b=0.05, gap_c=0.25):
    if any(
        _aabb_overlap(rect, (b["xmin"], b["ymin"], b["xmax"], b["ymax"]), gap_b)
        for b in buildings
    ):
        return False
    if any(_aabb_overlap(rect, _car_rect(car), gap_c) for car in cars):
        return False
    if landmark is not None:
        lx, ly = float(landmark["cx"]), float(landmark["cy"])
        if _aabb_overlap(rect, (lx - 3.3, ly - 3.3, lx + 3.3, ly + 3.3), 0.6):
            return False
    if any(_aabb_overlap(rect, other, 8.0) for other in placed):
        return False
    return True


def _open_toward_street(cx, cy, slab):
    if slab is None:
        return "e"
    if slab["along"] == "y":
        return "e" if cx < slab["cx"] else "w"
    return "n" if cy < slab["cy"] else "s"


def _ruin_piece(terrain, cx, cy, hx, hy, hz, orn, rgba, kind, lift=0.0):
    z_sit = _sit_oriented(terrain, cx, cy, hx, hy, hz, orn)
    if kind == "floor":
        gz = z_sit - hz
        z = gz + lift
    else:
        z = z_sit
    return {
        "hx": float(hx),
        "hy": float(hy),
        "hz": float(hz),
        "cx": float(cx),
        "cy": float(cy),
        "z": float(z),
        "orn": list(orn),
        "rgba": list(rgba),
        "kind": kind,
        "lift": float(lift),
    }


def _build_varied_ruin_shell(terrain, cx, cy, sx, sy, height, open_side, index, buildings, cars):
    """A charred shell whose walls, slab, and debris are not a rotated copy."""
    rng = np.random.default_rng(50_003 + int(index) * 97)
    hx, hy = 0.5 * sx, 0.5 * sy
    xmin, xmax = cx - hx, cx + hx
    ymin, ymax = cy - hy, cy + hy
    t = float(rng.uniform(0.38, 0.85))
    z0 = _support_z(
        terrain,
        [(xmin, ymin), (xmax, ymin), (xmin, ymax), (xmax, ymax), (cx, cy)],
    )
    opposite = {"n": "s", "s": "n", "e": "w", "w": "e"}[open_side]
    choices = [s for s in ("n", "s", "e", "w") if s != open_side]
    rng.shuffle(choices)
    n_walls = int(rng.integers(2, 4))
    if opposite not in choices[:n_walls] and rng.random() < 0.7:
        choices = [opposite] + [s for s in choices if s != opposite]
    sides = choices[:n_walls]
    yaw = float(rng.uniform(-0.22, 0.22))
    orn = _quat_axis_angle(0.0, 0.0, 1.0, yaw)
    pieces = []
    for wi, side in enumerate(sides):
        if side == "w":
            x0, x1, y0, y1 = xmin, xmin + t, ymin, ymax
        elif side == "e":
            x0, x1, y0, y1 = xmax - t, xmax, ymin, ymax
        elif side == "s":
            x0, x1, y0, y1 = xmin, xmax, ymin, ymin + t
        else:
            x0, x1, y0, y1 = xmin, xmax, ymax - t, ymax
        wh = height * float(rng.uniform(0.40, 1.0))
        pieces.append(
            _ruin_piece(
                terrain,
                0.5 * (x0 + x1),
                0.5 * (y0 + y1),
                0.5 * (x1 - x0),
                0.5 * (y1 - y0),
                0.5 * wh,
                orn,
                list(_RUIN_WALL_RGB[int(rng.integers(0, len(_RUIN_WALL_RGB)))]),
                "wall",
            )
        )

    ix0, ix1 = xmin + t, xmax - t
    iy0, iy1 = ymin + t, ymax - t
    span_x, span_y = max(ix1 - ix0, 1.0), max(iy1 - iy0, 1.0)
    frac = float(rng.uniform(0.35, 0.78))
    if open_side == "e":
        fx0, fx1, fy0, fy1 = ix0, ix0 + frac * span_x, iy0, iy1
    elif open_side == "w":
        fx0, fx1, fy0, fy1 = ix1 - frac * span_x, ix1, iy0, iy1
    elif open_side == "n":
        fx0, fx1, fy0, fy1 = ix0, ix1, iy0, iy0 + frac * span_y
    else:
        fx0, fx1, fy0, fy1 = ix0, ix1, iy1 - frac * span_y, iy1
    pieces.append(
        _ruin_piece(
            terrain,
            0.5 * (fx0 + fx1),
            0.5 * (fy0 + fy1),
            0.5 * (fx1 - fx0),
            0.5 * (fy1 - fy0),
            float(rng.uniform(0.16, 0.34)),
            orn,
            list(_RUIN_SLAB_RGB),
            "floor",
            lift=height * float(rng.uniform(0.22, 0.62)),
        )
    )

    tip = math.radians(float(rng.uniform(24.0, 56.0)))
    length = float(rng.uniform(6.4, 11.2))
    width = float(rng.uniform(4.0, 7.2))
    thick = float(rng.uniform(0.38, 0.7))
    if open_side in ("e", "w"):
        sign = 1.0 if open_side == "e" else -1.0
        tcx = (xmax if open_side == "e" else xmin) + sign * float(rng.uniform(0.6, 2.8))
        tcy = cy + float(rng.uniform(-2.5, 2.5))
        tip_orn = _quat_axis_angle(0.0, 1.0, 0.0, sign * tip)
        thx, thy, thz = 0.5 * length, 0.5 * width, 0.5 * thick
    else:
        sign = 1.0 if open_side == "n" else -1.0
        tcx = cx + float(rng.uniform(-2.5, 2.5))
        tcy = (ymax if open_side == "n" else ymin) + sign * float(rng.uniform(0.6, 2.8))
        tip_orn = _quat_axis_angle(1.0, 0.0, 0.0, -sign * tip)
        thx, thy, thz = 0.5 * width, 0.5 * length, 0.5 * thick
    pieces.append(
        _ruin_piece(terrain, tcx, tcy, thx, thy, thz, tip_orn, list(_RUIN_SLAB_RGB), "tip")
    )

    n_chunks = int(rng.integers(4, 7))
    catalog = (
        (3.6, 3.1, 1.7),
        (4.4, 3.2, 2.4),
        (3.2, 4.1, 1.5),
        (5.1, 3.4, 2.2),
        (3.8, 3.8, 1.9),
        (4.6, 2.9, 2.6),
    )
    order = rng.permutation(len(catalog))
    for ci in range(n_chunks):
        full = catalog[int(order[ci])]
        chx = 0.5 * full[0] * float(rng.uniform(0.85, 1.15))
        chy = 0.5 * full[1] * float(rng.uniform(0.85, 1.15))
        chx = max(chx, 1.6)
        chy = max(chy, 1.6)
        ang = float(rng.uniform(0.0, 2.0 * math.pi))
        rad = float(rng.uniform(0.15 * min(sx, sy), 0.55 * min(sx, sy)))
        px = cx + math.cos(ang) * rad
        py = cy + math.sin(ang) * rad
        if open_side == "e":
            px += float(rng.uniform(0.0, 3.0))
        elif open_side == "w":
            px -= float(rng.uniform(0.0, 3.0))
        elif open_side == "n":
            py += float(rng.uniform(0.0, 3.0))
        else:
            py -= float(rng.uniform(0.0, 3.0))
        crect = (px - chx, py - chy, px + chx, py + chy)
        if not _footprint_clear(crect, buildings, cars, None, [], gap_b=0.0, gap_c=0.15):
            px = 0.55 * px + 0.45 * cx
            py = 0.55 * py + 0.45 * cy
        pieces.append(
            _ruin_piece(
                terrain,
                px,
                py,
                chx,
                chy,
                0.5 * full[2],
                _quat_axis_angle(0.0, 0.0, 1.0, float(rng.uniform(-0.8, 0.8))),
                list(_RUIN_CHUNK_RGB[ci % len(_RUIN_CHUNK_RGB)]),
                "chunk",
            )
        )

    return {
        "cx": float(cx),
        "cy": float(cy),
        "sx": float(sx),
        "sy": float(sy),
        "height": float(height),
        "z0": float(z0),
        "xmin": float(xmin),
        "xmax": float(xmax),
        "ymin": float(ymin),
        "ymax": float(ymax),
        "open_side": open_side,
        "pieces": pieces,
    }


def _build_ruin_shell(terrain, cx, cy, sx, sy, height, open_side, index, buildings, cars, vary=False):
    """Broken shell: 2–3 walls, one attached floor, one tipped slab, 4–6 chunks."""
    if vary:
        return _build_varied_ruin_shell(
            terrain, cx, cy, sx, sy, height, open_side, index, buildings, cars
        )
    index = int(index) % 4
    hx, hy = 0.5 * sx, 0.5 * sy
    xmin, xmax = cx - hx, cx + hx
    ymin, ymax = cy - hy, cy + hy
    t = float(_RUIN_WALL_T[index])
    z0 = _support_z(
        terrain,
        [(xmin, ymin), (xmax, ymin), (xmin, ymax), (xmax, ymax), (cx, cy)],
    )
    n_walls = 3 if index % 2 == 0 else 2
    opposite = {"n": "s", "s": "n", "e": "w", "w": "e"}[open_side]
    sides = ["n", "s", "e", "w"]
    sides.remove(open_side)
    if n_walls == 2:
        adj = [s for s in sides if s != opposite]
        sides = [opposite, adj[index % len(adj)]]

    wall_rgb = list(_RUIN_WALL_RGB[index])
    pieces = []
    for wi, side in enumerate(sides):
        if side == "w":
            x0, x1, y0, y1 = xmin, xmin + t, ymin, ymax
        elif side == "e":
            x0, x1, y0, y1 = xmax - t, xmax, ymin, ymax
        elif side == "s":
            x0, x1, y0, y1 = xmin, xmax, ymin, ymin + t
        else:
            x0, x1, y0, y1 = xmin, xmax, ymax - t, ymax
        # Last wall a bit shorter so the shell reads as broken, not a neat box
        wh = height if wi < n_walls - 1 or n_walls == 2 else height * 0.78
        if n_walls == 2 and wi == 1:
            wh = height * 0.86
        pieces.append(
            _ruin_piece(
                terrain,
                0.5 * (x0 + x1),
                0.5 * (y0 + y1),
                0.5 * (x1 - x0),
                0.5 * (y1 - y0),
                0.5 * wh,
                [0.0, 0.0, 0.0, 1.0],
                wall_rgb if wi == 0 else _RUIN_WALL_RGB[(index + 1) % 4],
                "wall",
            )
        )

    # Attached floor slab on the back half of the interior
    ix0, ix1 = xmin + t, xmax - t
    iy0, iy1 = ymin + t, ymax - t
    span_x, span_y = max(ix1 - ix0, 1.0), max(iy1 - iy0, 1.0)
    if open_side == "e":
        fx0, fx1, fy0, fy1 = ix0, ix0 + 0.58 * span_x, iy0, iy1
    elif open_side == "w":
        fx0, fx1, fy0, fy1 = ix1 - 0.58 * span_x, ix1, iy0, iy1
    elif open_side == "n":
        fx0, fx1, fy0, fy1 = ix0, ix1, iy0, iy0 + 0.58 * span_y
    else:
        fx0, fx1, fy0, fy1 = ix0, ix1, iy1 - 0.58 * span_y, iy1
    attach_h = 0.46 * height
    pieces.append(
        _ruin_piece(
            terrain,
            0.5 * (fx0 + fx1),
            0.5 * (fy0 + fy1),
            0.5 * (fx1 - fx0),
            0.5 * (fy1 - fy0),
            0.24,
            [0.0, 0.0, 0.0, 1.0],
            _RUIN_SLAB_RGB,
            "floor",
            lift=attach_h,
        )
    )

    # Tipped floor slab, 30–50°, into the street through the missing wall
    tip = math.radians(_RUIN_TIP_DEG[index % len(_RUIN_TIP_DEG)])
    length, width, thick = 9.2, 5.8, 0.50
    if open_side in ("e", "w"):
        sign = 1.0 if open_side == "e" else -1.0
        tcx = (xmax if open_side == "e" else xmin) + sign * 1.8
        tcy = cy
        orn = _quat_axis_angle(0.0, 1.0, 0.0, sign * tip)
        thx, thy, thz = 0.5 * length, 0.5 * width, 0.5 * thick
    else:
        sign = 1.0 if open_side == "n" else -1.0
        tcx = cx
        tcy = (ymax if open_side == "n" else ymin) + sign * 1.8
        orn = _quat_axis_angle(1.0, 0.0, 0.0, -sign * tip)
        thx, thy, thz = 0.5 * width, 0.5 * length, 0.5 * thick
    pieces.append(
        _ruin_piece(
            terrain, tcx, tcy, thx, thy, thz, orn, _RUIN_SLAB_RGB, "tip"
        )
    )

    n_chunks = (5, 4, 6, 5)[index % 4]
    chunk_sizes = (
        (3.5, 3.2, 1.9),
        (4.2, 3.4, 2.3),
        (3.3, 3.9, 1.6),
        (3.8, 3.1, 2.1),
        (3.6, 3.6, 1.8),
        (4.0, 3.3, 2.0),
    )
    if open_side == "e":
        spots = (
            (xmax - 2.2, cy - 3.2),
            (xmax + 1.4, cy + 0.8),
            (cx + 1.6, cy + 3.6),
            (xmax - 0.6, cy + 3.4),
            (cx - 1.0, cy - 3.8),
            (xmax + 2.2, cy - 2.4),
        )
    elif open_side == "w":
        spots = (
            (xmin + 2.2, cy + 3.2),
            (xmin - 1.4, cy - 0.8),
            (cx - 1.6, cy - 3.6),
            (xmin + 0.6, cy - 3.4),
            (cx + 1.0, cy + 3.8),
            (xmin - 2.2, cy + 2.4),
        )
    elif open_side == "n":
        spots = (
            (cx - 3.2, ymax - 2.2),
            (cx + 0.8, ymax + 1.4),
            (cx + 3.6, cy + 1.6),
            (cx + 3.4, ymax - 0.6),
            (cx - 3.8, cy - 1.0),
            (cx - 2.4, ymax + 2.2),
        )
    else:
        spots = (
            (cx + 3.2, ymin + 2.2),
            (cx - 0.8, ymin - 1.4),
            (cx - 3.6, cy - 1.6),
            (cx - 3.4, ymin + 0.6),
            (cx + 3.8, cy + 1.0),
            (cx + 2.4, ymin - 2.2),
        )
    for ci in range(n_chunks):
        chx = 0.5 * chunk_sizes[ci][0]
        chy = 0.5 * chunk_sizes[ci][1]
        px, py = spots[ci]
        crect = (px - chx, py - chy, px + chx, py + chy)
        if not _footprint_clear(crect, buildings, cars, None, [], gap_b=0.0, gap_c=0.15):
            px = 0.55 * px + 0.45 * cx
            py = 0.55 * py + 0.45 * cy
        pieces.append(
            _ruin_piece(
                terrain,
                px,
                py,
                chx,
                chy,
                0.5 * chunk_sizes[ci][2],
                [0.0, 0.0, 0.0, 1.0],
                list(_RUIN_CHUNK_RGB[ci % len(_RUIN_CHUNK_RGB)]),
                "chunk",
            )
        )

    return {
        "cx": float(cx),
        "cy": float(cy),
        "sx": float(sx),
        "sy": float(sy),
        "height": float(height),
        "z0": float(z0),
        "xmin": float(xmin),
        "ymin": float(ymin),
        "xmax": float(xmax),
        "ymax": float(ymax),
        "open": open_side,
        "pieces": pieces,
    }


def _street_lot_candidates(roads, center, size):
    """Lot centers parked along both curbs of each street slab."""
    cx, cy = float(center[0]), float(center[1])
    half = 0.5 * size
    out = []
    for sx, sy in ((_RUIN_SX, _RUIN_SY), (_RUIN_SY, _RUIN_SX)):
        hx, hy = 0.5 * sx, 0.5 * sy
        for slab in roads:
            if slab["along"] == "y":
                y0 = slab["ymin"] + hy + 1.0
                y1 = slab["ymax"] - hy - 1.0
                if y1 < y0:
                    continue
                ys = np.linspace(y0, y1, max(3, int((y1 - y0) / 3.0) + 1))
                for sign in (-1.0, 1.0):
                    px = slab["cx"] + sign * (0.5 * _ROAD_WIDTH + 0.7 + hx)
                    if abs(px - cx) > half - hx - 1.0:
                        continue
                    for py in ys:
                        out.append((float(px), float(py), sx, sy, slab))
            else:
                x0 = slab["xmin"] + hx + 1.0
                x1 = slab["xmax"] - hx - 1.0
                if x1 < x0:
                    continue
                xs = np.linspace(x0, x1, max(3, int((x1 - x0) / 3.0) + 1))
                for sign in (-1.0, 1.0):
                    py = slab["cy"] + sign * (0.5 * _ROAD_WIDTH + 0.7 + hy)
                    if abs(py - cy) > half - hy - 1.0:
                        continue
                    for px in xs:
                        out.append((float(px), float(py), sx, sy, slab))
    return out


def _grid_lot_candidates(roads, center, size):
    cx, cy = float(center[0]), float(center[1])
    out = []
    for sx, sy in ((_RUIN_SX, _RUIN_SY), (_RUIN_SY, _RUIN_SX)):
        hx, hy = 0.5 * sx, 0.5 * sy
        for x in np.arange(cx - _RUIN_MAX_R, cx + _RUIN_MAX_R + 0.001, 2.0):
            for y in np.arange(cy - _RUIN_MAX_R, cy + _RUIN_MAX_R + 0.001, 2.0):
                if math.hypot(x - cx, y - cy) > _RUIN_MAX_R:
                    continue
                out.append((float(x), float(y), sx, sy, None))
    return out


def _lot_ok(px, py, sx, sy, roads, buildings, cars, landmark, placed, center, size):
    hx, hy = 0.5 * sx, 0.5 * sy
    rect = (px - hx, py - hy, px + hx, py + hy)
    corners = (
        (rect[0], rect[1]),
        (rect[2], rect[1]),
        (rect[0], rect[3]),
        (rect[2], rect[3]),
        (px, py),
    )
    if math.hypot(px - center[0], py - center[1]) > _RUIN_MAX_R:
        return False, None, rect
    if not all(_inside_square(x, y, center, size, margin=1.0) for x, y in corners):
        return False, None, rect
    beside, slab = _beside_street(rect, px, py, roads)
    if not beside:
        return False, None, rect
    if not _footprint_clear(rect, buildings, cars, landmark, placed):
        return False, None, rect
    return True, slab, rect


def _layout_ruins(terrain, buildings, roads, cars, landmark, center, size, seed):
    """Four 16×14 ruined shells in leftover street-edge lots near center."""
    cx, cy = float(center[0]), float(center[1])
    seen = set()
    ranked = []
    for px, py, sx, sy, hint in _street_lot_candidates(roads, center, size) + _grid_lot_candidates(
        roads, center, size
    ):
        key = (round(px, 1), round(py, 1), round(sx, 1), round(sy, 1))
        if key in seen:
            continue
        seen.add(key)
        ok, slab, rect = _lot_ok(
            px, py, sx, sy, roads, buildings, cars, landmark, [], center, size
        )
        if not ok:
            continue
        use = slab or hint
        dist = math.hypot(px - cx, py - cy)
        curb = 0.0 if hint is not None else 4.0
        ranked.append((dist + curb, dist, px, py, sx, sy, use, rect))
    ranked.sort(key=lambda t: t[0])

    picked = []
    placed = []
    for min_sep in (22.0, 16.0, 12.0, 8.0):
        picked = []
        placed = []
        for _score, dist, px, py, sx, sy, slab, rect in ranked:
            if any(math.hypot(px - r["cx"], py - r["cy"]) < min_sep for r in picked):
                continue
            # Re-check against already chosen ruins
            if any(_aabb_overlap(rect, other, 4.0) for other in placed):
                continue
            open_side = _open_toward_street(px, py, slab)
            height = float(_RUIN_H[len(picked)])
            ruin = _build_ruin_shell(
                terrain, px, py, sx, sy, height, open_side, len(picked), buildings, cars
            )
            picked.append(ruin)
            placed.append(rect)
            if len(picked) >= _N_RUINS:
                break
        if len(picked) >= _N_RUINS:
            break
    if len(picked) < _N_RUINS:
        raise RuntimeError(
            f"could only place {len(picked)} / {_N_RUINS} ruins within "
            f"{_RUIN_MAX_R:.0f} m of district center"
        )
    return picked


def _print_district_ruins(district):
    cx, cy = district.center
    print(f"[DISTRICT] center=({cx:.3f}, {cy:.3f})")
    for i, ruin in enumerate(district.ruins):
        print(
            f"[DISTRICT] ruin[{i}] x={ruin['cx']:.3f} y={ruin['cy']:.3f} "
            f"height={ruin['height']:.1f}"
        )


def nearest_ruin(district, origin=None):
    """Ruin closest to origin (district center if omitted)."""
    ox, oy = district.center if origin is None else origin
    return min(
        district.ruins,
        key=lambda r: math.hypot(r["cx"] - ox, r["cy"] - oy),
    )


def verify_district_ruins(district):
    """Numeric checks: 4 ruins near center, clear of intact interiors and cars."""
    cx, cy = district.center
    half = 0.5 * district.size
    if len(district.ruins) != _N_RUINS:
        raise RuntimeError(f"expected {_N_RUINS} ruins, got {len(district.ruins)}")
    max_r = 0.0
    for i, ruin in enumerate(district.ruins):
        dist = math.hypot(ruin["cx"] - cx, ruin["cy"] - cy)
        max_r = max(max_r, dist)
        if dist > _RUIN_MAX_R + 1e-6:
            raise RuntimeError(f"ruin[{i}] is {dist:.1f} m from center (>{_RUIN_MAX_R:.0f})")
        if abs(ruin["cx"] - cx) > half - 1.0 or abs(ruin["cy"] - cy) > half - 1.0:
            raise RuntimeError(f"ruin[{i}] is outside the district square")
        if not (12.0 <= ruin["height"] <= 18.0):
            raise RuntimeError(f"ruin[{i}] height {ruin['height']:.1f} not in 12–18 m")
        rect = (ruin["xmin"], ruin["ymin"], ruin["xmax"], ruin["ymax"])
        for b in district.buildings:
            if _aabb_overlap(rect, (b["xmin"], b["ymin"], b["xmax"], b["ymax"]), 0.0):
                raise RuntimeError(f"ruin[{i}] overlaps intact {b['kind']}")
        for car in district.cars:
            if _aabb_overlap(rect, _car_rect(car), 0.0):
                raise RuntimeError(f"ruin[{i}] overlaps a car")
        n_wall = sum(1 for p in ruin["pieces"] if p["kind"] == "wall")
        n_floor = sum(1 for p in ruin["pieces"] if p["kind"] == "floor")
        n_tip = sum(1 for p in ruin["pieces"] if p["kind"] == "tip")
        n_chunk = sum(1 for p in ruin["pieces"] if p["kind"] == "chunk")
        if not (2 <= n_wall <= 3 and n_floor == 1 and n_tip == 1 and 4 <= n_chunk <= 6):
            raise RuntimeError(
                f"ruin[{i}] pieces walls={n_wall} floor={n_floor} tip={n_tip} chunks={n_chunk}"
            )
        for piece in ruin["pieces"]:
            if piece["kind"] == "chunk" and 2.0 * max(piece["hx"], piece["hy"]) < 3.0 - 1e-6:
                raise RuntimeError(f"ruin[{i}] chunk is under 3 m across")
    print(
        f"[DISTRICT] ruin check ok: n={len(district.ruins)}  "
        f"max_r={max_r:.1f} m  (center {cx:.1f}, {cy:.1f})"
    )
    return max_r


def matching_tile_centers(terrain, center, square=DISTRICT_SIZE):
    """
    Four world centers: the given district plus the same local offset in
    each of the other three 2x2 DEM tiles. Tiles meet at the origin.
    """
    sx = float(terrain.size_x)
    sy = float(terrain.size_y)
    tile_w, tile_h = 0.5 * sx, 0.5 * sy
    cx, cy = float(center[0]), float(center[1])
    ix = 0 if cx < 0.0 else 1
    iy = 0 if cy < 0.0 else 1
    local_x = cx - (-tile_w + ix * tile_w)
    local_y = cy - (-tile_h + iy * tile_h)
    half = 0.5 * float(square)
    world_xmin, world_xmax = -0.5 * sx, 0.5 * sx
    world_ymin, world_ymax = -0.5 * sy, 0.5 * sy

    centers = []
    for jy in (0, 1):
        for ix_t in (0, 1):
            ox = -tile_w + ix_t * tile_w
            oy = -tile_h + jy * tile_h
            nx, ny = ox + local_x, oy + local_y
            if (
                nx - half < world_xmin - 1e-6
                or nx + half > world_xmax + 1e-6
                or ny - half < world_ymin - 1e-6
                or ny + half > world_ymax + 1e-6
            ):
                raise RuntimeError(
                    f"copied district at ({nx:.1f}, {ny:.1f}) leaves the map"
                )
            centers.append((float(nx), float(ny)))

    # Original first, then the three copies
    orig = (cx, cy)
    ordered = [orig] + [c for c in centers if hypot_sep(c, orig) > 1.0]
    if len(ordered) != 4:
        raise RuntimeError(f"expected 4 tile centers, got {ordered}")
    for i in range(4):
        for k in range(i + 1, 4):
            if abs(ordered[i][0] - ordered[k][0]) < square and abs(
                ordered[i][1] - ordered[k][1]
            ) < square:
                raise RuntimeError("copied district squares overlap")
    return ordered


def hypot_sep(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _window_texture_path(filename=_WINDOW_TEX_NAME, tex_w=512, tex_h=512):
    """
    Light wall, dense dark window grid, darker ground-floor strip at the
    bottom. PyBullet GEOM_BOX UVs are world-scaled (~32 m per 0–1), so a
    sparse grid only shows one or two huge panes; a dense grid still reads
    as windows on shops, mid-rises, and towers.
    """
    path = os.path.join(os.path.dirname(__file__), filename)
    wall = np.array([236, 232, 224], dtype=np.uint8)
    band = np.array([168, 158, 148], dtype=np.uint8)
    pane = np.array([28, 32, 38], dtype=np.uint8)
    rgb = np.empty((tex_h, tex_w, 3), dtype=np.uint8)
    rgb[:] = wall

    gf = int(round(0.08 * tex_h))
    rgb[tex_h - gf :, :] = band
    lintel = max(2, tex_h // 80)
    rgb[tex_h - gf : tex_h - gf + lintel, :] = (wall.astype(np.int16) * 0.72).astype(
        np.uint8
    )

    cols, rows = 16, 16
    top_pad = int(round(0.02 * tex_h))
    bot = tex_h - gf
    usable_h = max(1, bot - top_pad)
    cell_w = tex_w / cols
    cell_h = usable_h / rows
    inset_x = 0.20 * cell_w
    inset_y = 0.18 * cell_h
    for r in range(rows):
        for c in range(cols):
            x0 = int(round(c * cell_w + inset_x))
            x1 = int(round((c + 1) * cell_w - inset_x))
            y0 = int(round(top_pad + r * cell_h + inset_y))
            y1 = int(round(top_pad + (r + 1) * cell_h - inset_y))
            if x1 > x0 and y1 > y0:
                rgb[y0:y1, x0:x1] = pane

    _write_png_rgb(path, rgb)
    return path


def _road_texture_path(filename=_ROAD_TEX_NAME, tex_w=64, tex_h=256):
    """Dark pavement, yellow dashed center line, lighter sidewalks on both edges."""
    path = os.path.join(os.path.dirname(__file__), filename)
    pavement = np.array([48, 48, 52], dtype=np.uint8)
    walk = np.array([158, 154, 146], dtype=np.uint8)
    yellow = np.array([230, 200, 50], dtype=np.uint8)
    rgb = np.empty((tex_h, tex_w, 3), dtype=np.uint8)
    rgb[:] = pavement
    sw = max(4, tex_w // 8)
    rgb[:, :sw] = walk
    rgb[:, tex_w - sw :] = walk
    c0 = tex_w // 2 - max(1, tex_w // 16)
    c1 = tex_w // 2 + max(1, tex_w // 16)
    dash = int(round(0.40 * tex_h))
    rgb[:dash, c0:c1] = yellow
    _write_png_rgb(path, rgb)
    return path


def _road_slab_geometry(hx, hy, hz, along):
    """Thin box; top-face U is across the street, V tiles along it."""
    hx, hy, hz = float(hx), float(hy), float(hz)
    verts = [
        [-hx, -hy, -hz],
        [hx, -hy, -hz],
        [hx, hy, -hz],
        [-hx, hy, -hz],
        [-hx, -hy, hz],
        [hx, -hy, hz],
        [hx, hy, hz],
        [-hx, hy, hz],
    ]
    # 6 faces, two triangles each (CCW outward)
    faces = (
        (4, 5, 6, 7),  # top +Z
        (0, 3, 2, 1),  # bottom -Z
        (0, 1, 5, 4),  # -Y
        (2, 3, 7, 6),  # +Y
        (0, 4, 7, 3),  # -X
        (1, 2, 6, 5),  # +X
    )
    indices = []
    uvs = []
    mesh_verts = []
    for fi, (a, b, c, d) in enumerate(faces):
        base = len(mesh_verts)
        quad = (a, b, c, d)
        for vi in quad:
            x, y, z = verts[vi]
            mesh_verts.append([x, y, z])
            if fi == 0:
                if along == "y":
                    u = (x + hx) / max(2.0 * hx, 1e-6)
                    v = (y + hy) / _ROAD_REPEAT_M
                else:
                    u = (y + hy) / max(2.0 * hy, 1e-6)
                    v = (x + hx) / _ROAD_REPEAT_M
            else:
                u = 0.5
                v = 0.2
            uvs.append([float(u), float(v)])
        indices.extend([base, base + 1, base + 2, base, base + 2, base + 3])
    return mesh_verts, indices, uvs


def _aabb_gap(a, b):
    """Closest-point distance between AABBs (xmin, ymin, xmax, ymax). 0 if overlap."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    dx = max(0.0, bx0 - ax1, ax0 - bx1)
    dy = max(0.0, by0 - ay1, ay0 - by1)
    if dx == 0.0 and dy == 0.0:
        return 0.0
    if dx == 0.0:
        return dy
    if dy == 0.0:
        return dx
    return math.hypot(dx, dy)


def _square_hits_keepouts(cx, cy, half, keepouts):
    for kx, ky, r in keepouts:
        dx = max((cx - half) - kx, 0.0, kx - (cx + half))
        dy = max((cy - half) - ky, 0.0, ky - (cy + half))
        if math.hypot(dx, dy) <= float(r) + 1e-6:
            return True
    return False


def rubble_town_keepouts(scenery):
    """Keep the rubble-town square off groves, camps, huts, and the archer."""
    out = []
    scenery = scenery or {}
    for gx, gy in scenery.get("groves") or []:
        out.append((float(gx), float(gy), 16.0))
    for cx, cy in scenery.get("camps") or []:
        out.append((float(cx), float(cy), 12.0))
    for hx, hy in scenery.get("huts") or []:
        out.append((float(hx), float(hy), 10.0))
    archers = scenery.get("archers") or []
    if archers:
        for rec in archers:
            out.append((float(rec[0]), float(rec[1]), 8.0))
    else:
        for ax, ay in scenery.get("archery") or []:
            out.append((float(ax) + 3.0, float(ay) - 6.0, 8.0))
    rubble = scenery.get("rubble") or {}
    for rec in rubble.get("piles") or []:
        out.append((float(rec["x"]), float(rec["y"]), 8.0))
    for rec in rubble.get("arches") or []:
        out.append((float(rec["x"]), float(rec["y"]), 10.0))
    for rec in rubble.get("walls") or []:
        out.append((float(rec["x"]), float(rec["y"]), 6.0))
    ruins = scenery.get("ruins") or {}
    for rec in ruins.get("placements") or []:
        out.append((float(rec["x"]), float(rec["y"]), 12.0))
    return out


def find_rubble_town_site(
    terrain, district, size=RUBBLE_TOWN_SIZE, min_gap=RUBBLE_TOWN_GAP, keepouts=None
):
    """
    Flattest `size` x `size` square fully on the map, ≥ min_gap from the
    district AABB, and clear of scenery keepouts. Score is mean |slope|
    then relief; no hard slope cap so the search loosens rather than fails.
    """
    keepouts = list(keepouts or [])
    Z = np.asarray(terrain.heightmap, dtype=np.float64)
    rx = float(getattr(terrain, "resolution_x", terrain.resolution))
    ry = float(getattr(terrain, "resolution_y", terrain.resolution))
    gy, gx = Z.shape
    nx = max(2, int(round(size / rx)))
    ny = max(2, int(round(size / ry)))
    if nx > gx or ny > gy:
        raise ValueError(f"rubble town {size:.0f} m does not fit the {gx}x{gy} heightmap")

    dzdy, dzdx = np.gradient(Z, ry, rx)
    slope = np.hypot(dzdx, dzdy)

    half_x = 0.5 * float(terrain.size_x)
    half_y = 0.5 * float(terrain.size_y)
    half = 0.5 * size
    dcx, dcy = float(district.center[0]), float(district.center[1])
    dh = 0.5 * float(district.size)
    dbox = (dcx - dh, dcy - dh, dcx + dh, dcy + dh)

    step_j = max(1, int(round(8.0 / rx)))
    step_i = max(1, int(round(8.0 / ry)))

    best = None
    for i in range(0, gy - ny + 1, step_i):
        for j in range(0, gx - nx + 1, step_j):
            cx = -half_x + (j + 0.5 * nx) * rx
            cy = -half_y + (i + 0.5 * ny) * ry
            if (
                cx - half < -half_x
                or cx + half > half_x
                or cy - half < -half_y
                or cy + half > half_y
            ):
                continue
            gap = _aabb_gap(
                (cx - half, cy - half, cx + half, cy + half),
                dbox,
            )
            if gap < min_gap - 1e-6:
                continue
            if _square_hits_keepouts(cx, cy, half, keepouts):
                continue
            patch_s = slope[i : i + ny, j : j + nx]
            patch_z = Z[i : i + ny, j : j + nx]
            mean_slope = float(np.mean(patch_s))
            relief = float(np.max(patch_z) - np.min(patch_z))
            key = (mean_slope, relief, abs(cx) + abs(cy))
            if best is None or key < best[0]:
                best = (key, float(cx), float(cy), gap, mean_slope, relief)

    if best is None:
        raise RuntimeError(
            f"no in-bounds {size:.0f} m rubble-town site "
            f"≥{min_gap:.0f} m from the district AABB"
        )
    return {
        "center": (best[1], best[2]),
        "gap": float(best[3]),
        "mean_slope": float(best[4]),
        "relief": float(best[5]),
    }


def _broken_road_texture_path(filename=_BROKEN_ROAD_TEX_NAME, tex_w=96, tex_h=256):
    """Dark cracked asphalt — no sidewalks, no center-line polish."""
    path = os.path.join(os.path.dirname(__file__), filename)
    pavement = np.array([36, 34, 32], dtype=np.uint8)
    stain = np.array([48, 44, 40], dtype=np.uint8)
    crack = np.array([18, 16, 15], dtype=np.uint8)
    rgb = np.empty((tex_h, tex_w, 3), dtype=np.uint8)
    rgb[:] = pavement
    rng = np.random.default_rng(19)
    for _ in range(18):
        x0 = int(rng.integers(0, tex_w))
        y0 = int(rng.integers(0, tex_h))
        rw = int(rng.integers(4, 18))
        rh = int(rng.integers(6, 28))
        rgb[y0 : min(tex_h, y0 + rh), x0 : min(tex_w, x0 + rw)] = stain
    for _ in range(14):
        x = float(rng.integers(2, tex_w - 2))
        y = float(rng.integers(0, tex_h))
        vx, vy = float(rng.normal(0.0, 0.35)), float(rng.choice([-1.0, 1.0]))
        n = int(rng.integers(40, 90))
        for _s in range(n):
            ix, iy = int(round(x)), int(round(y))
            if 0 <= iy < tex_h and 0 <= ix < tex_w:
                rgb[iy, ix] = crack
                if 0 <= ix + 1 < tex_w and rng.random() < 0.45:
                    rgb[iy, ix + 1] = crack
            x += vx
            y += vy
            vx += float(rng.normal(0.0, 0.08))
    _write_png_rgb(path, rgb)
    return path


def _bezier(p0, p1, p2, p3, n=28):
    pts = []
    for i in range(n):
        t = i / (n - 1)
        u = 1.0 - t
        pts.append((
            u ** 3 * p0[0] + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + t ** 3 * p3[0],
            u ** 3 * p0[1] + 3 * u * u * t * p1[1] + 3 * u * t * t * p2[1] + t ** 3 * p3[1],
        ))
    return pts


def _rubble_curves(center, size, seed):
    """Two bends only: a gentle avenue and a short corner connector."""
    rng = np.random.default_rng(int(seed) + 23)
    cx, cy = float(center[0]), float(center[1])
    h = 0.5 * float(size) - 28.0
    bow = float(rng.uniform(22.0, 34.0))
    return [
        _bezier(
            (cx - 6.0, cy - 0.85 * h),
            (cx + bow, cy - 0.25 * h),
            (cx + bow, cy + 0.25 * h),
            (cx - 4.0, cy + 0.85 * h),
            n=22,
        ),
        _bezier(
            (cx - 0.72 * h, cy - 0.08 * h),
            (cx - 0.58 * h, cy - 0.42 * h),
            (cx - 0.22 * h, cy - 0.55 * h),
            (cx + 0.05 * h, cy - 0.78 * h),
            n=16,
        ),
    ]


def _resample_polyline(points, step):
    """Evenly spaced samples, keeping the original corners."""
    pts = [(float(x), float(y)) for x, y in points]
    if len(pts) < 2:
        return pts
    out = [pts[0]]
    remain = float(step)
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        dx, dy = x1 - x0, y1 - y0
        seg = math.hypot(dx, dy)
        if seg < 1e-6:
            continue
        ux, uy = dx / seg, dy / seg
        traveled = 0.0
        while traveled + remain <= seg + 1e-6:
            traveled += remain
            if traveled >= seg - 1e-4:
                break
            out.append((x0 + ux * traveled, y0 + uy * traveled))
            remain = float(step)
        remain -= seg - traveled
        if remain <= 1e-4:
            remain = float(step)
        if math.hypot(out[-1][0] - x1, out[-1][1] - y1) > 0.4:
            out.append((x1, y1))
    return out


def _dist_to_polyline(px, py, pts):
    best = 1e9
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        dx, dy = x1 - x0, y1 - y0
        seg2 = dx * dx + dy * dy
        if seg2 < 1e-8:
            dist = math.hypot(px - x0, py - y0)
        else:
            t = max(0.0, min(1.0, ((px - x0) * dx + (py - y0) * dy) / seg2))
            dist = math.hypot(px - (x0 + t * dx), py - (y0 + t * dy))
        if dist < best:
            best = dist
    return best


def _road_ribbon(terrain, points, z_lift):
    """One mesh for a whole street, so the asphalt is a single surface."""
    samples = _resample_polyline(points, 3.0)
    if len(samples) < 2:
        return None
    half_w = 0.5 * _BROKEN_ROAD_WIDTH
    thick = 0.05
    n = len(samples)
    left, right, dist_along = [], [], []
    traveled = 0.0
    for i, (x, y) in enumerate(samples):
        if i == 0:
            tx, ty = samples[1][0] - x, samples[1][1] - y
        elif i == n - 1:
            tx, ty = x - samples[i - 1][0], y - samples[i - 1][1]
        else:
            tx = samples[i + 1][0] - samples[i - 1][0]
            ty = samples[i + 1][1] - samples[i - 1][1]
        ln = math.hypot(tx, ty) or 1.0
        nx, ny = -ty / ln, tx / ln
        scale = 1.0
        if 0 < i < n - 1:
            pdx = x - samples[i - 1][0]
            pdy = y - samples[i - 1][1]
            pl = math.hypot(pdx, pdy) or 1.0
            dot = nx * (-pdy / pl) + ny * (pdx / pl)
            scale = min(2.0, 1.0 / max(abs(dot), 0.45))
        left.append((x + nx * half_w * scale, y + ny * half_w * scale))
        right.append((x - nx * half_w * scale, y - ny * half_w * scale))
        if i:
            traveled += math.hypot(x - samples[i - 1][0], y - samples[i - 1][1])
        dist_along.append(traveled)

    verts, uvs = [], []
    for i in range(n):
        lx, ly = left[i]
        rx, ry = right[i]
        zl = _sample_heightmap(terrain, lx, ly) + z_lift
        zr = _sample_heightmap(terrain, rx, ry) + z_lift
        v = dist_along[i] / _ROAD_REPEAT_M
        verts.append([lx, ly, zl])
        uvs.append([0.0, v])
        verts.append([rx, ry, zr])
        uvs.append([1.0, v])
    bot = len(verts)
    for i in range(n):
        lx, ly, zl = verts[2 * i]
        rx, ry, zr = verts[2 * i + 1]
        v = uvs[2 * i][1]
        verts.append([lx, ly, zl - thick])
        uvs.append([0.0, v])
        verts.append([rx, ry, zr - thick])
        uvs.append([1.0, v])

    indices = []
    for i in range(n - 1):
        a, b = 2 * i, 2 * i + 1
        c, d = 2 * (i + 1), 2 * (i + 1) + 1
        indices.extend([a, b, d, a, d, c])
        ba, bb, bc, bd = bot + a, bot + b, bot + c, bot + d
        indices.extend([ba, bd, bb, ba, bc, bd])
    xs = [p[0] for p in left + right]
    ys = [p[1] for p in left + right]
    return {
        "ribbon": True,
        "verts": verts,
        "indices": indices,
        "uvs": uvs,
        "along": "y",
        "cx": float(sum(xs) / len(xs)),
        "cy": float(sum(ys) / len(ys)),
        "hx": float(0.5 * (max(xs) - min(xs))),
        "hy": float(0.5 * (max(ys) - min(ys))),
        "hz": float(thick),
        "z": float(z_lift),
        "orn": [0.0, 0.0, 0.0, 1.0],
        "xmin": float(min(xs)),
        "xmax": float(max(xs)),
        "ymin": float(min(ys)),
        "ymax": float(max(ys)),
        "centerline": samples,
    }


def _city_block_axes(center, size, seed):
    """Mostly even streets, with a little stagger so the blocks are not identical."""
    rng = np.random.default_rng(int(seed) + 19)
    pitch = 34.0
    n = 6
    span = (n - 1) * pitch
    base = np.linspace(-0.5 * span, 0.5 * span, n)
    cx, cy = float(center[0]), float(center[1])
    ns = cx + base + rng.normal(0.0, 1.6, n)
    ew = cy + base + rng.normal(0.0, 1.6, n)
    return ns, ew


def _rubble_road_lines(center, size, seed):
    """Grid polylines plus two bends. Each entry is one continuous street."""
    cx, cy = float(center[0]), float(center[1])
    reach = 0.5 * float(size) - 18.0
    ns, ew = _city_block_axes(center, size, seed)
    y0, y1 = cy - reach, cy + reach
    x0, x1 = cx - reach, cx + reach
    lines = []
    for i, coord in enumerate(ns):
        if i == 1:
            lines.append(([(coord, y0), (coord, cy - 6.0)], 0.08))
        elif i == 3:
            lines.append((
                [(coord, y0), (coord, cy - 10.0), (coord + 11.0, cy + 10.0), (coord + 11.0, y1)],
                0.08,
            ))
        else:
            lines.append(([(coord, y0), (coord, y1)], 0.08))
    for i, coord in enumerate(ew):
        if i == 2:
            lines.append(([(x0, coord), (cx - 18.0, coord)], 0.10))
            lines.append(([(cx + 18.0, coord), (x1, coord)], 0.10))
        else:
            lines.append(([(x0, coord), (x1, coord)], 0.10))
    for curve in _rubble_curves(center, size, seed):
        lines.append((curve, 0.13))
    return lines


def _layout_broken_streets(terrain, center, size, seed=RUBBLE_TOWN_SEED):
    """Continuous streets: a block grid, one jog, one dead end, one gap, two bends."""
    roads = []
    for points, lift in _rubble_road_lines(center, size, seed):
        ribbon = _road_ribbon(terrain, points, lift)
        if ribbon is not None:
            roads.append(ribbon)
    return roads


def _near_roads(px, py, lines, clearance):
    for points, _lift in lines:
        if _dist_to_polyline(px, py, points) < clearance:
            return True
    return False


def _layout_rubble_shells(terrain, center, size, seed):
    """One ruin in most blocks. A few lots stay empty, and the bends stay clear."""
    rng = np.random.default_rng(int(seed) + 41)
    cx, cy = float(center[0]), float(center[1])
    half = 0.5 * float(size)
    xs, ys = _city_block_axes(center, size, seed)
    lines = _rubble_road_lines(center, size, seed)
    empty = {(1, 2), (3, 0)}
    sides = ("n", "s", "e", "w")
    ruins = []
    for iy in range(len(ys) - 1):
        for ix in range(len(xs) - 1):
            if (ix, iy) in empty:
                continue
            cell_w = float(xs[ix + 1] - xs[ix])
            cell_h = float(ys[iy + 1] - ys[iy])
            sx = float(np.clip(0.48 * cell_w, 11.0, 16.5))
            sy = float(np.clip(0.48 * cell_h, 10.0, 15.5))
            px = 0.5 * (xs[ix] + xs[ix + 1]) + float(rng.uniform(-2.5, 2.5))
            py = 0.5 * (ys[iy] + ys[iy + 1]) + float(rng.uniform(-2.5, 2.5))
            px = float(np.clip(px, cx - half + 0.5 * sx + 2.0, cx + half - 0.5 * sx - 2.0))
            py = float(np.clip(py, cy - half + 0.5 * sy + 2.0, cy + half - 0.5 * sy - 2.0))
            if _near_roads(px, py, lines, 8.0):
                continue
            ruins.append(
                _build_ruin_shell(
                    terrain,
                    px,
                    py,
                    sx,
                    sy,
                    float(rng.uniform(12.0, 18.0)),
                    sides[(ix + 2 * iy) % 4],
                    len(ruins),
                    [],
                    [],
                    vary=True,
                )
            )
    return ruins


def _print_rubble_town(town):
    cx, cy = town.center
    print(
        f"[RUBBLE-TOWN] center=({cx:.3f}, {cy:.3f})  size={town.size:.0f} m"
    )
    for i, ruin in enumerate(town.ruins):
        print(
            f"[RUBBLE-TOWN] shell[{i}] x={ruin['cx']:.3f} y={ruin['cy']:.3f} "
            f"height={ruin['height']:.1f}"
        )


def verify_rubble_town(town, district=None):
    """Numeric checks: a full ruined block, 220–250 m square, ≥200 m from the district."""
    if not (18 <= len(town.ruins) <= 42):
        raise RuntimeError(f"expected 18–42 rubble shells, got {len(town.ruins)}")
    if not (220.0 <= town.size <= 250.0):
        raise RuntimeError(f"rubble town size {town.size:.0f} m not in 220–250 m")
    half = 0.5 * town.size
    cx, cy = town.center
    gap = None
    if district is not None:
        dh = 0.5 * float(district.size)
        dcx, dcy = float(district.center[0]), float(district.center[1])
        gap = _aabb_gap(
            (cx - half, cy - half, cx + half, cy + half),
            (dcx - dh, dcy - dh, dcx + dh, dcy + dh),
        )
        if gap < RUBBLE_TOWN_GAP - 1e-3:
            raise RuntimeError(f"rubble town gap {gap:.1f} m < {RUBBLE_TOWN_GAP:.0f} m")
    for i, ruin in enumerate(town.ruins):
        if abs(ruin["cx"] - cx) > half - 1.0 or abs(ruin["cy"] - cy) > half - 1.0:
            raise RuntimeError(f"shell[{i}] is outside the rubble-town square")
        if not (12.0 <= ruin["height"] <= 18.0):
            raise RuntimeError(f"shell[{i}] height {ruin['height']:.1f} not in 12–18 m")
        n_wall = sum(1 for p in ruin["pieces"] if p["kind"] == "wall")
        n_floor = sum(1 for p in ruin["pieces"] if p["kind"] == "floor")
        n_tip = sum(1 for p in ruin["pieces"] if p["kind"] == "tip")
        n_chunk = sum(1 for p in ruin["pieces"] if p["kind"] == "chunk")
        if not (2 <= n_wall <= 3 and n_floor == 1 and n_tip == 1 and 4 <= n_chunk <= 6):
            raise RuntimeError(
                f"shell[{i}] pieces walls={n_wall} floor={n_floor} tip={n_tip} chunks={n_chunk}"
            )
        for piece in ruin["pieces"]:
            if piece["kind"] == "chunk" and 2.0 * max(piece["hx"], piece["hy"]) < 3.0 - 1e-6:
                raise RuntimeError(f"shell[{i}] chunk is under 3 m across")
            rgb = piece["rgba"][:3]
            if rgb[0] > 0.55 or rgb[1] > 0.50 or rgb[2] > 0.45:
                raise RuntimeError(f"shell[{i}] piece is not charred gray: {rgb}")
    gap_txt = "n/a" if gap is None else f"{gap:.1f} m"
    print(
        f"[RUBBLE-TOWN] check ok: n={len(town.ruins)}  size={town.size:.0f} m  "
        f"gap={gap_txt}  (center {cx:.1f}, {cy:.1f})"
    )
    return gap


def _piece_aabb(piece):
    """World axis-aligned bounds of one oriented ruin piece."""
    hx, hy, hz = float(piece["hx"]), float(piece["hy"]), float(piece["hz"])
    cx, cy, z = float(piece["cx"]), float(piece["cy"]), float(piece["z"])
    orn = piece["orn"]
    xs, ys, zs = [], [], []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                rx, ry, rz = _quat_rotate(orn, (sx * hx, sy * hy, sz * hz))
                xs.append(cx + rx)
                ys.append(cy + ry)
                zs.append(z + rz)
    return (min(xs), min(ys), min(zs), max(xs), max(ys), max(zs))


class RubbleTownLayout:
    """~240 m collapsed town: a block grid with two bends and a few broken streets."""

    def __init__(
        self,
        terrain,
        district=None,
        scenery=None,
        size=RUBBLE_TOWN_SIZE,
        seed=RUBBLE_TOWN_SEED,
        center=None,
    ):
        self.size = float(size)
        self.seed = int(seed)
        self.buildings = []
        self.cars = []
        self.trees = []
        self.landmark = None
        if center is None:
            if district is None:
                raise ValueError("RubbleTownLayout needs a district or an explicit center")
            site = find_rubble_town_site(
                terrain,
                district,
                size=self.size,
                min_gap=RUBBLE_TOWN_GAP,
                keepouts=rubble_town_keepouts(scenery),
            )
            self.center = site["center"]
            self.gap_from_district = float(site["gap"])
            self.mean_slope = float(site["mean_slope"])
            self.relief = float(site["relief"])
            print(
                f"[RUBBLE-TOWN] center=({self.center[0]:.3f}, {self.center[1]:.3f})  "
                f"size={self.size:.0f} m  gap={self.gap_from_district:.1f} m  "
                f"slope={self.mean_slope:.4f}  relief={self.relief:.2f} m"
            )
        else:
            self.center = (float(center[0]), float(center[1]))
            self.gap_from_district = None
            self.mean_slope = 0.0
            self.relief = 0.0
            print(
                f"[RUBBLE-TOWN] center=({self.center[0]:.3f}, {self.center[1]:.3f})  "
                f"size={self.size:.0f} m"
            )
        self.roads = _layout_broken_streets(terrain, self.center, self.size, self.seed)
        self.ruins = _layout_rubble_shells(terrain, self.center, self.size, self.seed)
        self.ground_z = _sample_heightmap(terrain, self.center[0], self.center[1])
        self._bind_search_surface()
        _print_rubble_town(self)
        verify_rubble_town(self, district)

    def _bind_search_surface(self):
        """Street mask and ruin AABBs so search can walk roads and crash on shells."""
        res = 2.0
        ox, oy = float(self.center[0]), float(self.center[1])
        half = 0.5 * self.size
        n = max(2, int(round(self.size / res)))
        street = np.zeros((n, n), dtype=bool)
        half_w = 0.5 * _BROKEN_ROAD_WIDTH + 0.8
        reach = int(math.ceil(half_w / res)) + 1
        reach2 = half_w * half_w
        for road in self.roads:
            line = road.get("centerline") or []
            for (x0, y0), (x1, y1) in zip(line, line[1:]):
                length = math.hypot(x1 - x0, y1 - y0)
                steps = max(1, int(math.ceil(length / res)))
                for s in range(steps + 1):
                    t = s / steps
                    x = x0 + (x1 - x0) * t
                    y = y0 + (y1 - y0) * t
                    ix = int((x - (ox - half)) / res)
                    iy = int((y - (oy - half)) / res)
                    for dy in range(-reach, reach + 1):
                        jy = iy + dy
                        if jy < 0 or jy >= n:
                            continue
                        cy = oy - half + (jy + 0.5) * res
                        for dx in range(-reach, reach + 1):
                            jx = ix + dx
                            if jx < 0 or jx >= n:
                                continue
                            cx = ox - half + (jx + 0.5) * res
                            if (cx - x) * (cx - x) + (cy - y) * (cy - y) <= reach2:
                                street[jy, jx] = True
        for ruin in self.ruins:
            for iy in range(n):
                cy = oy - half + (iy + 0.5) * res
                if cy < ruin["ymin"] or cy > ruin["ymax"]:
                    continue
                for ix in range(n):
                    cx = ox - half + (ix + 0.5) * res
                    if ruin["xmin"] <= cx <= ruin["xmax"]:
                        street[iy, ix] = False
        self.origin = (ox, oy)
        self.walkable_resolution = res
        self.walkable = street
        self.street_mask = street
        rows = [_piece_aabb(piece) for ruin in self.ruins for piece in ruin["pieces"]]
        self.boxes = (
            np.asarray(rows, dtype=np.float64) if rows else np.zeros((0, 6), dtype=np.float64)
        )
        self.styles = ["concrete"] * len(self.boxes)
        self._ground_applied = True

    def apply_ground_heights(self, terrain):
        """Shells already sit on the heightmap they were built against."""
        del terrain
        self._ground_applied = True

    def is_walkable(self, x, y):
        """True on a street that is not inside a ruin footprint."""
        ox, oy = self.origin
        half = self.size / 2.0
        res = self.walkable_resolution
        gx = int((x - ox + half) / res)
        gy = int((y - oy + half) / res)
        if gx < 0 or gy < 0 or gx >= self.walkable.shape[1] or gy >= self.walkable.shape[0]:
            return False
        return bool(self.walkable[gy, gx])

    def sample_walkable(self, rng, n=1, prefer_street=True):
        """Sample n street positions. prefer_street matches TownLayout's signature."""
        del prefer_street
        ys, xs = np.where(self.walkable)
        if len(xs) == 0:
            return np.zeros((n, 2))
        idx = rng.choice(len(xs), size=n, replace=True)
        half = self.size / 2.0
        ox, oy = self.origin
        res = self.walkable_resolution
        return np.column_stack(
            [
                xs[idx] * res - half + res * 0.5 + ox,
                ys[idx] * res - half + res * 0.5 + oy,
            ]
        )


class DistrictLayout:
    """One 300 m plain-box district on an existing Terrain."""

    def __init__(self, terrain, size=DISTRICT_SIZE, seed=DISTRICT_SEED):
        self.size = float(size)
        self.seed = int(seed)
        self.center = find_flattest_square(terrain, self.size)
        self.buildings = _place_buildings(terrain, self.center, self.size, self.seed)
        self.roads = _layout_streets(self.buildings, self.center, self.size)
        for slab in self.roads:
            slab["z"] = _sample_heightmap(terrain, slab["cx"], slab["cy"]) + _ROAD_LIFT
        self.trees, self.cars, self.landmark = _layout_props(
            terrain, self.buildings, self.roads, self.center, self.size, self.seed
        )
        self.ruins = _layout_ruins(
            terrain,
            self.buildings,
            self.roads,
            self.cars,
            self.landmark,
            self.center,
            self.size,
            self.seed,
        )
        self.ground_z = _sample_heightmap(terrain, self.center[0], self.center[1])
        _print_district_ruins(self)
        verify_district_ruins(self)

    def counts(self):
        out = {"shop": 0, "midrise": 0, "tower": 0}
        for b in self.buildings:
            out[b["kind"]] += 1
        return out

    def copy_to(self, terrain, new_center):
        """Same relative XY layout at new_center; re-sample Z on this tile."""
        dx = float(new_center[0]) - self.center[0]
        dy = float(new_center[1]) - self.center[1]
        dest = copy.copy(self)
        dest.center = (float(new_center[0]), float(new_center[1]))
        dest.buildings = []
        for b in self.buildings:
            nb = dict(b)
            nb["cx"] = b["cx"] + dx
            nb["cy"] = b["cy"] + dy
            nb["xmin"] = b["xmin"] + dx
            nb["xmax"] = b["xmax"] + dx
            nb["ymin"] = b["ymin"] + dy
            nb["ymax"] = b["ymax"] + dy
            height = b["zmax"] - b["zmin"]
            nb["zmin"] = _sample_heightmap(terrain, nb["cx"], nb["cy"])
            nb["zmax"] = nb["zmin"] + height
            dest.buildings.append(nb)
        dest.roads = []
        for slab in self.roads:
            ns = dict(slab)
            ns["cx"] = slab["cx"] + dx
            ns["cy"] = slab["cy"] + dy
            ns["xmin"] = slab["xmin"] + dx
            ns["xmax"] = slab["xmax"] + dx
            ns["ymin"] = slab["ymin"] + dy
            ns["ymax"] = slab["ymax"] + dy
            ns["z"] = _sample_heightmap(terrain, ns["cx"], ns["cy"]) + _ROAD_LIFT
            dest.roads.append(ns)
        dest.trees = []
        for tr in self.trees:
            nt = dict(tr)
            nt["cx"] = tr["cx"] + dx
            nt["cy"] = tr["cy"] + dy
            gz = _sample_heightmap(terrain, nt["cx"], nt["cy"])
            pavement = None
            for slab in dest.roads:
                if (
                    slab["xmin"] - 1.6 <= nt["cx"] <= slab["xmax"] + 1.6
                    and slab["ymin"] - 1.6 <= nt["cy"] <= slab["ymax"] + 1.6
                ):
                    pavement = slab["z"] + _ROAD_HALF_THICK
                    break
            nt["z0"] = gz if pavement is None else max(gz, pavement)
            dest.trees.append(nt)
        dest.cars = []
        for car in self.cars:
            nc = dict(car)
            nc["cx"] = car["cx"] + dx
            nc["cy"] = car["cy"] + dy
            gz = _sample_heightmap(terrain, nc["cx"], nc["cy"])
            nc["z"] = gz + _ROAD_LIFT + nc["hz"]
            dest.cars.append(nc)
        lm = copy.deepcopy(self.landmark)
        old_z0 = float(lm["z0"])
        lm["cx"] = lm["cx"] + dx
        lm["cy"] = lm["cy"] + dy
        lm["z0"] = _sample_heightmap(terrain, lm["cx"], lm["cy"])
        dz = lm["z0"] - old_z0
        for box in lm["boxes"]:
            box["z"] = box["z"] + dz
        dest.landmark = lm
        dest.ruins = []
        for ruin in self.ruins:
            nr = dict(ruin)
            nr["cx"] = ruin["cx"] + dx
            nr["cy"] = ruin["cy"] + dy
            nr["xmin"] = ruin["xmin"] + dx
            nr["xmax"] = ruin["xmax"] + dx
            nr["ymin"] = ruin["ymin"] + dy
            nr["ymax"] = ruin["ymax"] + dy
            nr["z0"] = _sample_heightmap(terrain, nr["cx"], nr["cy"])
            nr["pieces"] = []
            for piece in ruin["pieces"]:
                npiece = _ruin_piece(
                    terrain,
                    piece["cx"] + dx,
                    piece["cy"] + dy,
                    piece["hx"],
                    piece["hy"],
                    piece["hz"],
                    piece["orn"],
                    piece["rgba"],
                    piece["kind"],
                    lift=piece.get("lift", 0.0),
                )
                nr["pieces"].append(npiece)
            dest.ruins.append(nr)
        dest.ground_z = _sample_heightmap(terrain, dest.center[0], dest.center[1])
        return dest


def _load_shared_district_textures(client):
    tex_path = _window_texture_path()
    try:
        tex_id = p.loadTexture(tex_path, physicsClientId=client)
    except Exception:
        tex_id = -1
    road_path = _road_texture_path()
    try:
        road_tex = p.loadTexture(road_path, physicsClientId=client)
    except Exception:
        road_tex = -1
    print(
        f"[DISTRICT] window texture={os.path.basename(tex_path)}  "
        f"road texture={os.path.basename(road_path)}  (loaded once)"
    )
    return tex_id, road_tex


def spawn_districts_in_pybullet(client, districts):
    """Spawn every district using one window PNG and one road PNG."""
    window_tex, road_tex = _load_shared_district_textures(client)
    bodies = []
    for district in districts:
        bodies.extend(
            spawn_district_in_pybullet(
                client, district, window_tex=window_tex, road_tex=road_tex
            )
        )
    return bodies


def spawn_district_in_pybullet(client, district, window_tex=None, road_tex=None):
    """
    Visual-only wall boxes plus one roof lid each.

    Window and road textures may be passed in so copies reuse the same
    loads. The shared window texture is tinted per type. A slightly larger
    dark lid hides the top face. Building footprints and heights are
    unchanged.
    """
    if window_tex is None or road_tex is None:
        loaded_w, loaded_r = _load_shared_district_textures(client)
        if window_tex is None:
            window_tex = loaded_w
        if road_tex is None:
            road_tex = loaded_r
    tex_id = window_tex

    wall_bodies = []
    roof_bodies = []
    for b in district.buildings:
        hx = 0.5 * (b["xmax"] - b["xmin"])
        hy = 0.5 * (b["ymax"] - b["ymin"])
        hz = max(0.5 * (b["zmax"] - b["zmin"]), 0.05)
        vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[hx, hy, hz],
            rgbaColor=[1.0, 1.0, 1.0, 1.0],
            physicsClientId=client,
        )
        bid = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=vis,
            basePosition=[b["cx"], b["cy"], 0.5 * (b["zmin"] + b["zmax"])],
            physicsClientId=client,
        )
        if tex_id >= 0:
            p.changeVisualShape(
                bid,
                -1,
                rgbaColor=b["rgba"],
                textureUniqueId=tex_id,
                physicsClientId=client,
            )
        else:
            p.changeVisualShape(
                bid, -1, rgbaColor=b["rgba"], physicsClientId=client
            )
        wall_bodies.append(bid)

        roof_rgb = [min(1.0, c * _ROOF_TINT) for c in b["rgba"][:3]] + [1.0]
        roof_vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[hx + _ROOF_OVERHANG, hy + _ROOF_OVERHANG, 0.5 * _ROOF_THICK],
            rgbaColor=roof_rgb,
            physicsClientId=client,
        )
        roof_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=roof_vis,
            basePosition=[b["cx"], b["cy"], b["zmax"] + 0.5 * _ROOF_THICK],
            physicsClientId=client,
        )
        roof_bodies.append(roof_id)

    print(
        f"[DISTRICT] walls={len(wall_bodies)}  roof_lids={len(roof_bodies)}  "
        f"center=({district.center[0]:.1f}, {district.center[1]:.1f})"
    )

    road_bodies = []
    half = 0.5 * district.size
    x_min, x_max = district.center[0] - half, district.center[0] + half
    y_min, y_max = district.center[1] - half, district.center[1] + half
    for slab in district.roads:
        assert slab["xmin"] >= x_min - 1e-6 and slab["xmax"] <= x_max + 1e-6
        assert slab["ymin"] >= y_min - 1e-6 and slab["ymax"] <= y_max + 1e-6
        z = float(slab["z"])
        verts, indices, uvs = _road_slab_geometry(
            slab["hx"], slab["hy"], _ROAD_HALF_THICK, slab["along"]
        )
        vis = p.createVisualShape(
            p.GEOM_MESH,
            vertices=verts,
            indices=indices,
            uvs=uvs,
            rgbaColor=[1.0, 1.0, 1.0, 1.0],
            physicsClientId=client,
        )
        bid = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=vis,
            basePosition=[slab["cx"], slab["cy"], z],
            physicsClientId=client,
        )
        if road_tex >= 0:
            p.changeVisualShape(
                bid,
                -1,
                rgbaColor=[1.0, 1.0, 1.0, 1.0],
                textureUniqueId=road_tex,
                physicsClientId=client,
            )
        else:
            p.changeVisualShape(
                bid, -1, rgbaColor=[0.20, 0.20, 0.22, 1.0], physicsClientId=client
            )
        road_bodies.append(bid)
    print(f"[DISTRICT] slabs={len(road_bodies)}")

    def _box(hx, hy, hz, pos, rgba):
        vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[hx, hy, hz],
            rgbaColor=rgba,
            physicsClientId=client,
        )
        return p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=vis,
            basePosition=pos,
            physicsClientId=client,
        )

    prop_bodies = []
    trunk_h, canopy_h = 2.3, 2.0
    for tr in district.trees:
        z0 = float(tr["z0"])
        prop_bodies.append(
            _box(0.22, 0.22, 0.5 * trunk_h, [tr["cx"], tr["cy"], z0 + 0.5 * trunk_h], _TRUNK_RGB)
        )
        prop_bodies.append(
            _box(
                1.15,
                1.15,
                0.5 * canopy_h,
                [tr["cx"], tr["cy"], z0 + trunk_h + 0.5 * canopy_h],
                _CANOPY_RGB,
            )
        )
    for car in district.cars:
        prop_bodies.append(
            _box(car["hx"], car["hy"], car["hz"], [car["cx"], car["cy"], car["z"]], car["rgba"])
        )
    for box in district.landmark["boxes"]:
        prop_bodies.append(
            _box(
                box["hx"],
                box["hy"],
                box["hz"],
                [district.landmark["cx"], district.landmark["cy"], box["z"]],
                box["rgba"],
            )
        )
    print(
        f"[DISTRICT] trees={len(district.trees)}  cars={len(district.cars)}  "
        f"landmark_boxes={len(district.landmark['boxes'])}  "
        f"prop_bodies={len(prop_bodies)}  "
        f"tower=({district.landmark['cx']:.1f}, {district.landmark['cy']:.1f})"
    )

    ruin_bodies = []
    for ruin in getattr(district, "ruins", []) or []:
        for piece in ruin["pieces"]:
            vis = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[piece["hx"], piece["hy"], piece["hz"]],
                rgbaColor=piece["rgba"],
                physicsClientId=client,
            )
            ruin_bodies.append(
                p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=-1,
                    baseVisualShapeIndex=vis,
                    basePosition=[piece["cx"], piece["cy"], piece["z"]],
                    baseOrientation=piece["orn"],
                    physicsClientId=client,
                )
            )
    if district.ruins:
        print(
            f"[DISTRICT] ruin_bodies={len(ruin_bodies)}  ruins={len(district.ruins)}"
        )
    return wall_bodies + roof_bodies + road_bodies + prop_bodies + ruin_bodies


def spawn_rubble_town_in_pybullet(client, town):
    """Continuous cracked-asphalt streets plus the ruined shells. No intact buildings."""
    road_path = _broken_road_texture_path()
    try:
        road_tex = p.loadTexture(road_path, physicsClientId=client)
    except Exception:
        road_tex = -1

    road_bodies = []
    for slab in town.roads:
        if slab.get("ribbon"):
            verts, indices, uvs = slab["verts"], slab["indices"], slab["uvs"]
            pos = [0.0, 0.0, 0.0]
            orn = [0.0, 0.0, 0.0, 1.0]
        else:
            verts, indices, uvs = _road_slab_geometry(
                slab["hx"], slab["hy"], slab.get("hz", _BROKEN_ROAD_HALF_THICK), slab["along"]
            )
            pos = [slab["cx"], slab["cy"], slab["z"]]
            orn = slab.get("orn", [0.0, 0.0, 0.0, 1.0])
        vis = p.createVisualShape(
            p.GEOM_MESH,
            vertices=verts,
            indices=indices,
            uvs=uvs,
            rgbaColor=[1.0, 1.0, 1.0, 1.0],
            physicsClientId=client,
        )
        bid = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=vis,
            basePosition=pos,
            baseOrientation=orn,
            physicsClientId=client,
        )
        if road_tex >= 0:
            p.changeVisualShape(
                bid,
                -1,
                rgbaColor=[0.92, 0.90, 0.86, 1.0],
                textureUniqueId=road_tex,
                physicsClientId=client,
            )
        else:
            p.changeVisualShape(
                bid, -1, rgbaColor=[0.16, 0.15, 0.14, 1.0], physicsClientId=client
            )
        road_bodies.append(bid)

    ruin_bodies = []
    for ruin in town.ruins:
        for piece in ruin["pieces"]:
            vis = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[piece["hx"], piece["hy"], piece["hz"]],
                rgbaColor=piece["rgba"],
                physicsClientId=client,
            )
            ruin_bodies.append(
                p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=-1,
                    baseVisualShapeIndex=vis,
                    basePosition=[piece["cx"], piece["cy"], piece["z"]],
                    baseOrientation=piece["orn"],
                    physicsClientId=client,
                )
            )
    print(
        f"[RUBBLE-TOWN] slabs={len(road_bodies)}  ruin_bodies={len(ruin_bodies)}  "
        f"shells={len(town.ruins)}  tex={os.path.basename(road_path)}"
    )
    return road_bodies + ruin_bodies


def _default_overview_eye(district):
    """Opening-view eye if the camera were still aimed at district center."""
    cx, cy = district.center
    dist, dz, yaw = 150.0, 80.0, 25.0
    pitch = -math.asin(dz / dist)
    return (
        cx + dist * math.sin(math.radians(yaw)) * math.cos(pitch),
        cy + dist * (-math.cos(math.radians(yaw))) * math.cos(pitch),
        float(district.ground_z) + 12.0 + dist * (-math.sin(pitch)),
    )


def frame_district_camera(client, district):
    """Look at the nearest ruined shell from ~100 m away and ~55 m up."""
    cx, cy = district.center
    ruins = getattr(district, "ruins", None) or []
    if ruins:
        eye = _default_overview_eye(district)
        ruin = nearest_ruin(district, origin=(eye[0], eye[1]))
        target = np.array(
            [ruin["cx"], ruin["cy"], float(ruin["z0"]) + 8.0],
            dtype=np.float64,
        )
        dist = 100.0
        dz = 55.0
        district.camera_ruin = ruin
    else:
        target = np.array(
            [cx, cy, float(district.ground_z) + 12.0], dtype=np.float64
        )
        dist = 150.0
        dz = 80.0
        district.camera_ruin = None
    yaw = 25.0
    pitch = -math.degrees(math.asin(dz / dist))
    p.resetDebugVisualizerCamera(
        cameraDistance=dist,
        cameraYaw=yaw,
        cameraPitch=pitch,
        cameraTargetPosition=target.tolist(),
        physicsClientId=client,
    )
    print(
        f"[CAMERA] district yaw={yaw:g} pitch={pitch:.1f} dist={dist:.0f} m  "
        f"target=({target[0]:.1f}, {target[1]:.1f}, {target[2]:.1f})"
        + (
            f"  ruin=({ruin['cx']:.1f}, {ruin['cy']:.1f})"
            if ruins
            else ""
        )
    )
    return yaw, pitch, dist, target
