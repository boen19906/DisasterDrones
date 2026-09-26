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


def _build_ruin_shell(terrain, cx, cy, sx, sy, height, open_side, index, buildings, cars):
    """Broken shell: 2–3 walls, one attached floor, one tipped slab, 4–6 chunks."""
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
    tip = math.radians(_RUIN_TIP_DEG[index])
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

    n_chunks = (5, 4, 6, 5)[index]
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
