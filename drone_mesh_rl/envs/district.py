"""
district.py — One 300 m block district, copied onto each 2x2 DEM tile.

Seeded axis-aligned boxes on the existing heightmap. Walls share one
window texture (light wall, dark window grid, darker ground-floor band).
A thin untinted roof lid covers the top face so roofs have no windows.
Streets are long pavement slabs through leftover space in the same 300 m
square (buildings are not moved). Sidewalk trees, street cars, and one
water tower sit in that square. The same relative layout is shifted onto
the other three terrain repeats.
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


def find_flattest_square(terrain, size=DISTRICT_SIZE):
    """
    Sliding-window search for the flattest `size` x `size` square fully
    inside the terrain AABB. Score is mean |gradient|, then relief (max-min),
    then distance to the origin so a 2x2-tiled DEM yields one site.
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
        self.ground_z = _sample_heightmap(terrain, self.center[0], self.center[1])

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
    return wall_bodies + roof_bodies + road_bodies + prop_bodies


def frame_district_camera(client, district):
    """Look at the district from ~150 m away and ~80 m up."""
    cx, cy = district.center
    target = np.array(
        [cx, cy, float(district.ground_z) + 12.0], dtype=np.float64
    )
    dist = 150.0
    dz = 80.0
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
    )
    return yaw, pitch, dist, target
