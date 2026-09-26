"""
scenery.py — Grouped Kenney OBJ props on tiled USGS terrain.

Many groves, rock outcrops, camps, and huts sit outside the four 300 m
districts. Meshes are Y-up; they are rotated so file-Y is world-Z and
scaled from each file AABB. Each unique OBJ is cleaned (xyz only) and
loaded once.
"""

from __future__ import annotations

import math
import os
import tempfile

import numpy as np
import pybullet as p

from .town import _sample_heightmap

_PREFABS = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "Prefabs"))
_BANNED = frozenset({"weapon-bow.obj", "weapon-arrow.obj", "platform.obj"})
_SEED = 19
_MARGIN = 30.0
_PATCH_LIFT = 0.05
_N_GROVES = 56
_N_OUTCROPS = 36
_N_CAMPS = 16
_N_HUTS = 8
_N_ARCHERY = 4
_N_BRIDGES_MAX = 6
_GROVE_SEP = 45.0
_OUTCROP_SEP = 35.0
_CAMP_SEP = 100.0

# File Y extents (Kenney Y-up). Used for scale and ground sit.
_MESH_H = {
    "tree": 1.684,
    "tree-high": 2.284,
    "plant": 0.192,
    "patch-grass": 0.171,
    "patch-dirt": 0.100,
    "rocks-high": 1.000,
    "rocks-low": 0.523,
    "rocks-ramp": 0.500,
    "stones": 0.454,
    "tent": 1.000,
    "flag": 0.750,
    "character-archer": 0.821,
    "target": 0.581,
    "building-platform": 0.500,
    "building-structure": 1.000,
    "building-roof": 1.050,
    "ladder": 1.000,
    "fence": 0.400,
    "bridge": 0.473,
}

_TINT = {
    "tree": [0.23, 0.44, 0.20, 1.0],
    "tree-high": [0.18, 0.38, 0.17, 1.0],
    "plant": [0.32, 0.56, 0.22, 1.0],
    "patch-grass": [0.36, 0.52, 0.24, 1.0],
    "patch-dirt": [0.45, 0.34, 0.22, 1.0],
    "rocks-high": [0.42, 0.40, 0.37, 1.0],
    "rocks-low": [0.46, 0.43, 0.39, 1.0],
    "rocks-ramp": [0.40, 0.38, 0.35, 1.0],
    "stones": [0.38, 0.36, 0.33, 1.0],
    "tent": [0.78, 0.62, 0.28, 1.0],
    "flag": [0.82, 0.18, 0.16, 1.0],
    "character-archer": [0.42, 0.34, 0.26, 1.0],
    "target": [0.86, 0.84, 0.80, 1.0],
    "building-platform": [0.55, 0.42, 0.28, 1.0],
    "building-structure": [0.62, 0.50, 0.34, 1.0],
    "building-roof": [0.38, 0.30, 0.24, 1.0],
    "ladder": [0.48, 0.36, 0.22, 1.0],
    "fence": [0.40, 0.30, 0.20, 1.0],
    "bridge": [0.36, 0.32, 0.28, 1.0],
}

# Uniform scale from file AABB: trees 5–8 m, tent 2 m, stacked hut 4 m.
_SCALE = {
    "tree": 6.0 / _MESH_H["tree"],
    "tree-high": 7.6 / _MESH_H["tree-high"],
    "plant": 1.05 / _MESH_H["plant"],
    "patch-grass": 4.2,
    "patch-dirt": 4.0,
    "rocks-high": 3.1,
    "rocks-low": 2.7,
    "rocks-ramp": 3.4,
    "stones": 2.1,
    "tent": 2.0 / _MESH_H["tent"],
    "flag": 3.2 / _MESH_H["flag"],
    "character-archer": 1.72 / _MESH_H["character-archer"],
    "target": 1.15 / _MESH_H["target"],
    "ladder": 4.0 / (_MESH_H["building-platform"] + _MESH_H["building-structure"] + _MESH_H["building-roof"]),
    "fence": 2.4,
    "bridge": 1.0,  # replaced if a span is found
}
_HUT_SCALE = 4.0 / (
    _MESH_H["building-platform"] + _MESH_H["building-structure"] + _MESH_H["building-roof"]
)
_SCALE["building-platform"] = _HUT_SCALE
_SCALE["building-structure"] = _HUT_SCALE
_SCALE["building-roof"] = _HUT_SCALE
_SCALE["ladder"] = _HUT_SCALE

_HUT_H_PLAT = _MESH_H["building-platform"] * _HUT_SCALE
_HUT_H_STRUCT = _MESH_H["building-structure"] * _HUT_SCALE
_HUT_STRUCT_HALF = 0.50 * _HUT_SCALE


def _district_boxes(districts):
    boxes = []
    for d in districts:
        h = 0.5 * float(d.size)
        cx, cy = float(d.center[0]), float(d.center[1])
        boxes.append((cx - h, cx + h, cy - h, cy + h))
    return boxes


def _dist_to_rect(x, y, box):
    xmin, xmax, ymin, ymax = box
    dx = max(xmin - x, 0.0, x - xmax)
    dy = max(ymin - y, 0.0, y - ymax)
    if dx == 0.0 and dy == 0.0:
        return 0.0
    return math.hypot(dx, dy)


def _min_district_dist(x, y, boxes):
    return min(_dist_to_rect(x, y, b) for b in boxes)


def _in_world(terrain, x, y, margin=_MARGIN):
    return (
        abs(x) <= 0.5 * terrain.size_x - margin
        and abs(y) <= 0.5 * terrain.size_y - margin
    )


def _slope_field(terrain):
    Z = np.asarray(terrain.heightmap, dtype=np.float64)
    rx = float(terrain.resolution_x)
    ry = float(terrain.resolution_y)
    dzdy, dzdx = np.gradient(Z, ry, rx)
    return np.hypot(dzdx, dzdy), Z


def _grid_xy(terrain, i, j):
    x = -0.5 * terrain.size_x + (j + 0.5) * terrain.resolution_x
    y = -0.5 * terrain.size_y + (i + 0.5) * terrain.resolution_y
    return float(x), float(y)


def _local_relief(Z, i, j, wi=2, wj=2):
    i0, i1 = max(0, i - wi), min(Z.shape[0], i + wi + 1)
    j0, j1 = max(0, j - wj), min(Z.shape[1], j + wj + 1)
    patch = Z[i0:i1, j0:j1]
    return float(patch.max() - patch.min())


def _candidates(terrain, boxes, slope, Z, kind, spec):
    """Grid cells for groves (gentle), outcrops (steep), or camps (flat)."""
    gy, gx = slope.shape
    step = int(spec.get("step", 2))
    pad = float(spec["pad"])
    out = []
    for i in range(2, gy - 2, step):
        for j in range(2, gx - 2, step):
            x, y = _grid_xy(terrain, i, j)
            if not _in_world(terrain, x, y):
                continue
            if _min_district_dist(x, y, boxes) < pad:
                continue
            s = float(slope[i, j])
            rel = _local_relief(Z, i, j)
            if kind == "grove":
                if s > spec["s_max"] or rel > spec["rel_max"]:
                    continue
            elif kind == "outcrop":
                if s < spec["s_min"] or s > spec["s_max"] or rel < spec["rel_min"]:
                    continue
            elif kind == "camp":
                if s > spec["s_max"] or rel > spec["rel_max"]:
                    continue
            else:
                raise ValueError(kind)
            out.append((x, y, s, rel))
    return out


def _gather(terrain, boxes, slope, Z, kind, need):
    """Widen slope/relief/district pad until there are enough sites."""
    if kind == "grove":
        specs = (
            {"step": 2, "s_max": 0.08, "rel_max": 5.5, "pad": 20.0},
            {"step": 2, "s_max": 0.12, "rel_max": 8.0, "pad": 16.0},
            {"step": 1, "s_max": 0.18, "rel_max": 12.0, "pad": 12.0},
            {"step": 1, "s_max": 0.28, "rel_max": 18.0, "pad": 8.0},
            {"step": 1, "s_max": 0.45, "rel_max": 28.0, "pad": 6.0},
        )
    elif kind == "outcrop":
        specs = (
            {"step": 2, "s_min": 0.14, "s_max": 0.60, "rel_min": 3.0, "pad": 12.0},
            {"step": 2, "s_min": 0.10, "s_max": 0.70, "rel_min": 2.0, "pad": 10.0},
            {"step": 1, "s_min": 0.07, "s_max": 0.85, "rel_min": 1.2, "pad": 8.0},
            {"step": 1, "s_min": 0.045, "s_max": 1.20, "rel_min": 0.6, "pad": 6.0},
        )
    elif kind == "camp":
        specs = (
            {"step": 2, "s_max": 0.045, "rel_max": 2.4, "pad": 200.0},
            {"step": 2, "s_max": 0.055, "rel_max": 3.0, "pad": 200.0},
            {"step": 2, "s_max": 0.06, "rel_max": 3.5, "pad": 150.0},
            {"step": 1, "s_max": 0.08, "rel_max": 4.5, "pad": 100.0},
            {"step": 1, "s_max": 0.10, "rel_max": 6.0, "pad": 40.0},
            {"step": 1, "s_max": 0.14, "rel_max": 8.0, "pad": 10.0},
        )
    else:
        raise ValueError(kind)
    last = []
    for spec in specs:
        last = _candidates(terrain, boxes, slope, Z, kind, spec)
        if len(last) >= need * 2:
            return last, spec
    return last, specs[-1]


def _pick(rng, cands, n, min_sep, existing=None, existing_sep=None, prefer_steep=False):
    existing = list(existing or [])
    existing_sep = min_sep if existing_sep is None else float(existing_sep)
    if prefer_steep:
        ranked = sorted(range(len(cands)), key=lambda i: cands[i][2], reverse=True)
    else:
        ranked = np.arange(len(cands))
        rng.shuffle(ranked)
    picked = []
    for idx in ranked:
        x, y = cands[idx][0], cands[idx][1]
        if any(math.hypot(x - px, y - py) < existing_sep for px, py in existing):
            continue
        if any(math.hypot(x - px, y - py) < min_sep for px, py in picked):
            continue
        picked.append((x, y))
        if len(picked) == n:
            break
    return picked


def _must_pick(rng, cands, n, min_sep, existing=None, existing_sep=None, prefer_steep=False):
    """Hit `n` sites; only shrink separation if the requested gap cannot fit."""
    seps = [min_sep, min_sep * 0.85, min_sep * 0.7, min_sep * 0.55, 20.0, 12.0]
    last = []
    for sep in seps:
        last = _pick(
            rng, cands, n, sep, existing=existing, existing_sep=existing_sep, prefer_steep=prefer_steep
        )
        if len(last) >= n:
            return last
    if len(last) < n:
        # Last resort: take remaining candidates ignoring existing blockers.
        extra = _pick(rng, cands, n - len(last), 12.0, existing=last, existing_sep=12.0)
        last = last + extra
    if len(last) < n:
        raise RuntimeError(f"could only place {len(last)}/{n} sites")
    return last[:n]


def _camp_drop(terrain, x, y):
    """Best downhill Δz along a 16–24 m ray from (x, y)."""
    z0 = _sample_heightmap(terrain, x, y)
    best = 0.0
    best_rec = None
    for ang in np.linspace(0.0, 2.0 * math.pi, 16, endpoint=False):
        ux, uy = math.cos(ang), math.sin(ang)
        for length in (16.0, 20.0, 24.0):
            x2 = x + length * ux
            y2 = y + length * uy
            if not _in_world(terrain, x2, y2, margin=20.0):
                continue
            z2 = _sample_heightmap(terrain, x2, y2)
            dz = z0 - z2
            if dz > best:
                best = float(dz)
                best_rec = (x, y, z0, x2, y2, z2, length, math.atan2(uy, ux))
    return best, best_rec


def _jitter(rng, x, y, radius):
    ang = float(rng.uniform(0.0, 2.0 * math.pi))
    r = radius * math.sqrt(float(rng.uniform(0.15, 1.0)))
    return x + r * math.cos(ang), y + r * math.sin(ang)


def _cluster_offsets(rng, n, radius, min_gap):
    """Irregular disk cluster, not a row."""
    pts = []
    for _ in range(max(80, n * 40)):
        if len(pts) >= n:
            break
        nx, ny = _jitter(rng, 0.0, 0.0, radius)
        if all(math.hypot(nx - px, ny - py) >= min_gap for px, py in pts):
            pts.append((nx, ny))
    while len(pts) < n:
        pts.append(_jitter(rng, 0.0, 0.0, radius))
    return pts[:n]


def _parse_bounds(path):
    xs, ys, zs = [], [], []
    with open(path) as f:
        for line in f:
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


def _cleaned_obj(stem):
    src = os.path.join(_PREFABS, f"{stem}.obj")
    if not os.path.isfile(src):
        raise FileNotFoundError(src)
    cache = os.path.join(tempfile.gettempdir(), "disasterdrones_kenney_xyz")
    os.makedirs(cache, exist_ok=True)
    dst = os.path.join(cache, f"{stem}.obj")
    if os.path.isfile(dst) and os.path.getmtime(dst) >= os.path.getmtime(src):
        return dst
    with open(src) as fin, open(dst, "w") as fout:
        for line in fin:
            if line.startswith("v "):
                parts = line.split()
                fout.write(f"v {parts[1]} {parts[2]} {parts[3]}\n")
            else:
                fout.write(line)
    return dst


def _q_yaw(yaw):
    q_fix = p.getQuaternionFromEuler([math.pi / 2.0, 0.0, 0.0])
    q_yaw = p.getQuaternionFromEuler([0.0, 0.0, float(yaw)])
    return p.multiplyTransforms([0, 0, 0], q_yaw, [0, 0, 0], q_fix)[1]


def _q_span(yaw, pitch):
    q_fix = p.getQuaternionFromEuler([math.pi / 2.0, 0.0, 0.0])
    q_pitch = p.getQuaternionFromEuler([0.0, float(pitch), 0.0])
    q_yaw = p.getQuaternionFromEuler([0.0, 0.0, float(yaw)])
    _, q = p.multiplyTransforms([0, 0, 0], q_pitch, [0, 0, 0], q_fix)
    return p.multiplyTransforms([0, 0, 0], q_yaw, [0, 0, 0], q)[1]


class _MeshBank:
    """One visual shape per unique Kenney mesh."""

    def __init__(self, client):
        self.client = client
        self.vis = {}
        self.meta = {}

    def load(self, stem, scale=None):
        if stem in self.vis:
            return self.vis[stem]
        fname = f"{stem}.obj"
        if fname in _BANNED:
            raise RuntimeError(f"refusing to load banned mesh {fname}")
        scale = float(_SCALE[stem] if scale is None else scale)
        path = _cleaned_obj(stem)
        bounds = _parse_bounds(path)
        vis = p.createVisualShape(
            p.GEOM_MESH,
            fileName=path,
            meshScale=[scale, scale, scale],
            rgbaColor=_TINT[stem],
            physicsClientId=self.client,
        )
        self.vis[stem] = vis
        self.meta[stem] = {"scale": scale, "ymin": bounds["ymin"], "bounds": bounds}
        return vis

    def spawn(self, stem, x, y, z, yaw=0.0, pitch=None):
        vis = self.vis[stem]
        if pitch is None:
            orn = _q_yaw(yaw)
        else:
            orn = _q_span(yaw, pitch)
        return p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=vis,
            basePosition=[float(x), float(y), float(z)],
            baseOrientation=orn,
            physicsClientId=self.client,
        )

    def sit(self, terrain, stem, x, y, yaw=0.0, lift=0.0):
        meta = self.meta[stem]
        gz = _sample_heightmap(terrain, x, y)
        z = gz - meta["ymin"] * meta["scale"] + lift
        return self.spawn(stem, x, y, z, yaw), gz


def _hut_beside(terrain, camp, boxes, rng, others):
    for _ in range(16):
        ang = float(rng.uniform(0.0, 2.0 * math.pi))
        hx = camp[0] + 11.0 * math.cos(ang)
        hy = camp[1] + 11.0 * math.sin(ang)
        if not _in_world(terrain, hx, hy):
            continue
        if _min_district_dist(hx, hy, boxes) < 8.0:
            continue
        if any(math.hypot(hx - ox, hy - oy) < 14.0 for ox, oy in others):
            continue
        return (hx, hy)
    return (camp[0] + 11.0, camp[1] + 4.0)


def _plan_scenery(terrain, districts, rng):
    boxes = _district_boxes(districts)
    slope, Z = _slope_field(terrain)

    grove_c, grove_spec = _gather(terrain, boxes, slope, Z, "grove", _N_GROVES)
    out_c, out_spec = _gather(terrain, boxes, slope, Z, "outcrop", _N_OUTCROPS)
    camp_c, camp_spec = _gather(terrain, boxes, slope, Z, "camp", _N_CAMPS)
    print(
        f"[SCENERY] sites grove={len(grove_c)} (pad={grove_spec['pad']:.0f})  "
        f"outcrop={len(out_c)} (pad={out_spec['pad']:.0f})  "
        f"camp={len(camp_c)} (pad={camp_spec['pad']:.0f})"
    )

    groves = _must_pick(rng, grove_c, _N_GROVES, _GROVE_SEP)
    outcrops = _must_pick(
        rng, out_c, _N_OUTCROPS, _OUTCROP_SEP,
        existing=groves, existing_sep=22.0, prefer_steep=True,
    )
    camps = _must_pick(rng, camp_c, _N_CAMPS, _CAMP_SEP)

    huts = []
    hut_camps = camps[:_N_HUTS]
    for camp in hut_camps:
        huts.append(_hut_beside(terrain, camp, boxes, rng, huts))

    archery = [camps[i] for i in range(0, _N_CAMPS, max(1, _N_CAMPS // _N_ARCHERY))][:_N_ARCHERY]
    return {
        "boxes": boxes,
        "groves": groves,
        "outcrops": outcrops,
        "camps": camps,
        "huts": huts,
        "archery": archery,
    }


def _find_bridges(terrain, camps, boxes, blockers, rng, n=_N_BRIDGES_MAX):
    """Up to n real 15–25 m drops from camps; skip flat spans."""
    headings = np.linspace(0.0, 2.0 * math.pi, 20, endpoint=False)
    rng.shuffle(headings)
    lengths = (16.0, 20.0, 24.0)
    found = []
    for camp in camps:
        if len(found) >= n:
            break
        z_camp = _sample_heightmap(terrain, camp[0], camp[1])
        best = None
        for ang in headings:
            ux, uy = math.cos(ang), math.sin(ang)
            for length in lengths:
                x1 = camp[0] + 4.5 * ux
                y1 = camp[1] + 4.5 * uy
                x2 = x1 + length * ux
                y2 = y1 + length * uy
                if not _in_world(terrain, x1, y1) or not _in_world(terrain, x2, y2):
                    continue
                if _min_district_dist(x1, y1, boxes) < 8.0:
                    continue
                if _min_district_dist(x2, y2, boxes) < 8.0:
                    continue
                if any(math.hypot(x2 - px, y2 - py) < 10.0 for px, py in blockers):
                    continue
                cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
                if any(math.hypot(cx - b["cx"], cy - b["cy"]) < 28.0 for b in found):
                    continue
                z1 = _sample_heightmap(terrain, x1, y1)
                z2 = _sample_heightmap(terrain, x2, y2)
                dz = abs(z2 - z1)
                if dz < 2.0:
                    continue
                downhill = min(z1, z2) <= z_camp - 0.4
                score = dz + (1.5 if downhill else 0.0)
                rec = {
                    "x1": x1,
                    "y1": y1,
                    "z1": z1,
                    "x2": x2,
                    "y2": y2,
                    "z2": z2,
                    "length": length,
                    "yaw": math.atan2(y2 - y1, x2 - x1),
                    "pitch": -math.atan2(z2 - z1, length),
                    "cx": cx,
                    "cy": cy,
                }
                if best is None or score > best[0]:
                    best = (score, rec)
        if best is not None:
            found.append(best[1])
    return found


def spawn_scenery_in_pybullet(client, terrain, districts, seed=_SEED):
    """
    Place grouped Kenney scenery after the four districts.

    Returns (body_ids, info) where info has group centers and the bridge flag.
    """
    rng = np.random.default_rng(seed)
    plan = _plan_scenery(terrain, districts, rng)
    blockers = list(plan["groves"]) + list(plan["outcrops"]) + list(plan["huts"])
    bridges = _find_bridges(terrain, plan["camps"], plan["boxes"], blockers, rng)

    used = {
        "tree",
        "tree-high",
        "plant",
        "patch-grass",
        "rocks-high",
        "rocks-low",
        "rocks-ramp",
        "stones",
        "patch-dirt",
        "tent",
        "flag",
        "character-archer",
        "target",
        "building-platform",
        "building-structure",
        "building-roof",
        "ladder",
        "fence",
    }
    bank = _MeshBank(client)
    for stem in sorted(used):
        bank.load(stem)
    if bridges:
        span = float(np.median([b["length"] for b in bridges]))
        bank.load("bridge", scale=span / 1.042)

    bodies = []
    print("[SCENERY] Kenney meshes loaded once (xyz-only, Y-up → Z-up)")

    n_archery = 0
    for i, (gx, gy) in enumerate(plan["groves"]):
        n_trees = int(rng.integers(12, 21))
        n_plants = int(rng.integers(6, 11))
        tree_off = _cluster_offsets(rng, n_trees, 12.5, 2.0)
        for k, (ox, oy) in enumerate(tree_off):
            px, py = gx + ox, gy + oy
            if _min_district_dist(px, py, plan["boxes"]) == 0.0:
                px, py = gx, gy
            stem = "tree-high" if (k % 3 == 0) else "tree"
            yaw = float(rng.uniform(0.0, 2.0 * math.pi))
            bid, _ = bank.sit(terrain, stem, px, py, yaw)
            bodies.append(bid)
        plant_off = _cluster_offsets(rng, n_plants, 9.0, 1.3)
        for ox, oy in plant_off:
            px, py = gx + ox, gy + oy
            if _min_district_dist(px, py, plan["boxes"]) == 0.0:
                px, py = gx, gy
            yaw = float(rng.uniform(0.0, 2.0 * math.pi))
            bid, _ = bank.sit(terrain, "plant", px, py, yaw)
            bodies.append(bid)
        bid, _ = bank.sit(
            terrain, "patch-grass", gx, gy, float(rng.uniform(0.0, 2.0 * math.pi)), lift=_PATCH_LIFT
        )
        bodies.append(bid)

    for i, (ox, oy) in enumerate(plan["outcrops"]):
        use_ramp = bool(rng.random() < 0.4)
        if use_ramp:
            yaw = float(rng.uniform(0.0, 2.0 * math.pi))
            bid, _ = bank.sit(terrain, "rocks-ramp", ox, oy, yaw)
            bodies.append(bid)
        else:
            n_high = int(rng.integers(1, 3))
            for k in range(n_high):
                px, py = ox, oy
                if k:
                    px, py = ox + float(rng.uniform(-1.8, 1.8)), oy + float(rng.uniform(-1.8, 1.8))
                yaw = float(rng.uniform(0.0, 2.0 * math.pi))
                bid, _ = bank.sit(terrain, "rocks-high", px, py, yaw)
                bodies.append(bid)
        for k in range(2):
            ang = float(rng.uniform(0.0, 2.0 * math.pi))
            r = float(rng.uniform(2.2, 3.4))
            bid, _ = bank.sit(
                terrain,
                "rocks-low",
                ox + r * math.cos(ang),
                oy + r * math.sin(ang),
                float(rng.uniform(0.0, 2.0 * math.pi)),
            )
            bodies.append(bid)
        ang = float(rng.uniform(0.0, 2.0 * math.pi))
        bid, _ = bank.sit(
            terrain,
            "stones",
            ox + 2.0 * math.cos(ang),
            oy + 2.0 * math.sin(ang),
            float(rng.uniform(0.0, 2.0 * math.pi)),
        )
        bodies.append(bid)

    archery_set = set(plan["archery"])
    for i, (cx, cy) in enumerate(plan["camps"]):
        yaw_tent = float(rng.uniform(-0.4, 0.4))
        bid, _ = bank.sit(terrain, "patch-dirt", cx, cy, yaw_tent, lift=_PATCH_LIFT)
        bodies.append(bid)
        bid, _ = bank.sit(terrain, "tent", cx, cy, yaw_tent)
        bodies.append(bid)
        bid, _ = bank.sit(
            terrain, "stones", cx + 2.6, cy + 0.4, float(rng.uniform(0.0, 0.8))
        )
        bodies.append(bid)
        bid, _ = bank.sit(terrain, "flag", cx - 2.2, cy + 3.4, float(rng.uniform(-0.2, 0.2)))
        bodies.append(bid)
        bid, _ = bank.sit(terrain, "plant", cx - 3.0, cy - 1.6, float(rng.uniform(0.0, 6.0)))
        bodies.append(bid)
        if (cx, cy) in archery_set:
            along = float(rng.uniform(8.0, 12.0))
            ax, ay = cx + 3.0, cy - 6.0
            tx, ty = ax + along, ay + 0.4
            yaw_a = math.atan2(tx - ax, -(ty - ay))
            yaw_t = math.atan2(ax - tx, -(ay - ty))
            bid, _ = bank.sit(terrain, "character-archer", ax, ay, yaw_a)
            bodies.append(bid)
            bid, _ = bank.sit(terrain, "target", tx, ty, yaw_t)
            bodies.append(bid)
            n_archery += 1

    yard = (
        (0.0, 5.2, 0.0),
        (0.0, -5.2, 0.0),
        (5.4, 0.4, math.pi / 2.0),
        (-5.4, -0.3, math.pi / 2.0),
    )
    for hx, hy in plan["huts"]:
        gz = _sample_heightmap(terrain, hx, hy)
        yaw_hut = 0.15
        z_plat = gz - bank.meta["building-platform"]["ymin"] * _HUT_SCALE
        bodies.append(bank.spawn("building-platform", hx, hy, z_plat, yaw_hut))
        z_struct = gz + _HUT_H_PLAT - bank.meta["building-structure"]["ymin"] * _HUT_SCALE
        bodies.append(bank.spawn("building-structure", hx, hy, z_struct, yaw_hut))
        z_roof = (
            gz
            + _HUT_H_PLAT
            + _HUT_H_STRUCT
            - bank.meta["building-roof"]["ymin"] * _HUT_SCALE
        )
        bodies.append(bank.spawn("building-roof", hx, hy, z_roof, yaw_hut))
        bodies.append(
            bank.sit(terrain, "ladder", hx + _HUT_STRUCT_HALF + 0.12, hy, yaw=math.pi / 2.0)[0]
        )
        for fx, fy, fyaw in yard:
            bodies.append(bank.sit(terrain, "fence", hx + fx, hy + fy, fyaw)[0])

    for br in bridges:
        meta = bank.meta["bridge"]
        z_mid = 0.5 * (br["z1"] + br["z2"]) - meta["ymin"] * meta["scale"]
        bodies.append(
            bank.spawn(
                "bridge",
                br["cx"],
                br["cy"],
                z_mid,
                yaw=br["yaw"],
                pitch=br["pitch"],
            )
        )
        print(
            f"[SCENERY] bridge ({br['cx']:.1f}, {br['cy']:.1f})  "
            f"span={br['length']:.0f} m  Δz={abs(br['z2'] - br['z1']):.1f} m"
        )
    if not bridges:
        print("[SCENERY] bridges skipped (no 15–25 m span with a real drop)")

    loaded = sorted(bank.vis)
    print(
        f"[SCENERY] groves={len(plan['groves'])}  outcrops={len(plan['outcrops'])}  "
        f"camps={len(plan['camps'])}  huts={len(plan['huts'])}  "
        f"archery={n_archery}  bridges={len(bridges)}  "
        f"bodies={len(bodies)}"
    )
    print(f"[SCENERY] meshes reused: {','.join(loaded)}")
    info = {
        "groves": plan["groves"],
        "outcrops": plan["outcrops"],
        "camps": plan["camps"],
        "huts": plan["huts"],
        "archery": plan["archery"],
        "bridges": bridges,
        "n_bodies": len(bodies),
        "n_archery": n_archery,
        "meshes": loaded,
    }
    return bodies, info
