"""
rubble.py — Large Kenney debris on the single-patch USGS district.

Broken arches, rubble piles, and fallen wall chunks are built from the
same rocks / building / fence OBJs as scenery.py. Nothing from
weapon-bow.obj or weapon-arrow.obj is loaded. Groups sit after the
district and scenery so buildings, roads, cars, groves, camps, and the
archer stay as they are.
"""

from __future__ import annotations

import math

import numpy as np

from .scenery import (
    _BANNED,
    _MeshBank,
    _cluster_offsets,
    _district_boxes,
    _in_world,
    _local_relief,
    _min_district_dist,
    _slope_field,
)
from .town import _sample_heightmap

_SEED = 31
_N_ARCHES = 2
_N_PILES = 8
_N_PILES_IN = 5
_N_PILES_OUT = 3
_N_WALLS = 4
_CLEAR = 4.0
_MIDLINE_CLEAR = 2.3
_PILE_SEP_IN = 36.0
_PILE_SEP_OUT = 48.0

# Dusty / broken tints — same stems as scenery, not the hut palette.
_TINT = {
    "rocks-high": [0.40, 0.37, 0.33, 1.0],
    "rocks-low": [0.44, 0.40, 0.36, 1.0],
    "rocks-ramp": [0.38, 0.36, 0.32, 1.0],
    "stones": [0.36, 0.34, 0.30, 1.0],
    "building-structure": [0.50, 0.46, 0.40, 1.0],
    "building-roof": [0.30, 0.26, 0.22, 1.0],
    "building-platform": [0.46, 0.38, 0.30, 1.0],
    "fence": [0.34, 0.26, 0.18, 1.0],
}


def _variant_key(stem, scale):
    if isinstance(scale, (int, float)):
        return f"{stem}@{float(scale):.2f}"
    return f"{stem}@{float(scale[0]):.2f}x{float(scale[1]):.2f}x{float(scale[2]):.2f}"


def _ensure(bank, stem, scale):
    fname = f"{stem}.obj"
    if fname in _BANNED:
        raise RuntimeError(f"refusing to load banned mesh {fname}")
    key = _variant_key(stem, scale)
    bank.load(stem, scale=scale, key=key, tint=_TINT[stem])
    return key


def _people_keepouts(scenery):
    """4 m clearance around the archer and any future people/survivor points."""
    pts = []
    for rec in scenery.get("archers") or []:
        pts.append((float(rec[0]), float(rec[1]), _CLEAR + 0.6))
    if not pts:
        for cx, cy in scenery.get("archery") or []:
            pts.append((float(cx) + 3.0, float(cy) - 6.0, _CLEAR + 0.6))
    for key in ("survivors", "people", "people_spawns"):
        for rec in scenery.get(key) or []:
            if isinstance(rec, dict):
                pts.append((float(rec.get("x", rec.get("cx"))), float(rec.get("y", rec.get("cy"))), _CLEAR + 0.6))
            elif isinstance(rec, (tuple, list)) and len(rec) >= 2:
                pts.append((float(rec[0]), float(rec[1]), _CLEAR + 0.6))
    return pts


def _camp_keepouts(scenery):
    out = []
    for cx, cy in scenery.get("camps") or []:
        out.append((float(cx), float(cy), _CLEAR + 2.0))
    for hx, hy in scenery.get("huts") or []:
        out.append((float(hx), float(hy), _CLEAR + 2.5))
    return out


def _cluster_keepouts(scenery, district):
    """Outside piles stay out of groves, camps, hut yards, and the landmark."""
    out = []
    for gx, gy in scenery.get("groves") or []:
        out.append((float(gx), float(gy), 16.0))
    for ox, oy in scenery.get("outcrops") or []:
        out.append((float(ox), float(oy), 10.0))
    for cx, cy in scenery.get("camps") or []:
        out.append((float(cx), float(cy), 12.0))
    for hx, hy in scenery.get("huts") or []:
        out.append((float(hx), float(hy), 10.0))
    if district is not None:
        lm = getattr(district, "landmark", None)
        if lm:
            out.append((float(lm["cx"]), float(lm["cy"]), 8.0))
    return out


def _keepout_ok(x, y, keepouts, extra_r=0.0):
    for kx, ky, r in keepouts:
        if math.hypot(x - kx, y - ky) < r + extra_r:
            return False
    return True


def _hits_building(x, y, buildings, pad):
    for b in buildings:
        if b["xmin"] - pad <= x <= b["xmax"] + pad and b["ymin"] - pad <= y <= b["ymax"] + pad:
            return True
    return False


def _hits_car(x, y, cars, pad):
    for car in cars:
        if abs(x - car["cx"]) <= car["hx"] + pad and abs(y - car["cy"]) <= car["hy"] + pad:
            return True
    return False


def _inside_district(x, y, district, margin=6.0):
    half = 0.5 * district.size - margin
    return abs(x - district.center[0]) <= half and abs(y - district.center[1]) <= half


def _midline_dist(x, y, roads):
    best = 1e9
    for slab in roads:
        if slab["along"] == "y":
            if slab["ymin"] - 2.0 <= y <= slab["ymax"] + 2.0:
                best = min(best, abs(x - slab["cx"]))
        else:
            if slab["xmin"] - 2.0 <= x <= slab["xmax"] + 2.0:
                best = min(best, abs(y - slab["cy"]))
    return best


def _nudge_from_midline(x, y, roads, min_d=_MIDLINE_CLEAR):
    """Push a point off the walkable street center without leaving the slab."""
    for slab in roads:
        if slab["along"] == "y":
            if slab["ymin"] - 3.0 <= y <= slab["ymax"] + 3.0 and abs(x - slab["cx"]) < min_d:
                sign = 1.0 if x >= slab["cx"] else -1.0
                x = slab["cx"] + sign * min_d
        else:
            if slab["xmin"] - 3.0 <= x <= slab["xmax"] + 3.0 and abs(y - slab["cy"]) < min_d:
                sign = 1.0 if y >= slab["cy"] else -1.0
                y = slab["cy"] + sign * min_d
    return x, y


def _ij(terrain, x, y):
    j = int((x + 0.5 * terrain.size_x) / terrain.resolution_x)
    i = int((y + 0.5 * terrain.size_y) / terrain.resolution_y)
    gy, gx = terrain.heightmap.shape
    return int(np.clip(i, 0, gy - 1)), int(np.clip(j, 0, gx - 1))


def _flat_ok(terrain, slope, Z, x, y, s_max=0.12, rel_max=8.0):
    i, j = _ij(terrain, x, y)
    return float(slope[i, j]) <= s_max and _local_relief(Z, i, j) <= rel_max


def _sep_ok(x, y, placed, sep):
    return all(math.hypot(x - px, y - py) >= sep for px, py in placed)


class _RubbleWorld:
    def __init__(self, client, terrain, districts, scenery, seed=_SEED):
        self.client = client
        self.terrain = terrain
        self.districts = districts
        self.district = districts[0]
        self.scenery = scenery
        self.rng = np.random.default_rng(seed)
        self.boxes = _district_boxes(districts)
        self.slope, self.Z = _slope_field(terrain)
        self.bank = _MeshBank(client)
        self.bodies = []
        self.pieces = []
        self.arches = []
        self.piles = []
        self.walls = []
        self.largest = {"size": 0.0, "stem": "", "kind": "", "height": 0.0}

        people = _people_keepouts(scenery)
        camps = _camp_keepouts(scenery)
        self.hard_keep = people + camps
        self.soft_keep = _cluster_keepouts(scenery, self.district) + self.hard_keep

    def record(self, key, x, y, yaw, pitch, roll, kind, group, radius):
        h, long_xy, longest = self.bank.extents(key, yaw, pitch, roll)
        stem = self.bank.meta[key]["stem"]
        rec = {
            "key": key,
            "stem": stem,
            "x": float(x),
            "y": float(y),
            "kind": kind,
            "group": group,
            "radius": float(radius),
            "height": float(h),
            "longest": float(longest),
            "long_xy": float(long_xy),
        }
        self.pieces.append(rec)
        if longest > self.largest["size"]:
            self.largest = {
                "size": float(longest),
                "height": float(h),
                "stem": stem,
                "kind": kind,
            }
        return rec

    def sit_piece(self, key, x, y, yaw, pitch, roll, kind, group, radius, lift=0.02):
        bid, _, _ = self.bank.sit_oriented(
            self.terrain, key, x, y, yaw=yaw, pitch=pitch, roll=roll, lift=lift
        )
        self.bodies.append(bid)
        return self.record(key, x, y, yaw, pitch, roll, kind, group, radius)

    def spawn_air(self, key, x, y, z, yaw, pitch, roll, kind, group, radius):
        bid = self.bank.spawn(key, x, y, z, yaw=yaw, pitch=pitch, roll=roll)
        self.bodies.append(bid)
        return self.record(key, x, y, yaw, pitch, roll, kind, group, radius)


def _street_edge_arch_sites(district):
    cx, cy = district.center
    half = 0.5 * district.size
    sites = []
    for slab in district.roads:
        if slab["along"] == "y":
            yaw = 0.0
            for y_end in (slab["ymin"] + 4.0, slab["ymax"] - 4.0):
                x = slab["cx"]
                edge = min(abs(y_end - (cy - half)), abs(y_end - (cy + half)))
                if edge > 32.0:
                    continue
                sites.append(
                    {
                        "x": x,
                        "y": float(y_end),
                        "yaw": yaw,
                        "edge": edge,
                        "center": math.hypot(x - cx, y_end - cy),
                    }
                )
        else:
            yaw = 0.5 * math.pi
            for x_end in (slab["xmin"] + 4.0, slab["xmax"] - 4.0):
                y = slab["cy"]
                edge = min(abs(x_end - (cx - half)), abs(x_end - (cx + half)))
                if edge > 32.0:
                    continue
                sites.append(
                    {
                        "x": float(x_end),
                        "y": y,
                        "yaw": yaw,
                        "edge": edge,
                        "center": math.hypot(x_end - cx, y - cy),
                    }
                )
    sites.sort(key=lambda s: (s["edge"], -s["center"]))
    return sites


def _arch_ok(world, x, y, yaw, half_span, extra_r):
    ux, uy = math.cos(yaw), math.sin(yaw)
    p1 = (x - ux * half_span, y - uy * half_span)
    p2 = (x + ux * half_span, y + uy * half_span)
    d = world.district
    for px, py in (p1, p2, (x, y)):
        if not _in_world(world.terrain, px, py, margin=12.0):
            return False
        if _hits_building(px, py, d.buildings, 1.6):
            return False
        if _hits_car(px, py, d.cars, 2.4):
            return False
        if not _keepout_ok(px, py, world.hard_keep, extra_r=extra_r):
            return False
    return True


def _place_arch(world, x, y, yaw, pillar_h, pillar_w, gap, lintel_stem, lintel_scale, kind):
    half_span = 0.5 * (pillar_w + gap)
    pillar_key = _ensure(world.bank, "building-structure", [pillar_w, pillar_h, pillar_w])
    lintel_key = _ensure(world.bank, lintel_stem, lintel_scale)
    ux, uy = math.cos(yaw), math.sin(yaw)
    p1 = (x - ux * half_span, y - uy * half_span)
    p2 = (x + ux * half_span, y + uy * half_span)
    yaw_a = yaw + float(world.rng.uniform(-0.12, 0.12))
    world.sit_piece(pillar_key, p1[0], p1[1], yaw_a, 0.0, 0.0, "arch-pillar", "arch", 0.55 * pillar_w)
    world.sit_piece(
        pillar_key,
        p2[0],
        p2[1],
        yaw_a + 0.08,
        0.0,
        0.0,
        "arch-pillar",
        "arch",
        0.55 * pillar_w,
    )
    pitch = float(world.rng.uniform(0.28, 0.52))
    roll = float(world.rng.uniform(-0.10, 0.10))
    orn = world.bank._orn(yaw + 0.5 * math.pi, pitch, roll)
    zmin0 = min(pt[2] for pt in world.bank.corners_world(lintel_key, 0.0, 0.0, 0.0, orn))
    gz = _sample_heightmap(world.terrain, x, y)
    z = gz + pillar_h * 0.70 - zmin0
    world.spawn_air(lintel_key, x, y, z, yaw + 0.5 * math.pi, pitch, roll, "arch-lintel", "arch", 6.0)
    world.arches.append(
        {
            "x": float(x),
            "y": float(y),
            "kind": kind,
            "yaw": float(yaw),
            "height": float(pillar_h),
            "width": float(2.0 * pillar_w + gap),
            "gap": float(gap),
        }
    )


def _place_arches(world):
    d = world.district
    placed = False
    for site in _street_edge_arch_sites(d):
        pillar_h, pillar_w, gap = 10.2, 3.2, 4.1
        half_span = 0.5 * (pillar_w + gap)
        if not _arch_ok(world, site["x"], site["y"], site["yaw"], half_span, extra_r=3.0):
            continue
        if not _inside_district(site["x"], site["y"], d, margin=2.0):
            continue
        _place_arch(
            world,
            site["x"],
            site["y"],
            site["yaw"],
            pillar_h,
            pillar_w,
            gap,
            "building-roof",
            [9.6, 1.85, 2.7],
            "street-edge",
        )
        placed = True
        break
    if not placed:
        raise RuntimeError("could not place street-edge arch")

    slope, Z = world.slope, world.Z
    cx, cy = d.center
    half = 0.5 * d.size
    cands = []
    for r in (8.0, 12.0, 16.0, 20.0, 24.0):
        for t in np.linspace(0.0, 1.0, 28, endpoint=False):
            # Perimeter just outside the 300 m square.
            if t < 0.25:
                x = cx - half - r
                y = cy - half + (t / 0.25) * (2.0 * half + 2.0 * r)
            elif t < 0.50:
                x = cx - half - r + ((t - 0.25) / 0.25) * (2.0 * half + 2.0 * r)
                y = cy + half + r
            elif t < 0.75:
                x = cx + half + r
                y = cy + half + r - ((t - 0.50) / 0.25) * (2.0 * half + 2.0 * r)
            else:
                x = cx + half + r - ((t - 0.75) / 0.25) * (2.0 * half + 2.0 * r)
                y = cy - half - r
            if not _in_world(world.terrain, x, y, margin=16.0):
                continue
            if _min_district_dist(x, y, world.boxes) < 6.0:
                continue
            if not _flat_ok(world.terrain, slope, Z, x, y, s_max=0.10, rel_max=5.5):
                continue
            if not _keepout_ok(x, y, world.soft_keep, extra_r=8.0):
                continue
            i, j = _ij(world.terrain, x, y)
            cands.append((float(slope[i, j]), _local_relief(Z, i, j), r, x, y))
    if not cands:
        raise RuntimeError("could not place outside arch (no flat ground)")
    cands.sort()
    pillar_h, pillar_w, gap = 11.0, 3.5, 4.6
    half_span = 0.5 * (pillar_w + gap)
    ax0, ay0 = world.arches[0]["x"], world.arches[0]["y"]
    chosen = None
    for _, _, _, x, y in cands:
        yaw = math.atan2(y - cy, x - cx) + 0.5 * math.pi
        if not _arch_ok(world, x, y, yaw, half_span, extra_r=3.0):
            continue
        if math.hypot(x - ax0, y - ay0) < 40.0:
            continue
        chosen = (x, y, yaw)
        break
    if chosen is None:
        _, _, _, x, y = cands[0]
        yaw = math.atan2(y - cy, x - cx) + 0.5 * math.pi
        chosen = (x, y, yaw)
    _place_arch(
        world,
        chosen[0],
        chosen[1],
        chosen[2],
        pillar_h,
        pillar_w,
        gap,
        "building-platform",
        [9.2, 2.3, 3.1],
        "outside",
    )


def _district_pile_candidates(district):
    cands = []
    for b in district.buildings:
        sx = b["xmax"] - b["xmin"]
        sy = b["ymax"] - b["ymin"]
        off = 3.5
        for t in (0.18, 0.50, 0.82):
            cands.append((b["xmin"] + t * sx, b["ymin"] - off, "wall"))
            cands.append((b["xmin"] + t * sx, b["ymax"] + off, "wall"))
            cands.append((b["xmin"] - off, b["ymin"] + t * sy, "wall"))
            cands.append((b["xmax"] + off, b["ymin"] + t * sy, "wall"))
    for slab in district.roads:
        # Sit just outside the pavement so the pile can spill into the street
        # without covering the walkable centerline.
        if slab["along"] == "y":
            y0, y1 = slab["ymin"] + 12.0, slab["ymax"] - 12.0
            if y1 <= y0:
                continue
            for y in np.linspace(y0, y1, max(2, int((y1 - y0) / 16.0))):
                cands.append((slab["xmin"] - 1.6, float(y), "street"))
                cands.append((slab["xmax"] + 1.6, float(y), "street"))
        else:
            x0, x1 = slab["xmin"] + 12.0, slab["xmax"] - 12.0
            if x1 <= x0:
                continue
            for x in np.linspace(x0, x1, max(2, int((x1 - x0) / 16.0))):
                cands.append((float(x), slab["ymin"] - 1.6, "street"))
                cands.append((float(x), slab["ymax"] + 1.6, "street"))
    return cands


def _pile_site_ok(world, x, y, extra_r, inside, midline=True):
    d = world.district
    if inside:
        if not _inside_district(x, y, d, margin=8.0):
            return False
        if _hits_building(x, y, d.buildings, 2.0):
            return False
        if _hits_car(x, y, d.cars, 3.2):
            return False
        if midline and _midline_dist(x, y, d.roads) < 5.0:
            return False
        if not _keepout_ok(x, y, world.hard_keep, extra_r=extra_r):
            return False
    else:
        if _min_district_dist(x, y, world.boxes) < 12.0:
            return False
        if not _in_world(world.terrain, x, y, margin=20.0):
            return False
        if not _flat_ok(world.terrain, world.slope, world.Z, x, y, s_max=0.11, rel_max=7.0):
            return False
        if not _keepout_ok(x, y, world.soft_keep, extra_r=extra_r):
            return False
    return True


def _fit_district_xy(world, x, y, inside):
    """Keep a debris piece off building interiors, cars, and street midlines."""
    if not inside:
        return x, y
    d = world.district
    px, py = _nudge_from_midline(x, y, d.roads, min_d=_MIDLINE_CLEAR + 0.3)
    if _hits_building(px, py, d.buildings, 1.4) or _hits_car(px, py, d.cars, 2.4):
        return None
    if _midline_dist(px, py, d.roads) < _MIDLINE_CLEAR:
        return None
    return px, py


def _rock_scale(stem, height, footprint):
    """Non-uniform scale: file-Y is height. Footprint stays tighter than height."""
    if stem == "rocks-high":
        return [footprint, height, footprint]
    if stem == "rocks-low":
        return [footprint, height / 0.523, footprint]
    if stem == "rocks-ramp":
        return [footprint, height / 0.500, footprint]
    return [footprint * 0.95, height / 0.454, footprint * 0.90]


def _place_pile(world, x, y, where):
    rng = world.rng
    inside = where.startswith("district")
    n_rocks = int(rng.integers(4, 7))
    peak_h = float(rng.uniform(6.4, 9.6))
    extras = {
        "rocks-low": float(rng.uniform(3.4, 5.2)),
        "rocks-ramp": float(rng.uniform(3.6, 5.6)),
        "stones": float(rng.uniform(2.6, 4.2)),
    }
    order = ["rocks-high"] + [str(s) for s in rng.choice(["rocks-low", "rocks-ramp", "stones"], size=n_rocks - 1)]
    offs = _cluster_offsets(rng, n_rocks, 2.8, 1.0)
    placed_rocks = 0
    for k, stem in enumerate(order):
        if stem == "rocks-high":
            height = peak_h if k == 0 else float(rng.uniform(6.0, min(peak_h, 8.0)))
            footprint = float(rng.uniform(4.2, 5.4))
        else:
            height = extras[stem] * float(rng.uniform(0.9, 1.05))
            footprint = float(rng.uniform(3.4, 4.8))
        key = _ensure(world.bank, stem, _rock_scale(stem, height, footprint))
        ox, oy = offs[k]
        px, py = x + ox, y + oy
        fitted = _fit_district_xy(world, px, py, inside)
        if fitted is None:
            fitted = _fit_district_xy(world, x + 0.4 * ox, y + 0.4 * oy, inside)
        if fitted is None:
            if k == 0:
                fitted = (x, y)
            else:
                continue
        px, py = fitted
        yaw = float(rng.uniform(0.0, 2.0 * math.pi))
        if stem == "rocks-high" and k == 0:
            pitch = float(rng.uniform(0.08, 0.28))
            roll = float(rng.uniform(-0.18, 0.18))
        else:
            pitch = float(rng.uniform(0.25, 0.95))
            roll = float(rng.uniform(-0.65, 0.65))
        world.sit_piece(key, px, py, yaw, pitch, roll, "pile-rock", "pile", 0.5 * footprint)
        placed_rocks += 1
    while placed_rocks < 4:
        height = float(rng.uniform(3.2, 4.8))
        footprint = 3.6
        key = _ensure(world.bank, "stones", _rock_scale("stones", height, footprint))
        ox, oy = _cluster_offsets(rng, 1, 2.2, 0.8)[0]
        fitted = _fit_district_xy(world, x + ox, y + oy, inside) or (x, y)
        world.sit_piece(
            key,
            fitted[0],
            fitted[1],
            float(rng.uniform(0.0, 6.28)),
            float(rng.uniform(0.3, 0.9)),
            float(rng.uniform(-0.5, 0.5)),
            "pile-rock",
            "pile",
            1.8,
        )
        placed_rocks += 1
    if rng.random() < 0.55:
        slab_stem, slab_scale = "building-roof", [5.2, 1.35, 3.6]
    else:
        slab_stem, slab_scale = "building-structure", [1.35, 6.4, 4.6]
    slab_key = _ensure(world.bank, slab_stem, slab_scale)
    ang = float(rng.uniform(0.0, 2.0 * math.pi))
    sx, sy = x + 1.4 * math.cos(ang), y + 1.4 * math.sin(ang)
    fitted = _fit_district_xy(world, sx, sy, inside) or (x, y)
    world.sit_piece(
        slab_key,
        fitted[0],
        fitted[1],
        float(rng.uniform(0.0, 2.0 * math.pi)),
        float(rng.uniform(0.85, 1.25)),
        float(rng.uniform(-0.35, 0.35)),
        "pile-slab",
        "pile",
        3.4,
    )
    world.piles.append({"x": float(x), "y": float(y), "where": where, "n_rocks": placed_rocks, "peak": peak_h})


def _place_piles(world):
    d = world.district
    blockers = [(a["x"], a["y"]) for a in world.arches]
    cands = _district_pile_candidates(d)
    world.rng.shuffle(cands)
    # Prefer wall-adjacent sites, then street edges.
    cands.sort(key=lambda t: 0 if t[2] == "wall" else 1)
    n_in = 0
    for sep in (_PILE_SEP_IN, 28.0, 20.0, 14.0):
        for x, y, tag in cands:
            if n_in >= _N_PILES_IN:
                break
            if not _pile_site_ok(world, x, y, extra_r=5.5, inside=True):
                continue
            if not _sep_ok(x, y, blockers, sep):
                continue
            _place_pile(world, x, y, f"district-{tag}")
            blockers.append((x, y))
            n_in += 1
        if n_in >= _N_PILES_IN:
            break
    if n_in < _N_PILES_IN:
        raise RuntimeError(f"could only place {n_in}/{_N_PILES_IN} district piles")

    cx, cy = d.center
    half = 0.5 * d.size
    out_cands = []
    for r in np.arange(28.0, 95.0, 10.0):
        for ang in np.linspace(0.0, 2.0 * math.pi, 36, endpoint=False):
            x = cx + (half + r) * math.cos(ang)
            y = cy + (half + r) * math.sin(ang)
            out_cands.append((x, y))
    world.rng.shuffle(out_cands)
    n_out = 0
    for sep in (_PILE_SEP_OUT, 36.0, 24.0, 16.0):
        for x, y in out_cands:
            if n_out >= _N_PILES_OUT:
                break
            if not _pile_site_ok(world, x, y, extra_r=6.0, inside=False):
                continue
            if not _sep_ok(x, y, blockers, sep):
                continue
            _place_pile(world, x, y, "outside")
            blockers.append((x, y))
            n_out += 1
        if n_out >= _N_PILES_OUT:
            break
    if n_out < _N_PILES_OUT:
        raise RuntimeError(f"could only place {n_out}/{_N_PILES_OUT} outside piles")


def _place_walls(world):
    hosts = [(a["x"], a["y"], a["kind"]) for a in world.arches] + [
        (p["x"], p["y"], p["where"]) for p in world.piles
    ]
    rng = world.rng
    n = 0
    used = []
    for hi, (hx, hy, hkind) in enumerate(hosts):
        if n >= _N_WALLS:
            break
        for _try in range(18):
            ang = float(rng.uniform(0.0, 2.0 * math.pi))
            dist = float(rng.uniform(6.5, 9.5))
            x = hx + dist * math.cos(ang)
            y = hy + dist * math.sin(ang)
            inside = _min_district_dist(x, y, world.boxes) == 0.0
            extra = 5.0
            if not _pile_site_ok(world, x, y, extra_r=extra, inside=inside, midline=inside):
                # Outside-of-district walls next to the outside arch/piles.
                if inside:
                    continue
                if not _in_world(world.terrain, x, y, margin=16.0):
                    continue
                if not _keepout_ok(x, y, world.hard_keep, extra_r=extra):
                    continue
                if _hits_building(x, y, world.district.buildings, 2.0):
                    continue
            if not _sep_ok(x, y, used, 8.0):
                continue
            yaw = ang + 0.5 * math.pi + float(rng.uniform(-0.2, 0.2))
            fitted = _fit_district_xy(world, x, y, inside)
            if fitted is None:
                continue
            x, y = fitted
            if n % 2 == 0:
                key = _ensure(world.bank, "building-structure", [1.45, 8.6, 6.4])
                pitch = float(rng.uniform(1.05, 1.35))
                roll = float(rng.uniform(-0.18, 0.18))
                world.sit_piece(key, x, y, yaw, pitch, roll, "fallen-wall", "wall", 4.5)
                _h, _long_xy, longest = world.bank.extents(key, yaw, pitch, roll)
                if longest < 6.0:
                    raise RuntimeError(f"fallen wall shorter than 6 m ({longest:.2f})")
            else:
                fence_scale = 6.15
                key = _ensure(world.bank, "fence", fence_scale)
                n_seg = 3
                spacing = 1.096 * fence_scale * 0.92
                ux, uy = math.cos(yaw), math.sin(yaw)
                pitch = float(rng.uniform(-0.12, 0.12))
                roll = 0.5 * math.pi + float(rng.uniform(-0.08, 0.08))
                segs = []
                for s in range(n_seg):
                    t = (s - 0.5 * (n_seg - 1)) * spacing
                    fx, fy = x + t * ux, y + t * uy
                    ff = _fit_district_xy(world, fx, fy, inside)
                    if ff is None:
                        segs = []
                        break
                    segs.append(ff)
                if len(segs) < n_seg:
                    continue
                for fx, fy in segs:
                    world.sit_piece(
                        key,
                        fx,
                        fy,
                        yaw + float(rng.uniform(-0.06, 0.06)),
                        pitch,
                        roll,
                        "fallen-fence",
                        "wall",
                        3.2,
                    )
            world.walls.append({"x": float(x), "y": float(y), "host": hkind})
            used.append((x, y))
            n += 1
            break
    if n < _N_WALLS:
        raise RuntimeError(f"could only place {n}/{_N_WALLS} fallen walls")


def _verify(world):
    d = world.district
    people = _people_keepouts(world.scenery)
    camps = _camp_keepouts(world.scenery)
    issues = []
    for rec in world.pieces:
        x, y, r = rec["x"], rec["y"], rec["radius"]
        if _hits_building(x, y, d.buildings, 1.2):
            issues.append(f"{rec['kind']} inside building at ({x:.1f},{y:.1f})")
        if _hits_car(x, y, d.cars, 1.6):
            issues.append(f"{rec['kind']} on car at ({x:.1f},{y:.1f})")
        for kx, ky, kr in people + camps:
            if math.hypot(x - kx, y - ky) < kr + r * 0.35:
                issues.append(
                    f"{rec['kind']} within 4 m keepout of ({kx:.1f},{ky:.1f})"
                )
        if rec["group"] in ("pile", "wall") and _inside_district(x, y, d, margin=0.0):
            if rec["kind"] != "arch-pillar" and _midline_dist(x, y, d.roads) < 2.0:
                issues.append(f"{rec['kind']} on street midline at ({x:.1f},{y:.1f})")
    if issues:
        raise RuntimeError("rubble verify failed: " + "; ".join(issues[:8]))


def spawn_rubble_in_pybullet(client, terrain, districts, scenery, seed=_SEED):
    """
    Place 2 broken arches, 8 rubble piles, and 4 fallen walls.

    Returns (body_ids, info).
    """
    world = _RubbleWorld(client, terrain, districts, scenery, seed=seed)
    _place_arches(world)
    _place_piles(world)
    _place_walls(world)
    _verify(world)

    loaded = sorted({world.bank.meta[k]["stem"] for k in world.bank.vis})
    banned_hit = [s for s in loaded if f"{s}.obj" in _BANNED]
    if banned_hit:
        raise RuntimeError(f"banned meshes loaded: {banned_hit}")

    largest = world.largest
    print(
        f"[RUBBLE] arches={len(world.arches)}  piles={len(world.piles)}  "
        f"fallen_walls={len(world.walls)}  bodies={len(world.bodies)}"
    )
    print(
        f"[RUBBLE] largest piece={largest['size']:.1f} m "
        f"(height={largest['height']:.1f} m, {largest['stem']}, {largest['kind']})"
    )
    for i, a in enumerate(world.arches):
        print(
            f"[RUBBLE] arch{i} {a['kind']} ({a['x']:.1f}, {a['y']:.1f})  "
            f"{a['height']:.1f} m tall  {a['width']:.1f} m wide  gap={a['gap']:.1f} m"
        )
    print(f"[RUBBLE] meshes: {','.join(loaded)}")

    info = {
        "arches": world.arches,
        "piles": world.piles,
        "walls": world.walls,
        "n_arches": len(world.arches),
        "n_piles": len(world.piles),
        "n_walls": len(world.walls),
        "n_bodies": len(world.bodies),
        "largest": largest,
        "meshes": loaded,
        "pieces": world.pieces,
    }
    return world.bodies, info
