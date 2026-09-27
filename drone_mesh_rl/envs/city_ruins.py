"""
city_ruins.py — Destroyed-city groups from Colonne_1.obj only.

The three Colonne_*.obj files are the same full scene. This module reads
Colonne_1.obj, splits on Wavefront `o ` names, and never opens
Colonne_2.obj or Colonne_3.obj. Each group is placed with one rigid
transform so floors, walls, and columns keep their file offsets.
"""

from __future__ import annotations

import math
import os
import re
import tempfile
import unicodedata

import numpy as np

from .rubble import (
    _camp_keepouts,
    _cluster_keepouts,
    _flat_ok,
    _hits_building,
    _hits_car,
    _inside_district,
    _keepout_ok,
    _midline_dist,
    _people_keepouts,
    _sep_ok,
)
from .scenery import (
    _MeshBank,
    _district_boxes,
    _in_world,
    _min_district_dist,
    _slope_field,
)
from .town import _sample_heightmap

_RUBBLE_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "RubblePrefabs"))
_SOURCE_NAME = "Colonne_1.obj"
_IGNORE_FILES = frozenset({"Colonne_2.obj", "Colonne_3.obj"})
_SEED = 47
_TARGET_H = 14.0
_LIFT = 0.04
_MIDLINE = 3.8
_CACHE_VER = "v1"

_COPIES = {
    "left": 2,
    "right": 2,
    "center": 2,
    "debris": 3,
}

# Prefer a mix in/just outside the district; at most one ruin + one debris far out.
_PLACEMENT_PLAN = (
    ("center", "against-building"),
    ("center", "street-edge"),
    ("left", "against-building"),
    ("left", "road-exit"),
    ("right", "against-building"),
    ("right", "open-ground"),
    ("debris", "street-edge"),
    ("debris", "road-exit"),
    ("debris", "open-ground"),
)

_TINT = {
    "etage": [0.58, 0.54, 0.47, 1.0],
    "etape": [0.58, 0.54, 0.47, 1.0],
    "face_pleine": [0.50, 0.45, 0.39, 1.0],
    "face": [0.54, 0.40, 0.34, 1.0],
    "colonne": [0.64, 0.60, 0.54, 1.0],
    "route": [0.30, 0.30, 0.32, 1.0],
    "rock": [0.40, 0.37, 0.33, 1.0],
    "debri": [0.44, 0.40, 0.35, 1.0],
    "debr": [0.44, 0.40, 0.35, 1.0],
}


def _fold(name):
    n = unicodedata.normalize("NFKD", name)
    return "".join(c for c in n if not unicodedata.combining(c)).lower()


def _slug(name):
    return re.sub(r"[^A-Za-z0-9_]+", "_", _fold(name)).strip("_") or "part"


def _tint_for(name):
    n = _fold(name)
    for key in ("face_pleine", "face", "colonne", "etage", "etape", "route", "rock", "debri", "debr"):
        if n.startswith(key):
            return list(_TINT[key])
    return [0.52, 0.48, 0.42, 1.0]


def _is_ruin_piece(name):
    n = _fold(name)
    return n.startswith("etage") or n.startswith("etape") or n.startswith("face") or n.startswith("colonne")


def _is_debris_piece(name):
    n = _fold(name)
    return bool(re.match(r"(route|rock|debri|debr)_\d", n)) or n.startswith("debri") or n.startswith("debr")


def _source_path():
    path = os.path.join(_RUBBLE_DIR, _SOURCE_NAME)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return path


def _parse_colonne1():
    """Split Colonne_1.obj on `o ` names. Vertices stay in file space."""
    src = _source_path()
    verts = []
    objects = []
    cur = None
    faces = []

    def flush():
        nonlocal cur, faces
        if cur is not None:
            objects.append((cur, faces))
        faces = []

    with open(src, encoding="utf-8") as f:
        for line in f:
            if line.startswith("o "):
                flush()
                cur = line[2:].strip()
            elif line.startswith("v "):
                parts = line.split()
                verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif line.startswith("f "):
                idxs = []
                for tok in line.split()[1:]:
                    vi = int(tok.split("/")[0])
                    if vi < 0:
                        vi = len(verts) + 1 + vi
                    idxs.append(vi)
                if len(idxs) >= 3:
                    faces.append(idxs)
    flush()
    return verts, objects


def _part_bounds(verts, faces):
    used = set()
    for face in faces:
        used.update(face)
    xs, ys, zs = [], [], []
    for i in used:
        x, y, z = verts[i - 1]
        xs.append(x)
        ys.append(y)
        zs.append(z)
    return {
        "xmin": min(xs),
        "xmax": max(xs),
        "ymin": min(ys),
        "ymax": max(ys),
        "zmin": min(zs),
        "zmax": max(zs),
        "cx": 0.5 * (min(xs) + max(xs)),
        "cy": 0.5 * (min(ys) + max(ys)),
        "cz": 0.5 * (min(zs) + max(zs)),
        "h": max(ys) - min(ys),
    }


def _write_part_obj(path, verts, faces):
    used = []
    seen = {}
    for face in faces:
        for vi in face:
            if vi not in seen:
                seen[vi] = len(used) + 1
                used.append(vi)
    with open(path, "w", encoding="utf-8") as fout:
        fout.write("# extracted from Colonne_1.obj\n")
        for vi in used:
            x, y, z = verts[vi - 1]
            fout.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        for face in faces:
            mapped = [seen[vi] for vi in face]
            if len(mapped) == 3:
                fout.write(f"f {mapped[0]} {mapped[1]} {mapped[2]}\n")
            else:
                for i in range(1, len(mapped) - 1):
                    fout.write(f"f {mapped[0]} {mapped[i]} {mapped[i + 1]}\n")


def _extract_parts(verts, objects):
    cache = os.path.join(tempfile.gettempdir(), f"disasterdrones_colonne1_{_CACHE_VER}")
    os.makedirs(cache, exist_ok=True)
    src = _source_path()
    stamp = os.path.join(cache, ".stamp")
    fresh = (
        os.path.isfile(stamp)
        and os.path.getmtime(stamp) >= os.path.getmtime(src)
        and open(stamp, encoding="utf-8").read().strip() == str(len(objects))
    )
    parts = []
    for name, faces in objects:
        path = os.path.join(cache, f"{_slug(name)}.obj")
        if not fresh:
            _write_part_obj(path, verts, faces)
        bounds = _part_bounds(verts, faces)
        parts.append({"name": name, "path": path, "bounds": bounds, "faces": len(faces)})
    if not fresh:
        with open(stamp, "w", encoding="utf-8") as f:
            f.write(str(len(objects)))
    return parts


def _classify(parts):
    groups = {k: [] for k in ("left", "right", "center", "debris")}
    tallest = 0.0
    for part in parts:
        b = part["bounds"]
        tallest = max(tallest, b["h"])
        name = part["name"]
        if _is_debris_piece(name):
            groups["debris"].append(part)
        elif _is_ruin_piece(name):
            cx = b["cx"]
            if cx < -2.0:
                groups["left"].append(part)
            elif cx > 1.5:
                groups["right"].append(part)
            else:
                groups["center"].append(part)
    if not all(groups.values()):
        raise RuntimeError(f"empty city-ruin group: {[k for k, v in groups.items() if not v]}")
    return groups, tallest


def _group_aabb(parts):
    xs = [p["bounds"]["xmin"] for p in parts] + [p["bounds"]["xmax"] for p in parts]
    ys = [p["bounds"]["ymin"] for p in parts] + [p["bounds"]["ymax"] for p in parts]
    zs = [p["bounds"]["zmin"] for p in parts] + [p["bounds"]["zmax"] for p in parts]
    return {
        "xmin": min(xs),
        "xmax": max(xs),
        "ymin": min(ys),
        "ymax": max(ys),
        "zmin": min(zs),
        "zmax": max(zs),
        "cx": 0.5 * (min(xs) + max(xs)),
        "cy": 0.5 * (min(ys) + max(ys)),
        "cz": 0.5 * (min(zs) + max(zs)),
        "hx": 0.5 * (max(xs) - min(xs)),
        "hy": 0.5 * (max(ys) - min(ys)),
        "hz": 0.5 * (max(zs) - min(zs)),
    }


def _world_xy(fx, fz, dest_x, dest_y, yaw, scale, gcx, gcz):
    ux = scale * (fx - gcx)
    uz = scale * (fz - gcz)
    c, s = math.cos(yaw), math.sin(yaw)
    return dest_x + c * ux + s * uz, dest_y + s * ux - c * uz


def _footprint_pts(aabb, dest_x, dest_y, yaw, scale, n=3):
    gcx, gcz = aabb["cx"], aabb["cz"]
    pts = []
    for fx in np.linspace(aabb["xmin"], aabb["xmax"], n):
        for fz in np.linspace(aabb["zmin"], aabb["zmax"], n):
            pts.append(_world_xy(fx, fz, dest_x, dest_y, yaw, scale, gcx, gcz))
    return pts


def _half_world(aabb, scale, yaw):
    hx = aabb["hx"] * scale
    hz = aabb["hz"] * scale
    c, s = abs(math.cos(yaw)), abs(math.sin(yaw))
    return c * hx + s * hz, s * hx + c * hz


def _rubble_keepouts(scenery):
    out = []
    rubble = scenery.get("rubble") or {}
    for rec in rubble.get("piles") or []:
        out.append((float(rec["x"]), float(rec["y"]), 8.0))
    for rec in rubble.get("arches") or []:
        out.append((float(rec["x"]), float(rec["y"]), 10.0))
    for rec in rubble.get("walls") or []:
        out.append((float(rec["x"]), float(rec["y"]), 6.0))
    return out


def _site_ok(world, x, y, yaw, aabb, scale, extra_r, far, keepouts):
    terrain = world["terrain"]
    district = world["district"]
    if not _in_world(terrain, x, y, margin=18.0):
        return False
    ddist = _min_district_dist(x, y, world["boxes"])
    if far:
        if ddist < 36.0:
            return False
        if not _flat_ok(terrain, world["slope"], world["Z"], x, y, s_max=0.14, rel_max=9.0):
            return False
    else:
        if ddist > 24.0 and not _inside_district(x, y, district, margin=0.0):
            return False
    if not _keepout_ok(x, y, keepouts, extra_r=extra_r):
        return False
    for px, py in _footprint_pts(aabb, x, y, yaw, scale):
        if not _in_world(terrain, px, py, margin=10.0):
            return False
        if _hits_building(px, py, district.buildings, 0.7):
            return False
        if _hits_car(px, py, district.cars, 1.6):
            return False
        if not _keepout_ok(px, py, world["hard_keep"], extra_r=0.5):
            return False
    if _inside_district(x, y, district, margin=0.0):
        if _midline_dist(x, y, district.roads) < _MIDLINE:
            return False
    return True


def _wall_sites(district, aabb, scale):
    hx = aabb["hx"] * scale
    hz = aabb["hz"] * scale
    sites = []
    for b in district.buildings:
        sx = b["xmax"] - b["xmin"]
        sy = b["ymax"] - b["ymin"]
        for t in (0.22, 0.50, 0.78):
            sites.append(
                {
                    "x": b["xmin"] + t * sx,
                    "y": b["ymin"] - hz - 1.8,
                    "yaw": 0.0,
                    "tag": "against-building",
                }
            )
            sites.append(
                {
                    "x": b["xmin"] + t * sx,
                    "y": b["ymax"] + hz + 1.8,
                    "yaw": 0.0,
                    "tag": "against-building",
                }
            )
            sites.append(
                {
                    "x": b["xmin"] - hz - 1.8,
                    "y": b["ymin"] + t * sy,
                    "yaw": 0.5 * math.pi,
                    "tag": "against-building",
                }
            )
            sites.append(
                {
                    "x": b["xmax"] + hz + 1.8,
                    "y": b["ymin"] + t * sy,
                    "yaw": 0.5 * math.pi,
                    "tag": "against-building",
                }
            )
    return sites


def _street_edge_sites(district, aabb, scale):
    sites = []
    for slab in district.roads:
        if slab["along"] == "y":
            yaw = 0.5 * math.pi
            wx, wy = _half_world(aabb, scale, yaw)
            y0, y1 = slab["ymin"] + 10.0, slab["ymax"] - 10.0
            if y1 <= y0:
                continue
            for y in np.linspace(y0, y1, max(2, int((y1 - y0) / 22.0))):
                sites.append({"x": slab["xmin"] - wx - 0.6, "y": float(y), "yaw": yaw, "tag": "street-edge"})
                sites.append({"x": slab["xmax"] + wx + 0.6, "y": float(y), "yaw": yaw, "tag": "street-edge"})
        else:
            yaw = 0.0
            wx, wy = _half_world(aabb, scale, yaw)
            x0, x1 = slab["xmin"] + 10.0, slab["xmax"] - 10.0
            if x1 <= x0:
                continue
            for x in np.linspace(x0, x1, max(2, int((x1 - x0) / 22.0))):
                sites.append({"x": float(x), "y": slab["ymin"] - wy - 0.6, "yaw": yaw, "tag": "street-edge"})
                sites.append({"x": float(x), "y": slab["ymax"] + wy + 0.6, "yaw": yaw, "tag": "street-edge"})
    return sites


def _road_exit_sites(district, aabb, scale):
    cx, cy = district.center
    half = 0.5 * district.size
    sites = []
    for slab in district.roads:
        if slab["along"] == "y":
            yaw = 0.5 * math.pi
            wx, wy = _half_world(aabb, scale, yaw)
            for y_edge, y_out in ((slab["ymin"], cy - half - 10.0), (slab["ymax"], cy + half + 10.0)):
                if min(abs(y_edge - (cy - half)), abs(y_edge - (cy + half))) > 18.0:
                    continue
                x_off = slab["cx"] + (wx + 2.4) * (1.0 if slab["cx"] >= cx else -1.0)
                sites.append({"x": x_off, "y": float(y_edge), "yaw": yaw, "tag": "road-exit"})
                sites.append({"x": x_off, "y": float(y_out), "yaw": yaw, "tag": "road-exit"})
        else:
            yaw = 0.0
            wx, wy = _half_world(aabb, scale, yaw)
            for x_edge, x_out in ((slab["xmin"], cx - half - 10.0), (slab["xmax"], cx + half + 10.0)):
                if min(abs(x_edge - (cx - half)), abs(x_edge - (cx + half))) > 18.0:
                    continue
                y_off = slab["cy"] + (wy + 2.4) * (1.0 if slab["cy"] >= cy else -1.0)
                sites.append({"x": float(x_edge), "y": y_off, "yaw": yaw, "tag": "road-exit"})
                sites.append({"x": float(x_out), "y": y_off, "yaw": yaw, "tag": "road-exit"})
    return sites


def _open_ground_sites(world, aabb, scale, rng):
    district = world["district"]
    cx, cy = district.center
    half = 0.5 * district.size
    sites = []
    yaws = (0.0, 0.5 * math.pi, 0.35, -0.55)
    for r in np.arange(42.0, 110.0, 12.0):
        for ang in np.linspace(0.0, 2.0 * math.pi, 28, endpoint=False):
            x = cx + (half + r) * math.cos(ang)
            y = cy + (half + r) * math.sin(ang)
            yaw = float(yaws[int(round((ang / (0.25 * math.pi)))) % len(yaws)])
            sites.append({"x": float(x), "y": float(y), "yaw": yaw, "tag": "open-ground"})
    rng.shuffle(sites)
    return sites


def _candidates_for(kind_tag, aabb, scale, world, rng):
    district = world["district"]
    if kind_tag == "against-building":
        return _wall_sites(district, aabb, scale)
    if kind_tag == "street-edge":
        return _street_edge_sites(district, aabb, scale)
    if kind_tag == "road-exit":
        return _road_exit_sites(district, aabb, scale)
    if kind_tag == "open-ground":
        return _open_ground_sites(world, aabb, scale, rng)
    raise ValueError(kind_tag)


def _fallback_tags(preferred, far):
    if far:
        return (preferred, "open-ground")
    order = ["against-building", "street-edge", "road-exit"]
    return (preferred,) + tuple(t for t in order if t != preferred)


def _pick_site(world, aabb, scale, preferred, placed, keepouts, rng):
    far = preferred == "open-ground"
    extra_r = max(aabb["hx"], aabb["hz"]) * scale * 0.55
    sep = 28.0 if far else 22.0
    for tag in _fallback_tags(preferred, far):
        cands = _candidates_for(tag, aabb, scale, world, rng)
        rng.shuffle(cands)
        for sep_try in (sep, 18.0, 12.0):
            for rec in cands:
                x, y, yaw = rec["x"], rec["y"], rec["yaw"]
                if not _sep_ok(x, y, placed, sep_try):
                    continue
                if not _site_ok(world, x, y, yaw, aabb, scale, extra_r, far, keepouts):
                    continue
                return {"x": float(x), "y": float(y), "yaw": float(yaw), "tag": rec["tag"]}
    raise RuntimeError(f"could not place city-ruin group ({preferred})")


def _sit_z(terrain, aabb, dest_x, dest_y, yaw, scale):
    zs = []
    for px, py in _footprint_pts(aabb, dest_x, dest_y, yaw, scale, n=3):
        zs.append(_sample_heightmap(terrain, px, py))
    return max(zs) - aabb["ymin"] * scale + _LIFT


def _spawn_group(bank, terrain, parts, aabb, scale, dest_x, dest_y, yaw):
    gcx, gcz = aabb["cx"], aabb["cz"]
    spawn_x = dest_x - scale * (math.cos(yaw) * gcx + math.sin(yaw) * gcz)
    spawn_y = dest_y - scale * (math.sin(yaw) * gcx - math.cos(yaw) * gcz)
    spawn_z = _sit_z(terrain, aabb, dest_x, dest_y, yaw, scale)
    bodies = []
    for part in parts:
        bodies.append(
            bank.spawn(part["name"], spawn_x, spawn_y, spawn_z, yaw=yaw, key=part["name"])
        )
    return bodies, (spawn_x, spawn_y, spawn_z)


def spawn_city_ruins_in_pybullet(client, terrain, districts, scenery, seed=_SEED):
    """
    Place 2 left / 2 right / 2 center ruins and 3 street-debris piles.

    Returns (body_ids, info).
    """
    verts, objects = _parse_colonne1()
    parts = _extract_parts(verts, objects)
    groups, tallest = _classify(parts)
    scale = _TARGET_H / tallest
    aabbs = {k: _group_aabb(v) for k, v in groups.items()}

    print(
        f"[RUINS] source={_SOURCE_NAME}  ignored={','.join(sorted(_IGNORE_FILES))}  "
        f"parts={len(parts)}"
    )
    print(
        f"[RUINS] groups  left={len(groups['left'])}  right={len(groups['right'])}  "
        f"center={len(groups['center'])}  debris={len(groups['debris'])}"
    )
    print(
        f"[RUINS] scale={scale:.3f}  (tallest piece {tallest:.2f} m → {_TARGET_H:.1f} m)"
    )

    rng = np.random.default_rng(seed)
    district = districts[0]
    slope, Z = _slope_field(terrain)
    hard = _people_keepouts(scenery) + _camp_keepouts(scenery)
    keepouts = (
        _cluster_keepouts(scenery, district)
        + hard
        + _rubble_keepouts(scenery)
    )
    world = {
        "terrain": terrain,
        "district": district,
        "boxes": _district_boxes(districts),
        "slope": slope,
        "Z": Z,
        "hard_keep": hard,
    }

    bank = _MeshBank(client)
    loaded = []
    for part in parts:
        bank.load(
            _slug(part["name"]),
            scale=scale,
            key=part["name"],
            tint=_tint_for(part["name"]),
            path=part["path"],
        )
        loaded.append(part["name"])

    bodies = []
    placements = []
    placed_xy = []
    counts = {k: 0 for k in _COPIES}
    for kind, tag in _PLACEMENT_PLAN:
        if counts[kind] >= _COPIES[kind]:
            continue
        site = _pick_site(world, aabbs[kind], scale, tag, placed_xy, keepouts, rng)
        bids, spawn = _spawn_group(
            bank, terrain, groups[kind], aabbs[kind], scale, site["x"], site["y"], site["yaw"]
        )
        bodies.extend(bids)
        placed_xy.append((site["x"], site["y"]))
        keepouts.append((site["x"], site["y"], 10.0))
        counts[kind] += 1
        rec = {
            "kind": kind,
            "x": site["x"],
            "y": site["y"],
            "yaw": site["yaw"],
            "tag": site["tag"],
            "n_parts": len(groups[kind]),
            "spawn": spawn,
        }
        placements.append(rec)
        print(
            f"[RUINS] {kind:6}  ({site['x']:.1f}, {site['y']:.1f})  {site['tag']}  "
            f"parts={len(groups[kind])}"
        )

    expected = sum(_COPIES.values())
    if len(placements) != expected:
        raise RuntimeError(f"placed {len(placements)}/{expected} city-ruin groups")

    print(
        f"[RUINS] copies={len(placements)}  bodies={len(bodies)}  "
        f"meshes={len(loaded)}"
    )
    info = {
        "groups": {k: len(v) for k, v in groups.items()},
        "scale": float(scale),
        "tallest": float(tallest),
        "placements": placements,
        "n_bodies": len(bodies),
        "meshes": loaded,
        "source": _SOURCE_NAME,
    }
    return bodies, info
