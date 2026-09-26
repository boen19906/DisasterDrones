"""
Earthquake disaster geometry as axis-aligned boxes.

These are the training-world primitives (LoS + collisions). PyBullet / the
radar only *draw* the same boxes — they are not the source of truth.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class DisasterBox:
    """One solid obstacle in world coordinates (meters)."""

    kind: str
    xmin: float
    ymin: float
    zmin: float
    xmax: float
    ymax: float
    zmax: float
    rgba: tuple
    label: str = ""

    @property
    def center(self):
        return np.array(
            [
                0.5 * (self.xmin + self.xmax),
                0.5 * (self.ymin + self.ymax),
                0.5 * (self.zmin + self.zmax),
            ]
        )

    @property
    def half_extents(self):
        return np.array(
            [
                0.5 * (self.xmax - self.xmin),
                0.5 * (self.ymax - self.ymin),
                0.5 * (self.zmax - self.zmin),
            ]
        )

    def contains_point(self, point, margin=0.0):
        x, y, z = point
        return (
            self.xmin - margin <= x <= self.xmax + margin
            and self.ymin - margin <= y <= self.ymax + margin
            and self.zmin - margin <= z <= self.zmax + margin
        )

    def contains_xy(self, x, y, margin=0.0):
        return (
            self.xmin - margin <= x <= self.xmax + margin
            and self.ymin - margin <= y <= self.ymax + margin
        )

    def to_dict(self):
        return {
            "kind": self.kind,
            "label": self.label,
            "xmin": round(self.xmin, 2),
            "ymin": round(self.ymin, 2),
            "xmax": round(self.xmax, 2),
            "ymax": round(self.ymax, 2),
            "zmin": round(self.zmin, 2),
            "zmax": round(self.zmax, 2),
        }


def ray_hits_aabb(point_a, point_b, box):
    """True if the segment AB intersects the box (slab test, t in [0, 1])."""
    p0 = np.asarray(point_a, dtype=np.float64)
    p1 = np.asarray(point_b, dtype=np.float64)
    direction = p1 - p0
    tmin, tmax = 0.0, 1.0
    bounds_min = (box.xmin, box.ymin, box.zmin)
    bounds_max = (box.xmax, box.ymax, box.zmax)

    for i in range(3):
        if abs(direction[i]) < 1e-12:
            if p0[i] < bounds_min[i] or p0[i] > bounds_max[i]:
                return False
            continue
        inv = 1.0 / direction[i]
        t1 = (bounds_min[i] - p0[i]) * inv
        t2 = (bounds_max[i] - p0[i]) * inv
        if t1 > t2:
            t1, t2 = t2, t1
        tmin = max(tmin, t1)
        tmax = min(tmax, t2)
        if tmin > tmax:
            return False
    return True


def _height(terrain, x, y):
    gx = int((x + terrain.size_x / 2) / terrain.resolution)
    gy = int((y + terrain.size_y / 2) / terrain.resolution)
    gx = int(np.clip(gx, 0, terrain.grid_x - 1))
    gy = int(np.clip(gy, 0, terrain.grid_y - 1))
    return float(terrain.heightmap[gy, gx])


def _box(terrain, cx, cy, sx, sy, height, kind, rgba, label="", z_lift=0.0):
    z0 = _height(terrain, cx, cy) + z_lift
    return DisasterBox(
        kind=kind,
        xmin=cx - sx * 0.5,
        ymin=cy - sy * 0.5,
        zmin=z0,
        xmax=cx + sx * 0.5,
        ymax=cy + sy * 0.5,
        zmax=z0 + height,
        rgba=tuple(rgba),
        label=label,
    )


CONCRETE = (0.55, 0.52, 0.48, 1.0)
DAMAGED = (0.42, 0.36, 0.30, 1.0)
SCHOOL = (0.78, 0.72, 0.55, 1.0)
ROOF = (0.38, 0.38, 0.40, 1.0)
RUBBLE = (0.32, 0.28, 0.24, 1.0)
METAL = (0.45, 0.48, 0.52, 1.0)


def generate_earthquake_layout(terrain, seed=42):
    """
    Procedural quake village: standing shells, a collapsed school, rubble.

    Kept as ~30-50 AABBs so LoS stays cheap during MAPPO.
    """
    rng = np.random.default_rng(seed)
    boxes = []

    # --- Collapsed elementary school (courtyard opening faces +Y) ---
    school = np.array([-18.0, 12.0])
    boxes.append(_box(terrain, school[0], school[1] + 7.5, 22.0, 1.4, 8.0, "school_wall", SCHOOL, "SCHOOL N"))
    boxes.append(_box(terrain, school[0] - 10.5, school[1], 1.4, 14.0, 8.0, "school_wall", SCHOOL, "SCHOOL W"))
    boxes.append(_box(terrain, school[0] + 10.5, school[1] - 2.0, 1.4, 10.0, 5.5, "school_wall", DAMAGED, "SCHOOL E"))
    # Collapsed roof slab sitting low — blocks overhead LoS into the classroom
    boxes.append(_box(terrain, school[0] - 2.0, school[1] - 1.0, 12.0, 8.0, 1.3, "roof_slab", ROOF, "ROOF", z_lift=2.8))
    # Fallen wall lying in the yard
    boxes.append(_box(terrain, school[0] + 4.0, school[1] + 3.5, 8.0, 1.6, 1.2, "fallen_wall", DAMAGED, z_lift=0.0))

    for _ in range(10):
        rx = school[0] + rng.uniform(-14.0, 14.0)
        ry = school[1] + rng.uniform(-10.0, 10.0)
        s = rng.uniform(0.8, 2.2)
        h = rng.uniform(0.6, 2.4)
        boxes.append(_box(terrain, rx, ry, s, s * rng.uniform(0.7, 1.4), h, "rubble", RUBBLE))

    # --- Standing (cracked) apartment block ---
    apt = np.array([22.0, 16.0])
    boxes.append(_box(terrain, apt[0], apt[1], 8.0, 10.0, 11.0, "building", CONCRETE, "APT"))
    boxes.append(_box(terrain, apt[0] + 7.0, apt[1] - 1.0, 3.5, 6.0, 4.0, "building", DAMAGED, "APT WING"))

    # --- House shells ---
    houses = [(16.0, -22.0, 7.0, 6.0, 6.5), (-28.0, -18.0, 6.0, 8.0, 7.0), (8.0, 28.0, 5.5, 5.5, 5.0)]
    for i, (hx, hy, sx, sy, hz) in enumerate(houses):
        boxes.append(_box(terrain, hx, hy, sx, sy, hz, "building", CONCRETE if i % 2 == 0 else DAMAGED, f"HOUSE {i+1}"))

    # --- Rubble fields / debris lanes ---
    for _ in range(14):
        rx = rng.uniform(-35.0, 35.0)
        ry = rng.uniform(-35.0, 35.0)
        if abs(rx) < 8.0 and abs(ry) < 8.0:
            continue
        s = rng.uniform(1.0, 3.0)
        h = rng.uniform(0.8, 2.8)
        boxes.append(_box(terrain, rx, ry, s, s * rng.uniform(0.6, 1.5), h, "rubble", RUBBLE))

    # Fallen comms mast (long thin box on its side)
    boxes.append(_box(terrain, 5.0, -8.0, 14.0, 0.8, 0.8, "fallen_mast", METAL, "MAST", z_lift=0.2))

    return boxes


def segment_blocked_by_structures(point_a, point_b, boxes):
    for box in boxes:
        if ray_hits_aabb(point_a, point_b, box):
            return True
    return False


def point_hits_structure(point, boxes, margin=0.15):
    for box in boxes:
        if box.contains_point(point, margin=margin):
            return True
    return False


def push_xy_out_of_structures(x, y, boxes, margin=0.6):
    """Nudge a ground agent out of solid footprints."""
    px, py = float(x), float(y)
    for _ in range(8):
        hit = None
        for box in boxes:
            if box.contains_xy(px, py, margin=margin):
                hit = box
                break
        if hit is None:
            return px, py
        cx = 0.5 * (hit.xmin + hit.xmax)
        cy = 0.5 * (hit.ymin + hit.ymax)
        dx = px - cx
        dy = py - cy
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            dx = 1.0
        n = np.hypot(dx, dy)
        px += (dx / n) * 1.5
        py += (dy / n) * 1.5
    return px, py
