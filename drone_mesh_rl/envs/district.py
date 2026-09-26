"""
district.py — One plain 300 m block district on USGS terrain.

No roads, trees, cars, landmarks, or window detail. Seeded axis-aligned
boxes only, sitting on the existing heightmap (the DEM is not flattened).
"""

from __future__ import annotations

import math

import numpy as np
import pybullet as p

from .town import _sample_heightmap

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


class DistrictLayout:
    """One 300 m plain-box district on an existing Terrain."""

    def __init__(self, terrain, size=DISTRICT_SIZE, seed=DISTRICT_SEED):
        self.size = float(size)
        self.seed = int(seed)
        self.center = find_flattest_square(terrain, self.size)
        self.buildings = _place_buildings(terrain, self.center, self.size, self.seed)
        self.ground_z = _sample_heightmap(terrain, self.center[0], self.center[1])

    def counts(self):
        out = {"shop": 0, "midrise": 0, "tower": 0}
        for b in self.buildings:
            out[b["kind"]] += 1
        return out


def spawn_district_in_pybullet(client, district):
    """Visual-only boxes. Returns body ids (one per building)."""
    bodies = []
    for b in district.buildings:
        hx = 0.5 * (b["xmax"] - b["xmin"])
        hy = 0.5 * (b["ymax"] - b["ymin"])
        hz = max(0.5 * (b["zmax"] - b["zmin"]), 0.05)
        vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[hx, hy, hz],
            rgbaColor=b["rgba"],
            physicsClientId=client,
        )
        bid = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=vis,
            basePosition=[b["cx"], b["cy"], 0.5 * (b["zmin"] + b["zmax"])],
            physicsClientId=client,
        )
        bodies.append(bid)
    return bodies


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
