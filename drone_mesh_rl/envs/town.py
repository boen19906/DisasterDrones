"""
town.py — Seeded damaged-town layout: streets, buildings, rubble, overpasses,
plus a green wilderness ring (trees, cabins, rocks) outside the city core.

Buildings and overpasses are axis-aligned boxes (AABB) so collision and
line-of-sight stay vectorized numpy — no PyBullet raycasts.
"""

from __future__ import annotations

import numpy as np
import pybullet as p

# Map sizing: keep the 250 m damaged city; add ~100 heightmap cells of green per side.
RESOLUTION = 2.0
CITY_SIZE = 250.0
GREEN_MARGIN_CELLS = 100
GREEN_MARGIN_M = GREEN_MARGIN_CELLS * RESOLUTION  # 200 m
DEFAULT_ENV_SIZE = CITY_SIZE + 2.0 * GREEN_MARGIN_M  # 650 m
DEFAULT_CITY_HALF = CITY_SIZE / 2.0  # 125 m

# Styles → RGBA for GUI only
BUILDING_COLORS = {
    "concrete": [0.55, 0.55, 0.52, 1.0],
    "brick": [0.55, 0.32, 0.25, 1.0],
    "charred": [0.22, 0.20, 0.18, 1.0],
    "rubble": [0.45, 0.40, 0.35, 1.0],
    "overpass": [0.40, 0.40, 0.42, 1.0],
}

ROAD_COLOR = [0.18, 0.18, 0.20, 1.0]
CITY_GROUND_COLOR = [0.45, 0.38, 0.28, 1.0]
WILD_GROUND_COLOR = [0.28, 0.48, 0.22, 1.0]

# Tree canopy / trunk palettes (several "species")
TREE_SPECIES = [
    {"canopy": [0.15, 0.42, 0.18, 1.0], "trunk": [0.40, 0.26, 0.14, 1.0], "bare": False},
    {"canopy": [0.22, 0.50, 0.20, 1.0], "trunk": [0.38, 0.24, 0.12, 1.0], "bare": False},
    {"canopy": [0.45, 0.55, 0.18, 1.0], "trunk": [0.42, 0.28, 0.15, 1.0], "bare": False},
    {"canopy": [0.12, 0.32, 0.14, 1.0], "trunk": [0.35, 0.22, 0.12, 1.0], "bare": False},
    {"canopy": [0.45, 0.32, 0.18, 1.0], "trunk": [0.32, 0.22, 0.14, 1.0], "bare": True},
]

CABIN_WOOD = [0.48, 0.32, 0.18, 1.0]
CABIN_ROOF = [0.35, 0.22, 0.12, 1.0]
ROCK_COLOR = [0.50, 0.48, 0.44, 1.0]
LOG_COLOR = [0.38, 0.26, 0.14, 1.0]


class TownLayout:
    """
    One seeded damaged-town layout for a square map.

    Attributes
    ----------
    boxes : np.ndarray, shape (N, 6)
        Axis-aligned boxes as [xmin, ymin, zmin, xmax, ymax, zmax].
    styles : list[str]
        Style key per box (for coloring).
    roads : list[tuple]
        Non-colliding road slabs as (cx, cy, half_x, half_y, z).
    walkable : np.ndarray, bool, shape (grid, grid)
        True where survivors may spawn / drift (streets and open lots).
    walkable_resolution : float
        Meters per walkable cell.
    city_half : float
        Half-extent of the built-up city core (wilderness outside).
    trees, cabins, rocks : list[dict]
        Wilderness decoration descriptors (visual props).
    """

    def __init__(self, size=None, seed=None, altitude_cap=40.0, city_half=None):
        if size is None:
            size = DEFAULT_ENV_SIZE
        self.size = float(size)
        self.altitude_cap = float(altitude_cap)
        self.rng = np.random.default_rng(seed)

        half = self.size / 2.0
        # Keep the original 250 m city when the map is large enough for the green rim.
        if city_half is None:
            if half >= DEFAULT_CITY_HALF + GREEN_MARGIN_M * 0.9:
                city_half = DEFAULT_CITY_HALF
            else:
                city_half = max(40.0, half - max(20.0, half * 0.15))
        self.city_half = float(min(city_half, half - 15.0))

        self.walkable_resolution = RESOLUTION
        grid = int(self.size / self.walkable_resolution)
        self.walkable = np.ones((grid, grid), dtype=bool)

        self.boxes = np.zeros((0, 6), dtype=np.float64)
        self.styles: list[str] = []
        self.roads: list[tuple] = []
        self.overpass_indices: list[int] = []
        self.trees: list[dict] = []
        self.cabins: list[dict] = []
        self.rocks: list[dict] = []
        self._ground_applied = False

        self._build_street_grid(half)
        self._place_buildings(half)
        self._place_rubble(half)
        self._place_overpasses(half)
        self._place_wilderness(half)
        self._finalize_walkable(half)

    # ------------------------------------------------------------------
    # Layout generation
    # ------------------------------------------------------------------

    def _build_street_grid(self, half):
        """Orthogonal street lattice inside the city core only (~10 m wide)."""
        street_width = 10.0
        spacing = 45.0
        centers = []
        ch = self.city_half
        c = -ch + street_width / 2 + 5.0
        while c < ch - street_width / 2 - 5.0:
            centers.append(c)
            c += spacing
        # Always include a near-center corridor
        if not any(abs(x) < 8.0 for x in centers):
            centers.append(0.0)
            centers.sort()

        self.street_centers_x = centers
        self.street_centers_y = list(centers)
        self.street_width = street_width

        # Road visual slabs span the city core only (not the wilderness rim)
        z_road = 0.05
        for cx in self.street_centers_x:
            self.roads.append((cx, 0.0, street_width / 2, ch, z_road))
        for cy in self.street_centers_y:
            self.roads.append((0.0, cy, ch, street_width / 2, z_road))

    def _block_rects(self, half):
        """Yield (xmin, xmax, ymin, ymax) for building blocks between streets."""
        ch = self.city_half
        xs = [-ch] + [c + self.street_width / 2 for c in self.street_centers_x]
        xs_end = [c - self.street_width / 2 for c in self.street_centers_x] + [ch]
        ys = [-ch] + [c + self.street_width / 2 for c in self.street_centers_y]
        ys_end = [c - self.street_width / 2 for c in self.street_centers_y] + [ch]

        for i in range(len(xs)):
            for j in range(len(ys)):
                x0, x1 = xs[i], xs_end[i]
                y0, y1 = ys[j], ys_end[j]
                if (x1 - x0) < 12.0 or (y1 - y0) < 12.0:
                    continue
                yield x0, x1, y0, y1

    def _add_box(self, xmin, ymin, zmin, xmax, ymax, zmax, style):
        row = np.array([[xmin, ymin, zmin, xmax, ymax, zmax]], dtype=np.float64)
        self.boxes = row if self.boxes.size == 0 else np.vstack([self.boxes, row])
        self.styles.append(style)
        return len(self.styles) - 1

    def _place_buildings(self, half):
        """2–4 buildings per block; footprints 8–18m; mix of heights/styles."""
        style_weights = ["concrete", "concrete", "brick", "charred", "concrete"]
        target_min, target_max = 80, 120
        blocks = list(self._block_rects(half))
        self.rng.shuffle(blocks)

        for x0, x1, y0, y1 in blocks:
            if len(self.styles) >= target_max:
                break
            n = int(self.rng.integers(2, 5))
            margin = 1.5
            usable_w = (x1 - x0) - 2 * margin
            usable_d = (y1 - y0) - 2 * margin
            if usable_w < 8.0 or usable_d < 8.0:
                continue

            placed = []
            for _ in range(n):
                if len(self.styles) >= target_max:
                    break
                fw = float(self.rng.uniform(8.0, min(18.0, usable_w)))
                fd = float(self.rng.uniform(8.0, min(18.0, usable_d)))
                cx = float(self.rng.uniform(x0 + margin + fw / 2, x1 - margin - fw / 2))
                cy = float(self.rng.uniform(y0 + margin + fd / 2, y1 - margin - fd / 2))

                xmin, xmax = cx - fw / 2, cx + fw / 2
                ymin, ymax = cy - fd / 2, cy + fd / 2
                overlap = False
                for px0, px1, py0, py1 in placed:
                    if not (xmax <= px0 or xmin >= px1 or ymax <= py0 or ymin >= py1):
                        overlap = True
                        break
                if overlap:
                    continue

                style = style_weights[int(self.rng.integers(0, len(style_weights)))]
                if style == "charred":
                    height = float(self.rng.uniform(3.0, 12.0))
                else:
                    roll = self.rng.random()
                    if roll < 0.08:
                        height = float(
                            self.rng.uniform(self.altitude_cap + 5.0, self.altitude_cap + 25.0)
                        )
                    elif roll < 0.35:
                        height = float(self.rng.uniform(12.0, 28.0))
                    else:
                        height = float(self.rng.uniform(4.0, 12.0))

                self._add_box(xmin, ymin, 0.0, xmax, ymax, height, style)
                placed.append((xmin, xmax, ymin, ymax))

        # If under target, sprinkle extra mid-block fillers inside the city
        ch = self.city_half
        attempts = 0
        while len(self.styles) < target_min and attempts < 400:
            attempts += 1
            fw = float(self.rng.uniform(8.0, 14.0))
            fd = float(self.rng.uniform(8.0, 14.0))
            cx = float(self.rng.uniform(-ch + 15, ch - 15))
            cy = float(self.rng.uniform(-ch + 15, ch - 15))
            if self._on_street(cx, cy):
                continue
            xmin, xmax = cx - fw / 2, cx + fw / 2
            ymin, ymax = cy - fd / 2, cy + fd / 2
            if self._footprint_overlaps(xmin, xmax, ymin, ymax):
                continue
            height = float(self.rng.uniform(4.0, 18.0))
            style = style_weights[int(self.rng.integers(0, len(style_weights)))]
            self._add_box(xmin, ymin, 0.0, xmax, ymax, height, style)

    def _place_rubble(self, half):
        """Low rubble piles as short AABBs near streets / lots (city only)."""
        ch = self.city_half
        n_rubble = int(self.rng.integers(12, 25))
        for _ in range(n_rubble):
            if len(self.styles) >= 120:
                break
            fw = float(self.rng.uniform(3.0, 8.0))
            fd = float(self.rng.uniform(3.0, 8.0))
            if self.rng.random() < 0.6 and self.street_centers_x:
                if self.rng.random() < 0.5:
                    cx = float(self.rng.choice(self.street_centers_x)) + float(
                        self.rng.choice([-1, 1])
                    ) * (self.street_width / 2 + fw / 2 + 1.0)
                    cy = float(self.rng.uniform(-ch + 10, ch - 10))
                else:
                    cy = float(self.rng.choice(self.street_centers_y)) + float(
                        self.rng.choice([-1, 1])
                    ) * (self.street_width / 2 + fd / 2 + 1.0)
                    cx = float(self.rng.uniform(-ch + 10, ch - 10))
            else:
                cx = float(self.rng.uniform(-ch + 10, ch - 10))
                cy = float(self.rng.uniform(-ch + 10, ch - 10))

            if max(abs(cx), abs(cy)) > ch - 2.0:
                continue
            xmin, xmax = cx - fw / 2, cx + fw / 2
            ymin, ymax = cy - fd / 2, cy + fd / 2
            if self._footprint_overlaps(xmin, xmax, ymin, ymax, pad=0.5):
                continue
            hz = float(self.rng.uniform(1.0, 3.5))
            self._add_box(xmin, ymin, 0.0, xmax, ymax, hz, "rubble")

    def _place_overpasses(self, half):
        """Elevated slabs spanning streets — fly under low, hit high."""
        ch = self.city_half
        n_over = int(self.rng.integers(3, 6))
        placed = 0
        attempts = 0
        while placed < n_over and attempts < 80:
            attempts += 1
            if self.rng.random() < 0.5:
                sx = float(self.rng.choice(self.street_centers_x))
                cy = float(self.rng.uniform(-ch + 30, ch - 30))
                span = float(self.rng.uniform(14.0, 22.0))
                width = float(self.rng.uniform(8.0, 14.0))
                xmin = sx - span / 2
                xmax = sx + span / 2
                ymin = cy - width / 2
                ymax = cy + width / 2
            else:
                sy = float(self.rng.choice(self.street_centers_y))
                cx = float(self.rng.uniform(-ch + 30, ch - 30))
                span = float(self.rng.uniform(14.0, 22.0))
                width = float(self.rng.uniform(8.0, 14.0))
                xmin = cx - width / 2
                xmax = cx + width / 2
                ymin = sy - span / 2
                ymax = sy + span / 2

            clearance = float(self.rng.uniform(6.0, 12.0))
            thickness = float(self.rng.uniform(1.5, 3.0))
            zmin = clearance
            zmax = clearance + thickness
            idx = self._add_box(xmin, ymin, zmin, xmax, ymax, zmax, "overpass")
            self.overpass_indices.append(idx)
            placed += 1

    def _in_wilderness(self, x, y, margin=6.0):
        """True if (x, y) is outside the city core (with margin)."""
        return max(abs(x), abs(y)) >= self.city_half + margin

    def _wilderness_footprint_clear(self, xmin, xmax, ymin, ymax, pad=2.0):
        """Avoid overlapping other wilderness props (cabins / rocks / tree trunks)."""
        for c in self.cabins:
            if not (
                xmax + pad < c["xmin"]
                or xmin - pad > c["xmax"]
                or ymax + pad < c["ymin"]
                or ymin - pad > c["ymax"]
            ):
                return False
        for r in self.rocks:
            if not (
                xmax + pad < r["xmin"]
                or xmin - pad > r["xmax"]
                or ymax + pad < r["ymin"]
                or ymin - pad > r["ymax"]
            ):
                return False
        for t in self.trees:
            tx, ty = t["x"], t["y"]
            tr = t.get("clear_r", 2.5)
            if (xmin - tr <= tx <= xmax + tr) and (ymin - tr <= ty <= ymax + tr):
                return False
        return True

    def _sample_wilderness_xy(self, half, margin=8.0):
        """Sample a point in the outer ring (not in the city square)."""
        ch = self.city_half + margin
        for _ in range(40):
            x = float(self.rng.uniform(-half + 4.0, half - 4.0))
            y = float(self.rng.uniform(-half + 4.0, half - 4.0))
            if max(abs(x), abs(y)) >= ch:
                return x, y
        edge = int(self.rng.integers(0, 4))
        t = float(self.rng.uniform(-half + 6.0, half - 6.0))
        d = float(self.rng.uniform(ch + 2.0, half - 5.0))
        if edge == 0:
            return d, t
        if edge == 1:
            return -d, t
        if edge == 2:
            return t, d
        return t, -d

    def _place_wilderness(self, half):
        """Scatter trees, cabins, and a few rocks/logs in the outer ring only."""
        # Modest counts for the larger rim — enough presence, not 10x props.
        n_cabins = int(self.rng.integers(10, 17))
        attempts = 0
        while len(self.cabins) < n_cabins and attempts < 400:
            attempts += 1
            cx, cy = self._sample_wilderness_xy(half, margin=12.0)
            fw = float(self.rng.uniform(4.0, 7.5))
            fd = float(self.rng.uniform(3.5, 6.5))
            xmin, xmax = cx - fw / 2, cx + fw / 2
            ymin, ymax = cy - fd / 2, cy + fd / 2
            if not self._in_wilderness(cx, cy, margin=10.0):
                continue
            if not self._wilderness_footprint_clear(xmin, xmax, ymin, ymax, pad=8.0):
                continue
            wall_h = float(self.rng.uniform(2.2, 3.2))
            roof_h = float(self.rng.uniform(1.2, 2.0))
            pitched = bool(self.rng.random() < 0.65)
            self.cabins.append(
                {
                    "x": cx,
                    "y": cy,
                    "xmin": xmin,
                    "xmax": xmax,
                    "ymin": ymin,
                    "ymax": ymax,
                    "fw": fw,
                    "fd": fd,
                    "wall_h": wall_h,
                    "roof_h": roof_h,
                    "pitched": pitched,
                }
            )

        n_trees = int(self.rng.integers(70, 111))
        attempts = 0
        while len(self.trees) < n_trees and attempts < 2000:
            attempts += 1
            x, y = self._sample_wilderness_xy(half, margin=8.0)
            if not self._in_wilderness(x, y, margin=6.0):
                continue
            too_close = False
            for c in self.cabins:
                if (c["xmin"] - 3.0 <= x <= c["xmax"] + 3.0) and (
                    c["ymin"] - 3.0 <= y <= c["ymax"] + 3.0
                ):
                    too_close = True
                    break
            if too_close:
                continue
            for t in self.trees:
                if (x - t["x"]) ** 2 + (y - t["y"]) ** 2 < 5.5**2:
                    too_close = True
                    break
            if too_close:
                continue

            species = TREE_SPECIES[int(self.rng.integers(0, len(TREE_SPECIES)))]
            if species["bare"] and self.rng.random() < 0.55:
                species = TREE_SPECIES[int(self.rng.integers(0, 4))]

            scale = float(self.rng.uniform(0.7, 1.55))
            trunk_r = 0.18 * scale
            trunk_h = float(self.rng.uniform(1.8, 4.5)) * scale
            canopy_kind = "cone" if self.rng.random() < 0.55 else "box"
            canopy_r = float(self.rng.uniform(1.2, 2.8)) * scale
            canopy_h = float(self.rng.uniform(2.0, 4.5)) * scale
            self.trees.append(
                {
                    "x": x,
                    "y": y,
                    "trunk_r": trunk_r,
                    "trunk_h": trunk_h,
                    "canopy_kind": canopy_kind,
                    "canopy_r": canopy_r,
                    "canopy_h": canopy_h,
                    "canopy_rgba": list(species["canopy"]),
                    "trunk_rgba": list(species["trunk"]),
                    "bare": bool(species["bare"]),
                    "clear_r": max(2.0, canopy_r * 0.7),
                }
            )

        n_rocks = int(self.rng.integers(6, 12))
        attempts = 0
        while len(self.rocks) < n_rocks and attempts < 250:
            attempts += 1
            cx, cy = self._sample_wilderness_xy(half, margin=10.0)
            kind = "rock" if self.rng.random() < 0.6 else "log"
            if kind == "rock":
                fw = float(self.rng.uniform(1.5, 4.0))
                fd = float(self.rng.uniform(1.2, 3.5))
                hz = float(self.rng.uniform(0.8, 2.4))
            else:
                fw = float(self.rng.uniform(3.0, 6.0))
                fd = float(self.rng.uniform(0.35, 0.7))
                hz = float(self.rng.uniform(0.35, 0.7))
            xmin, xmax = cx - fw / 2, cx + fw / 2
            ymin, ymax = cy - fd / 2, cy + fd / 2
            if not self._wilderness_footprint_clear(xmin, xmax, ymin, ymax, pad=2.5):
                continue
            self.rocks.append(
                {
                    "kind": kind,
                    "x": cx,
                    "y": cy,
                    "xmin": xmin,
                    "xmax": xmax,
                    "ymin": ymin,
                    "ymax": ymax,
                    "fw": fw,
                    "fd": fd,
                    "hz": hz,
                    "yaw": float(self.rng.uniform(0.0, 6.28)),
                }
            )

    def apply_ground_heights(self, terrain):
        """
        Shift building / overpass AABB z so bases follow the heightmap.

        Call after Terrain is built. Relative heights (building tallness,
        overpass clearance) are preserved. Idempotent.
        """
        if self._ground_applied or self.boxes.size == 0:
            return
        new_boxes = self.boxes.copy()
        for i, box in enumerate(self.boxes):
            xmin, ymin, zmin, xmax, ymax, zmax = box
            cx = 0.5 * (xmin + xmax)
            cy = 0.5 * (ymin + ymax)
            gz = terrain.height_at(cx, cy)
            if self.styles[i] == "overpass":
                clearance = zmin
                thickness = zmax - zmin
                new_boxes[i, 2] = gz + clearance
                new_boxes[i, 5] = gz + clearance + thickness
            else:
                height = zmax - zmin
                new_boxes[i, 2] = gz
                new_boxes[i, 5] = gz + height
        self.boxes = new_boxes

        lifted = []
        for cx, cy, hx, hy, _z in self.roads:
            gz = terrain.height_at(cx, cy)
            lifted.append((cx, cy, hx, hy, gz + 0.06))
        self.roads = lifted
        self._ground_applied = True

    def _on_street(self, x, y, pad=0.0):
        half_w = self.street_width / 2 + pad
        for sx in self.street_centers_x:
            if abs(x - sx) <= half_w:
                return True
        for sy in self.street_centers_y:
            if abs(y - sy) <= half_w:
                return True
        return False

    def _footprint_overlaps(self, xmin, xmax, ymin, ymax, pad=0.0):
        if self.boxes.size == 0:
            return False
        ground = self.boxes[self.boxes[:, 2] < 0.5]
        if ground.size == 0:
            return False
        return bool(
            np.any(
                (xmax + pad > ground[:, 0])
                & (xmin - pad < ground[:, 3])
                & (ymax + pad > ground[:, 1])
                & (ymin - pad < ground[:, 4])
            )
        )

    def _finalize_walkable(self, half):
        """Mark streets and open lots walkable; building footprints blocked."""
        res = self.walkable_resolution
        gy, gx = self.walkable.shape
        xs = (np.arange(gx) + 0.5) * res - half
        ys = (np.arange(gy) + 0.5) * res - half
        XX, YY = np.meshgrid(xs, ys)

        street = np.zeros_like(self.walkable, dtype=bool)
        half_w = self.street_width / 2
        for sx in self.street_centers_x:
            street |= np.abs(XX - sx) <= half_w
        for sy in self.street_centers_y:
            street |= np.abs(YY - sy) <= half_w

        # Streets only count inside the city core
        ch = self.city_half
        street &= (np.abs(XX) <= ch) & (np.abs(YY) <= ch)

        self.walkable[:] = True

        if self.boxes.size:
            ground = self.boxes[self.boxes[:, 2] < 0.5]
            for b in ground:
                xmin, ymin, _, xmax, ymax, _ = b
                mask = (XX >= xmin) & (XX <= xmax) & (YY >= ymin) & (YY <= ymax)
                self.walkable[mask] = False

        self.street_mask = street

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def is_walkable(self, x, y):
        """Return True if (x, y) is on a street or open lot (not in a building)."""
        half = self.size / 2.0
        res = self.walkable_resolution
        gx = int((x + half) / res)
        gy = int((y + half) / res)
        if gx < 0 or gy < 0 or gx >= self.walkable.shape[1] or gy >= self.walkable.shape[0]:
            return False
        return bool(self.walkable[gy, gx])

    def sample_walkable(self, rng, n=1, prefer_street=True):
        """Sample n walkable (x, y) positions."""
        if prefer_street:
            mask = self.walkable & self.street_mask
            if not mask.any():
                mask = self.walkable
        else:
            mask = self.walkable
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return np.zeros((n, 2))
        idx = rng.choice(len(xs), size=n, replace=True)
        half = self.size / 2.0
        res = self.walkable_resolution
        pts = np.column_stack(
            [
                xs[idx] * res - half + res * 0.5,
                ys[idx] * res - half + res * 0.5,
            ]
        )
        return pts

    def footprints_2d(self):
        """Return ground footprints for radar: list of {xmin,ymin,xmax,ymax,style}."""
        out = []
        for i, b in enumerate(self.boxes):
            out.append(
                {
                    "xmin": round(float(b[0]), 2),
                    "ymin": round(float(b[1]), 2),
                    "xmax": round(float(b[3]), 2),
                    "ymax": round(float(b[4]), 2),
                    "style": self.styles[i],
                    "zmin": round(float(b[2]), 2),
                    "zmax": round(float(b[5]), 2),
                }
            )
        return out

    def building_count(self):
        """Count concrete/brick/charred buildings (excludes rubble + overpass)."""
        return sum(1 for s in self.styles if s in ("concrete", "brick", "charred"))

    # ------------------------------------------------------------------
    # Vectorized collision / LoS (no PyBullet)
    # ------------------------------------------------------------------

    @staticmethod
    def sphere_hits_boxes(pos, radius, boxes):
        """
        True if sphere at pos with given radius intersects any AABB.

        Vectorized closest-point test.
        """
        if boxes is None or len(boxes) == 0:
            return False
        pos = np.asarray(pos, dtype=np.float64)
        closest = np.column_stack(
            [
                np.clip(pos[0], boxes[:, 0], boxes[:, 3]),
                np.clip(pos[1], boxes[:, 1], boxes[:, 4]),
                np.clip(pos[2], boxes[:, 2], boxes[:, 5]),
            ]
        )
        d2 = np.sum((closest - pos) ** 2, axis=1)
        return bool(np.any(d2 <= radius * radius))

    @staticmethod
    def sphere_hit_mask(positions, radius, boxes):
        """Boolean mask of which sphere centers intersect any box."""
        positions = np.asarray(positions, dtype=np.float64)
        n = len(positions)
        if boxes is None or len(boxes) == 0 or n == 0:
            return np.zeros(n, dtype=bool)
        mins = boxes[:, 0:3]
        maxs = boxes[:, 3:6]
        closest = np.clip(positions[:, None, :], mins[None, :, :], maxs[None, :, :])
        d2 = np.sum((closest - positions[:, None, :]) ** 2, axis=2)
        return np.any(d2 <= radius * radius, axis=1)

    @staticmethod
    def segment_hits_boxes(point_a, point_b, boxes):
        """
        True if the 3D segment AB intersects any AABB.

        Uses the slab method (Kay–Kajiya) vectorized over boxes.
        """
        if boxes is None or len(boxes) == 0:
            return False
        a = np.asarray(point_a, dtype=np.float64).reshape(3)
        b = np.asarray(point_b, dtype=np.float64).reshape(3)
        d = b - a
        if np.allclose(d, 0.0):
            return TownLayout.sphere_hits_boxes(a, 1e-6, boxes)

        n = len(boxes)
        tmin = np.zeros(n, dtype=np.float64)
        tmax = np.ones(n, dtype=np.float64)
        alive = np.ones(n, dtype=bool)

        for axis in range(3):
            amin = boxes[:, axis]
            amax = boxes[:, axis + 3]
            da = d[axis]
            parallel = np.abs(da) < 1e-12
            outside = parallel & ((a[axis] < amin) | (a[axis] > amax))
            alive &= ~outside

            inv = np.zeros(n, dtype=np.float64)
            if not np.all(parallel):
                inv[~parallel] = 1.0 / da
            t1 = (amin - a[axis]) * inv
            t2 = (amax - a[axis]) * inv
            t_enter = np.minimum(t1, t2)
            t_exit = np.maximum(t1, t2)

            tmin = np.where(parallel, tmin, np.maximum(tmin, t_enter))
            tmax = np.where(parallel, tmax, np.minimum(tmax, t_exit))
            alive &= tmin <= tmax
            if not np.any(alive):
                return False

        return bool(np.any(alive & (tmax >= 0.0) & (tmin <= 1.0)))

    @staticmethod
    def segment_clear(point_a, point_b, boxes):
        """True if segment does not hit any box."""
        return not TownLayout.segment_hits_boxes(point_a, point_b, boxes)


def connect_pybullet(gui=True, shadows=False):
    """Open a PyBullet client. Shadows default off for a lighter GUI."""
    client = p.connect(p.GUI if gui else p.DIRECT)
    if gui:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=client)
        p.configureDebugVisualizer(
            p.COV_ENABLE_SHADOWS, 1 if shadows else 0, physicsClientId=client
        )
        p.configureDebugVisualizer(
            p.COV_ENABLE_KEYBOARD_SHORTCUTS, 0, physicsClientId=client
        )
        p.configureDebugVisualizer(
            p.COV_ENABLE_WIREFRAME, 0, physicsClientId=client
        )
    p.setGravity(0, 0, -9.81, physicsClientId=client)
    return client


def _spawn_wilderness_ground(client, town, terrain, env_size):
    """
    Green visual patches over the wilderness rim.

    Heightfield is tinted green; coarse pads follow local hill height so the
    rim stays grassy without drowning the viewer in tiny bodies. Tile size
    scales with the green band so a 650 m map stays light.
    """
    half = float(env_size) / 2.0
    ch = town.city_half
    rim = half - ch
    # Larger tiles on the big rim (~200 m) keep body count modest
    tile = float(np.clip(rim / 5.0, 28.0, 48.0))
    bodies = []
    xs = np.arange(-half + tile / 2, half, tile)
    ys = np.arange(-half + tile / 2, half, tile)
    for cx in xs:
        for cy in ys:
            if max(abs(cx), abs(cy)) < ch + 2.0:
                continue
            if max(abs(cx), abs(cy)) + tile / 2 < ch:
                continue
            gz = terrain.height_at(cx, cy)
            hx = hy = tile / 2 - 0.2
            vis = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[hx, hy, 0.05],
                rgbaColor=WILD_GROUND_COLOR,
                physicsClientId=client,
            )
            bid = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=vis,
                basePosition=[float(cx), float(cy), gz + 0.04],
                physicsClientId=client,
            )
            bodies.append(bid)
    return bodies


def _spawn_trees(client, town, terrain):
    bodies = []
    for t in town.trees:
        gz = terrain.height_at(t["x"], t["y"])
        trunk_vis = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=t["trunk_r"],
            length=t["trunk_h"],
            rgbaColor=t["trunk_rgba"],
            physicsClientId=client,
        )
        trunk_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=trunk_vis,
            basePosition=[t["x"], t["y"], gz + t["trunk_h"] / 2.0],
            physicsClientId=client,
        )
        bodies.append(trunk_id)

        if t["bare"]:
            canopy_h = t["canopy_h"] * 0.35
            canopy_r = t["canopy_r"] * 0.45
        else:
            canopy_h = t["canopy_h"]
            canopy_r = t["canopy_r"]

        canopy_z = gz + t["trunk_h"] + canopy_h * 0.35
        if t["canopy_kind"] == "cone" and not t["bare"]:
            canopy_vis = p.createVisualShape(
                p.GEOM_CYLINDER,
                radius=canopy_r,
                length=canopy_h,
                rgbaColor=t["canopy_rgba"],
                physicsClientId=client,
            )
        else:
            canopy_vis = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[canopy_r, canopy_r, canopy_h / 2.0],
                rgbaColor=t["canopy_rgba"],
                physicsClientId=client,
            )
            canopy_z = gz + t["trunk_h"] + canopy_h / 2.0

        canopy_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=canopy_vis,
            basePosition=[t["x"], t["y"], canopy_z],
            physicsClientId=client,
        )
        bodies.append(canopy_id)
    return bodies


def _spawn_cabins(client, town, terrain):
    bodies = []
    for c in town.cabins:
        gz = terrain.height_at(c["x"], c["y"])
        hx, hy = c["fw"] / 2.0, c["fd"] / 2.0
        wall_h = c["wall_h"]
        wall_vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[hx, hy, wall_h / 2.0],
            rgbaColor=CABIN_WOOD,
            physicsClientId=client,
        )
        wall_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=wall_vis,
            basePosition=[c["x"], c["y"], gz + wall_h / 2.0],
            physicsClientId=client,
        )
        bodies.append(wall_id)

        roof_h = c["roof_h"]
        if c["pitched"]:
            if c["fw"] >= c["fd"]:
                half_y = hy * 0.55
                for sign in (-1.0, 1.0):
                    roof_vis = p.createVisualShape(
                        p.GEOM_BOX,
                        halfExtents=[hx * 1.05, half_y, roof_h / 2.0],
                        rgbaColor=CABIN_ROOF,
                        physicsClientId=client,
                    )
                    rid = p.createMultiBody(
                        baseMass=0,
                        baseCollisionShapeIndex=-1,
                        baseVisualShapeIndex=roof_vis,
                        basePosition=[
                            c["x"],
                            c["y"] + sign * hy * 0.45,
                            gz + wall_h + roof_h * 0.35,
                        ],
                        baseOrientation=p.getQuaternionFromEuler(
                            [sign * 0.45, 0.0, 0.0]
                        ),
                        physicsClientId=client,
                    )
                    bodies.append(rid)
            else:
                half_x = hx * 0.55
                for sign in (-1.0, 1.0):
                    roof_vis = p.createVisualShape(
                        p.GEOM_BOX,
                        halfExtents=[half_x, hy * 1.05, roof_h / 2.0],
                        rgbaColor=CABIN_ROOF,
                        physicsClientId=client,
                    )
                    rid = p.createMultiBody(
                        baseMass=0,
                        baseCollisionShapeIndex=-1,
                        baseVisualShapeIndex=roof_vis,
                        basePosition=[
                            c["x"] + sign * hx * 0.45,
                            c["y"],
                            gz + wall_h + roof_h * 0.35,
                        ],
                        baseOrientation=p.getQuaternionFromEuler(
                            [0.0, -sign * 0.45, 0.0]
                        ),
                        physicsClientId=client,
                    )
                    bodies.append(rid)
        else:
            roof_vis = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[hx * 1.08, hy * 1.08, roof_h / 2.0],
                rgbaColor=CABIN_ROOF,
                physicsClientId=client,
            )
            rid = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=roof_vis,
                basePosition=[c["x"], c["y"], gz + wall_h + roof_h / 2.0],
                physicsClientId=client,
            )
            bodies.append(rid)
    return bodies


def _spawn_rocks(client, town, terrain):
    bodies = []
    for r in town.rocks:
        gz = terrain.height_at(r["x"], r["y"])
        color = ROCK_COLOR if r["kind"] == "rock" else LOG_COLOR
        vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[r["fw"] / 2.0, r["fd"] / 2.0, r["hz"] / 2.0],
            rgbaColor=color,
            physicsClientId=client,
        )
        orn = p.getQuaternionFromEuler([0.0, 0.0, r["yaw"]])
        bid = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=vis,
            basePosition=[r["x"], r["y"], gz + r["hz"] / 2.0],
            baseOrientation=orn,
            physicsClientId=client,
        )
        bodies.append(bid)
    return bodies


def spawn_town_in_pybullet(client, town, terrain, env_size):
    """
    Load green wilderness heightfield + dusty city plate + roads + buildings + props.

    Collision for buildings stays in numpy (AABB); the heightfield is the
    only PyBullet collision mesh. Returns
    (terrain_body, road_bodies, building_bodies, prop_bodies).
    """
    grid = terrain.grid_x
    size = float(env_size)

    town.apply_ground_heights(terrain)
    if hasattr(terrain, "set_obstacle_boxes"):
        terrain.set_obstacle_boxes(town.boxes)

    terrain_shape = p.createCollisionShape(
        p.GEOM_HEIGHTFIELD,
        meshScale=[size / grid, size / grid, 1.0],
        heightfieldData=terrain.heightmap.flatten().tolist(),
        numHeightfieldRows=grid,
        numHeightfieldColumns=grid,
        physicsClientId=client,
    )
    # PyBullet centers the heightfield AABB at the body origin; lift so
    # world Z matches heightmap values used by buildings / props.
    mid = terrain.heightfield_mid()
    terrain_body = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=terrain_shape,
        basePosition=[0.0, 0.0, mid],
        physicsClientId=client,
    )
    # Tint the full heightfield green; dusty city plate covers downtown.
    p.changeVisualShape(
        terrain_body,
        -1,
        rgbaColor=WILD_GROUND_COLOR,
        physicsClientId=client,
    )

    prop_bodies = []
    prop_bodies.extend(_spawn_wilderness_ground(client, town, terrain, env_size))

    ch = town.city_half
    city_z = terrain.height_at(0.0, 0.0) + 0.03
    city_vis = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[ch, ch, 0.035],
        rgbaColor=CITY_GROUND_COLOR,
        physicsClientId=client,
    )
    city_pad = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=city_vis,
        basePosition=[0.0, 0.0, city_z],
        physicsClientId=client,
    )
    prop_bodies.append(city_pad)

    road_bodies = []
    for cx, cy, hx, hy, z in town.roads:
        vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[hx, hy, 0.04],
            rgbaColor=ROAD_COLOR,
            physicsClientId=client,
        )
        bid = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=vis,
            basePosition=[cx, cy, z],
            physicsClientId=client,
        )
        road_bodies.append(bid)

    building_bodies = []
    for i, box in enumerate(town.boxes):
        xmin, ymin, zmin, xmax, ymax, zmax = box
        hx = (xmax - xmin) / 2.0
        hy = (ymax - ymin) / 2.0
        hz = (zmax - zmin) / 2.0
        cx = (xmin + xmax) / 2.0
        cy = (ymin + ymax) / 2.0
        cz = (zmin + zmax) / 2.0
        style = town.styles[i]
        color = BUILDING_COLORS.get(style, BUILDING_COLORS["concrete"])
        vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[hx, hy, hz],
            rgbaColor=color,
            physicsClientId=client,
        )
        bid = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=vis,
            basePosition=[cx, cy, cz],
            physicsClientId=client,
        )
        building_bodies.append(bid)

    prop_bodies.extend(_spawn_trees(client, town, terrain))
    prop_bodies.extend(_spawn_cabins(client, town, terrain))
    prop_bodies.extend(_spawn_rocks(client, town, terrain))

    return terrain_body, road_bodies, building_bodies, prop_bodies


def frame_town_camera(client, env_size):
    """Overview framing for the full map (pull back for the larger wilderness rim)."""
    cam_dist = max(280.0, float(env_size) * 0.95)
    p.resetDebugVisualizerCamera(
        cameraDistance=cam_dist,
        cameraYaw=45,
        cameraPitch=-42,
        cameraTargetPosition=[0, 0, 14.0],
        physicsClientId=client,
    )
