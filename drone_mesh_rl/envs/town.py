"""
town.py — Seeded damaged-town layout: streets, buildings, rubble, overpasses.

Buildings and overpasses are axis-aligned boxes (AABB) so collision and
line-of-sight stay vectorized numpy — no PyBullet raycasts.
"""

from __future__ import annotations

import numpy as np

# Styles → RGBA for GUI only
BUILDING_COLORS = {
    "concrete": [0.55, 0.55, 0.52, 1.0],
    "brick": [0.55, 0.32, 0.25, 1.0],
    "charred": [0.22, 0.20, 0.18, 1.0],
    "rubble": [0.45, 0.40, 0.35, 1.0],
    "overpass": [0.40, 0.40, 0.42, 1.0],
}

ROAD_COLOR = [0.18, 0.18, 0.20, 1.0]


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
    """

    def __init__(self, size=250.0, seed=None, altitude_cap=40.0):
        self.size = float(size)
        self.altitude_cap = float(altitude_cap)
        self.rng = np.random.default_rng(seed)

        half = self.size / 2.0
        self.walkable_resolution = 2.0
        grid = int(self.size / self.walkable_resolution)
        self.walkable = np.ones((grid, grid), dtype=bool)

        self.boxes = np.zeros((0, 6), dtype=np.float64)
        self.styles: list[str] = []
        self.roads: list[tuple] = []
        self.overpass_indices: list[int] = []

        self._build_street_grid(half)
        self._place_buildings(half)
        self._place_rubble(half)
        self._place_overpasses(half)
        self._finalize_walkable(half)

    # ------------------------------------------------------------------
    # Layout generation
    # ------------------------------------------------------------------

    def _build_street_grid(self, half):
        """Rough orthogonal street lattice ~10m wide across the map."""
        street_width = 10.0
        # Street centerlines from near -half to +half
        spacing = 45.0
        centers = []
        c = -half + street_width / 2 + 5.0
        while c < half - street_width / 2 - 5.0:
            centers.append(c)
            c += spacing
        # Always include a near-center corridor
        if not any(abs(x) < 8.0 for x in centers):
            centers.append(0.0)
            centers.sort()

        self.street_centers_x = centers
        self.street_centers_y = list(centers)
        self.street_width = street_width

        # Road visual slabs (no collision)
        z_road = 0.05
        for cx in self.street_centers_x:
            self.roads.append((cx, 0.0, street_width / 2, half, z_road))
        for cy in self.street_centers_y:
            self.roads.append((0.0, cy, half, street_width / 2, z_road))

    def _block_rects(self, half):
        """Yield (xmin, xmax, ymin, ymax) for building blocks between streets."""
        xs = [-half] + [c + self.street_width / 2 for c in self.street_centers_x]
        xs_end = [c - self.street_width / 2 for c in self.street_centers_x] + [half]
        ys = [-half] + [c + self.street_width / 2 for c in self.street_centers_y]
        ys_end = [c - self.street_width / 2 for c in self.street_centers_y] + [half]

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

                # Avoid overlapping footprints inside the block
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
                # Damaged / shorter buildings more often for charred
                if style == "charred":
                    height = float(self.rng.uniform(3.0, 12.0))
                else:
                    roll = self.rng.random()
                    if roll < 0.08:
                        # Tower taller than altitude cap
                        height = float(self.rng.uniform(self.altitude_cap + 5.0, self.altitude_cap + 25.0))
                    elif roll < 0.35:
                        height = float(self.rng.uniform(12.0, 28.0))
                    else:
                        height = float(self.rng.uniform(4.0, 12.0))

                self._add_box(xmin, ymin, 0.0, xmax, ymax, height, style)
                placed.append((xmin, xmax, ymin, ymax))

        # If under target, sprinkle extra mid-block fillers
        attempts = 0
        while len(self.styles) < target_min and attempts < 400:
            attempts += 1
            fw = float(self.rng.uniform(8.0, 14.0))
            fd = float(self.rng.uniform(8.0, 14.0))
            cx = float(self.rng.uniform(-half + 15, half - 15))
            cy = float(self.rng.uniform(-half + 15, half - 15))
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
        """Low rubble piles as short AABBs near streets / lots."""
        n_rubble = int(self.rng.integers(12, 25))
        for _ in range(n_rubble):
            if len(self.styles) >= 120:
                break
            fw = float(self.rng.uniform(3.0, 8.0))
            fd = float(self.rng.uniform(3.0, 8.0))
            # Prefer near street edges
            if self.rng.random() < 0.6 and self.street_centers_x:
                if self.rng.random() < 0.5:
                    cx = float(self.rng.choice(self.street_centers_x)) + float(
                        self.rng.choice([-1, 1])
                    ) * (self.street_width / 2 + fw / 2 + 1.0)
                    cy = float(self.rng.uniform(-half + 10, half - 10))
                else:
                    cy = float(self.rng.choice(self.street_centers_y)) + float(
                        self.rng.choice([-1, 1])
                    ) * (self.street_width / 2 + fd / 2 + 1.0)
                    cx = float(self.rng.uniform(-half + 10, half - 10))
            else:
                cx = float(self.rng.uniform(-half + 10, half - 10))
                cy = float(self.rng.uniform(-half + 10, half - 10))

            xmin, xmax = cx - fw / 2, cx + fw / 2
            ymin, ymax = cy - fd / 2, cy + fd / 2
            if self._footprint_overlaps(xmin, xmax, ymin, ymax, pad=0.5):
                continue
            hz = float(self.rng.uniform(1.0, 3.5))
            self._add_box(xmin, ymin, 0.0, xmax, ymax, hz, "rubble")

    def _place_overpasses(self, half):
        """Elevated slabs spanning streets — fly under low, hit high."""
        n_over = int(self.rng.integers(3, 6))
        placed = 0
        attempts = 0
        while placed < n_over and attempts < 80:
            attempts += 1
            # Pick a street crossing and span across the street
            if self.rng.random() < 0.5:
                # Span in Y across an X-aligned street
                sx = float(self.rng.choice(self.street_centers_x))
                cy = float(self.rng.uniform(-half + 30, half - 30))
                span = float(self.rng.uniform(14.0, 22.0))
                width = float(self.rng.uniform(8.0, 14.0))
                xmin = sx - span / 2
                xmax = sx + span / 2
                ymin = cy - width / 2
                ymax = cy + width / 2
            else:
                sy = float(self.rng.choice(self.street_centers_y))
                cx = float(self.rng.uniform(-half + 30, half - 30))
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
        # Only ground-touching boxes block footprints (overpasses have zmin > 0)
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
        # Start: streets walkable, lots walkable; buildings carve out
        xs = (np.arange(gx) + 0.5) * res - half
        ys = (np.arange(gy) + 0.5) * res - half
        XX, YY = np.meshgrid(xs, ys)

        street = np.zeros_like(self.walkable, dtype=bool)
        half_w = self.street_width / 2
        for sx in self.street_centers_x:
            street |= np.abs(XX - sx) <= half_w
        for sy in self.street_centers_y:
            street |= np.abs(YY - sy) <= half_w

        # Lots = not streets; both walkable initially
        self.walkable[:] = True

        # Carve building footprints (ground-level boxes only)
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
            # Overpasses still show as footprint on radar
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
        # Closest point on each box to the sphere center
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
        # (N, B, 3)
        mins = boxes[:, 0:3]
        maxs = boxes[:, 3:6]
        # Broadcast: positions[:, None, :] vs mins[None, :, :]
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
