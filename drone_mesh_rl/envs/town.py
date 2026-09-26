"""
town.py — Seeded damaged-town layout: streets, buildings, rubble, overpasses.

Buildings and overpasses are axis-aligned boxes (AABB) so collision and
line-of-sight stay vectorized numpy — no PyBullet raycasts.

Downtown buildings are street-aligned typed volumes (shop, row, mid-rise,
tower, L-shape, setback) with cheap visual detail overlays.

People are visual-only: a capsule body and a sphere head, standing on
walkable ground. A small subset is flagged as survivors for later search
logic; they do not move and they do not affect collision or line-of-sight.
"""

from __future__ import annotations

import numpy as np
import pybullet as p

# Nominal story height for typed buildings
STORY = 3.5

# Styles → RGBA for GUI only
BUILDING_COLORS = {
    "concrete": [0.55, 0.55, 0.52, 1.0],
    "brick": [0.55, 0.32, 0.25, 1.0],
    "charred": [0.22, 0.20, 0.18, 1.0],
    "rubble": [0.45, 0.40, 0.35, 1.0],
    "overpass": [0.40, 0.40, 0.42, 1.0],
}

ROAD_COLOR = [0.18, 0.18, 0.20, 1.0]
# Pavement sits clearly above the heightfield so the two never share a plane.
ROAD_LIFT = 0.12  # meters above local (max) ground to slab center
ROAD_HALF_THICK = 0.035  # half-thickness → bottom ~0.085 m above peak in segment
ROAD_SEG_LEN = 5.0  # short segments follow the heightmap

BUILDING_TYPE_NAMES = ("shop", "row", "midrise", "tower", "lshape", "setback")

# Standing people. Capsule total height is BODY_LENGTH + 2*BODY_RADIUS.
# Head sinks slightly into the capsule so the figure reads as one body.
# Feet-to-crown is about 1.35 m; body width is 0.22 m.
HUMAN_COUNT = 70
HUMAN_SURVIVOR_COUNT = 9
HUMAN_BODY_RADIUS = 0.11
HUMAN_BODY_LENGTH = 0.95
HUMAN_HEAD_RADIUS = 0.105
HUMAN_HEAD_OVERLAP = 0.04
HUMAN_MIN_SEP = 1.5
HUMAN_EDGE_MARGIN = 3.0

# Muted clothing. Survivors wear a high-visibility vest so the subset is visible.
HUMAN_SHIRT_COLORS = (
    [0.22, 0.30, 0.45, 1.0],
    [0.38, 0.39, 0.41, 1.0],
    [0.34, 0.40, 0.30, 1.0],
    [0.45, 0.26, 0.24, 1.0],
    [0.48, 0.40, 0.30, 1.0],
    [0.28, 0.38, 0.42, 1.0],
)
HUMAN_VEST_COLORS = (
    [0.90, 0.48, 0.10, 1.0],
    [0.90, 0.78, 0.12, 1.0],
)
HUMAN_SKIN_COLORS = (
    [0.87, 0.72, 0.58, 1.0],
    [0.74, 0.56, 0.42, 1.0],
    [0.56, 0.40, 0.30, 1.0],
    [0.42, 0.30, 0.22, 1.0],
)


def _scale_rgb(rgba, factor):
    return [
        float(np.clip(rgba[0] * factor, 0.0, 1.0)),
        float(np.clip(rgba[1] * factor, 0.0, 1.0)),
        float(np.clip(rgba[2] * factor, 0.0, 1.0)),
        float(rgba[3]),
    ]


def _sample_heightmap(terrain, x, y):
    """Nearest heightmap sample; works without Terrain helpers."""
    half_x = terrain.size_x / 2.0
    half_y = terrain.size_y / 2.0
    j = int((x + half_x) / terrain.resolution)
    i = int((y + half_y) / terrain.resolution)
    j = int(np.clip(j, 0, terrain.grid_x - 1))
    i = int(np.clip(i, 0, terrain.grid_y - 1))
    return float(terrain.heightmap[i, j])


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
        # Logical downtown buildings (one entry per structure, may map to >1 AABB)
        self.building_types: list[str] = []
        self.buildings: list[dict] = []
        # Visual-only detail AABBs: (xmin, ymin, zmin, xmax, ymax, zmax, rgba)
        self.detail_parts: list[tuple] = []
        self._ground_applied = False
        # Visual people. Each dict: x, y, z (feet), shirt, skin, survivor.
        # human_rng is independent of the building RNG so layout edits do not
        # reshuffle the crowd for a given town seed.
        self.humans: list[dict] = []
        self._humans_placed = False
        self.human_rng = np.random.default_rng(None if seed is None else int(seed) + 17011)

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
        # Road visual segments are built in apply_ground_heights once terrain exists.
        self.roads = []

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

    def _add_detail(self, xmin, ymin, zmin, xmax, ymax, zmax, rgba):
        self.detail_parts.append(
            (
                float(xmin),
                float(ymin),
                float(zmin),
                float(xmax),
                float(ymax),
                float(zmax),
                [float(c) for c in rgba],
            )
        )

    @staticmethod
    def _rects_overlap(a, b, pad=0.0):
        ax0, ax1, ay0, ay1 = a
        bx0, bx1, by0, by1 = b
        return not (
            ax1 + pad <= bx0
            or ax0 - pad >= bx1
            or ay1 + pad <= by0
            or ay0 - pad >= by1
        )

    def _pick_style(self, damaged=False):
        if damaged:
            return "charred"
        return "brick" if self.rng.random() < 0.35 else "concrete"

    def _register_building(self, btype, footprints, box_indices, style, height):
        self.building_types.append(btype)
        self.buildings.append(
            {
                "type": btype,
                "footprints": [(float(a), float(b), float(c), float(d)) for a, b, c, d in footprints],
                "box_indices": list(box_indices),
                "style": style,
                "height": float(height),
            }
        )

    def _add_building_details(self, footprints, height, style, btype):
        """Cheap silhouette detail: darker ground floor, roof lip, inset tone."""
        base = BUILDING_COLORS.get(style, BUILDING_COLORS["concrete"])
        ground_rgba = _scale_rgb(base, 0.72)
        inset_rgba = _scale_rgb(base, 1.12 if style != "charred" else 0.85)
        lip_rgba = _scale_rgb(base, 0.88)
        gf_h = min(STORY, height * 0.45)
        lip_t = 0.35
        lip_over = 0.45

        for xmin, xmax, ymin, ymax in footprints:
            # Darker ground-floor band (slightly proud so it reads)
            if height > 2.5:
                self._add_detail(
                    xmin - 0.05,
                    ymin - 0.05,
                    0.0,
                    xmax + 0.05,
                    ymax + 0.05,
                    gf_h,
                    ground_rgba,
                )
            # Thin roof lip past the walls
            self._add_detail(
                xmin - lip_over,
                ymin - lip_over,
                height - lip_t,
                xmax + lip_over,
                ymax + lip_over,
                height + 0.05,
                lip_rgba,
            )

        # Inset volume so tall boxes are not one solid color (esp. towers)
        if btype in ("tower", "midrise", "setback") and height > 10.0:
            primary = footprints[0]
            xmin, xmax, ymin, ymax = primary
            inset = 1.2 if btype == "tower" else 0.8
            z0 = max(gf_h + 0.5, height * 0.28)
            z1 = height - lip_t - 0.2
            if z1 > z0 + 2.0 and (xmax - xmin) > 2 * inset + 2.0 and (ymax - ymin) > 2 * inset + 2.0:
                self._add_detail(
                    xmin + inset,
                    ymin + inset,
                    z0,
                    xmax - inset,
                    ymax - inset,
                    z1,
                    inset_rgba,
                )
        elif btype in ("shop", "row", "lshape") and height > 5.0:
            # Subtle upper-band inset on the longest footprint
            primary = max(footprints, key=lambda f: (f[1] - f[0]) * (f[3] - f[2]))
            xmin, xmax, ymin, ymax = primary
            inset = 0.55
            z0 = gf_h + 0.3
            z1 = height - lip_t - 0.15
            if z1 > z0 + 1.0:
                self._add_detail(
                    xmin + inset,
                    ymin + inset,
                    z0,
                    xmax - inset,
                    ymax - inset,
                    z1,
                    inset_rgba,
                )

    def _footprint_free(self, xmin, xmax, ymin, ymax, placed, pad=1.2):
        probe = (xmin, xmax, ymin, ymax)
        for other in placed:
            if self._rects_overlap(probe, other, pad=pad):
                return False
        if self._footprint_overlaps(xmin, xmax, ymin, ymax, pad=pad):
            return False
        return True

    def _place_rect_building(
        self,
        xmin,
        xmax,
        ymin,
        ymax,
        height,
        btype,
        style,
        placed,
    ):
        if not self._footprint_free(xmin, xmax, ymin, ymax, placed):
            return False
        idx = self._add_box(xmin, ymin, 0.0, xmax, ymax, height, style)
        fps = [(xmin, xmax, ymin, ymax)]
        self._register_building(btype, fps, [idx], style, height)
        self._add_building_details(fps, height, style, btype)
        placed.append((xmin, xmax, ymin, ymax))
        return True

    def _place_lshape(
        self,
        edge,
        along0,
        depth_limit,
        block,
        style,
        placed,
        damaged=False,
    ):
        """Two ground boxes sharing a corner, flush to the street edge."""
        x0, x1, y0, y1 = block
        long_w = float(self.rng.uniform(14.0, 22.0))
        short_w = float(self.rng.uniform(8.0, 12.0))
        d_main = float(self.rng.uniform(8.0, 11.0))
        d_wing = float(self.rng.uniform(10.0, 16.0))
        height = float(self.rng.uniform(3.0, 8.0)) if damaged else float(
            self.rng.uniform(2.0 * STORY, 4.0 * STORY)
        )
        wing_left = self.rng.random() < 0.5

        if edge in ("s", "n"):
            if along0 + long_w > depth_limit:
                return False
            a0, a1 = along0, along0 + long_w
            if edge == "s":
                main = (a0, a1, y0, y0 + d_main)
                if wing_left:
                    wing = (a0, a0 + short_w, y0, y0 + d_wing)
                else:
                    wing = (a1 - short_w, a1, y0, y0 + d_wing)
            else:
                main = (a0, a1, y1 - d_main, y1)
                if wing_left:
                    wing = (a0, a0 + short_w, y1 - d_wing, y1)
                else:
                    wing = (a1 - short_w, a1, y1 - d_wing, y1)
        else:
            if along0 + long_w > depth_limit:
                return False
            a0, a1 = along0, along0 + long_w
            if edge == "w":
                main = (x0, x0 + d_main, a0, a1)
                if wing_left:
                    wing = (x0, x0 + d_wing, a0, a0 + short_w)
                else:
                    wing = (x0, x0 + d_wing, a1 - short_w, a1)
            else:
                main = (x1 - d_main, x1, a0, a1)
                if wing_left:
                    wing = (x1 - d_wing, x1, a0, a0 + short_w)
                else:
                    wing = (x1 - d_wing, x1, a1 - short_w, a1)

        for rect in (main, wing):
            if rect[0] < x0 - 0.01 or rect[1] > x1 + 0.01 or rect[2] < y0 - 0.01 or rect[3] > y1 + 0.01:
                return False
            if not self._footprint_free(rect[0], rect[1], rect[2], rect[3], placed, pad=0.8):
                return False

        idxs = []
        fps = []
        for rect in (main, wing):
            xmin, xmax, ymin, ymax = rect
            idxs.append(self._add_box(xmin, ymin, 0.0, xmax, ymax, height, style))
            fps.append((xmin, xmax, ymin, ymax))
            placed.append((xmin, xmax, ymin, ymax))
        self._register_building("lshape", fps, idxs, style, height)
        self._add_building_details(fps, height, style, "lshape")
        return True

    def _place_setback(
        self,
        xmin,
        xmax,
        ymin,
        ymax,
        style,
        placed,
        damaged=False,
    ):
        """Wide base with a smaller upper box stacked on top."""
        if not self._footprint_free(xmin, xmax, ymin, ymax, placed):
            return False
        if damaged:
            base_h = float(self.rng.uniform(3.0, 6.0))
            top_h = float(self.rng.uniform(2.0, 4.0))
        else:
            base_h = float(self.rng.uniform(2.0 * STORY, 4.0 * STORY))
            top_h = float(self.rng.uniform(2.0 * STORY, 5.0 * STORY))
        inset = float(self.rng.uniform(1.5, 3.0))
        tx0, tx1 = xmin + inset, xmax - inset
        ty0, ty1 = ymin + inset, ymax - inset
        if tx1 - tx0 < 4.0 or ty1 - ty0 < 4.0:
            return False

        i0 = self._add_box(xmin, ymin, 0.0, xmax, ymax, base_h, style)
        i1 = self._add_box(tx0, ty0, base_h, tx1, ty1, base_h + top_h, style)
        fps = [(xmin, xmax, ymin, ymax), (tx0, tx1, ty0, ty1)]
        total_h = base_h + top_h
        self._register_building("setback", fps, [i0, i1], style, total_h)
        # Ground floor + lip on the wide base; inset/lip on the smaller crown
        base_color = BUILDING_COLORS.get(style, BUILDING_COLORS["concrete"])
        self._add_detail(
            xmin - 0.05,
            ymin - 0.05,
            0.0,
            xmax + 0.05,
            ymax + 0.05,
            min(STORY, base_h * 0.5),
            _scale_rgb(base_color, 0.72),
        )
        self._add_detail(
            tx0 - 0.4,
            ty0 - 0.4,
            total_h - 0.35,
            tx1 + 0.4,
            ty1 + 0.4,
            total_h + 0.05,
            _scale_rgb(base_color, 0.88),
        )
        inset = 0.7
        if (tx1 - tx0) > 2 * inset + 2.0 and (ty1 - ty0) > 2 * inset + 2.0:
            self._add_detail(
                tx0 + inset,
                ty0 + inset,
                base_h + 0.4,
                tx1 - inset,
                ty1 - inset,
                total_h - 0.4,
                _scale_rgb(base_color, 1.12 if style != "charred" else 0.85),
            )
        placed.append((xmin, xmax, ymin, ymax))
        return True

    def _building_dims(self, btype, edge_len, lot_depth, damaged=False):
        """Return (along_street, into_lot, height) for a rectangular type."""
        max_along = max(6.0, edge_len - 1.0)
        max_depth = max(6.0, min(lot_depth * 0.55, lot_depth - 2.0))

        def span(lo, hi_cap):
            hi = min(hi_cap, max_along)
            if hi < lo:
                return float(max(6.0, hi))
            return float(self.rng.uniform(lo, hi))

        def depth_span(lo, hi_cap):
            hi = min(hi_cap, max_depth)
            if hi < lo:
                return float(max(6.0, hi))
            return float(self.rng.uniform(lo, hi))

        if damaged:
            along = span(8.0, 14.0)
            depth = depth_span(7.0, 11.0)
            height = float(self.rng.uniform(2.5, 7.0))
            return along, depth, height

        if btype == "shop":
            along = span(10.0, 18.0)
            depth = depth_span(8.0, 12.0)
            height = float(self.rng.uniform(1.0 * STORY, 2.0 * STORY))
        elif btype == "row":
            lo = min(22.0, max_along * 0.7)
            hi = min(max_along * 0.92, max_along)
            if hi < lo:
                along = float(max_along)
            else:
                along = float(self.rng.uniform(lo, hi))
            depth = depth_span(9.0, 14.0)
            height = float(self.rng.uniform(2.0 * STORY, 3.2 * STORY))
        elif btype == "midrise":
            along = span(8.0, 14.0)
            depth = depth_span(8.0, 12.0)
            height = float(self.rng.uniform(4.0 * STORY, 8.0 * STORY))
        elif btype == "tower":
            along = span(8.0, 12.0)
            depth = depth_span(8.0, 12.0)
            height = float(
                self.rng.uniform(self.altitude_cap + 5.0, self.altitude_cap + 22.0)
            )
        else:
            along = span(10.0, 16.0)
            depth = depth_span(8.0, 12.0)
            height = float(self.rng.uniform(2.0 * STORY, 5.0 * STORY))
        return along, depth, height

    def _edge_rect(self, edge, block, along0, along1, depth):
        """Map an along-street span + depth into a footprint on a block edge."""
        x0, x1, y0, y1 = block
        if edge == "s":
            return along0, along1, y0, y0 + depth
        if edge == "n":
            return along0, along1, y1 - depth, y1
        if edge == "w":
            return x0, x0 + depth, along0, along1
        return x1 - depth, x1, along0, along1

    def _fill_block_edge(self, edge, block, placed, type_cycle, towers_left, damaged_left):
        """Pack street-aligned buildings along one block edge with gaps."""
        x0, x1, y0, y1 = block
        if edge in ("s", "n"):
            t0, t1 = x0 + 1.0, x1 - 1.0
            lot_depth = y1 - y0
        else:
            t0, t1 = y0 + 1.0, y1 - 1.0
            lot_depth = x1 - x0

        cursor = t0
        gap = float(self.rng.uniform(1.5, 3.0))
        cursor += float(self.rng.uniform(0.0, 1.5))
        placed_on_edge = 0
        max_on_edge = 5
        used_types = set()

        while cursor < t1 - 8.0 and placed_on_edge < max_on_edge:
            # Prefer unused types on this block edge for a visible mix
            candidates = [t for t in type_cycle if t not in used_types] or list(type_cycle)
            if towers_left[0] <= 0:
                candidates = [t for t in candidates if t != "tower"] or [
                    t for t in BUILDING_TYPE_NAMES if t != "tower"
                ]
            btype = candidates[int(self.rng.integers(0, len(candidates)))]

            damaged = False
            if damaged_left[0] > 0:
                # Bias so a handful of short charred buildings appear citywide
                p_dmg = 0.35 if damaged_left[0] > 2 else 0.75
                if self.rng.random() < p_dmg:
                    damaged = True
                    # Damaged buildings stay short shops / rows
                    btype = "shop" if self.rng.random() < 0.6 else "row"

            style = self._pick_style(damaged=damaged)
            edge_remaining = t1 - cursor

            if btype == "lshape":
                ok = self._place_lshape(
                    edge, cursor, t1, block, style, placed, damaged=damaged
                )
                if ok:
                    if damaged:
                        damaged_left[0] -= 1
                    # L long arm ~14–22 m
                    cursor += float(self.rng.uniform(15.0, 23.0)) + gap
                    placed_on_edge += 1
                    used_types.add("lshape")
                else:
                    cursor += 2.0
                continue

            along, depth, height = self._building_dims(
                btype, edge_remaining, lot_depth, damaged=damaged
            )
            if along > edge_remaining:
                if edge_remaining < 9.0:
                    break
                along = edge_remaining
                if btype == "tower":
                    cursor += 1.0
                    continue

            xmin, xmax, ymin, ymax = self._edge_rect(
                edge, block, cursor, cursor + along, depth
            )
            # Keep depth inside the lot
            xmin = max(xmin, x0)
            xmax = min(xmax, x1)
            ymin = max(ymin, y0)
            ymax = min(ymax, y1)
            if xmax - xmin < 6.0 or ymax - ymin < 6.0:
                cursor += 2.0
                continue

            if btype == "setback":
                ok = self._place_setback(
                    xmin, xmax, ymin, ymax, style, placed, damaged=damaged
                )
            else:
                ok = self._place_rect_building(
                    xmin, xmax, ymin, ymax, height, btype, style, placed
                )

            if ok:
                if damaged:
                    damaged_left[0] -= 1
                if btype == "tower":
                    towers_left[0] -= 1
                used_types.add(btype)
                placed_on_edge += 1
                cursor += along + gap
            else:
                cursor += 2.0

        return placed_on_edge

    def _place_buildings(self, half):
        """Street-aligned typed buildings on block edges (downtown only)."""
        blocks = list(self._block_rects(half))
        self.rng.shuffle(blocks)

        # Few towers / few damaged short buildings citywide
        towers_left = [max(2, min(5, len(blocks) // 10))]
        damaged_left = [max(4, min(10, len(blocks) // 5))]

        for block in blocks:
            x0, x1, y0, y1 = block
            if (x1 - x0) < 14.0 or (y1 - y0) < 14.0:
                continue
            placed = []
            edges = ["s", "n", "w", "e"]
            self.rng.shuffle(edges)
            n_edges = int(self.rng.integers(2, 5))

            # Per-block mix — include most types, tower only sometimes
            type_cycle = ["shop", "row", "midrise", "lshape", "setback", "shop", "midrise"]
            if towers_left[0] > 0 and self.rng.random() < 0.45:
                type_cycle.append("tower")
            self.rng.shuffle(type_cycle)

            for edge in edges[:n_edges]:
                self._fill_block_edge(
                    edge, block, placed, type_cycle, towers_left, damaged_left
                )

        # Guarantee a handful of short damaged buildings if RNG under-delivered
        n_charred = sum(1 for b in self.buildings if b["style"] == "charred")
        target_dmg = max(4, min(8, len(self.buildings) // 15))
        if n_charred < target_dmg:
            candidates = [
                i
                for i, b in enumerate(self.buildings)
                if b["style"] != "charred"
                and b["type"] in ("shop", "row")
                and b["height"] <= 8.0
            ]
            self.rng.shuffle(candidates)
            for bi in candidates[: target_dmg - n_charred]:
                b = self.buildings[bi]
                b["style"] = "charred"
                for idx in b["box_indices"]:
                    self.styles[idx] = "charred"
                # Shorten intact tall leftovers slightly so damage reads
                if b["height"] > 7.0:
                    new_h = float(self.rng.uniform(3.0, 6.5))
                    b["height"] = new_h
                    for idx in b["box_indices"]:
                        self.boxes[idx, 5] = self.boxes[idx, 2] + new_h


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

    def _build_road_segments(self, terrain):
        """
        Short elevated pavement segments that follow the heightmap.

        Full-map slabs at one Z pierce rolling ground and z-fight; X/Y streets
        also stacked two slabs at every intersection. Here each segment uses
        the max local ground along its span, sits ROAD_LIFT above that, and
        Y-streets skip X-street bands so only one face exists at each paved
        (x, y).
        """
        half = self.size / 2.0
        half_w = self.street_width / 2.0
        seg = ROAD_SEG_LEN
        roads = []

        def elev(points):
            gz = max(_sample_heightmap(terrain, x, y) for x, y in points)
            return gz + ROAD_LIFT

        def sample_grid(u0, u1, v_center, half_cross, along_is_y):
            """Dense samples over a pavement rectangle for max-height elev."""
            n_along = max(3, int(np.ceil((u1 - u0) / 1.0)) + 1)
            n_cross = max(3, int(np.ceil((2.0 * half_cross) / 1.5)) + 1)
            along = np.linspace(u0, u1, n_along)
            cross = np.linspace(v_center - half_cross, v_center + half_cross, n_cross)
            pts = []
            for a in along:
                for c in cross:
                    if along_is_y:
                        pts.append((float(c), float(a)))  # x=cross, y=along
                    else:
                        pts.append((float(a), float(c)))  # x=along, y=cross
            return pts

        # North–south streets (constant x): cover full run, including crossings.
        for cx in self.street_centers_x:
            y0 = -half
            while y0 < half - 1e-6:
                y1 = min(y0 + seg, half)
                hy = 0.5 * (y1 - y0)
                if hy < 0.05:
                    break
                cy = 0.5 * (y0 + y1)
                z = elev(sample_grid(y0, y1, cx, half_w, along_is_y=True))
                roads.append((float(cx), float(cy), float(half_w), float(hy), z))
                y0 = y1

        # East–west streets (constant y): skip cells already paved by N–S streets.
        for cy in self.street_centers_y:
            x0 = -half
            while x0 < half - 1e-6:
                x1 = min(x0 + seg, half)
                hx = 0.5 * (x1 - x0)
                if hx < 0.05:
                    break
                cx = 0.5 * (x0 + x1)
                if any(abs(cx - sx) <= half_w for sx in self.street_centers_x):
                    x0 = x1
                    continue
                z = elev(sample_grid(x0, x1, cy, half_w, along_is_y=False))
                roads.append((float(cx), float(cy), float(hx), float(half_w), z))
                x0 = x1

        return roads

    def _flatten_street_heightmap(self, terrain):
        """
        Soften heightmap noise on street corridors so pavement rides a smooth
        ribbon. Does not change wilderness outside the city footprint.
        """
        half_city = self.size / 2.0
        half_w = self.street_width / 2.0 + 0.75
        Z = terrain.heightmap
        gy, gx = Z.shape
        res = float(terrain.resolution)
        half_x = terrain.size_x / 2.0
        half_y = terrain.size_y / 2.0
        xs = (np.arange(gx) + 0.5) * res - half_x
        ys = (np.arange(gy) + 0.5) * res - half_y
        XX, YY = np.meshgrid(xs, ys)
        mask = np.zeros_like(Z, dtype=bool)
        for sx in self.street_centers_x:
            mask |= np.abs(XX - sx) <= half_w
        for sy in self.street_centers_y:
            mask |= np.abs(YY - sy) <= half_w
        mask &= (np.abs(XX) <= half_city + 1.0) & (np.abs(YY) <= half_city + 1.0)
        if not mask.any():
            return
        # Mild blur, then write only into street cells (lots / rim stay put).
        try:
            from .terrain import _box_blur
        except Exception:
            return
        radius = max(1, int(round(2.0 / res)))
        smooth = _box_blur(Z, radius=radius, passes=3)
        Z[mask] = smooth[mask]

    def apply_ground_heights(self, terrain):
        """
        Shift building / overpass / detail AABB z so bases follow the heightmap.

        Call after Terrain is built. Relative heights are preserved. Idempotent.
        Also rebuilds road segments so pavement tracks local ground without
        stacking a second full-ground mesh on the heightfield.
        """
        if self._ground_applied:
            return

        # Smooth street ribbons before sampling so roads and bases agree.
        self._flatten_street_heightmap(terrain)

        def lift_box_row(box, style):
            xmin, ymin, zmin, xmax, ymax, zmax = box
            cx = 0.5 * (xmin + xmax)
            cy = 0.5 * (ymin + ymax)
            gz = _sample_heightmap(terrain, cx, cy)
            if style == "overpass":
                clearance = zmin
                thickness = zmax - zmin
                return (xmin, ymin, gz + clearance, xmax, ymax, gz + clearance + thickness)
            height = zmax - zmin
            # Preserve relative stack offsets (setback crowns keep zmin > 0)
            return (xmin, ymin, gz + zmin, xmax, ymax, gz + zmin + height)

        if self.boxes.size:
            new_boxes = self.boxes.copy()
            for i, box in enumerate(self.boxes):
                style = self.styles[i] if i < len(self.styles) else "concrete"
                lifted = lift_box_row(box, style)
                new_boxes[i] = lifted
            self.boxes = new_boxes

        if self.detail_parts:
            lifted_details = []
            for xmin, ymin, zmin, xmax, ymax, zmax, rgba in self.detail_parts:
                cx = 0.5 * (xmin + xmax)
                cy = 0.5 * (ymin + ymax)
                gz = _sample_heightmap(terrain, cx, cy)
                # Keep ground-floor detail bottoms a hair above the heightfield
                # so the building skirt does not share a plane with terrain.
                z0 = zmin if zmin > 1e-6 else 0.02
                lifted_details.append(
                    (xmin, ymin, gz + z0, xmax, ymax, gz + zmax, rgba)
                )
            self.detail_parts = lifted_details

        self.roads = self._build_road_segments(terrain)
        self._ground_applied = True

    def _blocks_feet(self, x, y, pad):
        """True if a pad around (x, y) overlaps a ground building or rubble."""
        if self.boxes.size == 0:
            return False
        over = set(self.overpass_indices)
        xmin, xmax = x - pad, x + pad
        ymin, ymax = y - pad, y + pad
        for i, b in enumerate(self.boxes):
            if i in over:
                continue
            if xmax > b[0] and xmin < b[3] and ymax > b[1] and ymin < b[4]:
                return True
        return False

    def _feet_z(self, terrain, x, y):
        """
        Ground the feet on pavement when (x, y) is on a street, otherwise
        on the heightmap. Road slab center is `z`; the walking surface is
        one half-thickness above that.
        """
        if self._on_street(x, y):
            for cx, cy, hx, hy, z in self.roads:
                if abs(x - cx) <= hx + 0.05 and abs(y - cy) <= hy + 0.05:
                    return float(z + ROAD_HALF_THICK + 0.01)
            gz = _sample_heightmap(terrain, x, y)
            return gz + ROAD_LIFT + ROAD_HALF_THICK + 0.01
        return _sample_heightmap(terrain, x, y) + 0.02

    def place_humans(self, terrain, count=HUMAN_COUNT, survivor_count=HUMAN_SURVIVOR_COUNT):
        """
        Scatter static people on streets and open lots.

        Call after apply_ground_heights so feet sit on pavement or terrain.
        Idempotent. Does not add collision boxes.
        """
        if self._humans_placed:
            return
        if not self._ground_applied:
            self.apply_ground_heights(terrain)

        rng = self.human_rng
        half = self.size / 2.0
        margin = HUMAN_EDGE_MARGIN
        pad = HUMAN_BODY_RADIUS + 0.2
        placed = []
        attempts = 0
        max_attempts = max(count * 80, 1)
        while len(placed) < count and attempts < max_attempts:
            attempts += 1
            x = float(rng.uniform(-half + margin, half - margin))
            y = float(rng.uniform(-half + margin, half - margin))
            if not self.is_walkable(x, y):
                continue
            if self._blocks_feet(x, y, pad):
                continue
            if placed:
                xy = np.asarray(placed, dtype=np.float64)
                d2 = (xy[:, 0] - x) ** 2 + (xy[:, 1] - y) ** 2
                if np.any(d2 < HUMAN_MIN_SEP * HUMAN_MIN_SEP):
                    continue
            placed.append((x, y))

        shirts = HUMAN_SHIRT_COLORS
        skins = HUMAN_SKIN_COLORS
        humans = []
        for x, y in placed:
            shirt = shirts[int(rng.integers(0, len(shirts)))]
            skin = skins[int(rng.integers(0, len(skins)))]
            humans.append(
                {
                    "x": float(x),
                    "y": float(y),
                    "z": float(self._feet_z(terrain, x, y)),
                    "shirt": [float(c) for c in shirt],
                    "skin": [float(c) for c in skin],
                    "survivor": False,
                }
            )

        n_surv = min(int(survivor_count), len(humans))
        if n_surv:
            pick = rng.choice(len(humans), size=n_surv, replace=False)
            vests = HUMAN_VEST_COLORS
            for i, idx in enumerate(np.atleast_1d(pick)):
                vest = vests[int(i) % len(vests)]
                humans[int(idx)]["survivor"] = True
                humans[int(idx)]["shirt"] = [float(c) for c in vest]

        self.humans = humans
        self._humans_placed = True

    def human_positions(self, survivors_only=False):
        """Feet positions as (N, 3). Empty array if nobody has been placed."""
        people = self.humans
        if survivors_only:
            people = [h for h in people if h["survivor"]]
        if not people:
            return np.zeros((0, 3), dtype=np.float64)
        return np.array([[h["x"], h["y"], h["z"]] for h in people], dtype=np.float64)

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


def connect_pybullet(gui=True, shadows=False):
    """Open a PyBullet client. Shadows default off for a lighter GUI."""
    client = p.connect(p.GUI if gui else p.DIRECT)
    if gui:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=client)
        p.configureDebugVisualizer(
            p.COV_ENABLE_SHADOWS, 1 if shadows else 0, physicsClientId=client
        )
        # Disable built-in W/wireframe and other GUI hotkeys so app keys win.
        p.configureDebugVisualizer(
            p.COV_ENABLE_KEYBOARD_SHORTCUTS, 0, physicsClientId=client
        )
        p.configureDebugVisualizer(
            p.COV_ENABLE_WIREFRAME, 0, physicsClientId=client
        )
    p.setGravity(0, 0, -9.81, physicsClientId=client)
    return client


def _spawn_human_visuals(client, town):
    """
    One capsule body and one sphere head per person. Visual only (mass 0,
    no collision). Capsule length is the cylinder section; PyBullet aligns
    that axis with Z, so the figure stands upright.
    """
    if not town.humans:
        return []

    shirt_shapes = {}
    skin_shapes = {}
    bodies = []
    head_z = (
        HUMAN_BODY_LENGTH / 2.0
        + HUMAN_BODY_RADIUS
        + HUMAN_HEAD_RADIUS
        - HUMAN_HEAD_OVERLAP
    )
    body_center_z = HUMAN_BODY_LENGTH / 2.0 + HUMAN_BODY_RADIUS

    for h in town.humans:
        shirt_key = tuple(h["shirt"])
        if shirt_key not in shirt_shapes:
            shirt_shapes[shirt_key] = p.createVisualShape(
                p.GEOM_CAPSULE,
                radius=HUMAN_BODY_RADIUS,
                length=HUMAN_BODY_LENGTH,
                rgbaColor=h["shirt"],
                physicsClientId=client,
            )
        skin_key = tuple(h["skin"])
        if skin_key not in skin_shapes:
            skin_shapes[skin_key] = p.createVisualShape(
                p.GEOM_SPHERE,
                radius=HUMAN_HEAD_RADIUS,
                rgbaColor=h["skin"],
                physicsClientId=client,
            )
        bid = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=shirt_shapes[shirt_key],
            basePosition=[h["x"], h["y"], h["z"] + body_center_z],
            physicsClientId=client,
        )
        hid = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=skin_shapes[skin_key],
            basePosition=[h["x"], h["y"], h["z"] + body_center_z + head_z],
            physicsClientId=client,
        )
        bodies.append(bid)
        bodies.append(hid)
    return bodies


def spawn_town_in_pybullet(client, town, terrain, env_size):
    """
    Load one dusty heightfield plus elevated road segments and building visuals.

    The heightfield is the only full-map ground surface (collision + visual).
    Roads are short pavement slabs ~ROAD_LIFT above local ground — not a second
    ground pad. No city-wide or rim color plates. People are spawned as
    visual-only figures and stored on town.human_body_ids. Returns
    (terrain_body, road_bodies, building_bodies).

    Mesh scale uses terrain.size_x / terrain.grid_x so a grass-belt world
    larger than the city still matches height samples. env_size is the city
    extent (call-site clarity; not used for mesh scale).
    """
    del env_size  # city size; mesh scale uses the full terrain world
    grid = terrain.grid_x
    mesh_xy = float(terrain.size_x) / float(grid)

    town.apply_ground_heights(terrain)
    if hasattr(terrain, "set_obstacle_boxes"):
        terrain.set_obstacle_boxes(town.boxes)

    # One image covers the heightfield once (PyBullet convention).
    tex_scale = (grid - 1) / 2.0
    terrain_shape = p.createCollisionShape(
        p.GEOM_HEIGHTFIELD,
        meshScale=[mesh_xy, mesh_xy, 1.0],
        heightfieldData=terrain.heightmap.flatten().tolist(),
        numHeightfieldRows=grid,
        numHeightfieldColumns=grid,
        heightfieldTextureScaling=tex_scale,
        physicsClientId=client,
    )
    # PyBullet centers the heightfield AABB at the body origin; lift so
    # world Z matches heightmap samples used by buildings / roads.
    hmin = float(np.min(terrain.heightmap))
    hmax = float(np.max(terrain.heightmap))
    mid = 0.5 * (hmin + hmax)
    terrain_body = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=terrain_shape,
        basePosition=[0.0, 0.0, mid],
        physicsClientId=client,
    )

    # Ground texture: dusty city feathers to grass. White tint so rgba does
    # not multiply brown on top of the texture colors.
    texture_id = -1
    if hasattr(terrain, "get_ground_texture_path"):
        try:
            tex_path = terrain.get_ground_texture_path()
            texture_id = p.loadTexture(tex_path, physicsClientId=client)
        except Exception:
            texture_id = -1

    if texture_id >= 0:
        p.changeVisualShape(
            terrain_body,
            -1,
            rgbaColor=[1.0, 1.0, 1.0, 1.0],
            textureUniqueId=texture_id,
            physicsClientId=client,
        )
    else:
        p.changeVisualShape(
            terrain_body,
            -1,
            rgbaColor=[0.45, 0.38, 0.28, 1],
            physicsClientId=client,
        )

    road_bodies = []
    for cx, cy, hx, hy, z in town.roads:
        vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[hx, hy, ROAD_HALF_THICK],
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

    def _spawn_box(xmin, ymin, zmin, xmax, ymax, zmax, rgba):
        hx = (xmax - xmin) / 2.0
        hy = (ymax - ymin) / 2.0
        hz = max((zmax - zmin) / 2.0, 0.05)
        cx = (xmin + xmax) / 2.0
        cy = (ymin + ymax) / 2.0
        cz = (zmin + zmax) / 2.0
        vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[hx, hy, hz],
            rgbaColor=rgba,
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

    for i, box in enumerate(town.boxes):
        xmin, ymin, zmin, xmax, ymax, zmax = box
        style = town.styles[i]
        color = BUILDING_COLORS.get(style, BUILDING_COLORS["concrete"])
        _spawn_box(xmin, ymin, zmin, xmax, ymax, zmax, color)

    # Ground floors, roof lips, inset tones (visual only)
    for part in town.detail_parts:
        xmin, ymin, zmin, xmax, ymax, zmax, rgba = part
        _spawn_box(xmin, ymin, zmin, xmax, ymax, zmax, rgba)

    town.place_humans(terrain)
    town.human_body_ids = _spawn_human_visuals(client, town)

    return terrain_body, road_bodies, building_bodies


def frame_town_camera(client, env_size):
    """Overview framing for the full map."""
    cam_dist = max(180.0, float(env_size) * 0.85)
    p.resetDebugVisualizerCamera(
        cameraDistance=cam_dist,
        cameraYaw=45,
        cameraPitch=-40,
        cameraTargetPosition=[0, 0, 8.0],
        physicsClientId=client,
    )
