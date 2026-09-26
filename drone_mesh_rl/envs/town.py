"""
town.py — Seeded damaged-town layout: streets, buildings, rubble, overpasses.

Buildings and overpasses are axis-aligned boxes (AABB) so collision and
line-of-sight stay vectorized numpy — no PyBullet raycasts.

Downtown buildings are street-aligned typed volumes (shop, row, mid-rise,
tower, L-shape, setback) with cheap visual detail overlays.
"""

from __future__ import annotations

import numpy as np

try:
    import pybullet as p
except ImportError:
    p = None

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
ROAD_SEG_LEN = 40.0  # long slabs; streets are flattened before paving

# Center-line dashes (above pavement, never coplanar with ground)
DASH_COLOR = [0.92, 0.86, 0.28, 1.0]
DASH_HALF_THICK = 0.02
DASH_LEN = 2.4
DASH_WIDTH = 0.16
DASH_GAP = 3.6
# Dash center sits above pavement top
DASH_ABOVE_PAVEMENT = ROAD_HALF_THICK + DASH_HALF_THICK + 0.015

SIGN_POST_COLOR = [0.35, 0.35, 0.32, 1.0]
SIGN_BOARD_COLOR = [0.75, 0.22, 0.18, 1.0]
GAS_CANOPY_COLOR = [0.85, 0.75, 0.20, 1.0]
GAS_SHOP_COLOR = [0.62, 0.62, 0.58, 1.0]
GAS_PUMP_COLOR = [0.25, 0.25, 0.28, 1.0]

BUILDING_TYPE_NAMES = ("shop", "row", "midrise", "tower", "lshape", "setback")


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
    j = int((x + half_x) / getattr(terrain, "resolution_x", terrain.resolution))
    i = int((y + half_y) / getattr(terrain, "resolution_y", terrain.resolution))
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

    def __init__(self, size=250.0, seed=None, altitude_cap=40.0, origin=(0.0, 0.0)):
        self.size = float(size)
        self.altitude_cap = float(altitude_cap)
        self.origin = (float(origin[0]), float(origin[1]))
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
        # Center-line dashes: (cx, cy, hx, hy, z, rgba)
        self.road_marks: list[tuple] = []
        self._ground_applied = False

        self._build_street_grid(half)
        self._place_buildings(half)
        self._place_rubble(half)
        self._place_overpasses(half)
        self._finalize_walkable(half)
        self._apply_origin_offset()

    def _apply_origin_offset(self):
        """Shift local (-half..half) layout into world coordinates."""
        ox, oy = self.origin
        if abs(ox) < 1e-12 and abs(oy) < 1e-12:
            return
        if self.boxes.size:
            self.boxes[:, 0] += ox
            self.boxes[:, 1] += oy
            self.boxes[:, 3] += ox
            self.boxes[:, 4] += oy
        lifted = []
        for xmin, ymin, zmin, xmax, ymax, zmax, rgba in self.detail_parts:
            lifted.append(
                (xmin + ox, ymin + oy, zmin, xmax + ox, ymax + oy, zmax, rgba)
            )
        self.detail_parts = lifted
        for b in self.buildings:
            b["footprints"] = [
                (fx0 + ox, fx1 + ox, fy0 + oy, fy1 + oy)
                for fx0, fx1, fy0, fy1 in b["footprints"]
            ]
        self.street_centers_x = [c + ox for c in self.street_centers_x]
        self.street_centers_y = [c + oy for c in self.street_centers_y]

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
        Elevated pavement segments that follow the (flattened) heightmap.

        Full-map slabs at one Z pierce rolling ground and z-fight; X/Y streets
        also stacked two slabs at every intersection. Here each segment uses
        the max local ground along its span, sits ROAD_LIFT above that, and
        east–west runs skip north–south street bands so only one face exists
        at each paved (x, y).
        """
        half = self.size / 2.0
        ox, oy = self.origin
        y_lo, y_hi = oy - half, oy + half
        x_lo, x_hi = ox - half, ox + half
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
            y0 = y_lo
            while y0 < y_hi - 1e-6:
                y1 = min(y0 + seg, y_hi)
                hy = 0.5 * (y1 - y0)
                if hy < 0.05:
                    break
                cy = 0.5 * (y0 + y1)
                z = elev(sample_grid(y0, y1, cx, half_w, along_is_y=True))
                roads.append((float(cx), float(cy), float(half_w), float(hy), z))
                y0 = y1

        # East–west streets (constant y): carve out N–S bands, then long slabs.
        for cy in self.street_centers_y:
            intervals = [(x_lo, x_hi)]
            for sx in self.street_centers_x:
                band_lo, band_hi = sx - half_w, sx + half_w
                next_intervals = []
                for a, b in intervals:
                    if b <= band_lo or a >= band_hi:
                        next_intervals.append((a, b))
                        continue
                    if a < band_lo:
                        next_intervals.append((a, band_lo))
                    if band_hi < b:
                        next_intervals.append((band_hi, b))
                intervals = next_intervals
            for a, b in intervals:
                x0 = a
                while x0 < b - 1e-6:
                    x1 = min(x0 + seg, b)
                    hx = 0.5 * (x1 - x0)
                    if hx < 0.05:
                        break
                    cx = 0.5 * (x0 + x1)
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
        ox, oy = self.origin
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
        mask &= (np.abs(XX - ox) <= half_city + 1.0) & (
            np.abs(YY - oy) <= half_city + 1.0
        )
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
        self.road_marks = []
        self._ground_applied = True

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def is_walkable(self, x, y):
        """Return True if (x, y) is on a street or open lot (not in a building)."""
        ox, oy = self.origin
        half = self.size / 2.0
        res = self.walkable_resolution
        gx = int((x - ox + half) / res)
        gy = int((y - oy + half) / res)
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
        ox, oy = self.origin
        res = self.walkable_resolution
        pts = np.column_stack(
            [
                xs[idx] * res - half + res * 0.5 + ox,
                ys[idx] * res - half + res * 0.5 + oy,
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
    def sphere_clearance(pos, boxes):
        """Meters from pos to the nearest AABB. 0 if inside or touching."""
        if boxes is None or len(boxes) == 0:
            return 1e6
        pos = np.asarray(pos, dtype=np.float64).reshape(3)
        closest = np.column_stack(
            [
                np.clip(pos[0], boxes[:, 0], boxes[:, 3]),
                np.clip(pos[1], boxes[:, 1], boxes[:, 4]),
                np.clip(pos[2], boxes[:, 2], boxes[:, 5]),
            ]
        )
        d2 = np.sum((closest - pos) ** 2, axis=1)
        return float(np.sqrt(np.min(d2)))

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


def _elev_points(terrain, points):
    return max(_sample_heightmap(terrain, x, y) for x, y in points) + ROAD_LIFT


def _sample_rect_points(u0, u1, v_center, half_cross, along_is_y):
    n_along = max(3, int(np.ceil(abs(u1 - u0) / 1.0)) + 1)
    n_cross = max(3, int(np.ceil((2.0 * half_cross) / 1.5)) + 1)
    along = np.linspace(u0, u1, n_along)
    cross = np.linspace(v_center - half_cross, v_center + half_cross, n_cross)
    pts = []
    for a in along:
        for c in cross:
            if along_is_y:
                pts.append((float(c), float(a)))
            else:
                pts.append((float(a), float(c)))
    return pts


def _dashes_along_line(terrain, x0, y0, x1, y1, marks_out):
    """Append raised center-line dashes along a straight segment."""
    dx = x1 - x0
    dy = y1 - y0
    length = float(np.hypot(dx, dy))
    if length < DASH_LEN:
        return
    ux, uy = dx / length, dy / length
    # Perpendicular half-width axis: dash extent across the road
    px, py = -uy, ux
    step = DASH_LEN + DASH_GAP
    s = DASH_GAP * 0.5
    hx = 0.5 * DASH_LEN
    hy = 0.5 * DASH_WIDTH
    while s + DASH_LEN <= length + 1e-6:
        mid = s + 0.5 * DASH_LEN
        cx = x0 + ux * mid
        cy = y0 + uy * mid
        # Axis-aligned dash box oriented to the road (approx via extents)
        if abs(ux) >= abs(uy):
            # Mostly east–west: length along x
            half_x, half_y = hx, hy
        else:
            half_x, half_y = hy, hx
        gz = _sample_heightmap(terrain, cx, cy)
        z = gz + ROAD_LIFT + DASH_ABOVE_PAVEMENT
        marks_out.append(
            (float(cx), float(cy), float(half_x), float(half_y), float(z), list(DASH_COLOR))
        )
        s += step


def _add_sign(detail_parts, terrain, x, y, facing_x=True):
    """Post + board a bit above ground, clear of the roadway."""
    gz = _sample_heightmap(terrain, x, y)
    post_w = 0.08
    post_h = 2.4
    board_w = 1.1
    board_d = 0.06
    board_h = 0.7
    board_z0 = gz + 1.5
    detail_parts.append(
        (
            x - post_w,
            y - post_w,
            gz + 0.02,
            x + post_w,
            y + post_w,
            gz + post_h,
            list(SIGN_POST_COLOR),
        )
    )
    if facing_x:
        detail_parts.append(
            (
                x - board_d * 0.5,
                y - board_w * 0.5,
                board_z0,
                x + board_d * 0.5,
                y + board_w * 0.5,
                board_z0 + board_h,
                list(SIGN_BOARD_COLOR),
            )
        )
    else:
        detail_parts.append(
            (
                x - board_w * 0.5,
                y - board_d * 0.5,
                board_z0,
                x + board_w * 0.5,
                y + board_d * 0.5,
                board_z0 + board_h,
                list(SIGN_BOARD_COLOR),
            )
        )


class MetroLayout:
    """
    Two street-aligned downtowns on opposite sides of the map center, joined
    by one connector road with a midpoint gas station, center dashes, and
    scattered roadside signs.

    Presents the same box/road/detail API as TownLayout for spawning.
    """

    def __init__(
        self,
        downtown_size=250.0,
        seed=None,
        altitude_cap=40.0,
        city_a_center=(-800.0, 0.0),
        city_b_center=(800.0, 0.0),
    ):
        self.downtown_size = float(downtown_size)
        self.altitude_cap = float(altitude_cap)
        self.city_a_center = (float(city_a_center[0]), float(city_a_center[1]))
        self.city_b_center = (float(city_b_center[0]), float(city_b_center[1]))
        self.rng = np.random.default_rng(seed)
        seed0 = None if seed is None else int(seed)
        seed1 = None if seed is None else int(seed) + 101

        self.town_a = TownLayout(
            size=self.downtown_size,
            seed=seed0,
            altitude_cap=altitude_cap,
            origin=self.city_a_center,
        )
        self.town_b = TownLayout(
            size=self.downtown_size,
            seed=seed1,
            altitude_cap=altitude_cap,
            origin=self.city_b_center,
        )

        # Pick an E–W street near each downtown center and join them
        self.connector_y = self._pick_connector_y()
        half = self.downtown_size / 2.0
        ax, ay = self.city_a_center
        bx, by = self.city_b_center
        # West downtown is town_a when ax < bx
        if ax <= bx:
            self.connector_x0 = ax + half
            self.connector_x1 = bx - half
            self.west_town = self.town_a
            self.east_town = self.town_b
        else:
            self.connector_x0 = bx + half
            self.connector_x1 = ax - half
            self.west_town = self.town_b
            self.east_town = self.town_a

        self.street_width = float(self.town_a.street_width)
        self.size = float(
            max(
                abs(self.city_a_center[0]) + half,
                abs(self.city_b_center[0]) + half,
                abs(self.city_a_center[1]) + half,
                abs(self.city_b_center[1]) + half,
            )
            * 2.0
        )
        self.origin = (0.0, 0.0)

        self.boxes = np.zeros((0, 6), dtype=np.float64)
        self.styles: list[str] = []
        self.detail_parts: list[tuple] = []
        self.buildings: list[dict] = []
        self.building_types: list[str] = []
        self.overpass_indices: list[int] = []
        self.roads: list[tuple] = []
        self.road_marks: list[tuple] = []
        self.gas_station = None  # filled in apply_ground_heights
        self._ground_applied = False

        self._merge_towns()

    def _pick_connector_y(self):
        """Use the west downtown's near-center E–W street so the join is flush."""
        west = self.town_a if self.city_a_center[0] <= self.city_b_center[0] else self.town_b
        east = self.town_b if west is self.town_a else self.town_a
        oy = west.origin[1]
        y = min(west.street_centers_y, key=lambda v: abs(v - oy))
        # Ensure the east downtown also paves this corridor into its streets
        if not any(abs(sy - y) < 0.5 for sy in east.street_centers_y):
            east.street_centers_y.append(float(y))
            east.street_centers_y.sort()
        return float(y)

    def _merge_towns(self):
        """Concatenate building geometry from both downtowns."""
        parts = []
        styles = []
        details = []
        buildings = []
        btypes = []
        overpass = []
        offset = 0
        for town in (self.town_a, self.town_b):
            if town.boxes.size:
                parts.append(town.boxes.copy())
            styles.extend(town.styles)
            details.extend(list(town.detail_parts))
            for b in town.buildings:
                nb = dict(b)
                nb["box_indices"] = [i + offset for i in b["box_indices"]]
                buildings.append(nb)
            btypes.extend(town.building_types)
            overpass.extend(i + offset for i in town.overpass_indices)
            offset += len(town.styles)

        if parts:
            self.boxes = np.vstack(parts)
        else:
            self.boxes = np.zeros((0, 6), dtype=np.float64)
        self.styles = styles
        self.detail_parts = details
        self.buildings = buildings
        self.building_types = btypes
        self.overpass_indices = overpass

    def _flatten_connector(self, terrain):
        half_w = self.street_width / 2.0 + 0.75
        Z = terrain.heightmap
        gy, gx = Z.shape
        res = float(terrain.resolution)
        half_x = terrain.size_x / 2.0
        half_y = terrain.size_y / 2.0
        xs = (np.arange(gx) + 0.5) * res - half_x
        ys = (np.arange(gy) + 0.5) * res - half_y
        XX, YY = np.meshgrid(xs, ys)
        mask = (
            (XX >= self.connector_x0 - 1.0)
            & (XX <= self.connector_x1 + 1.0)
            & (np.abs(YY - self.connector_y) <= half_w)
        )
        if not mask.any():
            return
        try:
            from .terrain import _box_blur
        except Exception:
            return
        radius = max(1, int(round(2.0 / res)))
        smooth = _box_blur(Z, radius=radius, passes=3)
        Z[mask] = smooth[mask]

    def _build_connector_roads(self, terrain):
        half_w = self.street_width / 2.0
        seg = ROAD_SEG_LEN
        cy = self.connector_y
        roads = []
        x0 = self.connector_x0
        while x0 < self.connector_x1 - 1e-6:
            x1 = min(x0 + seg, self.connector_x1)
            hx = 0.5 * (x1 - x0)
            if hx < 0.05:
                break
            cx = 0.5 * (x0 + x1)
            z = _elev_points(
                terrain, _sample_rect_points(x0, x1, cy, half_w, along_is_y=False)
            )
            roads.append((float(cx), float(cy), float(hx), float(half_w), z))
            x0 = x1
        return roads

    def _place_gas_station(self, terrain):
        """Canopy + shop + pumps beside the connector midpoint (not in roadway)."""
        mx = 0.5 * (self.connector_x0 + self.connector_x1)
        my = self.connector_y
        half_w = self.street_width / 2.0
        # Sit north of the road
        side = 1.0 if self.rng.random() < 0.5 else -1.0
        lot_y = my + side * (half_w + 14.0)
        gz = _sample_heightmap(terrain, mx, lot_y)

        # Shop box (colliding)
        shop_w, shop_d, shop_h = 8.0, 6.0, 3.5
        sx0 = mx - shop_w * 0.5
        sx1 = mx + shop_w * 0.5
        # Shop farther from road than pumps
        sy_shop = lot_y + side * 6.0
        sy0 = sy_shop - shop_d * 0.5
        sy1 = sy_shop + shop_d * 0.5
        self._add_box(sx0, sy0, gz, sx1, sy1, gz + shop_h, "concrete")

        # Canopy (visual)
        canopy_w, canopy_d, canopy_h = 12.0, 8.0, 0.35
        canopy_z = gz + 4.2
        cy_can = lot_y - side * 1.5
        self.detail_parts.append(
            (
                mx - canopy_w * 0.5,
                cy_can - canopy_d * 0.5,
                canopy_z,
                mx + canopy_w * 0.5,
                cy_can + canopy_d * 0.5,
                canopy_z + canopy_h,
                list(GAS_CANOPY_COLOR),
            )
        )
        # Canopy posts
        for px in (mx - 4.5, mx + 4.5):
            for py in (cy_can - 3.0, cy_can + 3.0):
                self.detail_parts.append(
                    (
                        px - 0.12,
                        py - 0.12,
                        gz + 0.02,
                        px + 0.12,
                        py + 0.12,
                        canopy_z,
                        list(SIGN_POST_COLOR),
                    )
                )

        # Pump boxes under canopy
        pump_w, pump_d, pump_h = 1.0, 0.7, 1.4
        for i, ox in enumerate((-3.0, 3.0)):
            px = mx + ox
            py = cy_can
            self.detail_parts.append(
                (
                    px - pump_w * 0.5,
                    py - pump_d * 0.5,
                    gz + 0.02,
                    px + pump_w * 0.5,
                    py + pump_d * 0.5,
                    gz + pump_h,
                    list(GAS_PUMP_COLOR),
                )
            )

        self.gas_station = {
            "midpoint": (float(mx), float(my)),
            "lot": (float(mx), float(lot_y)),
            "side": float(side),
        }

    def _add_box(self, xmin, ymin, zmin, xmax, ymax, zmax, style):
        row = np.array([[xmin, ymin, zmin, xmax, ymax, zmax]], dtype=np.float64)
        self.boxes = row if self.boxes.size == 0 else np.vstack([self.boxes, row])
        self.styles.append(style)
        return len(self.styles) - 1

    def _build_dashes_and_signs(self, terrain):
        marks = []
        # Connector center line
        _dashes_along_line(
            terrain,
            self.connector_x0,
            self.connector_y,
            self.connector_x1,
            self.connector_y,
            marks,
        )

        # Main downtown streets: near-center N–S and E–W per town
        for town in (self.town_a, self.town_b):
            ox, oy = town.origin
            half = town.size / 2.0
            # Primary E–W (connector-aligned) and primary N–S (through origin)
            main_ey = min(town.street_centers_y, key=lambda y: abs(y - oy))
            main_sx = min(town.street_centers_x, key=lambda x: abs(x - ox))
            _dashes_along_line(
                terrain, ox - half, main_ey, ox + half, main_ey, marks
            )
            _dashes_along_line(
                terrain, main_sx, oy - half, main_sx, oy + half, marks
            )

        self.road_marks = marks

        # Modest roadside signs along connector + main streets
        sign_spacing = 140.0
        rng = self.rng
        half_w = self.street_width / 2.0

        def scatter_signs(x0, y0, x1, y1, along_x):
            length = float(np.hypot(x1 - x0, y1 - y0))
            if length < 40.0:
                return
            n = max(1, int(length / sign_spacing))
            for i in range(n):
                if rng.random() < 0.25:
                    continue  # skip some so it is not regular
                t = (i + 0.5) / n
                t += float(rng.uniform(-0.08, 0.08))
                t = float(np.clip(t, 0.05, 0.95))
                x = x0 + (x1 - x0) * t
                y = y0 + (y1 - y0) * t
                side = 1.0 if rng.random() < 0.5 else -1.0
                if along_x:
                    sx, sy = x, y + side * (half_w + 2.2)
                    facing_x = True
                else:
                    sx, sy = x + side * (half_w + 2.2), y
                    facing_x = False
                _add_sign(self.detail_parts, terrain, sx, sy, facing_x=facing_x)

        scatter_signs(
            self.connector_x0,
            self.connector_y,
            self.connector_x1,
            self.connector_y,
            along_x=True,
        )
        for town in (self.town_a, self.town_b):
            ox, oy = town.origin
            half = town.size / 2.0
            main_ey = min(town.street_centers_y, key=lambda y: abs(y - oy))
            main_sx = min(town.street_centers_x, key=lambda x: abs(x - ox))
            scatter_signs(ox - half, main_ey, ox + half, main_ey, along_x=True)
            scatter_signs(main_sx, oy - half, main_sx, oy + half, along_x=False)

    def apply_ground_heights(self, terrain):
        """Lift both downtowns, pave the connector, place gas / dashes / signs."""
        if self._ground_applied:
            return

        # Flatten street ribbons in each downtown + connector corridor
        self.town_a._flatten_street_heightmap(terrain)
        self.town_b._flatten_street_heightmap(terrain)
        self._flatten_connector(terrain)

        # Re-merge boxes/details from towns (still at z relative / local base)
        # Towns have not applied ground yet — lift unified lists ourselves.
        self._merge_towns()

        def lift_box_row(box, style):
            xmin, ymin, zmin, xmax, ymax, zmax = box
            cx = 0.5 * (xmin + xmax)
            cy = 0.5 * (ymin + ymax)
            gz = _sample_heightmap(terrain, cx, cy)
            if style == "overpass":
                clearance = zmin
                thickness = zmax - zmin
                return (
                    xmin,
                    ymin,
                    gz + clearance,
                    xmax,
                    ymax,
                    gz + clearance + thickness,
                )
            return (xmin, ymin, gz + zmin, xmax, ymax, gz + zmin + (zmax - zmin))

        if self.boxes.size:
            new_boxes = self.boxes.copy()
            for i, box in enumerate(self.boxes):
                style = self.styles[i] if i < len(self.styles) else "concrete"
                new_boxes[i] = lift_box_row(box, style)
            self.boxes = new_boxes

        if self.detail_parts:
            lifted_details = []
            for xmin, ymin, zmin, xmax, ymax, zmax, rgba in self.detail_parts:
                cx = 0.5 * (xmin + xmax)
                cy = 0.5 * (ymin + ymax)
                gz = _sample_heightmap(terrain, cx, cy)
                z0 = zmin if zmin > 1e-6 else 0.02
                lifted_details.append(
                    (xmin, ymin, gz + z0, xmax, ymax, gz + zmax, rgba)
                )
            self.detail_parts = lifted_details

        # Mark towns applied so they are not double-lifted if reused
        self.town_a._ground_applied = True
        self.town_b._ground_applied = True
        self.town_a.roads = self.town_a._build_road_segments(terrain)
        self.town_b.roads = self.town_b._build_road_segments(terrain)

        self.roads = list(self.town_a.roads) + list(self.town_b.roads)
        self.roads.extend(self._build_connector_roads(terrain))

        self._place_gas_station(terrain)
        self._build_dashes_and_signs(terrain)
        self._ground_applied = True

    def is_walkable(self, x, y):
        return self.town_a.is_walkable(x, y) or self.town_b.is_walkable(x, y)

    def sample_walkable(self, rng, n=1, prefer_street=True):
        n_a = n // 2
        n_b = n - n_a
        pts_a = self.town_a.sample_walkable(rng, n=max(n_a, 1), prefer_street=prefer_street)
        pts_b = self.town_b.sample_walkable(rng, n=max(n_b, 1), prefer_street=prefer_street)
        pts = np.vstack([pts_a[:n_a], pts_b[:n_b]]) if n > 1 else pts_a
        return pts[:n]

    def footprints_2d(self):
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


# Afternoon sun: incoming from high SW so shadows fall NE.
# +X east, +Y north, +Z up. Unit vector points toward the sun.
_SUN_DIR = np.array([-0.45, -0.40, 0.80], dtype=np.float64)
_SUN_DIR /= np.linalg.norm(_SUN_DIR)
# Warm afternoon, not white-hot. Printed even if this PyBullet build
# has no lightColor kwarg on configureDebugVisualizer.
_LIGHT_COLOR = (1.0, 0.90, 0.70)
_SKY_RGB = (0.46, 0.70, 0.94)
# Shadow camera looks from (lightPos + cameraTarget) at the target; the
# ortho far plane is 300 m, so this offset must stay well under 300.
_LIGHT_OFFSET_M = 110.0
_SUN_DISC_DIST_M = 620.0
_SUN_DISC_RADIUS_M = 18.0
_SHADOW_MAP_RES = 2048
_SHADOW_MAP_WORLD = 400
_SHADOW_INTENSITY = 0.28


def sunlight_direction():
    """Unit vector from the scene toward the afternoon sun (high SW)."""
    return _SUN_DIR.copy()


def _print_sun_line(shadows):
    d = _SUN_DIR
    c = _LIGHT_COLOR
    print(
        f"[SUN] direction=({d[0]:.3f}, {d[1]:.3f}, {d[2]:.3f})  "
        f"color=({c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f})  "
        f"shadows={'on' if shadows else 'off'}"
    )


_SUN_BODY = None


def spawn_sun_disc(client, origin=(0.0, 0.0, 0.0)):
    """Visual-only sun disc along the incoming light, inside GUI draw distance.

    `origin` is the city/camera focus. The disc sits 620 m toward the SW
    sun so a spectator over downtown still sees it inside the ~1000 m far
    plane. Calling again moves the existing disc if that body is still live.
    """
    global _SUN_BODY
    if p is None or client is None:
        return None
    pos = np.asarray(origin, dtype=np.float64) + _SUN_DIR * _SUN_DISC_DIST_M
    if _SUN_BODY is not None:
        try:
            p.resetBasePositionAndOrientation(
                _SUN_BODY,
                pos.tolist(),
                [0.0, 0.0, 0.0, 1.0],
                physicsClientId=client,
            )
            return _SUN_BODY
        except Exception:
            _SUN_BODY = None
    vis = p.createVisualShape(
        p.GEOM_SPHERE,
        radius=_SUN_DISC_RADIUS_M,
        rgbaColor=[1.0, 0.92, 0.38, 1.0],
        specularColor=[1.0, 0.95, 0.55],
        physicsClientId=client,
    )
    _SUN_BODY = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=vis,
        basePosition=pos.tolist(),
        physicsClientId=client,
    )
    return _SUN_BODY


def configure_sunlight(client, shadows=True):
    """GUI directional sun, warm fill, sky tint, and a town-scale shadow map."""
    if p is None or client is None:
        _print_sun_line(False)
        return
    shadows = bool(shadows)
    p.configureDebugVisualizer(
        p.COV_ENABLE_SHADOWS, 1 if shadows else 0, physicsClientId=client
    )
    light_pos = (_SUN_DIR * _LIGHT_OFFSET_M).tolist()
    kwargs = dict(
        lightPosition=light_pos,
        shadowMapResolution=_SHADOW_MAP_RES,
        shadowMapWorldSize=_SHADOW_MAP_WORLD,
        shadowMapIntensity=_SHADOW_INTENSITY,
        rgbBackground=list(_SKY_RGB),
        physicsClientId=client,
    )
    try:
        p.configureDebugVisualizer(lightColor=list(_LIGHT_COLOR), **kwargs)
    except TypeError:
        p.configureDebugVisualizer(**kwargs)
    _print_sun_line(shadows)


def apply_spectator_sun(client, shadows=True, spawn_disc=True, origin=(0.0, 0.0, 0.0)):
    """Apply spectator lighting. Headless callers must pass shadows=False and spawn_disc=False."""
    if shadows or spawn_disc:
        configure_sunlight(client, shadows=shadows)
        if spawn_disc:
            return spawn_sun_disc(client, origin=origin)
        return None
    _print_sun_line(False)
    return None


def connect_pybullet(gui=True, shadows=None):
    """Open a PyBullet client. GUI gets an afternoon sun and shadows; DIRECT does not."""
    global _SUN_BODY
    _SUN_BODY = None
    if shadows is None:
        shadows = bool(gui)
    client = p.connect(p.GUI if gui else p.DIRECT)
    if gui:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=client)
        # Disable built-in W/wireframe and other GUI hotkeys so app keys win.
        p.configureDebugVisualizer(
            p.COV_ENABLE_KEYBOARD_SHORTCUTS, 0, physicsClientId=client
        )
        p.configureDebugVisualizer(
            p.COV_ENABLE_WIREFRAME, 0, physicsClientId=client
        )
        apply_spectator_sun(client, shadows=bool(shadows), spawn_disc=True)
    else:
        apply_spectator_sun(client, shadows=False, spawn_disc=False)
    p.setGravity(0, 0, -9.81, physicsClientId=client)
    return client


def spawn_town_in_pybullet(client, town, terrain, env_size):
    """
    Load one dusty heightfield plus elevated road segments and building visuals.

    The heightfield is the only full-map ground surface (collision + visual).
    Roads are pavement slabs ~ROAD_LIFT above local ground — not a second
    ground pad. Center dashes are baked into the ground texture (not bodies).
    No city-wide or rim color plates. Returns
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

    # PyBullet GUI shared-memory upload refuses float buffers > 1 MiB.
    hf_bytes = int(terrain.grid_x) * int(terrain.grid_y) * 4
    if hf_bytes >= 1_000_000:
        raise ValueError(
            f"heightfield upload {hf_bytes} bytes "
            f"({terrain.grid_x}x{terrain.grid_y}) exceeds safe 1 MB limit; "
            f"coarsen resolution (got {terrain.resolution} m)"
        )

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

    # Ground texture: dusty city feathers to grass, plus yellow center dashes
    # baked in (no per-dash MultiBody). White tint so rgba does not multiply
    # brown on top of the texture colors.
    texture_id = -1
    if hasattr(terrain, "get_ground_texture_path"):
        try:
            marks = getattr(town, "road_marks", None) or None
            tex_path = terrain.get_ground_texture_path(road_marks=marks)
            texture_id = p.loadTexture(tex_path, physicsClientId=client)
        except Exception:
            texture_id = -1

    if texture_id >= 0:
        p.changeVisualShape(
            terrain_body,
            -1,
            rgbaColor=[1.0, 1.0, 1.0, 1.0],
            textureUniqueId=texture_id,
            specularColor=[0.08, 0.08, 0.08],
            physicsClientId=client,
        )
    else:
        p.changeVisualShape(
            terrain_body,
            -1,
            rgbaColor=[0.45, 0.38, 0.28, 1],
            specularColor=[0.08, 0.08, 0.08],
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

    # Center-line dashes are drawn into the ground texture (not spawned).

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
