"""
terrain.py — Ground heightmap plus texture, either procedural (dusty city,
grass belt, mountain rim) or loaded from a real elevation GeoTIFF via
Terrain.from_dem (USGS bare-earth DEM, heights used as-is from the
loaded patch once — no in-memory tiling).

Grid convention: heightmap[i, j] with i along +Y (row 0 = south edge,
north = +Y) and j along +X. That is also PyBullet's heightfield order when
numHeightfieldRows = grid_x and numHeightfieldColumns = grid_y.
"""

import glob
import numpy as np
import os
import struct
import tarfile
import zlib

from .town import TownLayout

# Dusty town center (matches previous flat tint)
_DUST_RGB = np.array([0.45, 0.38, 0.28], dtype=np.float64)
# Soft natural grass (no neon)
_GRASS_RGB = np.array([0.32, 0.46, 0.28], dtype=np.float64)
# Mountain rock (gray-brown) and thin snow caps
_ROCK_RGB = np.array([0.47, 0.43, 0.38], dtype=np.float64)
_SNOW_RGB = np.array([0.90, 0.91, 0.93], dtype=np.float64)


def _smoothstep(edge0, edge1, x):
    t = np.clip((x - edge0) / max(edge1 - edge0, 1e-12), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _signed_distance_to_square(X, Y, half):
    """SDF of an axis-aligned square; outside isocontours have rounded corners."""
    qx = np.abs(X) - half
    qy = np.abs(Y) - half
    outside = np.sqrt(np.maximum(qx, 0.0) ** 2 + np.maximum(qy, 0.0) ** 2)
    inside = np.minimum(np.maximum(qx, qy), 0.0)
    return outside + inside


def _distance_to_segment(X, Y, a, b):
    """Euclidean distance from grid points to the segment a-b."""
    ax, ay = a
    bx, by = b
    vx, vy = bx - ax, by - ay
    denom = max(vx * vx + vy * vy, 1e-12)
    t = np.clip(((X - ax) * vx + (Y - ay) * vy) / denom, 0.0, 1.0)
    return np.hypot(X - (ax + t * vx), Y - (ay + t * vy))


def _upsample_bilinear(coarse, out_h, out_w):
    """Bilinear resize of a 2D array to (out_h, out_w)."""
    src_h, src_w = coarse.shape
    if src_h == out_h and src_w == out_w:
        return coarse.astype(np.float64)

    ys = np.linspace(0.0, src_h - 1.0, out_h)
    xs = np.linspace(0.0, src_w - 1.0, out_w)
    y0 = np.floor(ys).astype(np.int64)
    x0 = np.floor(xs).astype(np.int64)
    y1 = np.minimum(y0 + 1, src_h - 1)
    x1 = np.minimum(x0 + 1, src_w - 1)
    wy = ys - y0
    wx = xs - x0

    top = coarse[y0][:, x0] * (1.0 - wx) + coarse[y0][:, x1] * wx
    bot = coarse[y1][:, x0] * (1.0 - wx) + coarse[y1][:, x1] * wx
    # Broadcast row blend: wy is (out_h,), top/bot are (out_h, out_w)
    return top * (1.0 - wy)[:, None] + bot * wy[:, None]


def _box_blur(Z, radius, passes=2):
    """Separable box blur (uniform kernel) applied several times ≈ Gaussian."""
    if radius <= 0:
        return Z.astype(np.float64)
    out = Z.astype(np.float64)
    k = 2 * radius + 1
    for _ in range(passes):
        # Horizontal (extra left pad so cumsum window length stays W)
        padded = np.pad(out, ((0, 0), (radius + 1, radius)), mode="edge")
        csum = np.cumsum(padded, axis=1)
        out = (csum[:, k:] - csum[:, :-k]) / k
        # Vertical
        padded = np.pad(out, ((radius + 1, radius), (0, 0)), mode="edge")
        csum = np.cumsum(padded, axis=0)
        out = (csum[k:, :] - csum[:-k, :]) / k
    return out


def _limit_slopes(Z, max_slope, resolution, passes=12):
    """Soften steep gradients by blending toward neighbors where too steep."""
    if max_slope <= 0:
        return Z.astype(np.float64)
    out = Z.astype(np.float64).copy()
    max_dz = float(max_slope) * float(resolution)
    for _ in range(passes):
        up = np.empty_like(out)
        down = np.empty_like(out)
        left = np.empty_like(out)
        right = np.empty_like(out)
        up[0, :] = out[0, :]
        up[1:, :] = out[:-1, :]
        down[-1, :] = out[-1, :]
        down[:-1, :] = out[1:, :]
        left[:, 0] = out[:, 0]
        left[:, 1:] = out[:, :-1]
        right[:, -1] = out[:, -1]
        right[:, :-1] = out[:, 1:]
        # Clamp each cell toward neighbors so |Δ| ≤ max_dz
        out = np.minimum(out, up + max_dz)
        out = np.maximum(out, up - max_dz)
        out = np.minimum(out, down + max_dz)
        out = np.maximum(out, down - max_dz)
        out = np.minimum(out, left + max_dz)
        out = np.maximum(out, left - max_dz)
        out = np.minimum(out, right + max_dz)
        out = np.maximum(out, right - max_dz)
    return out


def _write_png_rgb(path, rgb_u8):
    """Write an 8-bit RGB PNG using only the stdlib (struct + zlib)."""
    height, width, channels = rgb_u8.shape
    if channels != 3:
        raise ValueError("expected HxWx3 RGB array")
    raw = bytearray()
    for row in range(height):
        raw.append(0)  # filter: None
        raw.extend(rgb_u8[row].tobytes())

    def chunk(tag, data):
        crc = zlib.crc32(tag)
        crc = zlib.crc32(data, crc) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", ihdr)
    png += chunk(b"IDAT", zlib.compress(bytes(raw), 9))
    png += chunk(b"IEND", b"")
    with open(path, "wb") as f:
        f.write(png)


# PyBullet GUI uploads the heightfield through 1 MiB shared memory.
DEM_MAX_GRID_SIDE = 512
DEM_MIN_CELL_M = 8.0
DEFAULT_DEM_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
)


def find_dem_file(data_dir=DEFAULT_DEM_DIR):
    """
    Return the single .tif in data_dir. If there is none but exactly one
    .tar.gz, extract its .tif members into data_dir (archive is kept).
    """
    def _tifs():
        return sorted(
            glob.glob(os.path.join(data_dir, "*.tif"))
            + glob.glob(os.path.join(data_dir, "*.tiff"))
        )

    listing = sorted(os.listdir(data_dir)) if os.path.isdir(data_dir) else None
    tifs = _tifs()
    if not tifs:
        archives = sorted(glob.glob(os.path.join(data_dir, "*.tar.gz")))
        if len(archives) == 1:
            with tarfile.open(archives[0], "r:gz") as tar:
                members = [
                    m for m in tar.getmembers()
                    if m.isfile() and m.name.lower().endswith((".tif", ".tiff"))
                ]
                for m in members:
                    m.name = os.path.basename(m.name)
                    tar.extract(m, data_dir)
            tifs = _tifs()
    if len(tifs) != 1:
        raise FileNotFoundError(
            f"expected exactly one DEM .tif in {data_dir}; found {len(tifs)} "
            f"(directory listing: {listing})"
        )
    return tifs[0]


def _meters_per_degree(lat_deg):
    """(m per degree longitude, m per degree latitude) on the GRS80 ellipsoid."""
    a = 6378137.0
    f = 1.0 / 298.257222101
    e2 = f * (2.0 - f)
    phi = np.radians(lat_deg)
    w = 1.0 - e2 * np.sin(phi) ** 2
    m_lat = np.pi / 180.0 * a * (1.0 - e2) / w**1.5
    m_lon = np.pi / 180.0 * a * np.cos(phi) / np.sqrt(w)
    return float(m_lon), float(m_lat)


def _fill_nodata(Z, invalid):
    """Fill invalid cells inward from valid neighbors (8-neighbor mean, onion peel)."""
    Z = np.array(Z, dtype=np.float64)
    invalid = np.array(invalid, dtype=bool)
    if invalid.all():
        raise ValueError("DEM has no valid elevation cells")
    Z[invalid] = 0.0
    while invalid.any():
        valid_f = (~invalid).astype(np.float64)
        vals = np.pad(Z * valid_f, 1)
        wts = np.pad(valid_f, 1)
        h, w = Z.shape
        s = np.zeros_like(Z)
        n = np.zeros_like(Z)
        for di in (0, 1, 2):
            for dj in (0, 1, 2):
                if di == 1 and dj == 1:
                    continue
                s += vals[di:di + h, dj:dj + w]
                n += wts[di:di + h, dj:dj + w]
        grow = invalid & (n > 0)
        Z[grow] = s[grow] / n[grow]
        invalid &= ~grow
    return Z


def _area_downsample(Z, k):
    """Mean over k x k blocks; partial edge blocks average only their cells."""
    if k <= 1:
        return Z.astype(np.float64)
    h, w = Z.shape
    oh, ow = -(-h // k), -(-w // k)
    padded = np.full((oh * k, ow * k), np.nan, dtype=np.float64)
    padded[:h, :w] = Z
    return np.nanmean(padded.reshape(oh, k, ow, k), axis=(1, 3))


def load_dem(path, max_side=DEM_MAX_GRID_SIDE, min_cell=DEM_MIN_CELL_M):
    """
    Read band 1 of an elevation GeoTIFF with rasterio.

    Returns a dict with the filled, possibly area-downsampled elevation grid
    in raster order (row 0 = north), cell sizes in meters, and metadata.
    """
    import rasterio

    with rasterio.open(path) as ds:
        raw = ds.read(1).astype(np.float64)
        transform = ds.transform
        crs = ds.crs
        nodata = ds.nodata
        bounds = ds.bounds

    invalid = ~np.isfinite(raw)
    if nodata is not None:
        invalid |= raw == nodata
    n_filled = int(invalid.sum())
    elev = _fill_nodata(raw, invalid) if n_filled else raw

    px_x = abs(float(transform.a))
    px_y = abs(float(transform.e))
    if crs is not None and crs.is_geographic:
        lat_c = 0.5 * (bounds.top + bounds.bottom)
        m_lon, m_lat = _meters_per_degree(lat_c)
        cell_x, cell_y = px_x * m_lon, px_y * m_lat
    else:
        unit = crs.linear_units_factor[1] if crs is not None else 1.0
        cell_x, cell_y = px_x * unit, px_y * unit
    # Raster row 0 must be the north edge for the flip to +Y.
    if transform.e >= 0 or transform.a <= 0 or transform.b or transform.d:
        raise ValueError(f"unsupported DEM orientation / rotation: {transform}")

    rows, cols = elev.shape
    k = max(
        1,
        int(np.ceil(max(rows, cols) / float(max_side))),
        int(np.ceil(min_cell / min(cell_x, cell_y) - 1e-9)),
    )
    grid = _area_downsample(elev, k)
    return {
        "path": path,
        "name": os.path.basename(path),
        "raster_shape": (rows, cols),
        "native_pixel": (px_x, px_y),
        "native_cell_m": (cell_x, cell_y),
        "crs": crs.to_string() if crs is not None else "none",
        "crs_is_geographic": bool(crs is not None and crs.is_geographic),
        "transform": transform,
        "bounds": tuple(bounds),
        "nodata_value": nodata,
        "nodata_filled": n_filled,
        "downsample": k,
        "elevation": grid,
        "cell_m": (cell_x * k, cell_y * k),
    }


class Terrain:
    def __init__(
        self,
        size_x=3250,
        size_y=3250,
        # 8 m cells keep float heightfield under PyBullet GUI 1 MiB SHM limit
        # (side ≤ 512; 3250/8 → ~406 → ~659 KB).
        resolution=8.0,
        seed=None,
        flat=False,
        with_structures=False,
        obstacle_boxes=None,
        city_size=250.0,
        city_centers=None,
        blend_width=40.0,
        color_blend_width=55.0,
        # Quiet meadow undulation (almost flat)
        meadow_amp=0.30,
        # Gentle rolling where the region mask allows
        grass_cell=120.0,
        grass_amp=2.2,
        # Low-frequency region mask (~cell size in meters)
        region_cell=220.0,
        # Distinct hill clusters (meters) — broad and smooth
        hill_amp_min=5.0,
        hill_amp_max=11.0,
        n_hill_clusters=(5, 10),
        hill_sigma_min=90.0,
        hill_sigma_max=160.0,
        # City-edge fade for all grass relief (starts after blend_width)
        hill_fade=55.0,
        max_slope=0.12,
        # Belt landforms on top of the fine relief (meters). Not slope-limited
        # by max_slope; shaped so Gaussian slopes stay ≲ 0.3.
        belt_region_cell=520.0,
        roll_cell=320.0,
        roll_amp=(14.0, 20.0),
        tall_hill_amp=(25.0, 50.0),
        n_tall_hills=(6, 10),
        belt_fade=200.0,
        # Smooth road corridor ((x0, y0), (x1, y1)); "auto" joins the two
        # city centers, None disables.
        connector_segment="auto",
        connector_flat=35.0,
        connector_fade=160.0,
        # Mountain rim along the world border
        rim_width=400.0,
        n_rim_ranges=(7, 10),
        rim_peak_amp=(80.0, 160.0),
        rim_tall_amp=(165.0, 185.0),
        rim_tall_prob=0.15,
        # Per-axis cap; diagonal gradients land near 0.4
        rim_max_slope=0.34,
        # Search pad: fade pavement→grass and a LOW berm on the outer band
        # of each city square (not the world rim).
        # Not the mountain rim (leave rim_width=0 for that).
        edge_grass_width=0.0,
        edge_rise_amp=0.0,
        # Soften the world border so the map does not end as a cliff.
        # 0 leaves the rim as generated (metro / old 250 m pad).
        outer_fade_width=0.0,
        # Min city-outside distance for a rim peak. None = size-adaptive.
        rim_min_city_dist=None,
    ):
        self.size_x = float(size_x)
        self.size_y = float(size_y)
        self.resolution = float(resolution)
        self.city_size = float(city_size)
        if city_centers is None:
            self.city_centers = [(0.0, 0.0)]
        else:
            self.city_centers = [
                (float(c[0]), float(c[1])) for c in city_centers
            ]
        self.blend_width = float(blend_width)
        self.color_blend_width = float(color_blend_width)
        self.meadow_amp = float(meadow_amp)
        self.grass_cell = float(grass_cell)
        self.grass_amp = float(grass_amp)
        self.region_cell = float(region_cell)
        self.hill_amp_min = float(hill_amp_min)
        self.hill_amp_max = float(hill_amp_max)
        self.n_hill_clusters = tuple(n_hill_clusters)
        self.hill_sigma_min = float(hill_sigma_min)
        self.hill_sigma_max = float(hill_sigma_max)
        self.hill_fade = float(hill_fade)
        self.max_slope = float(max_slope)
        self.belt_region_cell = float(belt_region_cell)
        self.roll_cell = float(roll_cell)
        self.roll_amp = tuple(float(v) for v in roll_amp)
        self.tall_hill_amp = tuple(float(v) for v in tall_hill_amp)
        self.n_tall_hills = tuple(n_tall_hills)
        self.belt_fade = float(belt_fade)
        if connector_segment == "auto":
            connector_segment = (
                tuple(self.city_centers[:2]) if len(self.city_centers) == 2 else None
            )
        self.connector_segment = connector_segment
        self.connector_flat = float(connector_flat)
        self.connector_fade = float(connector_fade)
        self.rim_width = float(rim_width)
        self.n_rim_ranges = tuple(n_rim_ranges)
        self.rim_peak_amp = tuple(float(v) for v in rim_peak_amp)
        self.rim_tall_amp = tuple(float(v) for v in rim_tall_amp)
        self.rim_tall_prob = float(rim_tall_prob)
        self.rim_max_slope = float(rim_max_slope)
        self.edge_grass_width = float(edge_grass_width)
        self.edge_rise_amp = float(edge_rise_amp)
        self.outer_fade_width = float(outer_fade_width)
        self.rim_min_city_dist = (
            None if rim_min_city_dist is None else float(rim_min_city_dist)
        )
        self.grid_x = int(round(self.size_x / self.resolution))
        self.grid_y = int(round(self.size_y / self.resolution))
        self.resolution_x = self.resolution
        self.resolution_y = self.resolution
        self.source = "procedural"
        self.dem_info = None
        self.z_offset = 0.0
        self._seed = seed
        self.flat = bool(flat)
        self.obstacle_boxes = (
            np.asarray(obstacle_boxes, dtype=np.float64)
            if obstacle_boxes is not None and len(obstacle_boxes) > 0
            else np.zeros((0, 6), dtype=np.float64)
        )
        # Relative grass relief (above shared grade) — used for texture tint
        self._grass_relief = None
        # Fine relief + belt landforms + rim (above grade) — rock / snow tint
        self._relief = None
        self.peak_height_range = (0.0, 0.0)
        self.structures = []
        if self.flat:
            # Search trainer still asks for a flat field. Skip the city relief.
            self.heightmap = np.zeros((self.grid_y, self.grid_x), dtype=np.float64)
        else:
            self.heightmap = self._generate_heightmap(seed)
        if with_structures:
            from .structures import generate_earthquake_layout

            self.structures = generate_earthquake_layout(
                self, seed=seed if seed is not None else 42
            )
        self._texture_path = None
        self._cache_height_axes()

    def _cache_height_axes(self):
        """World coordinates of the heightmap columns and rows. Built once."""
        self._axis_x, self._axis_y = self.vertex_axes()

    @staticmethod
    def _fract_index(axis, value):
        """Same fractional index np.interp would return, without allocating an arange."""
        n = axis.shape[0]
        if n <= 1 or value <= axis[0]:
            return 0.0
        last = n - 1
        if value >= axis[last]:
            return float(last)
        j = int(np.searchsorted(axis, value, side="right") - 1)
        if j < 0:
            return 0.0
        if j >= last:
            return float(last)
        span = float(axis[j + 1] - axis[j])
        if span == 0.0:
            return float(j)
        return j + (float(value) - float(axis[j])) / span

    @classmethod
    def from_dem(cls, path=None, max_side=DEM_MAX_GRID_SIDE, min_cell=DEM_MIN_CELL_M):
        """
        Terrain whose heightmap is a real elevation raster, used directly
        (no procedural relief). One vertex per (downsampled) pixel center.

        After the GeoTIFF is loaded (file left unchanged), the patch is
        used once (no tiling). The footprint is centered on the origin,
        north stays +Y, east stays +X, and world z is elevation minus
        the lowest point.
        """
        if path is None:
            path = find_dem_file()
        info = load_dem(path, max_side=max_side, min_cell=min_cell)
        self = cls.__new__(cls)
        self.source = "dem"
        self.flat = False
        self.structures = []
        self.dem_info = info
        # Raster row 0 is north; flip so heightmap row 0 is the south (-Y) edge.
        patch = np.ascontiguousarray(np.flipud(info["elevation"]))
        # Single USGS patch, no tiling. ~175 x 93 stays under the 512-side /
        # 1 MiB PyBullet heightfield limit.
        self.dem_tile = (1, 1)
        elev = patch
        self.z_offset = float(np.min(elev))
        self.heightmap = elev - self.z_offset
        self.grid_y, self.grid_x = self.heightmap.shape
        self.resolution_x, self.resolution_y = (float(v) for v in info["cell_m"])
        self.resolution = min(self.resolution_x, self.resolution_y)
        self.size_x = self.grid_x * self.resolution_x
        self.size_y = self.grid_y * self.resolution_y
        self.city_centers = []
        self.obstacle_boxes = np.zeros((0, 6), dtype=np.float64)
        self._seed = None
        self._grass_relief = None
        self._relief = None
        self.peak_height_range = (0.0, float(np.max(self.heightmap)))
        self._texture_path = None
        self._cache_height_axes()
        return self

    def vertex_axes(self):
        """World X coords of heightmap columns and Y coords of heightmap rows."""
        if self.source == "dem":
            xs = -self.size_x / 2 + (np.arange(self.grid_x) + 0.5) * self.resolution_x
            ys = -self.size_y / 2 + (np.arange(self.grid_y) + 0.5) * self.resolution_y
            return xs, ys
        xs = np.linspace(-self.size_x / 2, self.size_x / 2, self.grid_x)
        ys = np.linspace(-self.size_y / 2, self.size_y / 2, self.grid_y)
        return xs, ys

    def get_height(self, x, y):
        """Bilinear ground height (heightmap z) at world (x, y), clamped to the grid."""
        xs = self._axis_x
        ys = self._axis_y
        fx = self._fract_index(xs, float(x))
        fy = self._fract_index(ys, float(y))
        j0, i0 = int(fx), int(fy)
        if j0 >= self.grid_x:
            j0 = self.grid_x - 1
        if i0 >= self.grid_y:
            i0 = self.grid_y - 1
        j1 = j0 + 1 if j0 + 1 < self.grid_x else j0
        i1 = i0 + 1 if i0 + 1 < self.grid_y else i0
        tx, ty = fx - j0, fy - i0
        Z = self.heightmap
        top = Z[i0, j0] * (1.0 - tx) + Z[i0, j1] * tx
        bot = Z[i1, j0] * (1.0 - tx) + Z[i1, j1] * tx
        return float(top * (1.0 - ty) + bot * ty)

    def world_to_crs(self, x, y):
        """DEM only: world (x, y) -> raster CRS coordinates (e.g. lon, lat)."""
        info = self.dem_info
        cx, cy = info["native_cell_m"]
        # Anchored at the raster's NW corner (downsample padding is south/east).
        col = (x + self.size_x / 2) / cx
        row = (self.size_y / 2 - y) / cy
        return info["transform"] * (col, row)

    def set_obstacle_boxes(self, boxes):
        """Attach town building/overpass AABBs used by check_los."""
        if boxes is None or len(boxes) == 0:
            self.obstacle_boxes = np.zeros((0, 6), dtype=np.float64)
        else:
            self.obstacle_boxes = np.asarray(boxes, dtype=np.float64)

    def _coords(self):
        x = np.linspace(-self.size_x / 2, self.size_x / 2, self.grid_x)
        y = np.linspace(-self.size_y / 2, self.size_y / 2, self.grid_y)
        return np.meshgrid(x, y)

    def _city_outside_distance(self, X, Y):
        """Signed distance outside the nearest downtown square footprint."""
        half = self.city_size / 2.0
        dist = None
        for cx, cy in self.city_centers:
            d = _signed_distance_to_square(X - cx, Y - cy, half)
            dist = d if dist is None else np.minimum(dist, d)
        return dist

    def _rim_city_clear(self):
        """How far a rim peak must sit outside the nearest city pad."""
        if self.rim_min_city_dist is not None:
            return float(self.rim_min_city_dist)
        half_w = 0.5 * min(self.size_x, self.size_y)
        city_half = 0.5 * self.city_size
        typical_peak_r = half_w - 0.4 * max(self.rim_width, 1.0)
        typical_d_c = typical_peak_r - city_half
        return min(450.0, max(120.0, 0.7 * typical_d_c))

    def _blurred_field(self, rng, cell, amp=1.0, blur_frac=0.55, passes=4):
        """Heavily blurred low-frequency lattice → smooth scalar field."""
        ncx = max(2, int(np.ceil(self.size_x / cell)) + 1)
        ncy = max(2, int(np.ceil(self.size_y / cell)) + 1)
        coarse = rng.uniform(-1.0, 1.0, size=(ncy, ncx))
        field = _upsample_bilinear(coarse, self.grid_y, self.grid_x)
        blur_radius = max(2, int(round((cell * blur_frac) / self.resolution)))
        field = _box_blur(field, blur_radius, passes=passes)
        peak = float(np.max(np.abs(field)))
        if peak > 1e-9:
            field = field * (amp / peak)
        return field

    def _region_masks(self, rng):
        """
        Soft flat / rolling / hilly weights from one blurred low-frequency field.
        Weights are smooth and sum to 1; hill weight is sparse (~20–30% of map).
        """
        raw = self._blurred_field(rng, self.region_cell, amp=1.0, blur_frac=0.7, passes=5)
        # Map to [0, 1]
        r = 0.5 * (raw + 1.0)
        # Flat meadows dominate low values; hills only on high plateaus
        flat_w = 1.0 - _smoothstep(0.18, 0.40, r)
        hill_w = _smoothstep(0.66, 0.86, r)
        roll_w = np.clip(1.0 - flat_w - hill_w, 0.0, 1.0)
        # Renormalize softly where rounding left a gap
        s = flat_w + roll_w + hill_w
        s = np.maximum(s, 1e-9)
        return flat_w / s, roll_w / s, hill_w / s

    def _anisotropic_hill(self, X, Y, cx, cy, amp, sx, sy, angle):
        """Wide smooth Gaussian hill, optionally elongated."""
        ca, sa = np.cos(angle), np.sin(angle)
        dx, dy = X - cx, Y - cy
        xr = ca * dx + sa * dy
        yr = -sa * dx + ca * dy
        return amp * np.exp(-0.5 * ((xr / sx) ** 2 + (yr / sy) ** 2))

    def _build_grass_relief(self, rng, X, Y, dist):
        """
        Variety in the belt: quiet meadows, gentle rolls, a few tall hill clusters.
        Relief is faded out through the blend band so the city seam stays clean.
        """
        flat_w, roll_w, hill_w = self._region_masks(rng)

        # Quiet base undulation — almost flat meadows everywhere in the belt
        meadow = self._blurred_field(
            rng, cell=70.0, amp=self.meadow_amp, blur_frac=0.65, passes=4
        )
        meadow = meadow * (0.35 + 0.65 * flat_w)  # stronger in flat regions

        # Gentle rolling (only where roll_w is up)
        rolls = self._blurred_field(
            rng, cell=self.grass_cell, amp=self.grass_amp, blur_frac=0.6, passes=4
        )
        # Shift to mostly positive gentle swells
        rolls = 0.55 * rolls + 0.45 * np.abs(rolls)
        rolls = rolls * roll_w

        # Distinct hill clusters — only in the outer belt so the city-edge
        # fade does not shear tall peaks into cliffs.
        hills = np.zeros_like(X)
        half = self.city_size / 2.0
        # Keep hill centers clear of every downtown pad + blend band
        r_min = half + self.blend_width + max(40.0, 0.55 * self.hill_sigma_min)
        r_max = min(self.size_x, self.size_y) / 2.0 - 16.0
        if r_min >= r_max - 5.0:
            r_min = half + self.blend_width + 25.0
        n_lo, n_hi = self.n_hill_clusters
        n_hills = int(rng.integers(n_lo, n_hi + 1))
        # Grid lookup for region gate
        xs = X[0, :]
        ys = Y[:, 0]

        def _sample_mask(cx, cy):
            j = int(np.clip(np.argmin(np.abs(xs - cx)), 0, self.grid_x - 1))
            i = int(np.clip(np.argmin(np.abs(ys - cy)), 0, self.grid_y - 1))
            return float(hill_w[i, j]), float(dist[i, j])

        def _near_any_city(cx, cy):
            for ox, oy in self.city_centers:
                if max(abs(cx - ox), abs(cy - oy)) < r_min:
                    return True
            return False

        placed = 0
        attempts = 0
        while placed < n_hills and attempts < n_hills * 80:
            attempts += 1
            cx = float(rng.uniform(-r_max, r_max))
            cy = float(rng.uniform(-r_max, r_max))
            if _near_any_city(cx, cy):
                continue
            hw, d_c = _sample_mask(cx, cy)
            if hw < 0.40 or d_c < self.blend_width + 20.0:
                continue
            amp = float(rng.uniform(self.hill_amp_min, self.hill_amp_max))
            sx = float(rng.uniform(self.hill_sigma_min, self.hill_sigma_max))
            sy = float(rng.uniform(self.hill_sigma_min, self.hill_sigma_max))
            if rng.random() < 0.65:
                sy = sx * float(rng.uniform(0.78, 0.95))
            angle = float(rng.uniform(0.0, np.pi))
            hills += self._anisotropic_hill(X, Y, cx, cy, amp, sx, sy, angle)
            if rng.random() < 0.55:
                ox = float(rng.uniform(0.4, 0.75) * sx * rng.choice([-1.0, 1.0]))
                oy = float(rng.uniform(0.4, 0.75) * sy * rng.choice([-1.0, 1.0]))
                amp2 = amp * float(rng.uniform(0.3, 0.55))
                sx2 = sx * float(rng.uniform(0.7, 0.95))
                sy2 = sy * float(rng.uniform(0.7, 0.95))
                hills += self._anisotropic_hill(
                    X, Y, cx + ox, cy + oy, amp2, sx2, sy2, angle + 0.4
                )
            placed += 1

        hills = hills * (0.55 + 0.45 * hill_w)
        peak_cap = self.hill_amp_max * 1.05
        hills = peak_cap * np.tanh(hills / max(peak_cap, 1e-6))
        # Extra radial soft gate: hill mass lives outward of the seam
        hill_gate = _smoothstep(
            self.blend_width + 10.0,
            self.blend_width + self.hill_fade + 10.0,
            dist,
        )
        hills = hills * hill_gate

        # Meadows / rolls may approach the seam gently (low amp → mild slopes)
        near_fade = _smoothstep(self.blend_width * 0.5, self.blend_width + 12.0, dist)
        relief = (meadow + rolls) * near_fade + hills
        relief = np.maximum(relief, -0.15 * self.meadow_amp)

        soft_r = max(2, int(round(4.0 / self.resolution)))
        relief = _box_blur(relief, soft_r, passes=3)
        relief = _limit_slopes(relief, self.max_slope, self.resolution, passes=20)

        # Hard zero through the blend band after smoothing
        edge_fade = _smoothstep(
            self.blend_width, self.blend_width + self.hill_fade * 0.5, dist
        )
        relief = relief * edge_fade
        return relief

    def _connector_distance(self, X, Y):
        """Distance to the connector centerline (inf when there is none)."""
        if self.connector_segment is None:
            return np.full_like(np.asarray(X, dtype=np.float64), np.inf)
        a, b = self.connector_segment
        return _distance_to_segment(X, Y, a, b)

    def _open_land_gate(self, dist, road_d, city_fade):
        """0 on downtown pads, blend band, and connector corridor; 1 in the open."""
        g = _smoothstep(self.blend_width, self.blend_width + city_fade, dist)
        g = g * _smoothstep(
            self.connector_flat, self.connector_flat + self.connector_fade, road_d
        )
        return g

    def _build_belt_relief(self, rng, X, Y, dist, road_d):
        """
        Belt landforms: quiet meadows, broad 8–20 m rolls, and 25–50 m hill
        clusters (main peak + lower shoulder). Faded to zero across the city
        blend and the connector corridor.
        """
        gate = self._open_land_gate(dist, road_d, self.belt_fade)

        # Meadow / roll / hill regions: split open land roughly in thirds
        raw = self._blurred_field(
            rng, self.belt_region_cell, amp=1.0, blur_frac=0.7, passes=5
        )
        open_land = gate > 0.5
        sample = raw[open_land] if open_land.any() else raw.ravel()
        q_lo, q_hi = np.quantile(sample, [0.33, 0.67])
        w = 0.3 * max(float(q_hi - q_lo), 1e-6)
        meadow_w = 1.0 - _smoothstep(q_lo - w, q_lo + w, raw)
        hill_w = _smoothstep(q_hi - w, q_hi + w, raw)
        roll_w = np.clip(1.0 - meadow_w - hill_w, 0.0, 1.0)

        swell = 0.5 * (
            self._blurred_field(rng, self.roll_cell, amp=1.0, blur_frac=0.6, passes=4)
            + 1.0
        )
        roll_amp = float(rng.uniform(*self.roll_amp))
        rolls = roll_amp * swell * (roll_w + 0.5 * hill_w + 0.1 * meadow_w)

        xs = X[0, :]
        ys = Y[:, 0]
        hx, hy = self.size_x / 2.0, self.size_y / 2.0
        amp_lo, amp_hi = self.tall_hill_amp

        def _clear(cx, cy, sigma):
            if min(hx - abs(cx), hy - abs(cy)) < self.rim_width + sigma:
                return False
            reach = 2.2 * sigma
            d_city = float(self._city_outside_distance(np.array(cx), np.array(cy)))
            if d_city < self.blend_width + reach:
                return False
            d_road = float(self._connector_distance(np.array(cx), np.array(cy)))
            return d_road >= self.connector_flat + reach

        hills = np.zeros_like(X)
        self._belt_hills = []
        n_lo, n_hi = self.n_tall_hills
        n_hills = int(rng.integers(n_lo, n_hi + 1))
        attempts = 0
        while len(self._belt_hills) < n_hills and attempts < n_hills * 300:
            attempts += 1
            cx = float(rng.uniform(-hx, hx))
            cy = float(rng.uniform(-hy, hy))
            i = int(np.argmin(np.abs(ys - cy)))
            j = int(np.argmin(np.abs(xs - cx)))
            if hill_w[i, j] < 0.5:
                continue
            amp = float(rng.uniform(amp_lo, amp_hi))
            # Minor-axis sigma ≥ 2.4·amp keeps Gaussian flank slope ≲ 0.25
            sy = amp * float(rng.uniform(2.4, 3.0))
            sx = sy * float(rng.uniform(1.0, 1.5))
            if not _clear(cx, cy, sx):
                continue
            if any(
                np.hypot(cx - px, cy - py) < 1.6 * max(sx, ps)
                for px, py, _, ps in self._belt_hills
            ):
                continue
            angle = float(rng.uniform(0.0, np.pi))
            hills += self._anisotropic_hill(X, Y, cx, cy, amp, sx, sy, angle)
            self._belt_hills.append((cx, cy, amp, sx))

            # Lower shoulder beside the main peak
            phi = float(rng.uniform(0.0, 2.0 * np.pi))
            off = sy * float(rng.uniform(0.9, 1.4))
            ox, oy = cx + off * np.cos(phi), cy + off * np.sin(phi)
            amp2 = amp * float(rng.uniform(0.45, 0.65))
            sx2 = sx * float(rng.uniform(0.75, 0.95))
            sy2 = sy * float(rng.uniform(0.75, 0.95))
            if _clear(ox, oy, sx2):
                hills += self._anisotropic_hill(
                    X, Y, ox, oy, amp2, sx2, sy2, angle + float(rng.uniform(-0.6, 0.6))
                )

        # Soft knee so overlapping clusters do not stack far past amp_hi
        knee = 0.9 * amp_hi
        over = np.maximum(hills - knee, 0.0)
        hills = np.minimum(hills, knee) + 0.2 * amp_hi * np.tanh(over / (0.2 * amp_hi))

        return (rolls + hills) * gate

    def _perimeter_point(self, s, inset):
        """Point `inset` meters inside the border at perimeter arc length s."""
        hx, hy = self.size_x / 2.0, self.size_y / 2.0
        w, h = 2.0 * hx, 2.0 * hy
        s = s % (2.0 * (w + h))
        if s < w:
            x, y, ang = -hx + s, -hy + inset, 0.0
        elif s < w + h:
            x, y, ang = hx - inset, -hy + (s - w), 0.5 * np.pi
        elif s < 2.0 * w + h:
            x, y, ang = hx - (s - w - h), hy - inset, 0.0
        else:
            x, y, ang = -hx + inset, hy - (s - 2.0 * w - h), 0.5 * np.pi
        x = float(np.clip(x, -hx + inset, hx - inset))
        y = float(np.clip(y, -hy + inset, hy - inset))
        return x, y, ang

    def _build_rim_relief(self, rng, X, Y, dist, road_d):
        """
        Separate mountain ranges along the border: chains of wide peaks with
        passes between ranges.
        """
        # Cells whose height must stay ~0: city blend ring + connector ribbon
        guard = (dist < self.blend_width + 120.0) | (road_d < self.connector_flat + 40.0)
        gX, gY = X[guard], Y[guard]
        min_city = self._rim_city_clear()

        perim = 2.0 * (self.size_x + self.size_y)
        n_lo, n_hi = self.n_rim_ranges
        n_ranges = int(rng.integers(n_lo, n_hi + 1)) if n_hi >= n_lo else 0
        if n_ranges < 1 or self.rim_width <= 0.0:
            self._rim_peaks = []
            return np.zeros_like(X)
        slot = perim / n_ranges
        offset = float(rng.uniform(0.0, slot))

        p4 = np.zeros_like(X)
        self._rim_peaks = []
        for r in range(n_ranges):
            start = offset + r * slot + float(rng.uniform(0.0, 0.12)) * slot
            length = float(rng.uniform(0.45, 0.7)) * slot
            k = max(2, int(round(length / 300.0)))
            for m in range(k):
                s = start + (m + 0.5 + float(rng.uniform(-0.15, 0.15))) * length / k
                min_in = 0.2
                fade_out = float(getattr(self, "outer_fade_width", 0.0) or 0.0)
                if fade_out > 1e-6:
                    min_in = min(0.55, fade_out / max(self.rim_width, 1.0))
                inset = float(rng.uniform(min_in, 0.75)) * self.rim_width
                if rng.random() < self.rim_tall_prob:
                    amp = float(rng.uniform(*self.rim_tall_amp))
                else:
                    amp = float(rng.uniform(*self.rim_peak_amp))
                if m == 0 or m == k - 1:
                    amp *= float(rng.uniform(0.7, 0.9))
                across_k = float(rng.uniform(1.6, 1.9))
                along_k = float(rng.uniform(1.2, 1.7))
                tilt = float(rng.uniform(-0.25, 0.25))

                placed = False
                for _ in range(2):
                    cx, cy, ang = self._perimeter_point(s, inset)
                    s_across = amp * across_k
                    s_along = s_across * along_k
                    d_c = float(
                        self._city_outside_distance(np.array(cx), np.array(cy))
                    )
                    if d_c >= min_city and gX.size:
                        leak = self._anisotropic_hill(
                            gX, gY, cx, cy, amp, s_along, s_across, ang + tilt
                        )
                        placed = float(np.max(leak)) < 2.0
                    elif d_c >= min_city:
                        placed = True
                    if placed:
                        break
                    # Retry lower and closer to the border (but inside the outer fade)
                    inset = min_in * self.rim_width
                    amp *= 0.7
                if not placed:
                    continue
                peak = self._anisotropic_hill(
                    X, Y, cx, cy, amp, s_along, s_across, ang + tilt
                )
                p4 += peak**4
                self._rim_peaks.append((cx, cy, amp))

        # Smooth max keeps saddles between neighbouring peaks
        rim = np.power(p4, 0.25)
        return rim * self._open_land_gate(dist, road_d, 300.0)

    def _generate_heightmap(self, seed=None):
        """
        Shared long grade + city mounds/noise inside; varied smooth grass belt
        outside (meadows / rolls / hill clusters), smoothstep-blended over
        blend_width so the city edge has no cliff.
        """
        rng = np.random.default_rng(seed)
        X, Y = self._coords()

        # Gentle grade across the full world (kept small so downtowns stay flat)
        grade = 0.0012 * X + 0.0009 * Y

        # --- City layer: nearly flat pads + tiny mounds inside each downtown ---
        city = grade.copy()
        half_city = self.city_size / 2.0
        n_mounds = max(4, int(self.city_size / 40))
        for cx0, cy0 in self.city_centers:
            for _ in range(n_mounds):
                cx = cx0 + rng.uniform(-half_city * 0.85, half_city * 0.85)
                cy = cy0 + rng.uniform(-half_city * 0.85, half_city * 0.85)
                sigma = rng.uniform(8.0, 22.0)
                amp = rng.uniform(0.08, 0.35)
                city += amp * np.exp(
                    -((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sigma**2)
                )
        city = city + rng.normal(0.0, 0.025, size=city.shape)

        # --- Grass layer: meadows + rolls + tall smooth hills (regional) ---
        dist = self._city_outside_distance(X, Y)
        road_d = self._connector_distance(X, Y)
        grass_relief = self._build_grass_relief(rng, X, Y, dist)
        grass_relief = grass_relief * _smoothstep(
            self.connector_flat, self.connector_flat + self.connector_fade, road_d
        )
        self._grass_relief = grass_relief

        # --- Belt landforms + mountain rim: lenient rim_max_slope only ---
        belt = self._build_belt_relief(rng, X, Y, dist, road_d)
        rim = self._build_rim_relief(rng, X, Y, dist, road_d)
        landforms = _limit_slopes(
            belt + rim, self.rim_max_slope, self.resolution, passes=80
        )
        landforms = _box_blur(landforms, 1, passes=2)
        self._relief = grass_relief + landforms
        grass = grade + self._relief

        # --- Soft seam: distance-to-nearest-downtown + smoothstep ---
        t = _smoothstep(0.0, self.blend_width, dist)
        Z = (1.0 - t) * city + t * grass
        # Keep the inner city pad on the city layer so streets stay flyable.
        fade_w = float(getattr(self, "edge_grass_width", 0.0) or 0.0)
        lock_in = -max(fade_w, 1.0)
        inner = dist < lock_in
        if inner.any():
            Z = np.where(inner, city, Z)
        # Low uneven berm on the outer band of each city square. Not a mountain rim.
        rise_amp = float(getattr(self, "edge_rise_amp", 0.0) or 0.0)
        if fade_w > 1e-6 and rise_amp > 1e-6:
            inward = np.maximum(-dist, 0.0)
            w = (1.0 - _smoothstep(0.0, fade_w, inward)) * (dist <= 0.0)
            n = 0.55 + 0.45 * (0.5 * (self._blurred_field(rng, 22.0, amp=1.0) + 1.0))
            Z = Z + w * n * rise_amp
        # Soften the world border so the heightfield does not end as a cliff.
        fade_out = float(getattr(self, "outer_fade_width", 0.0) or 0.0)
        if fade_out > 1e-6:
            hx, hy = self.size_x / 2.0, self.size_y / 2.0
            border = np.minimum(hx - np.abs(X), hy - np.abs(Y))
            outer_w = _smoothstep(0.0, fade_out, border)
            pad = inner if inner.any() else dist <= 0.0
            city_ref = float(np.median(Z[pad])) if np.any(pad) else 0.0
            Z = np.where(dist > 0.0, city_ref + (Z - city_ref) * outer_w, Z)
        # Shift up so the map stays non-negative without flattening the low side
        # of a long gentle grade (np.maximum(., 0) would zero half a km-scale map).
        Z = Z - float(np.min(Z))
        pad = dist <= 0.0
        city_z = float(np.median(Z[pad])) if np.any(pad) else 0.0
        realized = []
        if getattr(self, "_rim_peaks", None):
            xs, ys = X[0, :], Y[:, 0]
            for cx, cy, _amp in self._rim_peaks:
                j = int(np.clip(np.argmin(np.abs(xs - cx)), 0, self.grid_x - 1))
                i = int(np.clip(np.argmin(np.abs(ys - cy)), 0, self.grid_y - 1))
                realized.append(float(Z[i, j] - city_z))
        if realized:
            self.peak_height_range = (min(realized), max(realized))
        else:
            far = dist > max(80.0, 0.35 * self.city_size)
            above = (Z[far] - city_z) if far.any() else np.array([0.0])
            self.peak_height_range = (0.0, float(np.max(above)))
        return Z.astype(np.float64)

    def generate_ground_texture(
        self, filename="ground_texture.png", tex_res=None, road_marks=None
    ):
        """
        Dusty brown city center feathers to soft green grass.
        Large blurred lightness variation only (no speckles). Optional subtle
        height tint from grass relief; high / steep ground turns to rock and
        the tallest ridges get thin snow. Optional yellow center-line dashes from
        road_marks (same spacing/color as town DASH_*).
        """
        if tex_res is None:
            # One texel per meter is enough; clamp for huge maps
            tex_res = min(self.grid_x, 512)

        rng = np.random.default_rng(
            None if self._seed is None else (int(self._seed) + 7919)
        )
        xs = np.linspace(-self.size_x / 2, self.size_x / 2, tex_res)
        ys = np.linspace(-self.size_y / 2, self.size_y / 2, tex_res)
        X, Y = np.meshgrid(xs, ys)

        dist = self._city_outside_distance(X, Y)
        t = _smoothstep(0.0, self.color_blend_width, dist)

        # Large-scale lightness (~40 m) for grass variation
        cell = 40.0
        ncx = max(2, int(np.ceil(self.size_x / cell)) + 1)
        ncy = max(2, int(np.ceil(self.size_y / cell)) + 1)
        coarse = rng.uniform(-1.0, 1.0, size=(ncy, ncx))
        light = _upsample_bilinear(coarse, tex_res, tex_res)
        blur_r = max(1, int(round(tex_res * (cell * 0.4) / self.size_x)))
        light = _box_blur(light, blur_r, passes=3)
        peak = float(np.max(np.abs(light)))
        if peak > 1e-9:
            light = light / peak

        # Subtle smooth height tint (no speckle)
        height_tint = np.zeros((tex_res, tex_res), dtype=np.float64)
        if self._grass_relief is not None:
            relief = _upsample_bilinear(self._grass_relief, tex_res, tex_res)
            relief = _box_blur(
                relief,
                max(1, int(round(tex_res * 8.0 / self.size_x))),
                passes=2,
            )
            rmax = float(np.max(relief))
            if rmax > 1e-6:
                height_tint = np.clip(relief / rmax, 0.0, 1.0) * 0.08

        # ±12% lightness on grass; slight brightening on hilltops
        grass_rgb = _GRASS_RGB[None, None, :] * (
            1.0 + 0.12 * light[:, :, None] + height_tint[:, :, None]
        )
        # High / steep ground → rock; tallest ridges → thin snow
        if self._relief is not None:
            h = _upsample_bilinear(self._relief, tex_res, tex_res)
            gy, gx = np.gradient(
                self._relief, self.size_y / (self.grid_y - 1), self.size_x / (self.grid_x - 1)
            )
            slope = _upsample_bilinear(np.hypot(gx, gy), tex_res, tex_res)
            h_n = h + 8.0 * light
            h_top = float(np.max(self._relief))
            rock_lo = min(55.0, max(24.0, 0.40 * h_top))
            rock_hi = min(110.0, max(rock_lo + 18.0, 0.80 * h_top))
            slope_h_lo = min(40.0, max(16.0, 0.30 * h_top))
            slope_h_hi = min(70.0, max(slope_h_lo + 12.0, 0.55 * h_top))
            rock_w = np.maximum(
                _smoothstep(rock_lo, rock_hi, h_n),
                _smoothstep(0.24, 0.36, slope) * _smoothstep(slope_h_lo, slope_h_hi, h_n),
            )
            snow_lo = max(135.0, h_top - 22.0)
            snow_w = _smoothstep(snow_lo, snow_lo + 12.0, h + 6.0 * light)
            snow_w = snow_w * (1.0 - _smoothstep(0.34, 0.45, slope))
            rock_rgb = _ROCK_RGB[None, None, :] * (1.0 + 0.10 * light[:, :, None])
            grass_rgb = (
                (1.0 - rock_w)[:, :, None] * grass_rgb + rock_w[:, :, None] * rock_rgb
            )
            grass_rgb = (
                (1.0 - snow_w)[:, :, None] * grass_rgb
                + snow_w[:, :, None] * _SNOW_RGB[None, None, :]
            )

        dust_rgb = np.broadcast_to(_DUST_RGB, grass_rgb.shape).copy()
        dust_rgb = dust_rgb * (1.0 + 0.04 * light[:, :, None])

        rgb = (1.0 - t)[:, :, None] * dust_rgb + t[:, :, None] * grass_rgb
        fade_w = float(getattr(self, "edge_grass_width", 0.0) or 0.0)
        if fade_w > 1e-6:
            inward = np.maximum(-dist, 0.0)
            w = (1.0 - _smoothstep(0.0, fade_w, inward)) * (dist <= 0.0)
            rgb = (1.0 - w)[:, :, None] * rgb + w[:, :, None] * grass_rgb
        rgb = np.clip(rgb, 0.0, 1.0)
        rgb_u8 = (rgb * 255.0 + 0.5).astype(np.uint8)

        # Bake center-line dashes into the texture (no MultiBody per dash).
        if road_marks:
            sx = float(self.size_x)
            sy = float(self.size_y)
            n = float(tex_res - 1) if tex_res > 1 else 1.0
            for mark in road_marks:
                cx, cy, hx, hy = mark[0], mark[1], mark[2], mark[3]
                rgba = mark[5] if len(mark) > 5 else (0.92, 0.86, 0.28, 1.0)
                color = (
                    np.clip(np.asarray(rgba[:3], dtype=np.float64) * 255.0 + 0.5, 0, 255)
                    .astype(np.uint8)
                )
                j0 = int(np.floor((cx - hx + 0.5 * sx) / sx * n))
                j1 = int(np.ceil((cx + hx + 0.5 * sx) / sx * n)) + 1
                i0 = int(np.floor((cy - hy + 0.5 * sy) / sy * n))
                i1 = int(np.ceil((cy + hy + 0.5 * sy) / sy * n)) + 1
                j0 = int(np.clip(j0, 0, tex_res - 1))
                j1 = int(np.clip(max(j1, j0 + 1), 1, tex_res))
                i0 = int(np.clip(i0, 0, tex_res - 1))
                i1 = int(np.clip(max(i1, i0 + 1), 1, tex_res))
                rgb_u8[i0:i1, j0:j1] = color

        path = os.path.join(os.path.dirname(__file__), filename)
        _write_png_rgb(path, rgb_u8)
        self._texture_path = path
        return path

    def generate_dem_texture(self, filename="ground_texture.png", max_tex_side=1024):
        """
        Elevation / slope coloring for a DEM: green lowland, gray-brown rock
        on high or steep ground, a thin snow cap on the top few percent of
        this raster's relief. Mild NW hillshade so relief reads without shadows.
        """
        up = max(1, min(8, max_tex_side // max(self.grid_x, self.grid_y)))
        th, tw = self.grid_y * up, self.grid_x * up
        Z = self.heightmap
        dzdy, dzdx = np.gradient(Z, self.resolution_y, self.resolution_x)
        h = _upsample_bilinear(Z, th, tw)
        gx = _upsample_bilinear(dzdx, th, tw)
        gy = _upsample_bilinear(dzdy, th, tw)
        slope = np.hypot(gx, gy)

        h_lo, h_hi = float(np.min(Z)), float(np.max(Z))
        e = (h - h_lo) / max(h_hi - h_lo, 1e-6)

        rock_w = np.maximum(
            _smoothstep(0.55, 0.80, e),
            _smoothstep(0.35, 0.60, slope) * _smoothstep(0.25, 0.50, e),
        )
        snow_w = _smoothstep(0.94, 0.975, e) * (1.0 - _smoothstep(0.45, 0.70, slope))

        # Lowland grass darkens slightly toward the rock band
        grass = _GRASS_RGB[None, None, :] * (1.0 - 0.12 * _smoothstep(0.2, 0.6, e))[:, :, None]
        rgb = (1.0 - rock_w)[:, :, None] * grass + rock_w[:, :, None] * _ROCK_RGB
        rgb = (1.0 - snow_w)[:, :, None] * rgb + snow_w[:, :, None] * _SNOW_RGB

        # Hillshade: sun from azimuth 315 deg (NW), 45 deg elevation
        alt, az = np.radians(45.0), np.radians(315.0)
        light = np.array([np.sin(az) * np.cos(alt), np.cos(az) * np.cos(alt), np.sin(alt)])
        norm = np.sqrt(gx * gx + gy * gy + 1.0)
        shade = (-gx * light[0] - gy * light[1] + light[2]) / norm
        rgb = rgb * np.clip(1.0 + 1.2 * (shade - light[2]), 0.7, 1.2)[:, :, None]

        rgb_u8 = (np.clip(rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        # PyBullet maps PNG row 0 to the -Y edge and PNG column 0 to the +X
        # edge of a heightfield (checked in GUI and TinyRenderer), so rows
        # stay south-first and columns are mirrored.
        rgb_u8 = np.ascontiguousarray(rgb_u8[:, ::-1])
        path = os.path.join(os.path.dirname(__file__), filename)
        _write_png_rgb(path, rgb_u8)
        self._texture_path = path
        return path

    def get_ground_texture_path(self, road_marks=None):
        """Return path to ground texture PNG, generating it if needed."""
        if self.source == "dem":
            if self._texture_path and os.path.isfile(self._texture_path):
                return self._texture_path
            return self.generate_dem_texture()
        if road_marks is not None:
            return self.generate_ground_texture(road_marks=road_marks)
        if self._texture_path and os.path.isfile(self._texture_path):
            return self._texture_path
        return self.generate_ground_texture()

    def get_urdf_path(self, filename="terrain.obj"):
        """Dynamically generate an .obj and .urdf file for PyBullet and return the URDF path."""
        obj_path = os.path.join(os.path.dirname(__file__), filename)
        with open(obj_path, "w") as f:
            f.write("# Terrain OBJ generated dynamically\n")
            for i in range(self.grid_y):
                for j in range(self.grid_x):
                    x = (j * self.resolution_x) - self.size_x / 2
                    y = (i * self.resolution_y) - self.size_y / 2
                    z = self.heightmap[i, j]
                    f.write(f"v {x} {y} {z}\n")
            for i in range(self.grid_y - 1):
                for j in range(self.grid_x - 1):
                    v1 = i * self.grid_x + j + 1
                    v2 = v1 + 1
                    v3 = (i + 1) * self.grid_x + j + 1
                    v4 = v3 + 1
                    f.write(f"f {v1} {v3} {v2}\n")
                    f.write(f"f {v2} {v3} {v4}\n")

        urdf_path = os.path.join(os.path.dirname(__file__), "terrain.urdf")
        mesh_abs_path = obj_path.replace("\\", "/")
        # Gray-brown dusty ground (not forest green)
        urdf_content = f"""<?xml version="1.0"?>
<robot name="terrain">
  <link name="base_link">
    <collision>
      <geometry>
        <mesh filename="{mesh_abs_path}" scale="1 1 1"/>
      </geometry>
    </collision>
    <visual>
      <geometry>
        <mesh filename="{mesh_abs_path}" scale="1 1 1"/>
      </geometry>
      <material name="dusty_brown">
        <color rgba="0.45 0.38 0.28 1"/>
      </material>
    </visual>
  </link>
</robot>
"""
        with open(urdf_path, "w") as f:
            f.write(urdf_content)

        return urdf_path

    def check_los(self, point_A, point_B):
        """
        Line-of-sight: heightmap occlusion plus vectorized segment-vs-AABB
        against town building / overpass boxes. No PyBullet raycasts.
        """
        point_A = np.asarray(point_A, dtype=np.float64)
        point_B = np.asarray(point_B, dtype=np.float64)

        # Building / overpass volumes
        if TownLayout.segment_hits_boxes(point_A, point_B, self.obstacle_boxes):
            return False

        # Heightmap sampling
        dist = np.linalg.norm(point_A[:2] - point_B[:2])
        num_samples = int(np.ceil(dist / self.resolution)) * 2 + 5

        x_vals = np.linspace(point_A[0], point_B[0], num_samples)
        y_vals = np.linspace(point_A[1], point_B[1], num_samples)
        z_vals = np.linspace(point_A[2], point_B[2], num_samples)

        for idx in range(num_samples):
            x = x_vals[idx]
            y = y_vals[idx]
            z = z_vals[idx]

            grid_x_idx = int((x + self.size_x / 2) / self.resolution_x)
            grid_y_idx = int((y + self.size_y / 2) / self.resolution_y)

            if 0 <= grid_x_idx < self.grid_x and 0 <= grid_y_idx < self.grid_y:
                if self.heightmap[grid_y_idx, grid_x_idx] > z:
                    return False
        return True


def spawn_terrain_in_pybullet(client, terrain):
    """
    Spawn the terrain heightfield alone (no town) with its ground texture.
    Supports non-square grids and non-square cells. Returns the body id.
    """
    import pybullet as p

    hf_bytes = int(terrain.grid_x) * int(terrain.grid_y) * 4
    assert terrain.grid_x * terrain.grid_y * 4 < 1_000_000, (
        f"heightfield upload {hf_bytes} bytes ({terrain.grid_x}x{terrain.grid_y}) "
        f"exceeds PyBullet GUI 1 MiB shared-memory limit"
    )
    shape = p.createCollisionShape(
        p.GEOM_HEIGHTFIELD,
        meshScale=[terrain.resolution_x, terrain.resolution_y, 1.0],
        heightfieldData=np.ascontiguousarray(terrain.heightmap).ravel().tolist(),
        # Bullet's "rows" run along X: data[i * grid_x + j] is (x_j, y_i).
        numHeightfieldRows=int(terrain.grid_x),
        numHeightfieldColumns=int(terrain.grid_y),
        # 1.0 stretches the texture once over the heightfield's XY extent.
        heightfieldTextureScaling=1.0,
        physicsClientId=client,
    )
    # Bullet centers the heightfield's Z range on the body origin.
    mid = 0.5 * (float(np.min(terrain.heightmap)) + float(np.max(terrain.heightmap)))
    body = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=shape,
        basePosition=[0.0, 0.0, mid],
        physicsClientId=client,
    )
    try:
        texture_id = p.loadTexture(terrain.get_ground_texture_path(), physicsClientId=client)
    except Exception:
        texture_id = -1
    if texture_id >= 0:
        p.changeVisualShape(
            body, -1, rgbaColor=[1, 1, 1, 1], textureUniqueId=texture_id,
            specularColor=[0.08, 0.08, 0.08],
            physicsClientId=client,
        )
    else:
        p.changeVisualShape(
            body, -1, rgbaColor=[*_GRASS_RGB, 1.0],
            specularColor=[0.08, 0.08, 0.08],
            physicsClientId=client,
        )
    return body
