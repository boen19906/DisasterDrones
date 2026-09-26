"""
terrain.py — City dusty ground plus a soft grass belt heightmap and texture.
"""

import numpy as np
import os
import struct
import zlib

from .town import TownLayout

# Dusty town center (matches previous flat tint)
_DUST_RGB = np.array([0.45, 0.38, 0.28], dtype=np.float64)
# Soft natural grass (no neon)
_GRASS_RGB = np.array([0.32, 0.46, 0.28], dtype=np.float64)


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


class Terrain:
    def __init__(
        self,
        size_x=450,
        size_y=450,
        resolution=1.0,
        seed=None,
        obstacle_boxes=None,
        city_size=250.0,
        blend_width=30.0,
        color_blend_width=40.0,
        grass_cell=28.0,
        grass_amp=1.8,
    ):
        self.size_x = float(size_x)
        self.size_y = float(size_y)
        self.resolution = float(resolution)
        self.city_size = float(city_size)
        self.blend_width = float(blend_width)
        self.color_blend_width = float(color_blend_width)
        self.grass_cell = float(grass_cell)
        self.grass_amp = float(grass_amp)
        self.grid_x = int(round(self.size_x / self.resolution))
        self.grid_y = int(round(self.size_y / self.resolution))
        self._seed = seed
        self.obstacle_boxes = (
            np.asarray(obstacle_boxes, dtype=np.float64)
            if obstacle_boxes is not None and len(obstacle_boxes) > 0
            else np.zeros((0, 6), dtype=np.float64)
        )
        self.heightmap = self._generate_heightmap(seed)
        self._texture_path = None

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
        half = self.city_size / 2.0
        return _signed_distance_to_square(X, Y, half)

    def _generate_heightmap(self, seed=None):
        """
        Shared long grade + city mounds/noise inside, blurred grass rolls outside,
        smoothstep-blended over blend_width so the city edge has no cliff.
        """
        rng = np.random.default_rng(seed)
        X, Y = self._coords()

        # Gentle grade across the full world (~0.5–1.5 m over the city)
        grade = 0.004 * X + 0.003 * Y

        # --- City layer: dusty mounds + fine noise (mounds only in city) ---
        city = grade.copy()
        half_city = self.city_size / 2.0
        n_mounds = max(8, int(self.city_size / 20))
        for _ in range(n_mounds):
            cx = rng.uniform(-half_city, half_city)
            cy = rng.uniform(-half_city, half_city)
            sigma = rng.uniform(4.0, 14.0)
            amp = rng.uniform(0.3, 1.5)
            city += amp * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sigma**2))
        fine = rng.normal(0.0, 0.08, size=city.shape)
        city = city + fine

        # --- Grass layer: coarse lattice (~25–30 m), upsample + blur ---
        cell = self.grass_cell
        ncx = max(2, int(np.ceil(self.size_x / cell)) + 1)
        ncy = max(2, int(np.ceil(self.size_y / cell)) + 1)
        # Amplitude ~1–2.5 m after blur; draw then scale to target amp
        coarse = rng.uniform(-1.0, 1.0, size=(ncy, ncx))
        rolls = _upsample_bilinear(coarse, self.grid_y, self.grid_x)
        # Blur radius ~ half a cell in samples so facets disappear
        blur_radius = max(1, int(round((cell * 0.45) / self.resolution)))
        rolls = _box_blur(rolls, blur_radius, passes=3)
        # Normalize to roughly ±grass_amp
        peak = float(np.max(np.abs(rolls)))
        if peak > 1e-9:
            rolls = rolls * (self.grass_amp / peak)
        grass = grade + rolls

        # --- Soft seam: distance-to-square + smoothstep ---
        dist = self._city_outside_distance(X, Y)
        t = _smoothstep(0.0, self.blend_width, dist)
        Z = (1.0 - t) * city + t * grass
        Z = np.maximum(Z, 0.0)
        return Z.astype(np.float64)

    def generate_ground_texture(self, filename="ground_texture.png", tex_res=None):
        """
        Dusty brown city center feathers to soft green grass.
        Large blurred lightness variation only (no speckles). PNG via stdlib.
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
        # ±12% lightness on grass; city stays mostly flat dusty
        grass_rgb = _GRASS_RGB[None, None, :] * (1.0 + 0.12 * light[:, :, None])
        dust_rgb = np.broadcast_to(_DUST_RGB, grass_rgb.shape).copy()
        # Slight dust variation so the city lot isn't perfectly flat
        dust_rgb = dust_rgb * (1.0 + 0.04 * light[:, :, None])

        rgb = (1.0 - t)[:, :, None] * dust_rgb + t[:, :, None] * grass_rgb
        rgb = np.clip(rgb, 0.0, 1.0)
        rgb_u8 = (rgb * 255.0 + 0.5).astype(np.uint8)

        path = os.path.join(os.path.dirname(__file__), filename)
        _write_png_rgb(path, rgb_u8)
        self._texture_path = path
        return path

    def get_ground_texture_path(self):
        """Return path to ground texture PNG, generating it if needed."""
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
                    x = (j * self.resolution) - self.size_x / 2
                    y = (i * self.resolution) - self.size_y / 2
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

            grid_x_idx = int((x + self.size_x / 2) / self.resolution)
            grid_y_idx = int((y + self.size_y / 2) / self.resolution)

            if 0 <= grid_x_idx < self.grid_x and 0 <= grid_y_idx < self.grid_y:
                if self.heightmap[grid_y_idx, grid_x_idx] > z:
                    return False
        return True
