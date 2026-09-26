"""
terrain.py — Dusty city core + rolling wilderness rim heightmap, plus building-box LoS.
"""

import numpy as np
import os

from .town import TownLayout


class Terrain:
    def __init__(
        self,
        size_x=250,
        size_y=250,
        resolution=2.0,
        seed=None,
        obstacle_boxes=None,
        city_half=80.0,
    ):
        self.size_x = size_x
        self.size_y = size_y
        self.resolution = resolution
        self.grid_x = int(size_x / resolution)
        self.grid_y = int(size_y / resolution)
        self._seed = seed
        # Inner city half-extent (meters from center); outer band is wilderness hills.
        self.city_half = float(city_half)
        self.obstacle_boxes = (
            np.asarray(obstacle_boxes, dtype=np.float64)
            if obstacle_boxes is not None and len(obstacle_boxes) > 0
            else np.zeros((0, 6), dtype=np.float64)
        )
        self.heightmap = self._generate_heightmap(seed)

    def set_obstacle_boxes(self, boxes):
        """Attach town building/overpass AABBs used by check_los."""
        if boxes is None or len(boxes) == 0:
            self.obstacle_boxes = np.zeros((0, 6), dtype=np.float64)
        else:
            self.obstacle_boxes = np.asarray(boxes, dtype=np.float64)

    def height_at(self, x, y):
        """Bilinear sample of the heightmap at world (x, y)."""
        # Map world → continuous grid coords
        fx = (float(x) + self.size_x / 2.0) / self.resolution
        fy = (float(y) + self.size_y / 2.0) / self.resolution
        fx = float(np.clip(fx, 0.0, self.grid_x - 1.001))
        fy = float(np.clip(fy, 0.0, self.grid_y - 1.001))
        i0 = int(np.floor(fy))
        j0 = int(np.floor(fx))
        i1 = min(i0 + 1, self.grid_y - 1)
        j1 = min(j0 + 1, self.grid_x - 1)
        ty = fy - i0
        tx = fx - j0
        z00 = self.heightmap[i0, j0]
        z01 = self.heightmap[i0, j1]
        z10 = self.heightmap[i1, j0]
        z11 = self.heightmap[i1, j1]
        return float(
            (1 - ty) * ((1 - tx) * z00 + tx * z01)
            + ty * ((1 - tx) * z10 + tx * z11)
        )

    def heightfield_mid(self):
        """AABB midpoint of height values — PyBullet centers heightfields here."""
        return float(0.5 * (self.heightmap.min() + self.heightmap.max()))

    def _generate_heightmap(self, seed=None):
        """
        Relatively flat dusty city interior; rolling wilderness hills outside.

        City streets stay sensible; the outer band rises and dips so flying
        over the rim reads as hills / ridges / pockets.
        """
        rng = np.random.default_rng(seed)
        half_x = self.size_x / 2.0
        half_y = self.size_y / 2.0
        x = np.linspace(-half_x, half_x, self.grid_x)
        y = np.linspace(-half_y, half_y, self.grid_y)
        X, Y = np.meshgrid(x, y)

        # Chebyshev radius — matches the square city / wilderness ring
        R = np.maximum(np.abs(X), np.abs(Y))
        city_h = self.city_half
        rim = max(half_x, half_y) - city_h
        # Soft blend over ~12 m so the city edge is not a cliff
        blend_w = 12.0
        if rim > 1.0:
            wild = np.clip((R - (city_h - blend_w * 0.35)) / blend_w, 0.0, 1.0)
            wild = wild * wild * (3.0 - 2.0 * wild)  # smoothstep
        else:
            wild = np.zeros_like(R)

        # --- City: gentle grade + tiny dusty mounds ---
        grade = 0.003 * X + 0.002 * Y
        Z_city = grade.copy()
        n_city_mounds = max(4, int(city_h / 25))
        for _ in range(n_city_mounds):
            cx = rng.uniform(-city_h * 0.85, city_h * 0.85)
            cy = rng.uniform(-city_h * 0.85, city_h * 0.85)
            sigma = rng.uniform(4.0, 10.0)
            amp = rng.uniform(0.2, 0.9)
            Z_city += amp * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sigma**2))
        Z_city += rng.normal(0.0, 0.05, size=Z_city.shape)

        # --- Wilderness: rolling hills, ridges, pockets ---
        Z_wild = grade * 0.5

        # Broad rolling hills (mostly outside the city)
        n_hills = int(rng.integers(10, 16))
        for _ in range(n_hills):
            # Prefer placements in the outer band
            for _try in range(12):
                cx = rng.uniform(-half_x * 0.95, half_x * 0.95)
                cy = rng.uniform(-half_y * 0.95, half_y * 0.95)
                if max(abs(cx), abs(cy)) >= city_h * 0.7:
                    break
            sigma = rng.uniform(18.0, 38.0)
            amp = rng.uniform(3.5, 11.0)
            Z_wild += amp * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sigma**2))

        # A few taller ridges (elongated Gaussians)
        n_ridges = int(rng.integers(3, 6))
        for _ in range(n_ridges):
            cx = rng.uniform(-half_x * 0.9, half_x * 0.9)
            cy = rng.uniform(-half_y * 0.9, half_y * 0.9)
            if max(abs(cx), abs(cy)) < city_h * 0.75:
                # Push toward nearest edge
                if abs(cx) >= abs(cy):
                    cx = np.sign(cx) * rng.uniform(city_h * 0.85, half_x * 0.92)
                else:
                    cy = np.sign(cy) * rng.uniform(city_h * 0.85, half_y * 0.92)
            angle = rng.uniform(0.0, np.pi)
            ca, sa = np.cos(angle), np.sin(angle)
            xr = (X - cx) * ca + (Y - cy) * sa
            yr = -(X - cx) * sa + (Y - cy) * ca
            sig_long = rng.uniform(28.0, 55.0)
            sig_short = rng.uniform(8.0, 16.0)
            amp = rng.uniform(5.0, 14.0)
            Z_wild += amp * np.exp(-(xr**2) / (2 * sig_long**2) - (yr**2) / (2 * sig_short**2))

        # Lower pockets / bowls
        n_pockets = int(rng.integers(4, 8))
        for _ in range(n_pockets):
            for _try in range(10):
                cx = rng.uniform(-half_x * 0.9, half_x * 0.9)
                cy = rng.uniform(-half_y * 0.9, half_y * 0.9)
                if max(abs(cx), abs(cy)) >= city_h * 0.8:
                    break
            sigma = rng.uniform(12.0, 24.0)
            amp = rng.uniform(2.0, 6.0)
            Z_wild -= amp * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sigma**2))

        # Mild tendency to rise toward map edges (varies by angle)
        edge_t = np.clip((R - city_h) / max(rim, 1.0), 0.0, 1.0)
        ang = np.arctan2(Y, X)
        edge_rise = edge_t * edge_t * (
            2.5 + 3.5 * (0.5 + 0.5 * np.sin(2.0 * ang + rng.uniform(0, 6)))
            + 2.0 * (0.5 + 0.5 * np.cos(3.0 * ang + rng.uniform(0, 6)))
        )
        Z_wild += edge_rise

        # Medium-frequency wilderness texture
        n_bumps = int(rng.integers(18, 28))
        for _ in range(n_bumps):
            cx = rng.uniform(-half_x, half_x)
            cy = rng.uniform(-half_y, half_y)
            if max(abs(cx), abs(cy)) < city_h * 0.9:
                continue
            sigma = rng.uniform(6.0, 14.0)
            amp = rng.uniform(0.8, 3.0) * (1.0 if rng.random() < 0.7 else -1.0)
            Z_wild += amp * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sigma**2))

        Z_wild += rng.normal(0.0, 0.15, size=Z_wild.shape)

        # Blend: city interior stays flat-ish; rim becomes wilderness
        Z = (1.0 - wild) * Z_city + wild * Z_wild
        Z = np.maximum(Z, 0.0)
        return Z.astype(np.float64)

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
        # Default dusty brown; wilderness green is painted separately in spawn.
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
