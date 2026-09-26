"""
terrain.py — Flat dusty-town heightmap ground plus building-box LoS.
"""

import numpy as np
import os

from .town import TownLayout


class Terrain:
    def __init__(self, size_x=250, size_y=250, resolution=2.0, seed=None, obstacle_boxes=None):
        self.size_x = size_x
        self.size_y = size_y
        self.resolution = resolution
        self.grid_x = int(size_x / resolution)
        self.grid_y = int(size_y / resolution)
        self._seed = seed
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

    def _generate_heightmap(self, seed=None):
        """
        Flatter dusty town ground: slight grade, low rubble mounds.
        Uses the episode seed (not a hardcoded constant).
        """
        rng = np.random.default_rng(seed)
        x = np.linspace(-self.size_x / 2, self.size_x / 2, self.grid_x)
        y = np.linspace(-self.size_y / 2, self.size_y / 2, self.grid_y)
        X, Y = np.meshgrid(x, y)

        # Gentle grade across the map (~0.5–1.5 m)
        grade = 0.004 * X + 0.003 * Y
        Z = grade.copy()

        # Low dusty mounds / rubble piles (0.3–1.5 m), not mountain peaks
        n_mounds = max(8, int(self.size_x / 20))
        for _ in range(n_mounds):
            cx = rng.uniform(-self.size_x / 2, self.size_x / 2)
            cy = rng.uniform(-self.size_y / 2, self.size_y / 2)
            sigma = rng.uniform(4.0, 14.0)
            amp = rng.uniform(0.3, 1.5)
            Z += amp * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sigma**2))

        # Fine noise for dusty lot texture
        Z += rng.normal(0.0, 0.08, size=Z.shape)
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
