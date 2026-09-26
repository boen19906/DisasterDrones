import numpy as np
import os

from .structures import generate_earthquake_layout, segment_blocked_by_structures


class Terrain:
    def __init__(self, size_x=100, size_y=100, resolution=1.0, seed=42, flat=False, with_structures=True):
        self.size_x = size_x
        self.size_y = size_y
        self.resolution = resolution
        self.seed = seed
        self.flat = flat
        self.grid_x = int(size_x / resolution)
        self.grid_y = int(size_y / resolution)
        self.heightmap = self._generate_heightmap()
        self.structures = generate_earthquake_layout(self, seed=seed) if with_structures else []
        
    def _generate_heightmap(self):
        """Generate a 2D numpy array using overlapping 2D Gaussians."""
        x = np.linspace(-self.size_x/2, self.size_x/2, self.grid_x)
        y = np.linspace(-self.size_y/2, self.size_y/2, self.grid_y)
        X, Y = np.meshgrid(x, y)
        
        Z = np.zeros_like(X)
        if self.flat:
            return Z
        np.random.seed(42)
        for _ in range(15):
            cx = np.random.uniform(-self.size_x/2, self.size_x/2)
            cy = np.random.uniform(-self.size_y/2, self.size_y/2)
            sigma = np.random.uniform(10.0, 30.0)
            amp = np.random.uniform(2.0, 5.0) # Flatter mountains
            Z += amp * np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * sigma**2))
            
        return Z

    def get_urdf_path(self, filename="terrain.obj"):
        """Dynamically generate an .obj and .urdf file for PyBullet and return the URDF path."""
        obj_path = os.path.join(os.path.dirname(__file__), filename)
        with open(obj_path, 'w') as f:
            f.write("# Terrain OBJ generated dynamically\n")
            for i in range(self.grid_y):
                for j in range(self.grid_x):
                    x = (j * self.resolution) - self.size_x/2
                    y = (i * self.resolution) - self.size_y/2
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
        mesh_abs_path = obj_path.replace("\\", "/") # PyBullet likes forward slashes for absolute paths
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
      <material name="forest_green">
        <color rgba="0.2 0.5 0.2 1"/>
      </material>
    </visual>
  </link>
</robot>
"""
        with open(urdf_path, 'w') as f:
            f.write(urdf_content)
            
        return urdf_path

    def check_los(self, point_A, point_B):
        """Fast numpy raycast using the 2D heightmap array."""
        dist = np.linalg.norm(np.array(point_A[:2]) - np.array(point_B[:2]))
        num_samples = int(np.ceil(dist / self.resolution)) * 2 + 5
        
        x_vals = np.linspace(point_A[0], point_B[0], num_samples)
        y_vals = np.linspace(point_A[1], point_B[1], num_samples)
        z_vals = np.linspace(point_A[2], point_B[2], num_samples)
        
        for idx in range(num_samples):
            x = x_vals[idx]
            y = y_vals[idx]
            z = z_vals[idx]
            
            grid_x_idx = int((x + self.size_x/2) / self.resolution)
            grid_y_idx = int((y + self.size_y/2) / self.resolution)
            
            if 0 <= grid_x_idx < self.grid_x and 0 <= grid_y_idx < self.grid_y:
                if self.heightmap[grid_y_idx, grid_x_idx] > z:
                    return False
        if segment_blocked_by_structures(point_A, point_B, self.structures):
            return False
        return True
