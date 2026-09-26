"""
visualizer.py - Simple interactive drone demo over the mountain terrain.
Uses direct position control (teleport) so controls are always instant and responsive.
Real physics training happens in disaster_env.py / train.py.
"""
import time
import os
import importlib.util
import numpy as np
import pybullet as p
import pybullet_data

def build_terrain(client, grid=128, size=100.0, seed=42):
    """Build the green mountain heightfield."""
    x = np.linspace(-size/2, size/2, grid)
    y = np.linspace(-size/2, size/2, grid)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    np.random.seed(seed)
    for _ in range(12):
        cx = np.random.uniform(-size/2, size/2)
        cy = np.random.uniform(-size/2, size/2)
        sigma = np.random.uniform(8.0, 25.0)
        amp = np.random.uniform(3.0, 8.0)
        Z += amp * np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * sigma**2))

    shape = p.createCollisionShape(
        p.GEOM_HEIGHTFIELD,
        meshScale=[size/grid, size/grid, 1.0],
        heightfieldData=Z.flatten().tolist(),
        numHeightfieldRows=grid,
        numHeightfieldColumns=grid,
        physicsClientId=client
    )
    body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=shape, physicsClientId=client)
    p.changeVisualShape(body, -1, rgbaColor=[0.2, 0.65, 0.2, 1], physicsClientId=client)
    return float(Z.max())


if __name__ == "__main__":
    print("Starting visualizer...")
    client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, 0, physicsClientId=client)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=client)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0, physicsClientId=client)

    # Build green terrain
    max_z = build_terrain(client)
    hover_z = max_z + 8.0
    print(f"Terrain max={max_z:.1f}m  |  Drones hover at Z={hover_z:.1f}m")

    # Load the Crazyflie URDF from the assets directory
    gp = importlib.util.find_spec("gym_pybullet_drones")
    assets_dir = os.path.join(os.path.dirname(gp.origin), "assets")
    original_cwd = os.getcwd()
    os.chdir(assets_dir)
    drone_colors = [[1, 0.2, 0.2, 1], [0.2, 0.5, 1, 1], [1, 0.8, 0, 1]]
    drone_positions = [
        [0.0,  0.0, hover_z],
        [3.0,  0.0, hover_z],
        [-3.0, 0.0, hover_z],
    ]
    drone_ids = []
    for i, pos in enumerate(drone_positions):
        try:
            did = p.loadURDF("cf2x.urdf", pos, globalScaling=8.0, physicsClientId=client)
            p.changeVisualShape(did, -1, rgbaColor=drone_colors[i], physicsClientId=client)
        except Exception:
            col = p.createCollisionShape(p.GEOM_SPHERE, radius=0.4, physicsClientId=client)
            vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.4, rgbaColor=drone_colors[i], physicsClientId=client)
            did = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col,
                                    baseVisualShapeIndex=vis, basePosition=pos, physicsClientId=client)
        drone_ids.append(did)
    os.chdir(original_cwd)
    print(f"Loaded {len(drone_ids)} drones.")

    orn = p.getQuaternionFromEuler([0, 0, 0])
    pos0 = list(drone_positions[0])

    p.resetDebugVisualizerCamera(
        cameraDistance=10, cameraYaw=45, cameraPitch=-25,
        cameraTargetPosition=pos0, physicsClientId=client
    )

    print("=" * 45)
    print("  CLICK the 3D viewport, then use keys:")
    print("  Arrow Keys : Move Drone 0 horizontally")
    print("  I / K      : Ascend / Descend")
    print("  Close window or CTRL+C to quit")
    print("=" * 45)

    speed = 0.3
    while True:
        try:
            keys = p.getKeyboardEvents(physicsClientId=client)
        except p.error:
            break

        dx, dy, dz = 0.0, 0.0, 0.0
        if keys.get(p.B3G_UP_ARROW,    0) & p.KEY_IS_DOWN: dx =  speed
        if keys.get(p.B3G_DOWN_ARROW,  0) & p.KEY_IS_DOWN: dx = -speed
        if keys.get(p.B3G_LEFT_ARROW,  0) & p.KEY_IS_DOWN: dy =  speed
        if keys.get(p.B3G_RIGHT_ARROW, 0) & p.KEY_IS_DOWN: dy = -speed
        if keys.get(ord('i'), 0) & p.KEY_IS_DOWN or keys.get(ord('I'), 0) & p.KEY_IS_DOWN: dz =  speed
        if keys.get(ord('k'), 0) & p.KEY_IS_DOWN or keys.get(ord('K'), 0) & p.KEY_IS_DOWN: dz = -speed

        pos0 = [pos0[0]+dx, pos0[1]+dy, pos0[2]+dz]
        try:
            p.resetBasePositionAndOrientation(drone_ids[0], pos0, orn, physicsClientId=client)
            p.resetDebugVisualizerCamera(
                cameraDistance=10, cameraYaw=45, cameraPitch=-25,
                cameraTargetPosition=pos0, physicsClientId=client
            )
            p.stepSimulation(physicsClientId=client)
        except p.error:
            break

        time.sleep(1./60.)

    try:
        p.disconnect(physicsClientId=client)
    except Exception:
        pass
    print("Done.")
