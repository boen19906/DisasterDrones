import numpy as np
import pybullet as p
from gym_pybullet_drones.envs.VelocityAviary import VelocityAviary
from gym_pybullet_drones.utils.enums import DroneModel

class DisasterDroneEnv(VelocityAviary):
    def __init__(self, **kwargs):
        self.target_pos = np.array([2.0, 2.0, 1.0])
        super().__init__(**kwargs)
        
    def _addObstacles(self):
        super()._addObstacles()
        # Add a wall/debris
        p.loadURDF("cube.urdf", [1.0, 1.0, 0.5], p.getQuaternionFromEuler([0, 0, 0]), globalScaling=0.5, physicsClientId=self.CLIENT)
        p.loadURDF("cube.urdf", [1.0, -1.0, 0.5], p.getQuaternionFromEuler([0, 0, 0]), globalScaling=0.5, physicsClientId=self.CLIENT)
        # Add target marker
        p.loadURDF("sphere2.urdf", self.target_pos, p.getQuaternionFromEuler([0, 0, 0]), globalScaling=0.1, physicsClientId=self.CLIENT)

    def _computeReward(self):
        drone_pos = self._getDroneStateVector(0)[0:3]
        return -1.0 * np.linalg.norm(self.target_pos - drone_pos)
        
    def _computeTerminated(self):
        drone_pos = self._getDroneStateVector(0)[0:3]
        if np.linalg.norm(self.target_pos - drone_pos) < 0.2:
            print("Target reached!")
        if drone_pos[2] < 0.05:
            pass # Just let it hit the ground instead of resetting
        return False
        
    def _computeTruncated(self):
        return (self.step_counter / self.PYB_FREQ) > 600.0

    def _computeInfo(self):
        return {}

import time

if __name__ == "__main__":
    env = DisasterDroneEnv(
        drone_model=DroneModel.CF2X,
        num_drones=1,
        initial_xyzs=np.array([[0.0, 0.0, 1.0]]), # Start the drone 1 meter in the air
        gui=True,
        record=False,
        obstacles=True
    )

    obs, info = env.reset()
    
    # Move the camera back so we aren't looking from inside the drone!
    p.resetDebugVisualizerCamera(cameraDistance=3.0, cameraYaw=-45, cameraPitch=-30, cameraTargetPosition=[0.5, 0.5, 0.5], physicsClientId=env.CLIENT)

    print("Running visual sandbox. Press CTRL+C to stop.")
    print("Controls: ARROWS (Move Forward/Back/Left/Right), I/K (Throttle Up/Down)")

    try:
        while True:
            keys = p.getKeyboardEvents()
            vx, vy, vz = 0.0, 0.0, 0.0
            
            # Arrow keys for horizontal movement
            if keys.get(p.B3G_UP_ARROW, 0) & p.KEY_IS_DOWN: vx = 1.0
            if keys.get(p.B3G_DOWN_ARROW, 0) & p.KEY_IS_DOWN: vx = -1.0
            if keys.get(p.B3G_LEFT_ARROW, 0) & p.KEY_IS_DOWN: vy = 1.0
            if keys.get(p.B3G_RIGHT_ARROW, 0) & p.KEY_IS_DOWN: vy = -1.0
            
            # I/K for Throttle (Altitude)
            if keys.get(ord('i'), 0) & p.KEY_IS_DOWN or keys.get(ord('I'), 0) & p.KEY_IS_DOWN: vz = 1.0
            if keys.get(ord('k'), 0) & p.KEY_IS_DOWN or keys.get(ord('K'), 0) & p.KEY_IS_DOWN: vz = -1.0
            
            # The 4th value is the speed multiplier (fraction of MAX_SPEED), NOT yaw!
            # If we don't set this to 1.0, the drone thinks we want 0 speed.
            action = np.array([[vx, vy, vz, 1.0]]) 
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Slow down the simulation to real-time
            time.sleep(1. / env.PYB_FREQ)
            
            if terminated or truncated:
                obs, info = env.reset()
                # Reset camera again just in case reset clears it
                p.resetDebugVisualizerCamera(cameraDistance=3.0, cameraYaw=-45, cameraPitch=-30, cameraTargetPosition=[0.5, 0.5, 0.5], physicsClientId=env.CLIENT)
    except (KeyboardInterrupt, p.error):
        pass
    finally:
        try:
            env.close()
        except p.error:
            pass
