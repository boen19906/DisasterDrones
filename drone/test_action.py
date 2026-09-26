import numpy as np
import time
from gym_pybullet_drones.envs.VelocityAviary import VelocityAviary
from gym_pybullet_drones.utils.enums import DroneModel

env = VelocityAviary(drone_model=DroneModel.CF2X, num_drones=1, gui=False, record=False)
env.reset()

print("Initial Z:", env._getDroneStateVector(0)[2])
for i in range(100):
    action = np.array([[0.0, 0.0, 1.0, 0.0]]) # 1 m/s UP
    env.step(action)
    if i % 20 == 0:
        print(f"Step {i} Z:", env._getDroneStateVector(0)[2])

env.close()
