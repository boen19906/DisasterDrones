# SYSTEM PROMPT FOR AI CODING AGENT: Implementation of Drone Mesh RL Simulation

## Context & Objective
You are an expert Machine Learning Engineer and Python Developer. You are tasked with writing a Multi-Agent Reinforcement Learning (MARL) environment.

We will use **`gym-pybullet-drones`** (specifically the updated `gymnasium` and `PettingZoo` compatible version) as our core physics and rendering engine. 

The environment simulates a UAV swarm establishing a self-healing RF mesh network to route ground survivor data to a passing Low Earth Orbit (LEO) satellite.

**CRITICAL REQUIREMENT:** 
The RF and satellite routing math in `network.py` must be decoupled. The synthetic link budget equations must be written so they can easily be replaced by hardware-in-the-loop telemetry from a physical LEO satellite testbed in the future.

Please implement the following files exactly as specified. 

---

### File 1: `envs/terrain.py` (Topography & Line of Sight)

**Objective:** Generate a continuous 2.5D heightmap and calculate line-of-sight (LoS) interference to act as obstacles in the PyBullet world.

**Implementation Steps:**
1. Create a `Terrain` class.
2. **Heightmap Generation:** Generate a 2D numpy array representing rugged mountainous terrain using a combination of overlapping 2D Gaussian functions.
3. **PyBullet Integration:** Provide a helper method `get_urdf_path()` that dynamically generates a `.urdf` or `.obj` file representing this heightmap so it can be loaded into the PyBullet physics server as a static collision mesh.
4. **Raycasting (`check_los(point_A, point_B)`):**
   * Instead of PyBullet's built-in ray test (which can be slow for thousands of checks), write a fast `numpy` raycast using the 2D heightmap array. 
   * Logic: Interpolate the 3D line between A and B. If the terrain height array $> z$ at any point, return `False` (Line of Sight blocked). Otherwise, return `True`.

---

### File 2: `envs/network.py` (RF Mesh & LEO Satellite Failover)

**Objective:** Handle the graph topology, RF path loss, and moving LEO satellite connections.

**Implementation Steps:**
1. **The LEO Satellite (`Satellite` class):**
   * State: `orbit_altitude` (550,000 m), `velocity`, `current_x`, `current_y`.
   * Method `update_position(dt)`: Move the satellite along a linear vector.
   * Method `get_elevation_angle(drone_x, drone_y, drone_z)`: Calculate angle above horizon.
2. **Link Budget Logic (`calculate_link(drone_a, drone_b, has_los)`):**
   * Implement Free Space Path Loss (FSPL) formula using inverse-square distance decay. 
   * If `has_los == False`, subtract 30 dBm.
   * Return a boolean: `True` if signal $> -85$ dBm, else `False`.
3. **Graph Topology (`calculate_throughput(...)`):**
   * Use `networkx.Graph()`. Add drones and survivors as nodes. Add edges where `calculate_link` is `True`.
   * **Failover Logic:** A drone node can connect to the Satellite node if `role == Gateway`, `has_los == True`, AND the elevation angle $> 25^\circ$.
   * Return the number of survivor nodes that have a valid shortest path to the Satellite node.

---

### File 3: `envs/survivors.py` (Ground Agents)

**Objective:** Simulate moving ground targets (the search and rescue teams).

**Implementation Steps:**
1. Create a `SurvivorCluster` class.
2. **Movement (`step(dt)`):** Implement a simple 2D random walk (Brownian motion) restricted by the terrain boundaries. 
3. Provide a method `get_positions()` returning a `numpy` array of all `[x, y, z]` coordinates for the network graph to use.

---

### File 4: `envs/disaster_env.py` (The PyBullet Aviary Wrapper)

**Objective:** Bridge the custom disaster logic with `gym-pybullet-drones`.

**Implementation Steps:**
1. Inherit from `BaseMultiagentAviary` (from `gym_pybullet_drones.envs.BaseMultiagentAviary` or similar PettingZoo-compatible PyBullet aviary).
2. **Initialization (`__init__`)**:
   * Spawn $N$ Bitcraze Crazyflie (`CF2X`) drones.
   * Initialize `Terrain`, `Satellite`, and `Survivors`.
   * Load the generated Terrain URDF into the PyBullet server (`p.loadURDF`).
3. **Observation Space Customization:**
   * Override `_computeObs()` to append the satellite's relative position, local wind vector, nearest neighbor RSSI values, and current drone `role` (0=Relay, 1=Gateway) to the default PyBullet kinematic observations.
4. **Step Execution Customization:**
   * In `step()`, call the parent `super().step(actions)`.
   * Apply PyBullet wind forces (`p.applyExternalForce`) using an Ornstein-Uhlenbeck process to simulate mountain gusts.
   * Update the Satellite and Survivors.
   * Execute the `network.py` graph logic to determine connectivity.
5. **Battery Logic Override:**
   * Track custom battery levels. 
   * Gateway Mode penalty: Drones assigned the Gateway role drain battery $4\times$ faster to simulate the Starlink phased array power draw.
   * If a drone's battery hits 0, disable its motors in PyBullet (`p.setJointMotorControlArray`).
6. **Reward Function (`_computeReward()`)**:
   * `+10` for each connected survivor routed to the satellite.
   * `-50` for physical collision with terrain or other drones (use PyBullet's built-in collision detection).
   * `-40` for battery depletion.

---

### File 5: `train.py` (MAPPO Execution)

**Objective:** The entry point to train the model.

**Implementation Steps:**
1. Import the custom `DisasterEnv` aviary.
2. Wrap the environment using PyBullet-Drones' `PettingZoo` wrapper so it is compatible with standard MARL libraries.
3. Set up a MAPPO (Multi-Agent PPO) training loop using `ray.rllib` or `CleanRL`.
4. Run the training purely **headless** (no PyBullet GUI) to maximize iterations per second. Save model checkpoints every 100 iterations.