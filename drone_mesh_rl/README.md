# 🛰️ Drone Mesh RL: Autonomous LEO-Integrated Drone Swarm Network

Autonomous UAV Swarm establishing a self-healing RF mesh network to discover and route ground survivor telemetry to a passing Low Earth Orbit (LEO) satellite across mountain terrain.

---

## 🏗️ Architecture

```
drone_mesh_rl/
├── envs/
│   ├── terrain.py       # Continuous 2.5D Gaussian mountain heightmap & fast raycast Line-of-Sight
│   ├── weather.py       # Ornstein-Uhlenbeck wind vectors, ridgeline gusts, and battery drain
│   ├── network.py       # FSPL link budget, NetworkX routing, & LEO satellite orbital sweep
│   ├── survivors.py     # Clustered ground survivors with downhill drift and discovery state
│   └── disaster_env.py  # PettingZoo ParallelEnv multi-agent simulation with PyBullet physics
├── models/
│   ├── actor_critic.py  # Decentralized Actor + Centralized Critic (MAPPO CTDE architecture)
│   └── buffer.py        # Multi-agent rollout buffer with Generalized Advantage Estimation (GAE)
├── train.py             # Headless MAPPO training pipeline (~440+ FPS on CPU)
├── evaluate.py          # Benchmark policy vs random baseline across survivor & battery metrics
└── visualize_env.py     # 3D PyBullet visualizer with RF mesh links & LEO uplink beams
```

---

## ⚡ Quick Start

### 1. Training Swarm Policy (MAPPO)
Run headless training to train the cooperative mesh swarm:
```bash
python train.py --num_drones 5 --num_clusters 4 --total_timesteps 50000 --save_path models/mappo_drone_mesh.pt
```

Key arguments:
- `--num_drones`: Swarm size (default: 5)
- `--num_clusters`: Survivor clusters in the mountains (default: 4)
- `--env_size`: Terrain dimension in meters (default: 100.0)
- `--total_timesteps`: Environment steps to train (default: 50000)
- `--rollout_steps`: Steps per rollout buffer (default: 256)

### 2. Evaluating Policy
Benchmark a trained checkpoint or a random baseline:
```bash
# Evaluate trained model
python evaluate.py --model_path models/mappo_drone_mesh.pt --episodes 10

# Evaluate random baseline
python evaluate.py --random --episodes 5
```

Outputs metrics including:
- Survivor discovery rate (%)
- Number of connected survivors routed to LEO satellite
- Battery retention
- Swarm mission success rate

### 3. Interactive 3D PyBullet Visualizer
Watch the swarm navigate mountain ridges, establish RF mesh links, and uplink to the passing satellite (now stabilized without strobe flickering):
```bash
# Autonomous exploration demo
python visualize_env.py

# Playback trained model
python visualize_env.py --model_path models/mappo_drone_mesh.pt
```

### 4. Mission Operations Control Center (Web UI)
Launch the browser-based Command & Control Dashboard for clean, intuitive real-time monitoring:
```bash
# Start the Mission Control Center
python server/telemetry_server.py

# Optional: with 3D PyBullet physics window
python server/telemetry_server.py --gui
```
Open **`http://localhost:8000`** in your browser.

Control Center Features:
- **Tactical 2D Radar Map**: Mountain grid with drone positions, dynamic cyan RF links, coverage radius, stealth red unknown beacons, and green discovered survivor crosshairs.
- **LEO Satellite Tracker**: Orbital sweep vector, elevation angle gauge, and blackout window countdown.
- **Swarm Node Health Table**: Real-time battery gauges, role indicators (click to toggle Gateway/Relay), altitude, and coordinates.
- **Interactive Command Deck**: Pause, Resume, Reset mission, Force satellite pass, and Policy toggle.

---

## 🎯 Reward & Penalty Formulation

| Event | Reward | Description |
|---|---|---|
| **Cluster Discovery** | `+20.0` | Shared bonus when a new survivor cluster is located |
| **LEO Routing** | `+5.0 / step` | Per connected survivor maintaining a route to satellite |
| **All Survivors Discovered** | `+100.0` | Episode mission completion bonus |
| **Energy Efficiency** | `+1.0 / step` | Bonus for operating with battery > 50% |
| **Terrain Collision** | `-50.0` | Heavy penalty for crashing into mountain terrain |
| **Mid-air Collision** | `-30.0` | Drone-drone proximity collision penalty |
| **Battery Depletion** | `-40.0` | Penalty if drone runs out of battery |
| **Low Battery Alert** | `-2.0 / step` | Penalty when battery drops below 20% |
| **Broken Mesh Link** | `-5.0` | Penalty when a survivor loses connectivity |
| **Uplink Inaction** | `-3.0 / step` | Penalty if satellite is visible but no gateway is uplinking |
| **High Altitude** | `-1.0 / step` | Discourages excessive high altitude energy waste |
