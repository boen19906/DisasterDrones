"""
telemetry_server.py — Autonomous Swarm Mission Control Center.

FastAPI + WebSocket backend broadcasting real-time drone mesh telemetry
and hosting an interactive high-tech Mission Control Operations Dashboard.

Usage:
  python server/telemetry_server.py                      # Starts web dashboard on http://localhost:8000
  python server/telemetry_server.py --gui                # Also opens PyBullet 3D window
  python server/telemetry_server.py --model_path models/mappo_drone_mesh.pt
"""

import os
import sys
import time
import json
import asyncio
import argparse
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure parent directory is in path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from envs.disaster_env import DisasterMeshEnv
from models.actor_critic import MAPPOModel

app = FastAPI(title="Drone Mesh Mission Control")


class SwarmSimulationManager:
    """Manages the disaster mesh simulation loop and handles telemetry broadcasting."""

    def __init__(self, num_drones=5, num_clusters=4, env_size=100.0, model_path="", gui=False):
        self.num_drones = num_drones
        self.num_clusters = num_clusters
        self.env_size = env_size
        self.gui = gui
        self.model_path = model_path

        self.env = None
        self.model = None
        self.is_paused = False
        self.sim_speed = 1.0  # 1.0 = real-time
        self.mode = "heuristic"  # 'model' or 'heuristic'
        self.step_num = 0
        self.obs_dict = None
        self.info_dict = None

        self._init_env()
        self._init_model()

    def _init_env(self):
        if self.env is not None:
            self.env.close()
        self.env = DisasterMeshEnv(
            num_drones=self.num_drones,
            num_clusters=self.num_clusters,
            env_size=self.env_size,
            max_steps=5000,
            render_mode="human" if self.gui else None,
            seed=42,
        )
        self.obs_dict, self.info_dict = self.env.reset()
        self.step_num = 0

    def _init_model(self):
        if self.model_path and os.path.exists(self.model_path):
            try:
                self.model = MAPPOModel(num_drones=self.num_drones, obs_dim=25, act_dim=5)
                self.model.load(self.model_path, map_location="cpu")
                self.model.eval()
                self.mode = "model"
                print(f"[MODEL] Successfully loaded MAPPO checkpoint: {self.model_path}")
            except Exception as e:
                print(f"[WARNING] Could not load model ({e}), default to heuristic.")
                self.model = None
                self.mode = "heuristic"

    def reset_sim(self):
        self.obs_dict, self.info_dict = self.env.reset()
        self.step_num = 0

    def force_satellite_pass(self):
        """Forces the satellite to begin an immediate overhead pass."""
        self.env.satellite.current_x = -self.env_size / 2 * 0.9
        self.env.satellite.current_y = 0.0
        self.env.satellite.visible = True
        self.env.satellite.blackout_timer = 0.0

    def set_drone_role(self, drone_idx, role_int):
        if 0 <= drone_idx < self.num_drones:
            self.env.gateway_roles[drone_idx] = role_int

    def step(self):
        if self.is_paused or self.env is None:
            return

        self.step_num += 1
        agent_names = self.env.possible_agents

        if self.mode == "model" and self.model is not None:
            obs_array = np.array([self.obs_dict[a] for a in agent_names], dtype=np.float32)
            state = obs_array.flatten()
            with torch.no_grad():
                obs_t = torch.tensor(obs_array, dtype=torch.float32)
                state_t = torch.tensor(state, dtype=torch.float32)
                actions_t, _, _, _ = self.model.get_action_and_value(obs_t, state_t, deterministic=True)
            actions_np = actions_t.cpu().numpy()
            actions = {
                agent_names[i]: actions_np[i]
                for i in range(self.num_drones)
                if agent_names[i] in self.env.agents
            }
        else:
            # Autonomous cooperative patrol heuristic:
            # Gateway drone hovers/sweeps central corridor, Relays fan outwards in search spirals
            actions = {}
            t = self.step_num * 0.04
            for i in range(self.num_drones):
                if not self.env.drone_alive[i]:
                    continue
                agent = agent_names[i]
                is_gw = bool(self.env.gateway_roles[i])
                if is_gw:
                    # Gateway patrols central high ground
                    vx = 0.3 * np.cos(t * 0.5)
                    vy = 0.3 * np.sin(t * 0.5)
                    vz = 0.1 * np.sin(t)
                    actions[agent] = np.array([vx, vy, vz, 1.0, 0.0], dtype=np.float32)
                else:
                    angle = t + (2 * np.pi * i / self.num_drones)
                    radius_speed = 0.65
                    vx = np.cos(angle) * radius_speed
                    vy = np.sin(angle) * radius_speed
                    vz = 0.08 * np.sin(t * 2 + i)
                    actions[agent] = np.array([vx, vy, vz, -1.0, 0.0], dtype=np.float32)

        self.obs_dict, rews, terms, truncs, infos = self.env.step(actions)

        if all(terms.values()) or all(truncs.values()):
            self.reset_sim()

    def get_telemetry_payload(self):
        """Construct full JSON serializable telemetry state."""
        env = self.env
        if env is None:
            return {}

        # Drones
        drones_data = []
        for i in range(self.num_drones):
            pos = env.drone_positions[i].tolist()
            vel = env.drone_velocities[i].tolist()
            drones_data.append({
                "id": i,
                "x": round(pos[0], 2),
                "y": round(pos[1], 2),
                "z": round(pos[2], 2),
                "vx": round(vel[0], 2),
                "vy": round(vel[1], 2),
                "vz": round(vel[2], 2),
                "battery": round(float(env.battery_levels[i]), 1),
                "is_gateway": bool(env.gateway_roles[i]),
                "is_alive": bool(env.drone_alive[i]),
            })

        # Survivors
        surv_data = []
        surv_pos = env.survivors.get_positions()
        for s_idx in range(env.survivors.num_survivors):
            surv_data.append({
                "id": s_idx,
                "x": round(float(surv_pos[s_idx, 0]), 2),
                "y": round(float(surv_pos[s_idx, 1]), 2),
                "z": round(float(surv_pos[s_idx, 2]), 2),
                "cluster_id": int(env.survivors.cluster_ids[s_idx]),
                "discovered": bool(env.survivors.discovered[s_idx]),
            })

        # Active mesh edges
        active_links = []
        if env.mesh_graph is not None:
            for u, v, data in env.mesh_graph.edges(data=True):
                # Filter out SAT links
                if u != "SAT" and v != "SAT":
                    rssi = data.get("rssi", -70.0)
                    active_links.append({"source": str(u), "target": str(v), "rssi": round(float(rssi), 1)})

        # Satellite
        sat = env.satellite
        drone_gw_indices = [i for i in range(self.num_drones) if env.gateway_roles[i] and env.drone_alive[i]]
        best_elev = -90.0
        uplink_active = False
        if drone_gw_indices:
            gw_pos = env.drone_positions[drone_gw_indices[0]]
            best_elev = sat.get_elevation_angle(gw_pos[0], gw_pos[1], gw_pos[2])
            uplink_active = sat.can_uplink(gw_pos[0], gw_pos[1], gw_pos[2])

        satellite_data = {
            "x": round(float(sat.current_x), 1),
            "y": round(float(sat.current_y), 1),
            "altitude": round(float(sat.orbit_altitude), 0),
            "visible": bool(sat.visible),
            "elevation_deg": round(float(best_elev), 1),
            "blackout_timer": round(float(sat.blackout_timer), 1),
            "uplink_active": bool(uplink_active),
            "pass_count": int(sat.pass_count),
        }

        # Weather
        wind = env.weather.get_global_wind().tolist()
        weather_data = {
            "wind_x": round(float(wind[0]), 3),
            "wind_y": round(float(wind[1]), 3),
            "wind_z": round(float(wind[2]), 3),
            "wind_mag": round(float(np.linalg.norm(wind)), 3),
            "storm_active": bool(env.weather.is_storm_active()),
        }

        # Stats
        disc_stats = env.survivors.get_discovery_stats()
        stats_data = {
            "step": self.step_num,
            "total_survivors": disc_stats["total_survivors"],
            "discovered_survivors": disc_stats["discovered_survivors"],
            "discovery_rate": round(disc_stats["discovery_rate"] * 100.0, 1),
            "connected_survivors": int(env.connected_survivors),
            "mode": self.mode,
            "paused": self.is_paused,
            "mean_battery": round(float(env.battery_levels.mean()), 1),
        }

        return {
            "type": "telemetry",
            "drones": drones_data,
            "survivors": surv_data,
            "satellite": satellite_data,
            "links": active_links,
            "weather": weather_data,
            "stats": stats_data,
            "env_size": self.env_size,
            "structures": [
                box.to_dict() for box in getattr(env.terrain, "structures", [])
            ],
        }


# Global sim manager instance
sim_manager = None


# ---------------------------------------------------------------------------
# Mission Control Dashboard HTML/CSS/JS Frontend
# ---------------------------------------------------------------------------

CONTROL_CENTER_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>🛰️ Drone Mesh Network — Mission Operations Control Center</title>
  <style>
    :root {
      --bg: #090d16;
      --card-bg: rgba(16, 24, 40, 0.85);
      --card-border: rgba(45, 65, 105, 0.45);
      --accent-cyan: #00f2fe;
      --accent-blue: #4facfe;
      --accent-green: #00e676;
      --accent-red: #ff3366;
      --accent-yellow: #ffd600;
      --text-main: #f0f4f8;
      --text-muted: #8ba2c4;
      --radar-grid: rgba(0, 242, 254, 0.12);
    }

    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", monospace; }

    body {
      background: var(--bg);
      color: var(--text-main);
      overflow-x: hidden;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }

    /* Header Bar */
    header {
      background: rgba(10, 15, 28, 0.95);
      border-bottom: 1px solid var(--card-border);
      padding: 12px 24px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      position: sticky;
      top: 0;
      z-index: 100;
      backdrop-filter: blur(10px);
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
    }

    .brand-icon {
      font-size: 24px;
      background: linear-gradient(135deg, var(--accent-cyan), var(--accent-blue));
      border-radius: 8px;
      width: 40px;
      height: 40px;
      display: flex;
      align-items: center;
      justify-content: center;
    }

    .brand h1 {
      font-size: 18px;
      letter-spacing: 1px;
      text-transform: uppercase;
      font-weight: 700;
      background: linear-gradient(90deg, #fff, var(--accent-cyan));
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }

    .brand p { font-size: 11px; color: var(--text-muted); }

    .header-pills {
      display: flex;
      gap: 14px;
      align-items: center;
    }

    .status-pill {
      background: rgba(0, 230, 118, 0.15);
      border: 1px solid var(--accent-green);
      color: var(--accent-green);
      padding: 5px 12px;
      border-radius: 20px;
      font-size: 11px;
      font-weight: 600;
      display: flex;
      align-items: center;
      gap: 6px;
    }

    .status-pill.blackout {
      background: rgba(255, 51, 102, 0.15);
      border-color: var(--accent-red);
      color: var(--accent-red);
    }

    .dot { width: 8px; height: 8px; border-radius: 50%; background: currentColor; animation: pulse 1.5s infinite; }
    @keyframes pulse { 0%, 100% { opacity: 1; transform: scale(1); } 50% { opacity: 0.4; transform: scale(0.8); } }

    /* Main Grid Layout */
    .dashboard-grid {
      display: grid;
      grid-template-columns: 1fr 390px;
      gap: 16px;
      padding: 16px 24px;
      flex: 1;
    }

    /* Card Panels */
    .panel {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 16px;
      backdrop-filter: blur(16px);
      display: flex;
      flex-direction: column;
      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
    }

    .panel-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 14px;
      border-bottom: 1px solid rgba(255,255,255,0.06);
      padding-bottom: 8px;
    }

    .panel-title {
      font-size: 13px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.8px;
      color: var(--text-muted);
      display: flex;
      align-items: center;
      gap: 8px;
    }

    /* Radar Canvas Area */
    .radar-container {
      position: relative;
      flex: 1;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      background: #040810;
      border-radius: 8px;
      border: 1px solid rgba(0, 242, 254, 0.2);
      overflow: hidden;
      min-height: 540px;
    }

    #radarCanvas {
      width: 100%;
      height: 100%;
      display: block;
      cursor: crosshair;
    }

    .radar-overlay-stats {
      position: absolute;
      top: 12px;
      left: 12px;
      background: rgba(10, 15, 28, 0.75);
      border: 1px solid var(--card-border);
      border-radius: 6px;
      padding: 8px 12px;
      font-size: 11px;
      line-height: 1.6;
      backdrop-filter: blur(6px);
      pointer-events: none;
    }

    .radar-legend {
      position: absolute;
      bottom: 12px;
      left: 12px;
      background: rgba(10, 15, 28, 0.8);
      border: 1px solid var(--card-border);
      border-radius: 6px;
      padding: 6px 12px;
      font-size: 11px;
      display: flex;
      gap: 12px;
      backdrop-filter: blur(6px);
    }

    .legend-item { display: flex; align-items: center; gap: 6px; }
    .legend-box { width: 10px; height: 10px; border-radius: 2px; }

    /* Side Column Panels */
    .side-col {
      display: flex;
      flex-direction: column;
      gap: 14px;
    }

    /* KPI Cards Row */
    .kpi-row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }

    .kpi-card {
      background: rgba(255, 255, 255, 0.03);
      border: 1px solid rgba(255, 255, 255, 0.07);
      border-radius: 8px;
      padding: 10px 12px;
    }

    .kpi-label { font-size: 10px; text-transform: uppercase; color: var(--text-muted); font-weight: 600; }
    .kpi-value { font-size: 20px; font-weight: 700; margin-top: 4px; color: var(--text-main); }
    .kpi-sub { font-size: 10px; color: var(--text-muted); margin-top: 2px; }

    /* Drone Table */
    .drone-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 11px;
    }

    .drone-table th {
      text-align: left;
      padding: 6px 8px;
      color: var(--text-muted);
      border-bottom: 1px solid var(--card-border);
      font-weight: 600;
    }

    .drone-table td {
      padding: 8px;
      border-bottom: 1px solid rgba(255,255,255,0.04);
    }

    .badge-role {
      padding: 2px 7px;
      border-radius: 4px;
      font-weight: 700;
      font-size: 9px;
      text-transform: uppercase;
      cursor: pointer;
    }

    .badge-gateway { background: rgba(255, 51, 102, 0.2); color: var(--accent-red); border: 1px solid var(--accent-red); }
    .badge-relay { background: rgba(0, 242, 254, 0.15); color: var(--accent-cyan); border: 1px solid var(--accent-cyan); }
    .badge-dead { background: rgba(255, 255, 255, 0.1); color: #888; }

    .battery-bar-container {
      width: 60px;
      height: 8px;
      background: rgba(255,255,255,0.1);
      border-radius: 4px;
      overflow: hidden;
      display: inline-block;
      vertical-align: middle;
      margin-right: 6px;
    }

    .battery-fill { height: 100%; border-radius: 4px; transition: width 0.3s ease; }
    .bat-high { background: var(--accent-green); }
    .bat-med { background: var(--accent-yellow); }
    .bat-low { background: var(--accent-red); }

    /* Control Buttons */
    .btn-row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin-top: 8px;
    }

    button {
      background: rgba(0, 242, 254, 0.1);
      border: 1px solid var(--accent-cyan);
      color: var(--accent-cyan);
      padding: 9px 12px;
      border-radius: 6px;
      font-size: 11px;
      font-weight: 700;
      cursor: pointer;
      transition: all 0.2s;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 6px;
    }

    button:hover {
      background: var(--accent-cyan);
      color: #000;
      box-shadow: 0 0 15px rgba(0, 242, 254, 0.4);
    }

    button.btn-danger {
      border-color: var(--accent-red);
      color: var(--accent-red);
      background: rgba(255, 51, 102, 0.1);
    }

    button.btn-danger:hover {
      background: var(--accent-red);
      color: #fff;
      box-shadow: 0 0 15px rgba(255, 51, 102, 0.4);
    }

    button.btn-warn {
      border-color: var(--accent-yellow);
      color: var(--accent-yellow);
      background: rgba(255, 214, 0, 0.1);
    }

    button.btn-warn:hover {
      background: var(--accent-yellow);
      color: #000;
      box-shadow: 0 0 15px rgba(255, 214, 0, 0.4);
    }
  </style>
</head>
<body>

  <header>
    <div class="brand">
      <div class="brand-icon">🛰️</div>
      <div>
        <h1>Autonomous Swarm Operations Center</h1>
        <p>LEO-Integrated RF Mesh Network | Mountain Search & Rescue</p>
      </div>
    </div>
    <div class="header-pills">
      <div class="status-pill" id="satStatusPill">
        <div class="dot"></div>
        <span id="satStatusText">LEO SATELLITE: SCANNING</span>
      </div>
      <div class="status-pill" id="weatherPill">
        <span id="weatherText">WIND: 0.0 N</span>
      </div>
      <div class="status-pill" id="modePill" style="border-color: var(--accent-cyan); color: var(--accent-cyan);">
        <span id="modeText">MODE: HEURISTIC</span>
      </div>
    </div>
  </header>

  <div class="dashboard-grid">

    <!-- Left: Tactical 2D Radar Canvas -->
    <div class="panel">
      <div class="panel-header">
        <div class="panel-title">
          <span>📡</span> Tactical Radar & RF Mesh Topology Map (100m x 100m)
        </div>
        <div style="font-size: 11px; color: var(--accent-cyan);" id="fpsCounter">30 FPS</div>
      </div>

      <div class="radar-container">
        <canvas id="radarCanvas"></canvas>

        <div class="radar-overlay-stats">
          <div><strong>SWARM:</strong> <span id="overlaySwarm">5 Drones</span></div>
          <div><strong>SAT ELEV:</strong> <span id="overlayElev">--°</span></div>
          <div><strong>ROUTED TO SAT:</strong> <span id="overlayConn" style="color: var(--accent-green);">0 Survivors</span></div>
        </div>

        <div class="radar-legend">
          <div class="legend-item"><div class="legend-box" style="background: var(--accent-red);"></div> Gateway (Uplink)</div>
          <div class="legend-item"><div class="legend-box" style="background: var(--accent-cyan);"></div> Relay Drone</div>
          <div class="legend-item"><div class="legend-box" style="background: var(--accent-green);"></div> Found Survivor</div>
          <div class="legend-item"><div class="legend-box" style="background: #ff5252;"></div> Unknown Survivor</div>
          <div class="legend-item"><div class="legend-box" style="background: var(--accent-yellow);"></div> LEO Orbit Beam</div>
        </div>
      </div>
    </div>

    <!-- Right: Operations Panel -->
    <div class="side-col">

      <!-- Mission KPIs -->
      <div class="panel">
        <div class="panel-header">
          <div class="panel-title"><span>🎯</span> Mission Progress</div>
          <span id="stepCounter" style="font-size: 11px; color: var(--text-muted);">Step: 0</span>
        </div>
        <div class="kpi-row">
          <div class="kpi-card">
            <div class="kpi-label">Survivors Located</div>
            <div class="kpi-value" id="kpiFound" style="color: var(--accent-green);">0 / 0</div>
            <div class="kpi-sub" id="kpiRate">0.0% Discovered</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-label">LEO Link Delivery</div>
            <div class="kpi-value" id="kpiConn" style="color: var(--accent-cyan);">0</div>
            <div class="kpi-sub">Survivors Streamed</div>
          </div>
        </div>
      </div>

      <!-- Satellite Telemetry -->
      <div class="panel">
        <div class="panel-header">
          <div class="panel-title"><span>🛰️</span> LEO Satellite Telemetry</div>
          <span id="passCounter" style="font-size: 11px; color: var(--accent-yellow);">Pass #0</span>
        </div>
        <div class="kpi-row">
          <div class="kpi-card">
            <div class="kpi-label">Elevation Angle</div>
            <div class="kpi-value" id="satElevValue">--°</div>
            <div class="kpi-sub">Threshold: > 25.0°</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-label">Orbit Status</div>
            <div class="kpi-value" id="satStatusValue" style="font-size: 14px;">IN PASS</div>
            <div class="kpi-sub" id="satTimerValue">Visible overhead</div>
          </div>
        </div>
      </div>

      <!-- Drone Swarm Health Table -->
      <div class="panel" style="flex: 1;">
        <div class="panel-header">
          <div class="panel-title"><span>🚁</span> Swarm Node Telemetry</div>
          <span id="meanBatLabel" style="font-size: 11px; color: var(--accent-green);">Avg Bat: 100%</span>
        </div>
        <div style="overflow-y: auto; max-height: 220px;">
          <table class="drone-table">
            <thead>
              <tr>
                <th>Drone</th>
                <th>Role</th>
                <th>Battery</th>
                <th>Alt</th>
                <th>Coords</th>
              </tr>
            </thead>
            <tbody id="droneTableBody">
              <!-- Dynamically populated -->
            </tbody>
          </table>
        </div>
      </div>

      <!-- Command Deck -->
      <div class="panel">
        <div class="panel-header">
          <div class="panel-title"><span>⚙️</span> Swarm Command Deck</div>
        </div>
        <div class="btn-row">
          <button id="btnPause">⏸️ Pause</button>
          <button id="btnReset" class="btn-warn">🔄 Reset Mission</button>
        </div>
        <div class="btn-row">
          <button id="btnForceSat">🛰️ Trigger LEO Pass</button>
          <button id="btnToggleMode">🤖 Toggle Policy</button>
        </div>
      </div>

    </div>
  </div>

  <script>
    const canvas = document.getElementById('radarCanvas');
    const ctx = canvas.getContext('2d');

    let ws = null;
    let lastData = null;
    let frameCount = 0;
    let lastFpsTime = performance.now();

    function resizeCanvas() {
      const rect = canvas.parentElement.getBoundingClientRect();
      canvas.width = rect.width;
      canvas.height = rect.height;
    }
    window.addEventListener('resize', resizeCanvas);
    resizeCanvas();

    // Connect WebSocket
    function connectWS() {
      const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
      ws = new WebSocket(`${proto}//${location.host}/ws/telemetry`);

      ws.onopen = () => {
        console.log('[WS] Connected to Drone Mesh Telemetry server.');
      };

      ws.onmessage = (event) => {
        const payload = JSON.parse(event.data);
        if (payload.type === 'telemetry') {
          lastData = payload;
          updateDashboard(payload);
        }
      };

      ws.onclose = () => {
        setTimeout(connectWS, 1500);
      };
    }
    connectWS();

    function sendCommand(cmd, data = {}) {
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ command: cmd, ...data }));
      }
    }

    // Button Listeners
    document.getElementById('btnPause').onclick = () => {
      sendCommand('toggle_pause');
    };
    document.getElementById('btnReset').onclick = () => {
      sendCommand('reset');
    };
    document.getElementById('btnForceSat').onclick = () => {
      sendCommand('force_sat');
    };
    document.getElementById('btnToggleMode').onclick = () => {
      sendCommand('toggle_mode');
    };

    // Update Dashboard UI Elements
    function updateDashboard(data) {
      const stats = data.stats;
      const sat = data.satellite;
      const weather = data.weather;

      document.getElementById('stepCounter').innerText = `Step: ${stats.step}`;
      document.getElementById('kpiFound').innerText = `${stats.discovered_survivors} / ${stats.total_survivors}`;
      document.getElementById('kpiRate').innerText = `${stats.discovery_rate}% Located`;
      document.getElementById('kpiConn').innerText = `${stats.connected_survivors}`;
      document.getElementById('meanBatLabel').innerText = `Avg Bat: ${stats.mean_battery}%`;

      // Satellite Header & Card
      const satPill = document.getElementById('satStatusPill');
      const satStatusText = document.getElementById('satStatusText');
      if (sat.visible && sat.elevation_deg > 25.0) {
        satPill.className = 'status-pill';
        satStatusText.innerText = `LEO SAT: UPLINK ACTIVE (${sat.elevation_deg}°)`;
        document.getElementById('satStatusValue').innerText = 'UPLINK ACTIVE';
        document.getElementById('satStatusValue').style.color = 'var(--accent-yellow)';
        document.getElementById('satTimerValue').innerText = `Overhead at x=${sat.x}m`;
      } else if (sat.visible) {
        satPill.className = 'status-pill';
        satStatusText.innerText = `LEO SAT: IN PASS (${sat.elevation_deg}°)`;
        document.getElementById('satStatusValue').innerText = 'IN PASS';
        document.getElementById('satStatusValue').style.color = 'var(--accent-cyan)';
        document.getElementById('satTimerValue').innerText = `Angle below 25°`;
      } else {
        satPill.className = 'status-pill blackout';
        satStatusText.innerText = `LEO SAT: BLACKOUT (${sat.blackout_timer}s)`;
        document.getElementById('satStatusValue').innerText = 'BLACKOUT';
        document.getElementById('satStatusValue').style.color = 'var(--accent-red)';
        document.getElementById('satTimerValue').innerText = `Next pass in ${sat.blackout_timer}s`;
      }

      document.getElementById('satElevValue').innerText = `${sat.elevation_deg}°`;
      document.getElementById('passCounter').innerText = `Pass #${sat.pass_count}`;

      // Weather Pill
      const storm = weather.storm_active;
      document.getElementById('weatherText').innerText = `WIND: ${weather.wind_mag.toFixed(2)}N ${storm ? '⚡ STORM' : ''}`;
      if (storm) {
        document.getElementById('weatherPill').className = 'status-pill blackout';
      } else {
        document.getElementById('weatherPill').className = 'status-pill';
      }

      // Mode pill & button label
      document.getElementById('modeText').innerText = `MODE: ${stats.mode.toUpperCase()}`;
      document.getElementById('btnPause').innerText = stats.paused ? '▶️ Resume' : '⏸️ Pause';

      // Overlay text on radar
      document.getElementById('overlaySwarm').innerText = `${data.drones.length} Drones (${data.drones.filter(d => d.is_alive).length} Alive)`;
      document.getElementById('overlayElev').innerText = `${sat.elevation_deg}°`;
      document.getElementById('overlayConn').innerText = `${stats.connected_survivors} Survivors`;

      // Drone Table
      const tbody = document.getElementById('droneTableBody');
      tbody.innerHTML = '';
      data.drones.forEach(d => {
        const tr = document.createElement('tr');
        const roleClass = !d.is_alive ? 'badge-dead' : (d.is_gateway ? 'badge-gateway' : 'badge-relay');
        const roleLabel = !d.is_alive ? 'DEAD' : (d.is_gateway ? 'GATEWAY' : 'RELAY');

        let batClass = 'bat-high';
        if (d.battery < 20) batClass = 'bat-low';
        else if (d.battery < 50) batClass = 'bat-med';

        tr.innerHTML = `
          <td><strong>D${d.id}</strong></td>
          <td><span class="badge-role ${roleClass}" onclick="toggleRole(${d.id}, ${d.is_gateway ? 0 : 1})">${roleLabel}</span></td>
          <td>
            <div class="battery-bar-container">
              <div class="battery-fill ${batClass}" style="width: ${d.battery}%;"></div>
            </div>
            ${d.battery}%
          </td>
          <td>${d.z}m</td>
          <td style="color: var(--text-muted); font-size: 10px;">[${d.x}, ${d.y}]</td>
        `;
        tbody.appendChild(tr);
      });
    }

    window.toggleRole = (id, newRole) => {
      sendCommand('set_role', { drone_id: id, role: newRole });
    };

    // Render Tactical 2D Radar Canvas
    function renderRadar() {
      requestAnimationFrame(renderRadar);

      frameCount++;
      const now = performance.now();
      if (now - lastFpsTime >= 1000) {
        document.getElementById('fpsCounter').innerText = `${frameCount} FPS`;
        frameCount = 0;
        lastFpsTime = now;
      }

      const w = canvas.width;
      const h = canvas.height;
      ctx.clearRect(0, 0, w, h);

      if (!lastData) return;

      const envSize = lastData.env_size;
      const scale = Math.min(w, h) * 0.88 / envSize;
      const cx = w / 2;
      const cy = h / 2;

      function toScreen(x, y) {
        return [cx + x * scale, cy - y * scale];
      }

      // 1. Draw Mountain Topography Grid Lines
      ctx.strokeStyle = 'rgba(0, 242, 254, 0.08)';
      ctx.lineWidth = 1;
      const gridSteps = 10;
      for (let i = -50; i <= 50; i += 10) {
        const p1 = toScreen(i, -50);
        const p2 = toScreen(i, 50);
        ctx.beginPath(); ctx.moveTo(p1[0], p1[1]); ctx.lineTo(p2[0], p2[1]); ctx.stroke();

        const p3 = toScreen(-50, i);
        const p4 = toScreen(50, i);
        ctx.beginPath(); ctx.moveTo(p3[0], p3[1]); ctx.lineTo(p4[0], p4[1]); ctx.stroke();
      }

      // Map boundary box
      const bTopLeft = toScreen(-50, 50);
      const bBottomRight = toScreen(50, -50);
      ctx.strokeStyle = 'rgba(0, 242, 254, 0.35)';
      ctx.lineWidth = 2;
      ctx.strokeRect(bTopLeft[0], bTopLeft[1], bBottomRight[0] - bTopLeft[0], bBottomRight[1] - bTopLeft[1]);

      // Earthquake structures (footprints)
      (lastData.structures || []).forEach(s => {
        const p1 = toScreen(s.xmin, s.ymax);
        const p2 = toScreen(s.xmax, s.ymin);
        const wRect = p2[0] - p1[0];
        const hRect = p2[1] - p1[1];
        if (s.kind === 'rubble' || s.kind === 'fallen_wall' || s.kind === 'fallen_mast') {
          ctx.fillStyle = 'rgba(90, 70, 52, 0.55)';
        } else if (s.kind === 'school_wall' || s.kind === 'roof_slab') {
          ctx.fillStyle = 'rgba(210, 185, 120, 0.55)';
        } else {
          ctx.fillStyle = 'rgba(130, 125, 118, 0.55)';
        }
        ctx.fillRect(p1[0], p1[1], wRect, hRect);
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.18)';
        ctx.lineWidth = 1;
        ctx.strokeRect(p1[0], p1[1], wRect, hRect);
      });

      // 2. Draw Survivors
      lastData.survivors.forEach(s => {
        const [sx, sy] = toScreen(s.x, s.y);
        ctx.beginPath();
        if (s.discovered) {
          // Bright green crosshair
          ctx.fillStyle = '#00e676';
          ctx.arc(sx, sy, 4.5, 0, Math.PI * 2);
          ctx.fill();

          ctx.strokeStyle = '#00e676';
          ctx.lineWidth = 1.2;
          ctx.beginPath();
          ctx.moveTo(sx - 7, sy); ctx.lineTo(sx + 7, sy);
          ctx.moveTo(sx, sy - 7); ctx.lineTo(sx, sy + 7);
          ctx.stroke();
        } else {
          // Undiscovered stealth red beacon
          ctx.fillStyle = 'rgba(255, 51, 102, 0.65)';
          ctx.arc(sx, sy, 3.5, 0, Math.PI * 2);
          ctx.fill();
        }
      });

      // 3. Draw RF Mesh Links (Cyan lines)
      const droneMap = {};
      lastData.drones.forEach(d => { droneMap[d.id] = d; });

      lastData.links.forEach(link => {
        // Source and target can be drone_i or surv_i
        let p1 = null, p2 = null;
        if (link.source.startsWith('drone_') && link.target.startsWith('drone_')) {
          const id1 = parseInt(link.source.split('_')[1]);
          const id2 = parseInt(link.target.split('_')[1]);
          if (droneMap[id1] && droneMap[id2]) {
            p1 = toScreen(droneMap[id1].x, droneMap[id1].y);
            p2 = toScreen(droneMap[id2].x, droneMap[id2].y);
          }
        }
        if (p1 && p2) {
          ctx.strokeStyle = 'rgba(0, 242, 254, 0.65)';
          ctx.lineWidth = 1.8;
          ctx.beginPath();
          ctx.moveTo(p1[0], p1[1]);
          ctx.lineTo(p2[0], p2[1]);
          ctx.stroke();
        }
      });

      // 4. Draw Drones
      lastData.drones.forEach(d => {
        const [dx, dy] = toScreen(d.x, d.y);

        if (!d.is_alive) {
          ctx.fillStyle = '#555';
          ctx.beginPath();
          ctx.arc(dx, dy, 5, 0, Math.PI * 2);
          ctx.fill();
          return;
        }

        // Coverage radius circle
        ctx.strokeStyle = d.is_gateway ? 'rgba(255, 51, 102, 0.15)' : 'rgba(0, 242, 254, 0.12)';
        ctx.fillStyle = d.is_gateway ? 'rgba(255, 51, 102, 0.03)' : 'rgba(0, 242, 254, 0.03)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.arc(dx, dy, 32 * scale, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();

        // Drone core circle
        ctx.fillStyle = d.is_gateway ? '#ff3366' : '#00f2fe';
        ctx.beginPath();
        ctx.arc(dx, dy, 6, 0, Math.PI * 2);
        ctx.fill();

        // Pulsing ring for Gateway
        if (d.is_gateway) {
          ctx.strokeStyle = '#ff3366';
          ctx.lineWidth = 1.5;
          ctx.beginPath();
          ctx.arc(dx, dy, 9 + Math.sin(now * 0.005) * 3, 0, Math.PI * 2);
          ctx.stroke();
        }

        // Drone Label
        ctx.fillStyle = '#ffffff';
        ctx.font = '10px monospace';
        ctx.fillText(`D${d.id} [${d.battery}%]`, dx + 8, dy - 8);
      });

      // 5. Draw Satellite Trajectory & Uplink Beam
      const sat = lastData.satellite;
      const [satScrX, satScrY] = toScreen(sat.x, 48); // Near top boundary
      ctx.fillStyle = sat.visible ? '#ffd600' : '#ff3366';
      ctx.beginPath();
      ctx.arc(satScrX, satScrY, 8, 0, Math.PI * 2);
      ctx.fill();
      ctx.font = '11px monospace';
      ctx.fillText(sat.visible ? '🛰️ LEO SAT' : '🛰️ [BLACKOUT]', satScrX + 12, satScrY + 4);

      // Yellow Uplink Beam to Gateway
      if (sat.uplink_active) {
        const gw = lastData.drones.find(d => d.is_gateway && d.is_alive);
        if (gw) {
          const [gwx, gwy] = toScreen(gw.x, gw.y);
          ctx.strokeStyle = 'rgba(255, 214, 0, 0.85)';
          ctx.lineWidth = 2.5;
          ctx.setLineDash([6, 4]);
          ctx.beginPath();
          ctx.moveTo(gwx, gwy);
          ctx.lineTo(satScrX, satScrY);
          ctx.stroke();
          ctx.setLineDash([]);
        }
      }
    }
    requestAnimationFrame(renderRadar);
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def serve_control_center():
    """Serves the Mission Operations Control Center dashboard."""
    return HTMLResponse(content=CONTROL_CENTER_HTML)


@app.websocket("/ws/telemetry")
async def websocket_telemetry_endpoint(websocket: WebSocket):
    """Real-time bi-directional telemetry streaming to the dashboard."""
    await websocket.accept()
    print("[WS] Client connected to telemetry stream.")

    try:
        while True:
            # Check for incoming control commands without blocking
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=0.001)
                msg = json.loads(data)
                cmd = msg.get("command")
                if cmd == "toggle_pause":
                    sim_manager.is_paused = not sim_manager.is_paused
                    print(f"[CMD] Pause toggled: {sim_manager.is_paused}")
                elif cmd == "reset":
                    sim_manager.reset_sim()
                    print("[CMD] Simulation reset.")
                elif cmd == "force_sat":
                    sim_manager.force_satellite_pass()
                    print("[CMD] Forced LEO satellite pass.")
                elif cmd == "set_role":
                    sim_manager.set_drone_role(msg.get("drone_id", 0), msg.get("role", 0))
                    print(f"[CMD] Role updated for drone {msg.get('drone_id')}: {msg.get('role')}")
                elif cmd == "toggle_mode":
                    sim_manager.mode = "heuristic" if sim_manager.mode == "model" else "model"
                    print(f"[CMD] Switched mode to: {sim_manager.mode}")
            except asyncio.TimeoutError:
                pass

            # Step simulation & push telemetry
            sim_manager.step()
            payload = sim_manager.get_telemetry_payload()
            await websocket.send_text(json.dumps(payload))

            # Maintain smooth 20 Hz telemetry tick
            await asyncio.sleep(0.05)

    except WebSocketDisconnect:
        print("[WS] Client disconnected.")
    except Exception as e:
        print(f"[WS Error] {e}")


def main():
    parser = argparse.ArgumentParser(description="Drone Mesh RL Mission Control Center Server")
    parser.add_argument("--port", type=int, default=8000, help="Web server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    parser.add_argument("--gui", action="store_true", help="Launch PyBullet 3D window alongside web dashboard")
    parser.add_argument("--model_path", type=str, default="models/mappo_drone_mesh.pt", help="MAPPO model checkpoint")
    parser.add_argument("--num_drones", type=int, default=5, help="Number of drones")
    parser.add_argument("--num_clusters", type=int, default=4, help="Number of survivor clusters")
    parser.add_argument("--env_size", type=float, default=100.0, help="Environment size in meters")
    args = parser.parse_args()

    global sim_manager
    sim_manager = SwarmSimulationManager(
        num_drones=args.num_drones,
        num_clusters=args.num_clusters,
        env_size=args.env_size,
        model_path=args.model_path,
        gui=args.gui,
    )

    print("=" * 68)
    print(" 🛰️  AUTONOMOUS SWARM MISSION OPERATIONS CONTROL CENTER")
    print(f" Dashboard URL: http://localhost:{args.port}")
    print(f" Swarm Size:    {args.num_drones} Drones | Map: {args.env_size}m x {args.env_size}m")
    print("=" * 68)

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
