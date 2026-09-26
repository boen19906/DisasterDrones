"""
network.py — RF mesh topology, link budget, and LEO satellite with orbital sweep.

CRITICAL REQUIREMENT (from spec):
  The RF and satellite routing math must be decoupled. The synthetic link budget
  equations are written so they can be easily replaced by hardware-in-the-loop
  telemetry from a physical LEO satellite testbed in the future.

Satellite Model:
  - Sweeps across the simulation boundary on a linear pass
  - After exiting, a configurable blackout period before the next pass
  - Elevation angle > 25° required for gateway uplink
  - Tracks visibility state for observation space
"""

import numpy as np
import networkx as nx


class Satellite:
    """
    LEO satellite that sweeps across the simulation boundary.

    The satellite enters from one edge, crosses overhead, and exits the
    opposite edge. After a blackout period, a new pass begins from the
    entry edge.
    """

    def __init__(
        self,
        orbit_altitude=550_000.0,
        velocity=7500.0,
        sim_boundary_x=100.0,
        blackout_duration=30.0,
        min_elevation_angle=25.0,
    ):
        """
        Parameters
        ----------
        orbit_altitude : float
            Altitude in meters (550 km for Starlink).
        velocity : float
            Ground-track velocity in m/s.
        sim_boundary_x : float
            Width of the simulation area — satellite sweeps across this.
        blackout_duration : float
            Seconds between satellite passes (no uplink available).
        min_elevation_angle : float
            Minimum elevation angle (degrees) for gateway uplink.
        """
        self.orbit_altitude = orbit_altitude
        self.velocity = velocity
        self.sim_boundary_x = sim_boundary_x
        self.blackout_duration = blackout_duration
        self.min_elevation_angle = min_elevation_angle

        # Satellite starts a pass from the left edge
        self.current_x = -sim_boundary_x / 2
        self.current_y = 0.0  # Passes directly overhead

        self.visible = True
        self.blackout_timer = 0.0
        self.pass_count = 0

        # Scale velocity to sim scale
        # Real LEO moves at ~7500 m/s over 550km altitude, but our sim is 100m wide.
        # We want a pass to take ~60 seconds of sim time across 100m.
        self.effective_velocity = sim_boundary_x / 60.0  # Cross sim in ~60s

    def update_position(self, dt):
        """
        Move the satellite. Handles pass completion and blackout periods.

        Parameters
        ----------
        dt : float
            Timestep in seconds.
        """
        if self.visible:
            self.current_x += self.effective_velocity * dt

            # Check if satellite has exited the boundary
            if self.current_x > self.sim_boundary_x / 2:
                self.visible = False
                self.blackout_timer = self.blackout_duration
                self.pass_count += 1
        else:
            # In blackout — count down
            self.blackout_timer -= dt
            if self.blackout_timer <= 0:
                # Start new pass from opposite edge
                self.current_x = -self.sim_boundary_x / 2
                self.current_y = 0.0
                self.visible = True

    def get_elevation_angle(self, drone_x, drone_y, drone_z):
        """
        Calculate the elevation angle (degrees) from a drone to the satellite.

        Parameters
        ----------
        drone_x, drone_y, drone_z : float
            Drone world-space coordinates.

        Returns
        -------
        elevation_deg : float
            Angle above the horizon in degrees. Returns -90 if satellite
            is not visible (in blackout).
        """
        if not self.visible:
            return -90.0

        dx = self.current_x - drone_x
        dy = self.current_y - drone_y
        dz = self.orbit_altitude - drone_z

        ground_dist = np.sqrt(dx**2 + dy**2)
        elevation_rad = np.arctan2(dz, ground_dist)
        return np.degrees(elevation_rad)

    def can_uplink(self, drone_x, drone_y, drone_z):
        """
        Check if a gateway drone can establish uplink.

        Returns
        -------
        bool
            True if satellite is visible AND elevation angle > threshold.
        """
        if not self.visible:
            return False
        elev = self.get_elevation_angle(drone_x, drone_y, drone_z)
        return elev > self.min_elevation_angle

    def get_relative_position(self, drone_x, drone_y, drone_z):
        """
        Get the satellite's position relative to a drone.

        Returns
        -------
        rel_pos : np.ndarray, shape (3,)
        """
        return np.array([
            self.current_x - drone_x,
            self.current_y - drone_y,
            self.orbit_altitude - drone_z,
        ])

    def reset(self):
        """Reset satellite to start of a new pass."""
        self.current_x = -self.sim_boundary_x / 2
        self.current_y = 0.0
        self.visible = True
        self.blackout_timer = 0.0
        self.pass_count = 0


# ---------------------------------------------------------------------------
# RF Link Budget  (decoupled — replace with HW-in-the-loop telemetry later)
# ---------------------------------------------------------------------------

def calculate_link(pos_a, pos_b, has_los, freq_hz=2.4e9):
    """
    Calculate whether an RF link is maintained between two nodes.

    Uses Free Space Path Loss (FSPL) with inverse-square distance decay.
    If LoS is blocked, a 30 dBm penalty is applied.

    This function is intentionally decoupled so it can be replaced with
    hardware-in-the-loop telemetry from a physical testbed.

    Parameters
    ----------
    pos_a, pos_b : array-like, shape (3,)
        3D positions of the two nodes.
    has_los : bool
        Whether line-of-sight exists between the nodes.
    freq_hz : float
        Operating frequency in Hz.

    Returns
    -------
    link_active : bool
        True if received signal > -85 dBm.
    rssi : float
        Received signal strength in dBm.
    """
    dist = np.linalg.norm(np.array(pos_a) - np.array(pos_b))
    dist = max(dist, 1e-3)  # Avoid log(0)

    dist_km = dist / 1000.0
    freq_mhz = freq_hz / 1e6
    fspl = 20 * np.log10(dist_km + 1e-9) + 20 * np.log10(freq_mhz) + 32.44

    tx_power = 20  # dBm
    rx_sensitivity = -85  # dBm

    received_power = tx_power - fspl
    if not has_los:
        received_power -= 30  # dBm penalty for obstructed LoS

    return received_power > rx_sensitivity, received_power


def calculate_throughput(drone_positions, survivor_positions, gateway_roles,
                         los_matrix, satellite, terrain=None):
    """
    Build the NetworkX graph and return the number of survivor nodes
    that have a valid shortest path to the Satellite node.

    Parameters
    ----------
    drone_positions : list of array-like [x, y, z]
    survivor_positions : np.ndarray, shape (M, 3)
    gateway_roles : np.ndarray, shape (N,)
        1 = Gateway, 0 = Relay.
    los_matrix : np.ndarray, shape (N, N), dtype=bool
        Line-of-sight between each drone pair.
    satellite : Satellite
    terrain : Terrain or None
        Used for drone-to-survivor LoS checks.

    Returns
    -------
    connected_survivors : int
        Number of survivors with a valid path to the satellite.
    rssi_values : dict
        RSSI values for each active link (for observation space).
    graph : nx.Graph
        The constructed network graph (for visualization).
    """
    G = nx.Graph()
    N = len(drone_positions)
    M = len(survivor_positions)

    # Add nodes
    for i in range(N):
        G.add_node(f"drone_{i}", type="drone", pos=drone_positions[i])
    for i in range(M):
        G.add_node(f"surv_{i}", type="survivor", pos=survivor_positions[i])
    G.add_node("SAT", type="satellite")

    rssi_values = {}

    # Drone-to-drone links
    for i in range(N):
        for j in range(i + 1, N):
            has_los = los_matrix[i, j]
            link_ok, rssi = calculate_link(
                drone_positions[i], drone_positions[j], has_los
            )
            if link_ok:
                G.add_edge(f"drone_{i}", f"drone_{j}", rssi=rssi)
                rssi_values[(i, j)] = rssi

    # Survivor-to-drone links
    for i in range(M):
        for j in range(N):
            # Check LoS between survivor and drone
            has_los = True
            if terrain is not None:
                has_los = terrain.check_los(survivor_positions[i], drone_positions[j])
            link_ok, rssi = calculate_link(
                survivor_positions[i], drone_positions[j], has_los
            )
            if link_ok:
                G.add_edge(f"surv_{i}", f"drone_{j}", rssi=rssi)

    # Gateway-to-satellite links
    for i in range(N):
        if gateway_roles[i]:
            pos = drone_positions[i]
            if satellite.can_uplink(pos[0], pos[1], pos[2]):
                G.add_edge(f"drone_{i}", "SAT")

    # Count connected survivors
    connected_survivors = 0
    for i in range(M):
        if nx.has_path(G, f"surv_{i}", "SAT"):
            connected_survivors += 1

    return connected_survivors, rssi_values, G
