"""
weather.py — Dynamic weather system with wind vectors and battery drain model.

Wind Model:
  - Global base wind: Ornstein-Uhlenbeck mean-reverting process
  - Localized gusts: terrain-gradient amplification near ridgelines
  - Storm bursts: random high-wind events lasting 50-100 timesteps

Battery Drain Model:
  - Base drain per step
  - Wind-fighting drain proportional to thrust needed to counter wind
  - Gateway mode 4x multiplier (Starlink phased array power draw)
  - Altitude penalty (thinner air = more thrust)
"""

import numpy as np


class WeatherSystem:
    """Simulates dynamic mountain weather affecting drone flight."""

    def __init__(
        self,
        terrain=None,
        # OU process parameters
        ou_theta=0.15,
        ou_mu=None,
        ou_sigma=0.02,
        # Storm parameters
        storm_probability=0.005,
        storm_duration_range=(50, 100),
        storm_intensity=3.0,
        # Battery drain rates (scaled for simulated episode timescales)
        base_drain_rate=1.2,          # 1.2% per second for relay drones (~80s total endurance)
        wind_drain_coefficient=0.15,   # Additional drain when fighting wind vectors
        gateway_multiplier=3.5,        # Gateway drone drains ~4.2% per sec (~24s before role handoff)
        altitude_drain_coefficient=0.01, # Extra thrust required at high altitude
        seed=None,
    ):
        """
        Parameters
        ----------
        terrain : Terrain or None
            Used for terrain-gradient gust amplification.
        ou_theta : float
            Mean-reversion speed for OU process.
        ou_mu : array-like or None
            Long-term mean wind vector [wx, wy, wz]. Defaults to gentle NE wind.
        ou_sigma : float
            Volatility of OU process.
        storm_probability : float
            Per-step probability of a storm burst starting.
        storm_duration_range : tuple(int, int)
            Min/max duration of storm bursts in timesteps.
        storm_intensity : float
            Wind multiplier during storms.
        base_drain_rate : float
            Base battery drain per step (percentage points).
        wind_drain_coefficient : float
            Additional drain per unit of wind force magnitude.
        gateway_multiplier : float
            Battery drain multiplier for gateway-role drones.
        altitude_drain_coefficient : float
            Additional drain per meter of altitude.
        seed : int or None
            Random seed.
        """
        self.terrain = terrain
        self.ou_theta = ou_theta
        self.ou_mu = np.array(ou_mu) if ou_mu is not None else np.array([0.03, 0.02, 0.0])
        self.ou_sigma = ou_sigma

        self.storm_probability = storm_probability
        self.storm_duration_range = storm_duration_range
        self.storm_intensity = storm_intensity

        self.base_drain_rate = base_drain_rate
        self.wind_drain_coefficient = wind_drain_coefficient
        self.gateway_multiplier = gateway_multiplier
        self.altitude_drain_coefficient = altitude_drain_coefficient

        self.rng = np.random.default_rng(seed)

        # State
        self.global_wind = np.array([0.0, 0.0, 0.0])
        self.storm_active = False
        self.storm_timer = 0
        self.storm_direction = np.array([0.0, 0.0, 0.0])

    def _get_terrain_gust_multiplier(self, x, y):
        """
        Compute a gust multiplier based on terrain gradient at (x, y).
        Ridgelines (high gradient magnitude) amplify wind.
        Returns a scalar >= 1.0.
        """
        if self.terrain is None:
            return 1.0

        gx = int((x + self.terrain.size_x / 2) / self.terrain.resolution)
        gy = int((y + self.terrain.size_y / 2) / self.terrain.resolution)
        gx = np.clip(gx, 1, self.terrain.grid_x - 2)
        gy = np.clip(gy, 1, self.terrain.grid_y - 2)

        dzdx = (
            self.terrain.heightmap[gy, gx + 1] - self.terrain.heightmap[gy, gx - 1]
        ) / (2 * self.terrain.resolution)
        dzdy = (
            self.terrain.heightmap[gy + 1, gx] - self.terrain.heightmap[gy - 1, gx]
        ) / (2 * self.terrain.resolution)

        gradient_mag = np.sqrt(dzdx**2 + dzdy**2)
        # Scale: flat terrain → 1.0, steep ridgeline → up to 3.0
        return 1.0 + 2.0 * np.clip(gradient_mag / 0.5, 0.0, 1.0)

    def step(self, dt):
        """
        Advance the weather state by one timestep.

        Parameters
        ----------
        dt : float
            Timestep duration in seconds.
        """
        # Update global wind via Ornstein-Uhlenbeck process
        dW = self.rng.normal(0, 1, size=3)
        self.global_wind += (
            self.ou_theta * (self.ou_mu - self.global_wind) * dt
            + self.ou_sigma * np.sqrt(dt) * dW
        )

        # Storm burst logic
        if self.storm_active:
            self.storm_timer -= 1
            if self.storm_timer <= 0:
                self.storm_active = False
        else:
            if self.rng.random() < self.storm_probability:
                self.storm_active = True
                self.storm_timer = self.rng.integers(
                    self.storm_duration_range[0], self.storm_duration_range[1] + 1
                )
                # Random storm direction
                angle = self.rng.uniform(0, 2 * np.pi)
                self.storm_direction = np.array([np.cos(angle), np.sin(angle), 0.0])

    def get_wind_at(self, x, y, z):
        """
        Get the effective wind vector at a given 3D position.
        Combines global wind + terrain gust amplification + storm burst.

        Parameters
        ----------
        x, y, z : float
            World-space coordinates.

        Returns
        -------
        wind : np.ndarray, shape (3,)
            Wind force vector in Newtons (for a ~27g Crazyflie).
        """
        gust_mult = self._get_terrain_gust_multiplier(x, y)
        wind = self.global_wind * gust_mult

        # Storm contribution
        if self.storm_active:
            wind += self.storm_direction * self.storm_intensity * self.ou_sigma * 10.0

        # Altitude effect: wind is stronger at higher altitudes
        altitude_factor = 1.0 + 0.01 * max(z, 0)
        wind *= altitude_factor

        return wind

    def compute_battery_drain(self, wind_at_drone, altitude, is_gateway, dt):
        """
        Compute battery drain for a single drone for this timestep.

        Parameters
        ----------
        wind_at_drone : np.ndarray, shape (3,)
            Wind vector at the drone's position.
        altitude : float
            Drone altitude in meters.
        is_gateway : bool
            Whether the drone is in gateway mode.
        dt : float
            Timestep.

        Returns
        -------
        drain : float
            Battery percentage points drained this step.
        """
        # Base drain
        drain = self.base_drain_rate * dt

        # Wind-fighting drain (proportional to wind magnitude the drone must counter)
        wind_mag = np.linalg.norm(wind_at_drone)
        drain += self.wind_drain_coefficient * wind_mag * dt

        # Altitude drain (thinner air = more thrust)
        drain += self.altitude_drain_coefficient * max(altitude, 0) * dt

        # Gateway multiplier
        if is_gateway:
            drain *= self.gateway_multiplier

        return drain

    def get_global_wind(self):
        """Return the current global wind vector."""
        return self.global_wind.copy()

    def is_storm_active(self):
        """Return whether a storm burst is currently active."""
        return self.storm_active

    def reset(self, seed=None):
        """Reset weather state."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.global_wind = np.array([0.0, 0.0, 0.0])
        self.storm_active = False
        self.storm_timer = 0
        self.storm_direction = np.array([0.0, 0.0, 0.0])
