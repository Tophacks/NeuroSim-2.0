"""A simplified continuous-control 2D landing environment.

Coordinates: x horizontal, z upward; angle is clockwise-positive tilt from vertical.
Actions: [throttle request 0..1, signed gimbal -1..1].
No claims of flight fidelity. Use only for software experimentation.
"""
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .config import PLANETS, get_vehicle


class RocketLandingEnv(gym.Env):
    metadata = {"render_modes": ["ansi"], "render_fps": 30}

    def __init__(self, planet="earth", vehicle="falcon9_inspired",
                 max_steps=800, seed_offset=0, render_mode=None, **vehicle_overrides):
        super().__init__()
        self.planet = PLANETS[planet]
        self.vehicle = get_vehicle(vehicle, **vehicle_overrides)
        self.max_steps = max_steps
        self.seed_offset = seed_offset
        self.render_mode = render_mode
        self.dt = 0.1
        self.action_space = spaces.Box(
            low=np.array([0, -1], dtype=np.float32),
            high=np.array([1, 1], dtype=np.float32),
        )
        # [x/100, z/300, vx/50, vz/50, angle/pi, omega/2,
        #  fuel_fraction, target_offset/100] clipped for stable learning.
        self.observation_space = spaces.Box(-10, 10, shape=(8,), dtype=np.float32)
        self.state = None
        self.steps = 0
        self.prev_shaping = 0.0
        self.last_action = np.zeros(2)

    def _observation(self):
        x, z, vx, vz, theta, omega, fuel = self.state
        return np.clip(np.array(
            [x/100, z/300, vx/50, vz/50, theta/math.pi, omega/2,
             fuel/max(self.vehicle.fuel_mass, 1e-6), -x/100],
            dtype=np.float32), -10, 10)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        v = self.vehicle
        self.state = np.array([
            rng.uniform(-35, 35), rng.uniform(90, 180),
            rng.uniform(-7, 7), rng.uniform(-20, -8),
            rng.uniform(-0.16, 0.16), rng.uniform(-0.05, 0.05),
            v.fuel_mass], dtype=np.float64)
        self.steps = 0
        self.prev_shaping = self._shaping()
        self.last_action = np.zeros(2)
        return self._observation(), {}

    def _shaping(self):
        x, z, vx, vz, theta, omega, _ = self.state
        return (-0.35 * abs(x) - 0.09 * abs(z) - 1.7 * abs(vx)
                - 1.1 * abs(vz) - 24 * abs(theta) - 5 * abs(omega))

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), self.action_space.low,
                         self.action_space.high)
        throttle, gimbal = action
        x, z, vx, vz, theta, omega, fuel = self.state
        v = self.vehicle
        p = self.planet
        actual_throttle = float(throttle) if fuel > 0 else 0.0
        fuel_used = min(fuel, v.burn_rate * actual_throttle * self.dt)
        fuel = max(0.0, fuel - fuel_used)
        mass = v.dry_mass + fuel + 0.5 * fuel_used
        thrust = v.max_thrust * (fuel_used / (v.burn_rate*self.dt)) if v.burn_rate > 0 else 0.0
        angle = theta + float(gimbal) * math.radians(v.max_gimbal_deg)
        # Simplified tilt dynamics; damping represents unmodelled stabilization.
        angular_acc = (-2.0 * (thrust / max(v.max_thrust, 1.0)) * gimbal
                       - 0.12 * omega)
        omega += angular_acc * self.dt
        theta += omega * self.dt
        cross_area = math.pi * (v.diameter/2)**2
        drag_x = -0.5*p.air_density*v.drag_coefficient*cross_area*vx*abs(vx)/mass
        drag_z = -0.5*p.air_density*v.drag_coefficient*cross_area*vz*abs(vz)/mass
        ax = thrust * math.sin(angle)/mass + drag_x
        az = thrust * math.cos(angle)/mass - p.gravity + drag_z
        vx += ax * self.dt
        vz += az * self.dt
        x += vx * self.dt
        z += vz * self.dt
        self.state = np.array([x,z,vx,vz,theta,omega,fuel], dtype=np.float64)
        self.last_action = action.copy()
        self.steps += 1

        shaping = self._shaping()
        reward = float((shaping - self.prev_shaping) * 0.8 - 0.008 * fuel_used)
        self.prev_shaping = shaping
        touchdown = z <= v.height / 2
        out_of_bounds = abs(x) > 300 or z > 450 or abs(theta) > 1.4
        safe = (touchdown and abs(x) <= 8 and abs(vx) < 2.0
                and abs(vz) < 3.0 and abs(theta) < 0.12 and abs(omega) < 0.12)
        terminated = bool(touchdown or out_of_bounds)
        truncated = bool(self.steps >= self.max_steps and not terminated)
        if terminated:
            reward += 250.0 if safe else -180.0
        elif truncated:
            reward -= 100.0
        outcome = "success" if safe else ("crash" if terminated else ("timeout" if truncated else "flying"))
        info = {
            "outcome": outcome,
            "landing_error_m": float(abs(x)) if touchdown else None,
            "fuel_used_kg": float(v.fuel_mass - fuel),
            "horizontal_speed_m_s": float(abs(vx)),
            "vertical_speed_m_s": float(abs(vz)),
        }
        return self._observation(), reward, terminated, truncated, info

    def render(self):
        if self.state is None:
            return ""
        x,z,vx,vz,theta,omega,fuel = self.state
        return f"x={x:7.1f}m z={z:7.1f}m vx={vx:6.1f} vz={vz:6.1f} pitch={math.degrees(theta):6.1f}° fuel={fuel:7.1f}kg"

    def close(self):
        pass
