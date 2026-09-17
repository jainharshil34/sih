"""
Unscented Kalman Filter (UKF) for Vehicle Navigation using FilterPy.

Tech Stack Requirement:
Uses filterpy's UnscentedKalmanFilter with MerweScaledSigmaPoints rather than
hand-rolling matrix algebra. Fuses non-linear kinematic state transitions,
AI velocity estimates, ZUPT stops, and GNSS observations.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
from filterpy.kalman import MerweScaledSigmaPoints, UnscentedKalmanFilter


def fx_vehicle(x: np.ndarray, dt: float, u: np.ndarray) -> np.ndarray:
    """Nonlinear state transition function: x_k+1 = f(x_k, u_k).
    
    State: x = [pos_x, pos_y, forward_vel, heading_rad, acc_bias, gyro_bias]^T
    Input: u = [raw_acc_fwd, raw_gyro_z]
    """
    pos_x, pos_y, v, heading, b_a, b_g = x
    a_meas, w_meas = u

    # True acceleration and turn rate
    a_true = a_meas - b_a
    w_true = w_meas - b_g

    # Kinematic propagation
    new_heading = (heading + w_true * dt) % (2.0 * math.pi)
    new_v = max(0.0, float(v + a_true * dt))
    new_x = float(pos_x + v * math.sin(heading) * dt)
    new_y = float(pos_y + v * math.cos(heading) * dt)

    return np.array([new_x, new_y, new_v, new_heading, b_a, b_g], dtype=float)


def hx_gnss(x: np.ndarray) -> np.ndarray:
    """Measurement function for GNSS position [x, y]."""
    return np.array([x[0], x[1]], dtype=float)


def hx_ai_vel(x: np.ndarray) -> np.ndarray:
    """Measurement function for AI forward velocity [v]."""
    return np.array([x[2]], dtype=float)


def hx_zupt(x: np.ndarray) -> np.ndarray:
    """Measurement function for Zero-Velocity stop [v]."""
    return np.array([x[2]], dtype=float)


class VehicleUKF:
    """Production FilterPy Unscented Kalman Filter for Resilient Vehicle Dead Reckoning."""

    def __init__(self, dt: float = 0.1,
                 q_pos: float = 0.05,
                 q_vel: float = 0.2,
                 q_head: float = 0.02,
                 q_ba: float = 1e-4,
                 q_bg: float = 1e-5):
        self.dt = dt
        self.dim_x = 6

        # Sigma point generator
        points = MerweScaledSigmaPoints(n=self.dim_x, alpha=0.1, beta=2.0, kappa=0.0)
        
        self.ukf = UnscentedKalmanFilter(
            dim_x=self.dim_x,
            dim_z=2,  # default GNSS pos dimension
            dt=dt,
            fx=fx_vehicle,
            hx=hx_gnss,
            points=points
        )

        # Initial state covariance P
        self.ukf.P = np.diag([
            1.0, 1.0,      # pos_x, pos_y (m^2)
            0.5,           # vel (m/s)^2
            0.05,          # heading (rad^2)
            0.01,          # acc bias (m/s^2)^2
            1e-4           # gyro bias (rad/s)^2
        ])

        # Process noise covariance Q
        self.ukf.Q = np.diag([
            q_pos * dt, q_pos * dt,
            q_vel * dt,
            q_head * dt,
            q_ba * dt,
            q_bg * dt
        ])

    def initialize(self, x0: float, y0: float, v0: float, heading0: float,
                   ba0: float = 0.0, bg0: float = 0.0):
        """Seed filter state."""
        self.ukf.x = np.array([x0, y0, v0, heading0, ba0, bg0], dtype=float)

    @property
    def position(self) -> Tuple[float, float]:
        return float(self.ukf.x[0]), float(self.ukf.x[1])

    @property
    def velocity(self) -> float:
        return float(self.ukf.x[2])

    @property
    def heading(self) -> float:
        return float(self.ukf.x[3])

    @property
    def gyro_bias(self) -> float:
        return float(self.ukf.x[5])

    def predict(self, raw_acc_fwd: float, raw_gyro_z: float, dt: Optional[float] = None):
        """Propagate state and sigma points using strapdown kinematics."""
        actual_dt = dt if dt is not None else self.dt
        u = np.array([raw_acc_fwd, raw_gyro_z], dtype=float)
        self.ukf.predict(dt=actual_dt, u=u)

    def update_gnss(self, pos_x: float, pos_y: float, sigma_pos: float = 2.5):
        """Fuse incoming GNSS coordinate fix."""
        z = np.array([pos_x, pos_y], dtype=float)
        R = np.eye(2) * (sigma_pos ** 2)
        self.ukf.update(z, R=R, hx=hx_gnss)

    def update_ai_velocity(self, v_ai: float, var_ai: float = 0.25):
        """Fuse AI-regressed speed with heteroscedastic uncertainty."""
        z = np.array([v_ai], dtype=float)
        R = np.array([[max(0.01, var_ai)]], dtype=float)
        self.ukf.update(z, R=R, hx=hx_ai_vel)

    def update_zupt(self, sigma_vel: float = 0.02):
        """Clamp velocity to zero when stopped."""
        z = np.array([0.0], dtype=float)
        R = np.array([[sigma_vel ** 2]], dtype=float)
        self.ukf.update(z, R=R, hx=hx_zupt)
