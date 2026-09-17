"""
NavResilient EKF & UKF: Multi-Rate Sensor Fusion Module.

Implements:
1. NavResilientUKF: FilterPy Unscented Kalman Filter with MerweScaledSigmaPoints
   fusing IMU angular rates, forward linear acceleration, AI velocity estimates,
   and intermittent GNSS observations with smooth no-jump handoff.
2. NavResilientEKF: Classical Extended Kalman Filter baseline for comparative benchmark.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
from filterpy.kalman import MerweScaledSigmaPoints, UnscentedKalmanFilter


def fx_vehicle_ukf(x: np.ndarray, dt: float, u: np.ndarray) -> np.ndarray:
    """Non-linear vehicle kinematic propagation.
    
    State vector (dim 6):
        x[0]: pos_east (m)
        x[1]: pos_north (m)
        x[2]: forward_speed (m/s)
        x[3]: heading_rad (rad, 0 = North, pi/2 = East)
        x[4]: bias_acc (m/s^2)
        x[5]: bias_gyro (rad/s)
    
    Control input:
        u[0]: forward linear acceleration (m/s^2)
        u[1]: yaw rate (rad/s)
    """
    px, py, v, psi, b_a, b_g = x
    a_raw, w_raw = u

    # Unbiased physical signals
    a_fwd = a_raw - b_a
    w_yaw = w_raw - b_g

    # Update heading (continuous radians)
    new_psi = float(psi + w_yaw * dt)
    
    # Update forward speed with non-negative lower bound
    new_v = max(0.0, float(v + a_fwd * dt))

    # Kinematic position projection in ENU
    # East (x) = v * sin(psi), North (y) = v * cos(psi)
    new_px = float(px + new_v * math.sin(new_psi) * dt)
    new_py = float(py + new_v * math.cos(new_psi) * dt)

    return np.array([new_px, new_py, new_v, new_psi, b_a, b_g], dtype=float)


def hx_pos(x: np.ndarray) -> np.ndarray:
    """GNSS position measurement [East, North]."""
    return np.array([x[0], x[1]], dtype=float)


def hx_speed(x: np.ndarray) -> np.ndarray:
    """Forward speed measurement [v]."""
    return np.array([x[2]], dtype=float)


def hx_heading(x: np.ndarray) -> np.ndarray:
    """Course / heading measurement [psi]."""
    return np.array([x[3]], dtype=float)


def robust_cholesky(P: np.ndarray) -> np.ndarray:
    """Robust matrix square root with positive eigenvalue floor."""
    P_sym = 0.5 * (P + P.T)
    try:
        return np.linalg.cholesky(P_sym)
    except (np.linalg.LinAlgError, Exception):
        eigvals, eigvecs = np.linalg.eigh(P_sym)
        eigvals = np.maximum(eigvals, 1e-6)
        return eigvecs @ np.diag(np.sqrt(eigvals))


class NavResilientUKF:
    """FilterPy Unscented Kalman Filter for robust GNSS + INS dead-reckoning fusion.
    
    Features:
    - MerweScaledSigmaPoints capture high non-linearity in vehicle turns
    - Smooth handoff transition when GNSS signal is lost or re-acquired (no jump)
    - Innovation gating to reject GPS multipath spikes
    - Dynamic covariance propagation during prolonged outages
    """

    def __init__(
        self,
        dt: float = 0.1,
        q_pos: float = 0.05,
        q_vel: float = 0.15,
        q_head: float = 0.015,
        q_ba: float = 1e-4,
        q_bg: float = 1e-5,
    ):
        self.dt = dt
        self.dim_x = 6
        self.is_gnss_available = True
        self.outage_duration_s = 0.0
        self.smooth_handoff_steps = 0
        self.handoff_window_steps = 10  # 1 second smooth blending window

        # Merwe scaled sigma points (optimal for non-linear robotics/navigation)
        points = MerweScaledSigmaPoints(
            n=self.dim_x, alpha=0.1, beta=2.0, kappa=0.0, sqrt_method=robust_cholesky
        )

        self.ukf = UnscentedKalmanFilter(
            dim_x=self.dim_x,
            dim_z=2,
            dt=dt,
            fx=fx_vehicle_ukf,
            hx=hx_pos,
            points=points
        )

        # Initial state covariance
        self.ukf.P = np.diag([
            2.0, 2.0,      # pos (m^2)
            0.5,           # speed (m/s)^2
            0.05,          # heading (rad^2)
            0.01,          # acc bias
            1e-4           # gyro bias
        ])

        # Base process noise covariance Q
        self.q_base = np.diag([
            q_pos * dt, q_pos * dt,
            q_vel * dt,
            q_head * dt,
            q_ba * dt,
            q_bg * dt
        ])
        self.ukf.Q = self.q_base.copy()

    def initialize(
        self,
        x0: float,
        y0: float,
        v0: float,
        heading0: float,
        ba0: float = 0.0,
        bg0: float = 0.0
    ):
        """Seed filter initial state."""
        self.ukf.x = np.array([x0, y0, v0, heading0, ba0, bg0], dtype=float)
        self.outage_duration_s = 0.0
        self.smooth_handoff_steps = 0

    @property
    def position(self) -> Tuple[float, float]:
        """(East, North) in meters."""
        return float(self.ukf.x[0]), float(self.ukf.x[1])

    @property
    def forward_speed(self) -> float:
        """Forward velocity in m/s."""
        return float(self.ukf.x[2])

    @property
    def heading_rad(self) -> float:
        """Heading angle in radians."""
        return float(self.ukf.x[3])

    @property
    def gyro_bias(self) -> float:
        """Estimated gyroscope bias in rad/s."""
        return float(self.ukf.x[5])

    @property
    def position_uncertainty(self) -> float:
        """1-sigma position standard deviation in meters."""
        return float(math.sqrt(max(0.0, self.ukf.P[0, 0] + self.ukf.P[1, 1])))

    def predict(self, acc_fwd: float, gyro_z: float, dt: Optional[float] = None):
        """High-rate prediction step driven by IMU measurements."""
        actual_dt = dt if dt is not None else self.dt
        u = np.array([acc_fwd, gyro_z], dtype=float)
        
        # During outage, gently expand Q to reflect cumulative unmodeled drift
        if not self.is_gnss_available:
            self.outage_duration_s += actual_dt
            scale = 1.0 + min(5.0, self.outage_duration_s * 0.05)
            self.ukf.Q = self.q_base * scale
        else:
            self.ukf.Q = self.q_base.copy()

        # Enforce positive definiteness before sigma point generation
        self.ukf.P = 0.5 * (self.ukf.P + self.ukf.P.T) + np.eye(self.dim_x) * 1e-5
        self.ukf.predict(dt=actual_dt, u=u)
        self.ukf.P = 0.5 * (self.ukf.P + self.ukf.P.T) + np.eye(self.dim_x) * 1e-5

    def update_gnss(
        self,
        pos_east: float,
        pos_north: float,
        sigma_pos: float = 2.5,
        course_rad: Optional[float] = None
    ) -> bool:
        """Fuse incoming GNSS fix with Mahalanobis innovation check & smooth handoff."""
        z = np.array([pos_east, pos_north], dtype=float)
        
        # Smooth handoff on GNSS re-acquisition after outage
        if not self.is_gnss_available:
            self.is_gnss_available = True
            self.smooth_handoff_steps = self.handoff_window_steps
            self.outage_duration_s = 0.0

        # Adjust measurement noise during smooth handoff to prevent jarring step jumps
        effective_sigma = sigma_pos
        if self.smooth_handoff_steps > 0:
            weight = self.smooth_handoff_steps / self.handoff_window_steps
            effective_sigma = sigma_pos * (1.0 + 3.0 * weight)
            self.smooth_handoff_steps -= 1

        R = np.eye(2) * (effective_sigma ** 2)

        # Innovation gating (Chi-square 2-DoF 99% threshold = 9.21)
        z_pred = hx_pos(self.ukf.x)
        innov = z - z_pred
        S = self.ukf.P[:2, :2] + R
        try:
            d2 = float(innov.T @ np.linalg.inv(S) @ innov)
            if d2 > 25.0 and self.smooth_handoff_steps == 0:
                # Suspect multipath outlier - reject
                return False
        except np.linalg.LinAlgError:
            pass

        self.ukf.update(z, R=R, hx=hx_pos)

        # Optional heading alignment from GNSS course when moving fast enough
        if course_rad is not None and self.forward_speed > 2.0:
            d_head = (course_rad - self.ukf.x[3] + math.pi) % (2.0 * math.pi) - math.pi
            self.ukf.x[3] = (self.ukf.x[3] + 0.25 * d_head) % (2.0 * math.pi)

        return True

    def mark_gnss_lost(self):
        """Flag GNSS outage for smooth dead reckoning transition."""
        self.is_gnss_available = False

    def update_ai_velocity(self, v_ai: float, var_ai: float = 0.20):
        """Fuse AI-regressed forward velocity with learned heteroscedastic uncertainty."""
        z = np.array([max(0.0, float(v_ai))], dtype=float)
        R = np.array([[max(0.01, float(var_ai))]], dtype=float)
        self.ukf.update(z, R=R, hx=hx_speed)

    def update_zupt(self, sigma_vel: float = 0.02):
        """Apply Zero-Velocity Update when stationary."""
        z = np.array([0.0], dtype=float)
        R = np.array([[sigma_vel ** 2]], dtype=float)
        self.ukf.update(z, R=R, hx=hx_speed)


class NavResilientEKF:
    """Classical Extended Kalman Filter for comparison with UKF."""

    def __init__(self, dt: float = 0.1):
        self.dt = dt
        # State: [px, py, v, psi, b_a, b_g]
        self.x = np.zeros(6, dtype=float)
        self.P = np.diag([2.0, 2.0, 0.5, 0.05, 0.01, 1e-4])
        self.Q = np.diag([0.05 * dt, 0.05 * dt, 0.15 * dt, 0.015 * dt, 1e-4 * dt, 1e-5 * dt])

    def initialize(self, x0: float, y0: float, v0: float, heading0: float):
        self.x = np.array([x0, y0, v0, heading0, 0.0, 0.0], dtype=float)

    @property
    def position(self) -> Tuple[float, float]:
        return float(self.x[0]), float(self.x[1])

    @property
    def forward_speed(self) -> float:
        return float(self.x[2])

    @property
    def heading_rad(self) -> float:
        return float(self.x[3])

    def predict(self, acc_fwd: float, gyro_z: float, dt: Optional[float] = None):
        dt = dt or self.dt
        px, py, v, psi, ba, bg = self.x
        a = acc_fwd - ba
        w = gyro_z - bg

        new_psi = float(psi + w * dt)
        new_v = max(0.0, float(v + a * dt))
        new_px = float(px + new_v * math.sin(new_psi) * dt)
        new_py = float(py + new_v * math.cos(new_psi) * dt)

        # Jacobian F = df/dx
        F = np.eye(6)
        F[0, 2] = math.sin(new_psi) * dt
        F[0, 3] = new_v * math.cos(new_psi) * dt
        F[0, 4] = -math.sin(new_psi) * dt * dt
        F[0, 5] = -new_v * math.cos(new_psi) * dt * dt
        F[1, 2] = math.cos(new_psi) * dt
        F[1, 3] = -new_v * math.sin(new_psi) * dt
        F[1, 4] = -math.cos(new_psi) * dt * dt
        F[1, 5] = new_v * math.sin(new_psi) * dt * dt
        F[2, 4] = -dt
        F[3, 5] = -dt

        self.x = np.array([new_px, new_py, new_v, new_psi, ba, bg], dtype=float)
        self.P = F @ self.P @ F.T + self.Q

    def update_gnss(self, pos_east: float, pos_north: float, sigma_pos: float = 2.5):
        z = np.array([pos_east, pos_north], dtype=float)
        H = np.zeros((2, 6))
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        R = np.eye(2) * (sigma_pos ** 2)

        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ H) @ self.P

    def update_speed(self, v_meas: float, sigma_v: float = 0.5):
        H = np.zeros((1, 6))
        H[0, 2] = 1.0
        R = np.array([[sigma_v ** 2]])
        y = np.array([v_meas]) - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + (K @ y).ravel()
        self.P = (np.eye(6) - K @ H) @ self.P
