"""
Plain 3D Strapdown Inertial Navigation System (INS) Dead Reckoning.

This is the classic unconstrained double-integration baseline that demonstrates
the failure mode of raw MEMS IMUs without AI velocity estimation or non-holonomic constraints.

Equations:
  q_{k+1} = q_k * exp(0.5 * w_b * dt)
  a_n = R(q_{k+1}) * f_b + g_n
  v_{k+1} = v_k + a_n * dt
  p_{k+1} = p_k + v_k * dt + 0.5 * a_n * dt^2
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from .esekf import quat_multiply, quat_to_rot, rpy_to_quat


class StrapdownINS:
    """Plain 3D Strapdown Dead Reckoning without constraints or AI assistance."""

    def __init__(self, gravity: float = 9.80665):
        self.g_n = np.array([0.0, 0.0, -gravity], dtype=float)
        self.p = np.zeros(3, dtype=float)       # [East, North, Up]
        self.v = np.zeros(3, dtype=float)       # [v_e, v_n, v_u]
        self.q = np.array([1.0, 0.0, 0.0, 0.0]) # [qw, qx, qy, qz]

    def reset(self, p0: np.ndarray, v0: np.ndarray, heading0_rad: float,
              pitch0_rad: float = 0.0, roll0_rad: float = 0.0):
        """Initialize position, velocity, and orientation."""
        self.p = np.array(p0, dtype=float).copy()
        self.v = np.array(v0, dtype=float).copy()
        self.q = rpy_to_quat(pitch0_rad, roll0_rad, heading0_rad)

    def step(self, raw_acc: np.ndarray, raw_gyro: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """Integrate one IMU step (accelerometer with gravity, gyro in body frame)."""
        # 1. Quaternion Attitude Update
        angle = float(np.linalg.norm(raw_gyro)) * dt
        if angle > 1e-8:
            axis = raw_gyro / np.linalg.norm(raw_gyro)
            dq = np.array([math.cos(angle * 0.5), *(axis * math.sin(angle * 0.5))])
        else:
            dq = np.array([1.0, 0.5 * raw_gyro[0] * dt, 0.5 * raw_gyro[1] * dt, 0.5 * raw_gyro[2] * dt])
            dq /= np.linalg.norm(dq)

        self.q = quat_multiply(self.q, dq)
        self.q = self.q / np.linalg.norm(self.q)
        R = quat_to_rot(self.q)

        # 2. Specific force rotated to Nav Frame and gravity compensated
        f_n = R @ raw_acc
        a_n = f_n + self.g_n

        # 3. Double Integration
        self.p += self.v * dt + 0.5 * a_n * (dt ** 2)
        self.v += a_n * dt

        return self.p.copy(), self.v.copy()
