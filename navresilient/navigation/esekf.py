"""
15-State Error-State Extended Kalman Filter (ES-EKF) for Resilient GNSS/INS Navigation.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np


def skew(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric cross-product matrix."""
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0]
    ], dtype=float)


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    """Convert unit quaternion q = [qw, qx, qy, qz] to 3x3 rotation matrix R_b^n."""
    qw, qx, qy, qz = q
    return np.array([
        [1.0 - 2.0 * (qy*qy + qz*qz), 2.0 * (qx*qy - qw*qz), 2.0 * (qx*qz + qw*qy)],
        [2.0 * (qx*qy + qw*qz), 1.0 - 2.0 * (qx*qx + qz*qz), 2.0 * (qy*qz - qw*qx)],
        [2.0 * (qx*qz - qw*qy), 2.0 * (qy*qz + qw*qx), 1.0 - 2.0 * (qx*qx + qy*qy)]
    ], dtype=float)


def rot_to_quat(R: np.ndarray) -> np.ndarray:
    tr = np.trace(R)
    if tr > 0.0:
        s = 0.5 / math.sqrt(tr + 1.0)
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s
    q = np.array([qw, qx, qy, qz], dtype=float)
    return q / np.linalg.norm(q)


def quat_multiply(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = q
    pw, px, py, pz = p
    return np.array([
        qw*pw - qx*px - qy*py - qz*pz,
        qw*px + qx*pw + qy*pz - qz*py,
        qw*py - qx*pz + qy*pw + qz*px,
        qw*pz + qx*py - qy*px + qz*pw
    ], dtype=float)


def rpy_to_quat(pitch: float, roll: float, yaw: float) -> np.ndarray:
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return np.array([qw, qx, qy, qz], dtype=float)


class ESEKF15:
    """15-State Error-State Kalman Filter."""

    def __init__(self,
                 sigma_acc_noise: float = 0.08,
                 sigma_gyro_noise: float = 0.015,
                 sigma_acc_bias_rw: float = 1e-4,
                 sigma_gyro_bias_rw: float = 1e-5,
                 gravity: float = 9.80665):
        self.g_n = np.array([0.0, 0.0, -gravity], dtype=float)
        
        self.p = np.zeros(3, dtype=float)
        self.v = np.zeros(3, dtype=float)
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self.ba = np.zeros(3, dtype=float)
        self.bg = np.zeros(3, dtype=float)

        self.P = np.diag([
            1.0, 1.0, 1.0,
            0.5, 0.5, 0.5,
            0.05, 0.05, 0.05,
            0.01, 0.01, 0.01,
            1e-4, 1e-4, 1e-4
        ]).astype(float)

        self.q_a = sigma_acc_noise ** 2
        self.q_w = sigma_gyro_noise ** 2
        self.q_ba = sigma_acc_bias_rw ** 2
        self.q_bg = sigma_gyro_bias_rw ** 2

    def reset_state(self, p0: np.ndarray, v0: np.ndarray, q0: np.ndarray,
                    ba0: Optional[np.ndarray] = None, bg0: Optional[np.ndarray] = None):
        self.p = np.array(p0, dtype=float).copy()
        self.v = np.array(v0, dtype=float).copy()
        self.q = np.array(q0, dtype=float).copy()
        self.q = self.q / np.linalg.norm(self.q)
        if ba0 is not None:
            self.ba = np.array(ba0, dtype=float).copy()
        if bg0 is not None:
            self.bg = np.array(bg0, dtype=float).copy()

    @property
    def rotation_matrix(self) -> np.ndarray:
        return quat_to_rot(self.q)

    @property
    def heading(self) -> float:
        R = self.rotation_matrix
        return float(math.atan2(R[0, 0], R[1, 0]))

    def predict(self, raw_acc: np.ndarray, raw_gyro: np.ndarray, dt: float):
        f_b = raw_acc - self.ba
        w_b = raw_gyro - self.bg

        angle = np.linalg.norm(w_b) * dt
        if angle > 1e-8:
            axis = w_b / np.linalg.norm(w_b)
            dq = np.array([math.cos(angle * 0.5), *(axis * math.sin(angle * 0.5))])
        else:
            dq = np.array([1.0, 0.5 * w_b[0] * dt, 0.5 * w_b[1] * dt, 0.5 * w_b[2] * dt])
            dq /= np.linalg.norm(dq)
        
        q_new = quat_multiply(self.q, dq)
        self.q = q_new / np.linalg.norm(q_new)
        R = quat_to_rot(self.q)

        f_n = R @ f_b
        a_n = f_n + self.g_n

        self.p += self.v * dt + 0.5 * a_n * (dt ** 2)
        self.v += a_n * dt

        F = np.eye(15, dtype=float)
        F[0:3, 3:6] = np.eye(3) * dt
        F[3:6, 6:9] = -skew(f_n) * dt
        F[3:6, 9:12] = -R * dt
        F[6:9, 12:15] = -R * dt

        Q = np.zeros((15, 15), dtype=float)
        Q[3:6, 3:6] = np.eye(3) * (self.q_a * dt)
        Q[6:9, 6:9] = np.eye(3) * (self.q_w * dt)
        Q[9:12, 9:12] = np.eye(3) * (self.q_ba * dt)
        Q[12:15, 12:15] = np.eye(3) * (self.q_bg * dt)

        self.P = F @ self.P @ F.T + Q
        self.P = 0.5 * (self.P + self.P.T)

    def _apply_error_state(self, delta_x: np.ndarray):
        self.p += delta_x[0:3]
        self.v += delta_x[3:6]

        d_theta = delta_x[6:9]
        angle = np.linalg.norm(d_theta)
        if angle > 1e-8:
            axis = d_theta / angle
            dq = np.array([math.cos(angle * 0.5), *(axis * math.sin(angle * 0.5))])
        else:
            dq = np.array([1.0, 0.5 * d_theta[0], 0.5 * d_theta[1], 0.5 * d_theta[2]])
            dq /= np.linalg.norm(dq)

        self.q = quat_multiply(dq, self.q)
        self.q = self.q / np.linalg.norm(self.q)

        self.ba += delta_x[9:12]
        self.bg += delta_x[12:15]

        G = np.eye(15, dtype=float)
        G[6:9, 6:9] = np.eye(3) - 0.5 * skew(d_theta)
        self.P = G @ self.P @ G.T
        self.P = 0.5 * (self.P + self.P.T)

    def update_ai_velocity(self, v_ai: float, var_ai: float = 0.25):
        R_b2n = self.rotation_matrix
        R_n2b = R_b2n.T
        v_b = R_n2b @ self.v

        y = v_ai - v_b[0]

        H = np.zeros((1, 15), dtype=float)
        H[0, 3:6] = R_n2b[0, :]
        H[0, 6:9] = (R_n2b @ skew(self.v))[0, :]

        R_cov = max(1e-3, float(var_ai))
        S = float(H @ self.P @ H.T + R_cov)
        K = (self.P @ H.T) / S

        delta_x = (K * y).ravel()
        self._apply_error_state(delta_x)
        self.P = (np.eye(15) - K @ H) @ self.P
        self.P = 0.5 * (self.P + self.P.T)

    def update_nhc(self, sigma_lat: float = 0.15, sigma_vert: float = 0.2):
        R_b2n = self.rotation_matrix
        R_n2b = R_b2n.T
        v_b = R_n2b @ self.v

        y = np.array([0.0 - v_b[1], 0.0 - v_b[2]], dtype=float)

        H = np.zeros((2, 15), dtype=float)
        H[0, 3:6] = R_n2b[1, :]
        H[0, 6:9] = (R_n2b @ skew(self.v))[1, :]
        H[1, 3:6] = R_n2b[2, :]
        H[1, 6:9] = (R_n2b @ skew(self.v))[2, :]

        R_cov = np.diag([sigma_lat ** 2, sigma_vert ** 2])
        S = H @ self.P @ H.T + R_cov
        K = self.P @ H.T @ np.linalg.inv(S)

        delta_x = K @ y
        self._apply_error_state(delta_x)
        self.P = (np.eye(15) - K @ H) @ self.P
        self.P = 0.5 * (self.P + self.P.T)

    def update_zupt(self, sigma_vel: float = 0.05, sigma_gyro: float = 0.005):
        y_v = -self.v
        H_v = np.zeros((3, 15), dtype=float)
        H_v[:, 3:6] = np.eye(3)
        R_v = np.eye(3) * (sigma_vel ** 2)

        S_v = H_v @ self.P @ H_v.T + R_v
        K_v = self.P @ H_v.T @ np.linalg.inv(S_v)
        delta_x = K_v @ y_v
        self._apply_error_state(delta_x)
        self.P = (np.eye(15) - K_v @ H_v) @ self.P

        H_w = np.zeros((3, 15), dtype=float)
        H_w[:, 12:15] = np.eye(3)
        R_w = np.eye(3) * (sigma_gyro ** 2)
        y_w = -self.bg
        S_w = H_w @ self.P @ H_w.T + R_w
        K_w = self.P @ H_w.T @ np.linalg.inv(S_w)
        delta_x_w = K_w @ y_w
        self._apply_error_state(delta_x_w)
        self.P = (np.eye(15) - K_w @ H_w) @ self.P
        self.P = 0.5 * (self.P + self.P.T)

    def update_gnss(self, p_gnss: np.ndarray, v_gnss: Optional[np.ndarray] = None,
                    sigma_pos: float = 2.5, sigma_vel: float = 0.25,
                    chi2_gate_pos: float = 16.0) -> bool:
        y_p = p_gnss - self.p
        H_p = np.zeros((3, 15), dtype=float)
        H_p[:, 0:3] = np.eye(3)
        R_p = np.eye(3) * (sigma_pos ** 2)

        S_p = H_p @ self.P @ H_p.T + R_p
        mahalanobis_sq = float(y_p.T @ np.linalg.inv(S_p) @ y_p)
        if mahalanobis_sq > chi2_gate_pos:
            R_p *= max(1.0, mahalanobis_sq / chi2_gate_pos)
            S_p = H_p @ self.P @ H_p.T + R_p

        K_p = self.P @ H_p.T @ np.linalg.inv(S_p)
        delta_x = K_p @ y_p
        self._apply_error_state(delta_x)
        self.P = (np.eye(15) - K_p @ H_p) @ self.P

        if v_gnss is not None:
            y_v = v_gnss - self.v
            H_v = np.zeros((3, 15), dtype=float)
            H_v[:, 3:6] = np.eye(3)
            R_v = np.eye(3) * (sigma_vel ** 2)
            S_v = H_v @ self.P @ H_v.T + R_v
            K_v = self.P @ H_v.T @ np.linalg.inv(S_v)
            self._apply_error_state(K_v @ y_v)
            self.P = (np.eye(15) - K_v @ H_v) @ self.P

        self.P = 0.5 * (self.P + self.P.T)
        return True
