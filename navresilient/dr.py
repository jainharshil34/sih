"""
Alignment, velocity estimation and constrained planar dead reckoning.
"""

from __future__ import annotations

import numpy as np

G = 9.80665


# ----------------------------------------------------------------- alignment
def gravity_tilt(acc_with_g: np.ndarray) -> tuple[float, float]:
    """Pitch and roll (rad) of the phone from a mean gravity vector."""
    g = np.nanmean(acc_with_g, axis=0)
    g = g / (np.linalg.norm(g) + 1e-9)
    roll = np.arctan2(g[1], g[2])
    pitch = np.arctan2(-g[0], np.hypot(g[1], g[2]))
    return float(pitch), float(roll)


def level_rotation(pitch: float, roll: float) -> np.ndarray:
    """Rotation taking phone body axes to a gravity-levelled frame."""
    cp, sp, cr, sr = np.cos(pitch), np.sin(pitch), np.cos(roll), np.sin(roll)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Ry @ Rx


def yaw_alignment(acc_level: np.ndarray, gps_speed: np.ndarray, fs: float) -> float:
    """Heading offset (rad) between phone x-axis and vehicle forward axis."""
    dv = np.gradient(gps_speed) * fs
    m = np.isfinite(dv) & (np.abs(dv) > 0.25)
    if m.sum() < int(3 * fs):
        return 0.0
    ax, ay = acc_level[m, 0], acc_level[m, 1]
    dv = dv[m]
    return float(np.arctan2(float(np.dot(ay, dv)), float(np.dot(ax, dv))))


def to_vehicle_frame(acc: np.ndarray, gyro: np.ndarray, gps_speed: np.ndarray,
                     fs: float) -> tuple[np.ndarray, np.ndarray, dict]:
    """Rotate phone-frame IMU into the vehicle frame (x forward, z up)."""
    pitch, roll = gravity_tilt(acc + np.array([0.0, 0.0, G]))
    R = level_rotation(pitch, roll)
    a_lvl = acc @ R.T
    w_lvl = gyro @ R.T
    psi = yaw_alignment(a_lvl, gps_speed, fs)
    c, s = np.cos(-psi), np.sin(-psi)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return a_lvl @ Rz.T, w_lvl @ Rz.T, {"pitch": pitch, "roll": roll, "yaw": psi}


# ------------------------------------------------------------------ velocity
def lowpass(sig: np.ndarray, fs: float, fc: float = 2.0) -> np.ndarray:
    n = max(3, int(round(fs / (2.0 * fc))))
    k = np.ones(n) / n
    out = np.asarray(sig, float)
    pad = ((n, n),) + ((0, 0),) * (out.ndim - 1)
    for _ in range(2):
        p = np.pad(out, pad, mode="edge")
        if out.ndim == 1:
            p = np.convolve(p, k, "same")
        else:
            p = np.column_stack([np.convolve(p[:, i], k, "same") for i in range(p.shape[1])])
        out = p[n:-n]
    return out


def _moving_var(x: np.ndarray, n: int) -> np.ndarray:
    k = np.ones(n) / n
    m1 = np.convolve(x, k, "same")
    m2 = np.convolve(x * x, k, "same")
    return np.maximum(m2 - m1 * m1, 0.0)


def zero_velocity_mask(acc_v: np.ndarray, gyro_v: np.ndarray, fs: float,
                       acc_thresh: float = 0.12, gyro_thresh: float = 0.02,
                       win: float = 1.0) -> np.ndarray:
    a = lowpass(acc_v, fs)
    w = lowpass(gyro_v, fs)
    n = max(3, int(win * fs))
    a_var = _moving_var(np.linalg.norm(a, axis=1), n)
    w_mag = np.convolve(np.abs(w[:, 2]), np.ones(n) / n, "same")
    a_mag = np.convolve(np.abs(a[:, 0]), np.ones(n) / n, "same")
    return (a_var < acc_thresh ** 2) & (w_mag < gyro_thresh) & (a_mag < acc_thresh)


def integrate_velocity(acc_fwd: np.ndarray, fs: float, v0: float = 0.0,
                       zupt: np.ndarray | None = None) -> np.ndarray:
    v = v0 + np.cumsum(acc_fwd) / fs
    if zupt is not None:
        v = v.copy()
        for i in np.flatnonzero(zupt):
            v[i:] -= v[i]
            v[i] = 0.0
    return np.maximum(v, 0.0)


def deadreckon(speed: np.ndarray, yaw_rate: np.ndarray, fs: float,
               x0: float, y0: float, heading0: float,
               zupt: np.ndarray | None = None,
               gyro_bias: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Planar Non-Holonomic Constraint (NHC) dead reckoning."""
    # Android gyro Z is CCW positive; geographic heading is CW from North
    w = -(yaw_rate - gyro_bias)
    v = speed.copy()
    if zupt is not None:
        v[zupt] = 0.0
        w = np.where(zupt, 0.0, w)
    heading = heading0 + np.cumsum(w) / fs
    dx, dy = v * np.sin(heading) / fs, v * np.cos(heading) / fs
    return x0 + np.cumsum(dx), y0 + np.cumsum(dy), heading


def estimate_gyro_bias(gyro_z: np.ndarray, zupt: np.ndarray) -> float:
    if zupt.sum() < 20:
        return 0.0
    return float(np.median(gyro_z[zupt]))
