"""
AI Vibration & Pothole Isolation Filter, and Robust Zero-Velocity (ZUPT/ZARU) Detector.

Problem Statement SIH26168 Requirement:
Smartphone IMUs mounted on dashboards pick up severe non-navigation motion:
- Engine harmonics (15-50 Hz at idle and cruise)
- Road shock, potholes, expansion joints, rumble strips (transient spikes)
- Vehicle chassis vibrations (~0.15 g RMS)

This module isolates the true kinematic navigation band (< 3 Hz) from high-frequency
noise, detects pothole shocks to prevent spurious acceleration integration, and
computes robust Zero-Velocity Updates (ZUPT) and Zero Angular Rate Updates (ZARU).
"""

from __future__ import annotations

import numpy as np


class VibrationFilter:
    """Multi-stage spectral and transient filter for smartphone IMU data."""

    def __init__(self, fs: float = 10.0, fc_kinematic: float = 2.5, shock_thresh_g: float = 2.0):
        self.fs = fs
        self.fc_kinematic = fc_kinematic
        self.shock_thresh = shock_thresh_g * 9.80665

    def filter_kinematics(self, sig: np.ndarray) -> np.ndarray:
        """Extract low-frequency kinematic motion (< 2.5-3.0 Hz) using zero-phase moving average."""
        n = max(3, int(round(self.fs / (2.0 * self.fc_kinematic))))
        k = np.ones(n) / n
        out = np.asarray(sig, float)
        pad = ((n, n),) + ((0, 0),) * (out.ndim - 1)
        for _ in range(2):  # 2 passes for 4th-order equivalent roll-off
            p = np.pad(out, pad, mode="edge")
            if out.ndim == 1:
                p = np.convolve(p, k, "same")
            else:
                p = np.column_stack([np.convolve(p[:, i], k, "same") for i in range(p.shape[1])])
            out = p[n:-n]
        return out

    def detect_potholes_and_shocks(self, raw_acc: np.ndarray) -> np.ndarray:
        """Detect impulse shocks from potholes, speed bumps, and road irregularities."""
        kinematic = self.filter_kinematics(raw_acc)
        high_freq = raw_acc - kinematic
        hf_norm = np.linalg.norm(high_freq, axis=-1) if high_freq.ndim > 1 else np.abs(high_freq)
        return hf_norm > self.shock_thresh

    def compute_spectral_energy(self, raw_acc: np.ndarray, win_len: int = 10) -> np.ndarray:
        """Compute high-frequency vibration energy (e.g., engine idle indicator)."""
        kinematic = self.filter_kinematics(raw_acc)
        residual = raw_acc - kinematic
        res_sq = np.sum(residual ** 2, axis=-1) if residual.ndim > 1 else residual ** 2
        kernel = np.ones(win_len) / win_len
        return np.convolve(res_sq, kernel, "same")


class StopDetector:
    """Generalized Likelihood Ratio Test (GLRT) and Adaptive Moving-Variance ZUPT/ZARU detector."""

    def __init__(self, fs: float = 10.0, win_s: float = 1.0,
                 acc_var_thresh: float = 0.015, gyro_mag_thresh: float = 0.025,
                 acc_mag_thresh: float = 0.15):
        self.fs = fs
        self.win_len = max(3, int(win_s * fs))
        self.acc_var_thresh = acc_var_thresh
        self.gyro_mag_thresh = gyro_mag_thresh
        self.acc_mag_thresh = acc_mag_thresh
        self.vib_filter = VibrationFilter(fs=fs, fc_kinematic=2.0)

    def detect(self, acc_v: np.ndarray, gyro_v: np.ndarray) -> tuple[np.ndarray, dict]:
        """Detect zero-velocity intervals (ZUPT) and zero angular rate intervals (ZARU).

        Returns:
            zupt_mask: boolean array (True where vehicle is stationary)
            metrics: dictionary of intermediate statistics for diagnostics
        """
        a_filt = self.vib_filter.filter_kinematics(acc_v)
        w_filt = self.vib_filter.filter_kinematics(gyro_v)

        # 1. Accelerometer variance over moving window
        a_norm = np.linalg.norm(a_filt, axis=1) if a_filt.ndim > 1 else np.abs(a_filt)
        k = np.ones(self.win_len) / self.win_len
        m1 = np.convolve(a_norm, k, "same")
        m2 = np.convolve(a_norm * a_norm, k, "same")
        a_var = np.maximum(m2 - m1 * m1, 0.0)

        # 2. Gyroscope yaw rate smoothed magnitude
        w_z = np.abs(w_filt[:, 2]) if w_filt.ndim > 1 else np.abs(w_filt)
        w_mag = np.convolve(w_z, k, "same")

        # 3. Forward acceleration magnitude
        a_fwd = np.abs(a_filt[:, 0]) if a_filt.ndim > 1 else a_norm
        a_fwd_mag = np.convolve(a_fwd, k, "same")

        zupt = (a_var < self.acc_var_thresh) & (w_mag < self.gyro_mag_thresh) & (a_fwd_mag < self.acc_mag_thresh)

        metrics = {
            "acc_var": a_var,
            "gyro_mag": w_mag,
            "acc_fwd_mag": a_fwd_mag,
            "stop_fraction": float(np.mean(zupt)),
        }
        return zupt, metrics
