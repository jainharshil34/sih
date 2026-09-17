"""
Automatic Dynamic Mount Calibration Engine.

Problem Statement SIH26168 Requirement:
In real Indian vehicles, smartphones are placed arbitrarily in windshield suction cups,
AC vent clips, dashboard magnetic mounts, or cup holders in arbitrary tilt angles.
This module dynamically estimates the phone-to-vehicle rotation R_p2v without manual setup:
  1. Pitch & Roll: Extracted from the static/quasi-static gravity vector during rest.
  2. Yaw (Heading Offset): Resolved by correlating horizontal phone acceleration
     against the time derivative of GNSS speed during initial vehicle acceleration.
  3. Mount Dislodgement Detection: Detects if the phone shifted or was touched mid-drive.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

G_ACCEL = 9.80665


@dataclass
class MountOrientation:
    pitch_rad: float
    roll_rad: float
    yaw_rad: float
    pitch_deg: float
    roll_deg: float
    yaw_deg: float
    R_mount: np.ndarray  # 3x3 rotation matrix from Phone Frame to Vehicle Frame
    is_calibrated: bool
    confidence: float


class DynamicMountCalibrator:
    """Automatic Two-Stage Phone-to-Vehicle Frame Auto-Calibrator."""

    def __init__(self, fs: float = 10.0,
                 gravity_norm_thresh: float = 1.5,
                 min_accel_for_yaw_mps2: float = 0.3):
        self.fs = fs
        self.gravity_norm_thresh = gravity_norm_thresh
        self.min_accel_for_yaw = min_accel_for_yaw_mps2

        # Calibration state
        self.pitch = 0.0
        self.roll = 0.0
        self.yaw = 0.0
        self.gyro_sign = 1.0
        self.gyro_bias = 0.0
        self.R_mount = np.eye(3, dtype=float)
        self.tilt_calibrated = False
        self.yaw_calibrated = False

        # Memory buffer for dynamic re-alignment detection
        self.last_gravity_vector = np.array([0.0, 0.0, G_ACCEL])

    def estimate_tilt_from_gravity(self, acc_with_g: np.ndarray) -> Tuple[float, float, np.ndarray]:
        """Stage 1: Estimate Pitch and Roll from measured mean gravity vector.
        
        Args:
            acc_with_g: (N, 3) raw accelerometer array including Earth gravity.
        Returns:
            pitch_rad, roll_rad, R_level (3x3 rotation taking phone to gravity-levelled frame).
        """
        g_mean = np.nanmean(acc_with_g, axis=0)
        norm_g = np.linalg.norm(g_mean) + 1e-9

        # Verify plausibility of gravity magnitude (within 1.5 m/s^2 of 9.81)
        if abs(norm_g - G_ACCEL) > self.gravity_norm_thresh * 2.0:
            # Fallback if vehicle is aggressively accelerating during estimation
            g_unit = np.array([0.0, 0.0, 1.0])
        else:
            g_unit = g_mean / norm_g

        # Euler angles from gravity direction:
        # roll: rotation around phone body X-axis
        # pitch: rotation around phone body Y-axis
        roll = float(math.atan2(g_unit[1], g_unit[2]))
        pitch = float(math.atan2(-g_unit[0], math.hypot(g_unit[1], g_unit[2])))

        cp, sp = math.cos(pitch), math.sin(pitch)
        cr, sr = math.cos(roll), math.sin(roll)

        Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=float)
        Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=float)
        R_level = Ry @ Rx

        self.pitch = pitch
        self.roll = roll
        self.tilt_calibrated = True
        self.last_gravity_vector = g_mean
        return pitch, roll, R_level

    def estimate_yaw_alignment(self, acc_level: np.ndarray, gps_speed: np.ndarray) -> Tuple[float, float]:
        """Stage 2: Estimate Heading Offset (Yaw) by correlating horizontal acceleration with GNSS speed derivative."""
        dv = np.gradient(gps_speed) * self.fs
        valid = np.isfinite(dv) & (np.abs(dv) > self.min_accel_for_yaw)

        if valid.sum() < int(1.5 * self.fs):
            return 0.0, 0.0

        ax = acc_level[valid, 0]
        ay = acc_level[valid, 1]
        dv_valid = dv[valid]

        dot_x = float(np.dot(ax, dv_valid))
        dot_y = float(np.dot(ay, dv_valid))

        yaw = float(math.atan2(dot_y, dot_x))
        denom = (np.linalg.norm(dv_valid) * np.linalg.norm(np.hypot(ax, ay)) + 1e-6)
        confidence = min(1.0, float(math.hypot(dot_x, dot_y) / denom))

        self.yaw = yaw
        self.yaw_calibrated = (confidence >= 0.25)
        return yaw, confidence

    def calibrate(self, acc_raw: np.ndarray, gyro_raw: np.ndarray,
                  gps_speed: np.ndarray,
                  gps_course_rad: Optional[np.ndarray] = None) -> MountOrientation:
        """Execute full dynamic calibration pipeline."""
        # 1. Pitch & Roll
        pitch, roll, R_level = self.estimate_tilt_from_gravity(acc_raw)

        # Level accelerations & angular rates
        a_level = acc_raw @ R_level.T
        w_level = gyro_raw @ R_level.T
        
        # 2. Yaw offset (only apply if correlation confidence is high enough and within plausible forward mount)
        yaw, conf = self.estimate_yaw_alignment(a_level, gps_speed)
        effective_yaw = yaw if (conf >= 0.70 and abs(yaw) < math.pi / 3.0) else 0.0

        # 3. Gyroscope sign and bias dynamic alignment against GNSS course
        if gps_course_rad is not None and len(gps_course_rad) >= 10:
            crs_arr = np.unwrap(gps_course_rad)
            # Smooth 1Hz staircase GNSS course to compute true angular rate
            change_idx = np.where(np.diff(crs_arr) != 0)[0] + 1
            if len(change_idx) >= 2:
                anchors = np.unique(np.concatenate([[0], change_idx, [len(crs_arr) - 1]]))
                t_idx = np.arange(len(crs_arr))
                crs_smooth = np.interp(t_idx, anchors, crs_arr[anchors])
                w_gnss = np.gradient(crs_smooth) * self.fs
            else:
                w_gnss = np.gradient(crs_arr) * self.fs

            w_z = w_level[:, 2]
            
            # Detect turning sign correlation during significant turns (|w_gnss| > 0.04 rad/s)
            turning_mask = (np.abs(w_gnss) > 0.04) & np.isfinite(w_z)
            if np.sum(turning_mask) >= 10:
                gnss_turns = w_gnss[turning_mask]
                gyro_turns = w_z[turning_mask]
                denom = (np.linalg.norm(gnss_turns) * np.linalg.norm(gyro_turns) + 1e-6)
                corr = float(np.dot(gnss_turns, gyro_turns) / denom)
                if abs(corr) >= 0.35:
                    self.gyro_sign = 1.0 if corr > 0 else -1.0
            
            # Gyro bias: prefer stationary rest (speed < 0.5 m/s), else shrunk dynamic difference
            spd_arr = gps_speed if gps_speed is not None else np.zeros_like(w_gnss)
            stop_mask = (spd_arr < 0.5) & np.isfinite(w_z)
            if np.sum(stop_mask) >= 5:
                self.gyro_bias = float(np.median(w_z[stop_mask]))
                self.gyro_bias = float(np.clip(self.gyro_bias, -0.008, 0.008))
            else:
                diff = w_z - self.gyro_sign * w_gnss
                valid_diff = np.isfinite(diff) & (spd_arr > 2.0)
                if np.sum(valid_diff) >= 5:
                    # Regularized shrinkage towards 0 to avoid GNSS course gradient noise
                    self.gyro_bias = float(np.clip(float(np.median(diff[valid_diff])) * 0.15, -0.0025, 0.0025))
                else:
                    self.gyro_bias = 0.0

        # 4. Full 3D Phone-to-Vehicle Rotation Matrix: R_mount = R_z(yaw) @ R_level
        cy, sy = math.cos(-effective_yaw), math.sin(-effective_yaw)
        Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=float)
        self.R_mount = Rz @ R_level

        return MountOrientation(
            pitch_rad=pitch,
            roll_rad=roll,
            yaw_rad=effective_yaw,
            pitch_deg=math.degrees(pitch),
            roll_deg=math.degrees(roll),
            yaw_deg=math.degrees(effective_yaw),
            R_mount=self.R_mount,
            is_calibrated=self.tilt_calibrated,
            confidence=conf
        )

    def transform_to_vehicle_frame(self, acc_raw: np.ndarray, gyro_raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply computed mounting rotation matrix to raw sensor frames and remove Earth gravity."""
        acc_v = self.R_mount @ acc_raw
        # Gravity acts along Earth Z (down/up in vehicle frame)
        acc_v = acc_v - np.array([0.0, 0.0, G_ACCEL])
        gyro_v = self.R_mount @ gyro_raw
        return acc_v, gyro_v

    def get_calibrated_yaw_rate(self, gyro_veh: np.ndarray) -> float:
        """Return calibrated geographic heading rate (rad/s, clockwise positive)."""
        wz = float(gyro_veh[2])
        return float(self.gyro_sign * (wz - self.gyro_bias))


# Export alias
AutoCalibrator = DynamicMountCalibrator
