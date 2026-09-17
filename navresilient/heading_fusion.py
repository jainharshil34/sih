"""
Complementary Heading & Yaw Fusion Engine.

Problem Statement SIH26168 Requirement:
In low-cost consumer smartphone MEMS IMUs, unconstrained gyroscope integration
drifts at ~0.5° - 2.0°/second. Over a 40–60 second GNSS outage across a 1 km corridor,
a 10° heading error produces > 175 m of cross-track dead-reckoning drift.

This module provides a multi-cue heading fusion engine:
1. Continuous GNSS-locked auto-calibration of gyro bias, sign, and magnetic declination.
2. Tilt-compensated 3D magnetometer heading extraction.
3. Complementary filter:
     heading_k = α * (heading_{k-1} + gyro_z * dt) + (1 - α) * mag_heading
   with α ≈ 0.98.
4. Last GNSS course anchor & distance-based decay during straight cruise.
5. Strong Zero-Yaw / Stop Clamping during stationary rest (ZUPT).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


def wrap_to_pi(angle: float) -> float:
    """Wrap angle to [-pi, pi] interval."""
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


@dataclass
class HeadingState:
    heading_rad: float
    heading_deg: float
    yaw_rate_rads: float
    mag_heading_rad: Optional[float]
    is_mag_valid: bool
    confidence: float


class ComplementaryHeadingFilter:
    """High-rate complementary heading fusion filter (Gyro + Tilt-Compensated Mag + GNSS)."""

    def __init__(
        self,
        fs: float = 10.0,
        alpha: float = 0.995,
        min_mag_norm_ut: float = 8.0,
        max_mag_norm_ut: float = 90.0,
        enable_course_anchor: bool = True
    ):
        self.fs = fs
        self.dt = 1.0 / fs
        self.alpha = alpha
        self.min_mag_norm = min_mag_norm_ut
        self.max_mag_norm = max_mag_norm_ut
        self.enable_course_anchor = enable_course_anchor

        # Internal filter states
        self.current_heading_rad = 0.0
        self.last_gnss_course_rad = 0.0
        self.mag_declination_rad = 0.0
        self.mag_calibrated = False
        self.is_gnss_locked = False
        self.outage_elapsed_s = 0.0
        self.outage_distance_m = 0.0
        self.gyro_bias = 0.0
        self.gyro_sign = 1.0
        self.is_initialized = False

    def initialize(self, initial_heading_rad: float, gyro_sign: float = 1.0, gyro_bias: float = 0.0):
        """Initialize filter heading state."""
        self.current_heading_rad = wrap_to_pi(initial_heading_rad)
        self.last_gnss_course_rad = wrap_to_pi(initial_heading_rad)
        self.gyro_sign = gyro_sign
        self.gyro_bias = gyro_bias
        self.is_initialized = True
        self.outage_elapsed_s = 0.0
        self.outage_distance_m = 0.0

    def mark_gnss_lost(self):
        """Signal GNSS outage onset."""
        self.is_gnss_locked = False
        self.outage_elapsed_s = 0.0
        self.outage_distance_m = 0.0

    def update_gnss(self, course_rad: float, speed_mps: float):
        """Update GNSS-locked reference course and calibrate magnetic declination."""
        self.is_gnss_locked = True
        self.outage_elapsed_s = 0.0
        self.outage_distance_m = 0.0

        if speed_mps > 0.4:
            self.last_gnss_course_rad = wrap_to_pi(course_rad)
            if not self.is_initialized:
                self.current_heading_rad = self.last_gnss_course_rad
                self.is_initialized = True
            else:
                # Smooth blending towards GNSS course
                err = wrap_to_pi(course_rad - self.current_heading_rad)
                self.current_heading_rad = wrap_to_pi(self.current_heading_rad + 0.35 * err)

    def calibrate_mag_declination(
        self,
        acc_raw: np.ndarray,
        mag_raw: np.ndarray,
        gnss_course_rad: float,
        speed_mps: float
    ):
        """Estimate the angular offset between tilt-compensated magnetometer and true course."""
        if speed_mps < 0.4 or mag_raw is None or np.all(mag_raw == 0.0):
            return

        psi_mag, valid = self.compute_tilt_compensated_mag_heading(acc_raw, mag_raw)
        if valid and psi_mag is not None:
            delta = wrap_to_pi(gnss_course_rad - psi_mag)
            if not self.mag_calibrated:
                self.mag_declination_rad = delta
                self.mag_calibrated = True
            else:
                # Exponential moving average
                diff = wrap_to_pi(delta - self.mag_declination_rad)
                self.mag_declination_rad = wrap_to_pi(self.mag_declination_rad + 0.05 * diff)

    def compute_tilt_compensated_mag_heading(
        self,
        acc_raw: np.ndarray,
        mag_raw: Optional[np.ndarray]
    ) -> Tuple[Optional[float], bool]:
        """Compute tilt-compensated magnetic heading from raw 3D accelerometer and magnetometer."""
        if mag_raw is None or len(mag_raw) < 3 or np.all(mag_raw == 0.0):
            return None, False

        ax, ay, az = float(acc_raw[0]), float(acc_raw[1]), float(acc_raw[2])
        mx, my, mz = float(mag_raw[0]), float(mag_raw[1]), float(mag_raw[2])

        # Pitch & Roll from gravity vector
        roll = math.atan2(ay, az)
        pitch = math.atan2(-ax, math.hypot(ay, az))

        cp, sp = math.cos(pitch), math.sin(pitch)
        cr, sr = math.cos(roll), math.sin(roll)

        # De-tilt magnetometer into horizontal plane
        bx = mx * cp + my * sr * sp + mz * cr * sp
        by = my * cr - mz * sr

        norm_b = math.hypot(bx, by)
        if norm_b < self.min_mag_norm or norm_b > self.max_mag_norm:
            # Magnetic anomaly / disturbance (iron near sensor)
            return None, False

        # Magnetometer azimuth (clockwise from magnetic North)
        psi_mag = math.atan2(-by, bx)
        return float(psi_mag), True

    def step(
        self,
        gyro_z_raw: float,
        acc_raw: np.ndarray,
        mag_raw: Optional[np.ndarray] = None,
        dt: Optional[float] = None,
        is_stationary: bool = False,
        forward_speed_mps: float = 0.0
    ) -> HeadingState:
        """Execute one step of complementary heading estimation."""
        step_dt = dt if dt is not None else self.dt

        # 1. Zero-Yaw Clamping during stationary rest (ZUPT)
        if is_stationary:
            return HeadingState(
                heading_rad=self.current_heading_rad,
                heading_deg=math.degrees(self.current_heading_rad) % 360.0,
                yaw_rate_rads=0.0,
                mag_heading_rad=None,
                is_mag_valid=False,
                confidence=1.0
            )

        # 2. Calibrated Gyro Yaw Rate
        w_z = self.gyro_sign * (gyro_z_raw - self.gyro_bias)

        # 3. High-Rate Gyro Propagation
        psi_gyro = wrap_to_pi(self.current_heading_rad + w_z * step_dt)

        # 4. Magnetometer Tilt Compensation & Low-Frequency Correction
        psi_mag, is_mag_valid = self.compute_tilt_compensated_mag_heading(acc_raw, mag_raw)
        fused_heading = psi_gyro
        confidence = 0.85

        if is_mag_valid and psi_mag is not None and self.mag_calibrated:
            # Calibrated magnetic heading in geographic frame
            psi_mag_cal = wrap_to_pi(psi_mag + self.mag_declination_rad)
            innov = wrap_to_pi(psi_mag_cal - psi_gyro)
            
            # Fuse with complementary filter
            fused_heading = wrap_to_pi(psi_gyro + (1.0 - self.alpha) * innov)
            confidence = 0.95

        # 5. Last GNSS Course Soft Anchoring with Distance & Turn Decay
        if not self.is_gnss_locked and self.enable_course_anchor:
            self.outage_elapsed_s += step_dt
            self.outage_distance_m += forward_speed_mps * step_dt
            
            anchor_err = wrap_to_pi(self.last_gnss_course_rad - fused_heading)
            # Only anchor during initial straight road before turns (< 5 deg deviation)
            if abs(anchor_err) < 0.08:
                # Distance-based decay: active initially, drops smoothly to 0 by 150m
                dist_decay = max(0.0, 1.0 - (self.outage_distance_m / 150.0))
                if dist_decay > 0.0 and forward_speed_mps > 4.0 and abs(w_z) < 0.005:
                    gain = 0.004 * dist_decay
                    fused_heading = wrap_to_pi(fused_heading + gain * anchor_err)

        self.current_heading_rad = fused_heading

        return HeadingState(
            heading_rad=self.current_heading_rad,
            heading_deg=math.degrees(self.current_heading_rad) % 360.0,
            yaw_rate_rads=w_z,
            mag_heading_rad=psi_mag,
            is_mag_valid=is_mag_valid,
            confidence=confidence
        )


# Export alias
HeadingFusionEngine = ComplementaryHeadingFilter
