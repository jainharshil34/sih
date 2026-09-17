"""
Simulate Realistic GNSS Outage Scenarios on IO-VNBD / Inertial Drive Data.

Generates structured multi-rate sensor streams (IMU at 10-100Hz, GNSS at 1Hz)
with configurable blackout windows for SIH26168 benchmark verification.
"""

from __future__ import annotations

import argparse
import math
from typing import Generator, List, Optional, Tuple

import numpy as np

from navresilient.contract import GNSSFix, IMUFrame
from navresilient.io_vnbd_loader import ParsedDrive, load_smartphone_drive


class GNSSOutageSimulator:
    """Streams synchronized IMUFrames and intermittent GNSSFixes with simulated blackouts."""

    def __init__(
        self,
        drive: ParsedDrive,
        outage_start_s: float = 30.0,
        outage_duration_s: float = 60.0,
        gnss_rate_hz: float = 1.0,
        gnss_noise_std_m: float = 2.0
    ):
        self.drive = drive
        self.outage_start_s = outage_start_s
        self.outage_end_s = outage_start_s + outage_duration_s
        self.gnss_interval_s = 1.0 / gnss_rate_hz
        self.gnss_noise_std_m = gnss_noise_std_m

    def stream_sensors(self) -> Generator[Tuple[IMUFrame, Optional[GNSSFix], bool], None, None]:
        """Generator yielding (imu_frame, optional_gnss_fix, is_in_outage) at each IMU timestep."""
        n = len(self.drive.t_s)
        last_gnss_t = -1e9

        for i in range(n):
            t = float(self.drive.t_s[i])
            is_outage = (self.outage_start_s <= t <= self.outage_end_s)

            # IMU Frame (10 Hz)
            imu_frame = IMUFrame(
                timestamp_s=t,
                ax_mps2=float(self.drive.acc_raw[i, 0]),
                ay_mps2=float(self.drive.acc_raw[i, 1]),
                az_mps2=float(self.drive.acc_raw[i, 2]),
                gx_rads=float(self.drive.gyro[i, 0]),
                gy_rads=float(self.drive.gyro[i, 1]),
                gz_rads=float(self.drive.gyro[i, 2])
            )

            # GNSS Fix (1 Hz when signal available)
            gnss_fix = None
            if not is_outage and (t - last_gnss_t) >= self.gnss_interval_s:
                # Add small simulated horizontal GPS noise
                noise_lat = np.random.normal(0, self.gnss_noise_std_m / 111320.0)
                noise_lon = np.random.normal(0, self.gnss_noise_std_m / (111320.0 * math.cos(math.radians(self.drive.lat[i]))))
                
                gnss_fix = GNSSFix(
                    timestamp_s=t,
                    latitude=float(self.drive.lat[i] + noise_lat),
                    longitude=float(self.drive.lon[i] + noise_lon),
                    altitude_m=920.0,
                    speed_mps=float(self.drive.gps_speed_mps[i]),
                    heading_deg=float(math.degrees(self.drive.gps_course_rad[i])) % 360.0,
                    accuracy_m=self.gnss_noise_std_m
                )
                last_gnss_t = t

            yield imu_frame, gnss_fix, is_outage
