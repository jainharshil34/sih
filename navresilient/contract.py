"""
Integration Contract for Engineer B (App & UI Consumer).

This module defines the strict, typed input/output schema and streaming API
consumed by the mobile application, fleet telematics units, and edge dashboards.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple


class SystemStatus(str, Enum):
    INITIALIZING = "INITIALIZING"
    GNSS_LOCKED = "GNSS_LOCKED"
    GNSS_DENIED_INS = "GNSS_DENIED_INS"
    REACQUIRED = "REACQUIRED"


@dataclass(frozen=True)
class IMUFrame:
    """Incoming High-Rate IMU Sensor Observation (10 Hz up to 200 Hz)."""
    timestamp_s: float
    ax_mps2: float       # Phone body x acceleration (m/s^2)
    ay_mps2: float       # Phone body y acceleration (m/s^2)
    az_mps2: float       # Phone body z acceleration (m/s^2, gravity included or removed)
    gx_rads: float       # Gyro roll rate (rad/s)
    gy_rads: float       # Gyro pitch rate (rad/s)
    gz_rads: float       # Gyro yaw rate (rad/s)
    mx_ut: Optional[float] = None   # Magnetometer X (micro-Tesla)
    my_ut: Optional[float] = None   # Magnetometer Y (micro-Tesla)
    mz_ut: Optional[float] = None   # Magnetometer Z (micro-Tesla)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GNSSFix:
    """Incoming Low-Rate GNSS / NavIC Fix (1 Hz to 10 Hz)."""
    timestamp_s: float
    latitude: float      # WGS-84 Latitude in degrees
    longitude: float     # WGS-84 Longitude in degrees
    altitude_m: float    # Altitude above ellipsoid (meters)
    speed_mps: float     # Ground speed (m/s)
    heading_deg: float   # Course over ground (0 - 360 degrees, 0 = True North)
    accuracy_m: float    # 1-sigma horizontal accuracy estimate (meters)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DriftCorrectedState:
    """Output Stream delivered to Engineer B's Mobile UI / App."""
    timestamp_s: float
    latitude: float
    longitude: float
    altitude_m: float
    speed_mps: float
    speed_kmh: float
    heading_deg: float
    status: SystemStatus
    drift_pct: float
    pos_uncertainty_1sigma_m: float
    is_stationary: bool
    pothole_detected: bool
    map_matched_lat: Optional[float] = None
    map_matched_lon: Optional[float] = None
    engine_latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict())
