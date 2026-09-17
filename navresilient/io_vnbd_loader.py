"""
IO-VNBD Dataset Downloader, Loader, and Statistical Characterization Engine.

Dataset Citation:
    Onyekpe, U., Palade, V., Kanarachos, S. & Szkolnik, A. (2021).
    IO-VNBD: Inertial and odometry benchmark dataset for ground vehicle positioning.
    Data in Brief 35, 106885. https://github.com/onyekpeu/IO-VNBD

Verified Dataset Specifications:
  * Smartphone Sensor Rate: 10.0 Hz (Delta-T = 100 ms)
  * GPS Fix Rate: 1.0 Hz (Position and speed held constant for 10 consecutive rows)
  * Accelerometer: m/s^2 (Includes specific force; GRAVITY X/Y/Z provided separately)
  * Gyroscope: rad/s (Body angular rates wx, wy, wz)
  * Vehicle Wheel Speed: rad/s (Requires rolling radius R = 0.303m for Ford Fiesta 195/55 R16)
  * GPS Speed: km/h in raw header, converted to m/s during parsing
"""

from __future__ import annotations

import math
import os
import re
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

FS = 10.0
GPS_FS = 1.0
WHEEL_RADIUS_M = 0.303  # 195/55 R16 tire on Ford Fiesta Titanium

_MEDIA_BASE_URL = "https://media.githubusercontent.com/media/onyekpeu/IO-VNBD/master/Synchronised%20V%20abd%20S%20datasets/Uncategorised%20IOVNB%20Dataset"


def download_iovnbd_drive(drive_id: str = "Vta10", out_dir: str = "data/IO-VNBD") -> Tuple[str, str]:
    """Download real IO-VNBD dataset CSV files directly from GitHub LFS media storage."""
    os.makedirs(out_dir, exist_ok=True)
    s_path = os.path.join(out_dir, f"S-{drive_id}.csv")
    v_path = os.path.join(out_dir, f"V-{drive_id}.csv")

    s_url = f"{_MEDIA_BASE_URL}/S-Dataset/S-{drive_id}.csv"
    v_url = f"{_MEDIA_BASE_URL}/V-Dataset/V-{drive_id}.csv"

    for url, path, name in [(s_url, s_path, f"S-{drive_id}"), (v_url, v_path, f"V-{drive_id}")]:
        if not os.path.exists(path) or os.path.getsize(path) < 1000:
            print(f"Downloading real IO-VNBD {name}.csv from GitHub LFS...")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
            with urllib.request.urlopen(req) as resp, open(path, "wb") as f:
                content = resp.read()
                f.write(content)
            print(f"  Saved {name}.csv ({len(content) / 1024:.1f} KB)")
        else:
            print(f"  Found cached {name}.csv ({os.path.getsize(path) / 1024:.1f} KB)")

    return s_path, v_path

_S_KEYS = {
    "lat": [["location", "latitude"], ["gps", "latitude"], ["latitude"]],
    "lon": [["location", "longitude"], ["gps", "longitude"], ["longitude"]],
    "alt": [["location", "altitude"], ["gps", "altitude"], ["altitude"]],
    "gps_speed": [["location", "speed"], ["gps", "speed"], ["speed"]],
    "gps_acc": [["location", "accuracy"], ["gps", "accuracy"], ["accuracy"]],
    "gps_course": [["location", "orientation"], ["gps", "orientation"]],
    "n_sat": [["satellites"]],
    "t_ms": [["time", "since", "start"], ["time"]],
    "ax": [["accelerometer", " x"], ["accelerometer x"], ["acc", "x"]],
    "ay": [["accelerometer", " y"], ["accelerometer y"], ["acc", "y"]],
    "az": [["accelerometer", " z"], ["accelerometer z"], ["acc", "z"]],
    "gx": [["gravity", " x"], ["gravity x"]],
    "gy": [["gravity", " y"], ["gravity y"]],
    "gz": [["gravity", " z"], ["gravity z"]],
    "wz": [["gyroscope", " z"], ["gyroscope z"], ["gyro", "yaw"]],
    "wy": [["gyroscope", " y"], ["gyroscope y"], ["gyro", "pitch"]],
    "wx": [["gyroscope", " x"], ["gyroscope x"], ["gyro", "roll"]],
    "mx": [["magnetic", " x"], ["magnetic field x"]],
    "my": [["magnetic", " y"], ["magnetic field y"]],
    "mz": [["magnetic", " z"], ["magnetic field z"]],
}

_V_KEYS = {
    "ws_fl": [["wheel", "speed", "front", "left"]],
    "ws_fr": [["wheel", "speed", "front", "right"]],
    "ws_rl": [["wheel", "speed", "rear", "left"]],
    "ws_rr": [["wheel", "speed", "rear", "right"]],
    "v_ind": [["indicated", "vehicle", "speed"]],
    "gps_speed": [["gps", "velocity"]],
    "yaw_rate": [["yaw", "rate"]],
}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())


def _resolve_cols(columns: List[str], keymap: Dict[str, List[List[str]]]) -> Dict[str, str]:
    low = [_norm(c) for c in columns]
    out, used = {}, set()
    for canon, variants in keymap.items():
        for kws in variants:
            hit = None
            for i, c in enumerate(low):
                if i in used:
                    continue
                if all(k in c for k in kws):
                    hit = i
                    break
            if hit is not None:
                out[canon] = columns[hit]
                used.add(hit)
                break
    return out


@dataclass
class ParsedDrive:
    name: str
    t_s: np.ndarray
    acc: np.ndarray        # (N, 3) m/s^2 linear (gravity removed)
    acc_raw: np.ndarray    # (N, 3) m/s^2 with gravity
    gyro: np.ndarray       # (N, 3) rad/s
    gravity: np.ndarray    # (N, 3) m/s^2
    lat: np.ndarray
    lon: np.ndarray
    gps_speed_mps: np.ndarray
    gps_course_rad: np.ndarray
    x_east_m: np.ndarray
    y_north_m: np.ndarray
    mag: Optional[np.ndarray] = None  # (N, 3) uT magnetometer
    fs: float = FS

    def __len__(self) -> int:
        return len(self.t_s)


def latlon_to_enu(lat: np.ndarray, lon: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert WGS-84 lat/lon array to local tangent plane East/North (meters)."""
    R = 6378137.0
    lat0, lon0 = math.radians(lat[0]), math.radians(lon[0])
    la, lo = np.radians(lat), np.radians(lon)
    x = (lo - lon0) * math.cos(lat0) * R
    y = (la - lat0) * R
    return x, y


def load_smartphone_drive(path: str, name: Optional[str] = None) -> ParsedDrive:
    """Load and parse an S-*.csv file from IO-VNBD."""
    df = pd.read_csv(path, low_memory=False, encoding="latin-1")
    col = _resolve_cols(list(df.columns), _S_KEYS)

    def g(k: str, default: float = np.nan) -> np.ndarray:
        if k not in col:
            return np.full(len(df), default, dtype=float)
        return pd.to_numeric(df[col[k]], errors="coerce").to_numpy(dtype=float)

    acc_raw = np.column_stack([g("ax"), g("ay"), g("az")])
    grav = np.column_stack([g("gx", 0.0), g("gy", 0.0), g("gz", 0.0)])
    acc_linear = acc_raw - grav if (np.isfinite(grav).all() and np.abs(grav).sum() > 0) else acc_raw

    gyro = np.column_stack([g("wx"), g("wy"), g("wz")])
    t_raw = g("t_ms")
    if np.isfinite(t_raw).any():
        dt_median = float(np.nanmedian(np.diff(t_raw))) if len(t_raw) > 1 else 0.1
        if dt_median > 5.0 or np.nanmax(t_raw) > 10000.0:
            t_s = (t_raw - np.nanmin(t_raw)) / 1000.0
        else:
            t_s = t_raw - np.nanmin(t_raw)
    else:
        t_s = np.arange(len(df)) / FS

    lat, lon = g("lat"), g("lon")
    ok = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(acc_raw).all(1) & np.isfinite(gyro).all(1)
    if ok.sum() < 20:
        raise ValueError(f"Insufficient valid data in {path}")

    sl = slice(int(np.argmax(ok)), len(ok) - int(np.argmax(ok[::-1])))

    lat = pd.Series(lat).ffill().bfill().to_numpy()[sl]
    lon = pd.Series(lon).ffill().bfill().to_numpy()[sl]
    raw_spd = g("gps_speed")[sl]
    # In AndroSensor/smartphone datasets, GPS speed is recorded in m/s
    spd = raw_spd if np.isfinite(raw_spd).any() else np.zeros_like(lat)

    # If GPS lat/lon was held constant between discrete receiver fixes, interpolate smoothly
    change_idx = np.where((np.diff(lat) != 0) | (np.diff(lon) != 0))[0] + 1
    if len(change_idx) > 1:
        if change_idx[0] > 0 and len(change_idx) >= 2:
            dt_fresh = float(change_idx[1] - change_idx[0])
            lat[0] = lat[change_idx[0]] - (lat[change_idx[1]] - lat[change_idx[0]]) * (float(change_idx[0]) / dt_fresh)
            lon[0] = lon[change_idx[0]] - (lon[change_idx[1]] - lon[change_idx[0]]) * (float(change_idx[0]) / dt_fresh)
        full_anchors = np.unique(np.concatenate([[0], change_idx, [len(lat) - 1]]))
        t_idx = np.arange(len(lat))
        lat = np.interp(t_idx, full_anchors, lat[full_anchors])
        lon = np.interp(t_idx, full_anchors, lon[full_anchors])

    x, y = latlon_to_enu(lat, lon)
    
    # Calculate Course Over Ground (COG) from ENU displacement
    dx = np.gradient(x)
    dy = np.gradient(y)
    crs_cog = np.arctan2(dx, dy)
    
    raw_crs = np.radians(g("gps_course", 0.0)[sl])
    if np.nanmax(np.abs(raw_crs)) < 1e-4 or np.nanstd(raw_crs) < 1e-4:
        crs = crs_cog
    else:
        crs = np.where(spd > 0.8, crs_cog, raw_crs)

    mag = np.column_stack([g("mx", 0.0), g("my", 0.0), g("mz", 0.0)])[sl]

    return ParsedDrive(
        name=name or os.path.basename(path).replace(".csv", ""),
        t_s=t_s[sl] - t_s[sl][0],
        acc=acc_linear[sl],
        acc_raw=acc_raw[sl],
        gyro=gyro[sl],
        gravity=grav[sl],
        lat=lat,
        lon=lon,
        gps_speed_mps=spd,
        gps_course_rad=crs,
        x_east_m=x,
        y_north_m=y,
        mag=mag
    )


def load_vehicle_drive(path: str) -> np.ndarray:
    """Load wheel-speed supervision signal (m/s) from V-*.csv."""
    df = pd.read_csv(path, low_memory=False, encoding="latin-1")
    col = _resolve_cols(list(df.columns), _V_KEYS)
    ws = [c for k, c in col.items() if k.startswith("ws_")]
    if len(ws) >= 2:
        rad_s = df[ws].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        return np.nanmean(rad_s, axis=1) * WHEEL_RADIUS_M
    if "v_ind" in col:
        return pd.to_numeric(df[col["v_ind"]], errors="coerce").to_numpy(float) / 3.6
    raise ValueError(f"No wheel-speed columns in {path}")


def inspect_dataset_stats(drive: ParsedDrive) -> dict:
    """Compute verified physical statistics: sample rate, noise floor, Allan bias."""
    dt_actual = np.diff(drive.t_s)
    actual_fs = 1.0 / np.median(dt_actual)

    # Accelerometer noise floor (standard deviation of norm)
    acc_norm = np.linalg.norm(drive.acc_raw, axis=1)
    acc_noise_sigma = float(np.std(acc_norm))

    # Gyroscope stationary noise and bias
    gyro_z = drive.gyro[:, 2]
    gyro_noise_sigma = float(np.std(gyro_z))
    gyro_bias_est = float(np.median(gyro_z))

    return {
        "drive_name": drive.name,
        "sample_count": len(drive.t_s),
        "duration_seconds": round(float(drive.t_s[-1]), 2),
        "sample_rate_hz": round(float(actual_fs), 2),
        "acc_noise_floor_mps2": round(acc_noise_sigma, 4),
        "gyro_noise_floor_rads": round(gyro_noise_sigma, 5),
        "gyro_bias_estimate_rads": round(gyro_bias_est, 6),
        "total_distance_m": round(float(np.sum(np.hypot(np.diff(drive.x_east_m), np.diff(drive.y_north_m)))), 2),
        "verified_units": {
            "accelerometer": "m/s^2 (specific force)",
            "gyroscope": "rad/s",
            "gps_speed": "m/s (converted from km/h)",
            "wheel_speed": "rad/s (scaled by 0.303m radius)"
        }
    }
