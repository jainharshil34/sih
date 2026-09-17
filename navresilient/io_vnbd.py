"""
IO-VNBD loader.

Column layout follows Table 5 (smartphone, "S-" files, 24 columns) and Table 3
(vehicle CAN/ECU, "V-" files, 29 columns) of:

    Onyekpe, U., Palade, V., Kanarachos, S. & Szkolnik, A. (2021).
    IO-VNBD: Inertial and odometry benchmark dataset for ground vehicle
    positioning. Data in Brief 35, 106885.

The "S-" files are AndroSensor exports, so the real header strings are things
like "ACCELEROMETER X (m/s^2)" rather than the tidy names in the paper. We
resolve columns by keyword first and fall back to the documented positional
order, which makes the loader robust to the unicode / spacing variations that
differ between AndroSensor versions.

Facts that matter and are easy to get wrong:
  * Rows are 10 Hz. The GPS fix updates at 1 Hz, so latitude, longitude and
    speed are held constant for ten consecutive rows. Never treat consecutive
    GPS rows as independent measurements.
  * ACCELEROMETER X/Y/Z include gravity. The dataset ships GRAVITY X/Y/Z
    separately for exactly this reason. Linear acceleration = accel - gravity.
  * GPS speed is km/h. Everything else here is SI.
  * The repo ships a "GPS outages" file listing row indexes where the receiver
    lost the satellites. Those rows have no usable ground truth; drop them
    before scoring anything.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

FS = 10.0  # dataset sample rate, Hz
GPS_FS = 1.0  # GPS fix rate, Hz

# keyword -> canonical name. First keyword list that matches wins.
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
    "yaw": [["orientation", "azimuth"], ["orientation", " z"], ["orientation z"]],
    "pitch": [["orientation", "pitch"], ["orientation", " x"]],
    "roll": [["orientation", "roll"], ["orientation", " y"]],
}

# documented order from Table 5, used only when keyword matching fails
_S_POSITIONAL = [
    "lat", "lon", "alt", "gps_speed", "gps_acc", "gps_course", "n_sat", "t_ms",
    "date", "ax", "ay", "az", "gx", "gy", "gz", "wz", "wy", "wx",
    "mx", "my", "mz", "yaw", "pitch", "roll",
]

_V_KEYS = {
    "n_sat": [["satellite"]],
    "t_s": [["time", "day"]],
    "lat": [["latitude"]],
    "lon": [["longitude"]],
    "gps_speed": [["gps", "velocity"]],
    "gps_course": [["gps", "heading"]],
    "ws_fl": [["wheel", "speed", "front", "left"]],
    "ws_fr": [["wheel", "speed", "front", "right"]],
    "ws_rl": [["wheel", "speed", "rear", "left"]],
    "ws_rr": [["wheel", "speed", "rear", "right"]],
    "yaw_rate": [["yaw", "rate"]],
    "v_ind": [["indicated", "vehicle", "speed"]],
    "a_long": [["indicated", "longitudinal"]],
    "a_lat": [["indicated", "lateral"]],
    "steer": [["steering"]],
}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())


def _resolve(columns, keymap, positional=None):
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
    if positional:
        for i, canon in enumerate(positional):
            if canon != "date" and canon not in out and i < len(columns):
                out[canon] = columns[i]
    return out


@dataclass
class Drive:
    """One IO-VNBD recording, resampled onto a clean 10 Hz clock."""
    name: str
    t: np.ndarray          # s, from 0
    acc: np.ndarray        # (N,3) m/s^2, gravity removed, phone body frame
    gyro: np.ndarray       # (N,3) rad/s, phone body frame (x, y, z)
    mag: np.ndarray | None # (N,3) uT or None
    lat: np.ndarray
    lon: np.ndarray
    gps_speed: np.ndarray  # m/s
    gps_course: np.ndarray # rad, 0 = north, clockwise
    n_sat: np.ndarray
    x: np.ndarray          # m, local ENU east
    y: np.ndarray          # m, local ENU north
    fs: float = FS

    def __len__(self):
        return len(self.t)

    @property
    def distance(self) -> float:
        return float(np.sum(np.hypot(np.diff(self.x), np.diff(self.y))))


def _enu(lat, lon):
    """Local tangent-plane projection about the first fix. Exact enough over
    the few-km spans we score; avoids a pyproj dependency."""
    R = 6378137.0
    lat0, lon0 = math.radians(lat[0]), math.radians(lon[0])
    la, lo = np.radians(lat), np.radians(lon)
    x = (lo - lon0) * math.cos(lat0) * R
    y = (la - lat0) * R
    return x, y


def load_smartphone(path: str, name: str | None = None) -> Drive:
    """Load one S-*.csv."""
    df = pd.read_csv(path, low_memory=False, encoding="latin-1")
    col = _resolve(list(df.columns), _S_KEYS, _S_POSITIONAL)
    need = ["lat", "lon", "ax", "ay", "az", "wx", "wy", "wz"]
    missing = [k for k in need if k not in col]
    if missing:
        raise ValueError(
            f"{path}: could not resolve columns {missing}. "
            f"Header seen: {list(df.columns)[:6]}..."
        )

    def g(k, default=np.nan):
        if k not in col:
            return np.full(len(df), default, dtype=float)
        return pd.to_numeric(df[col[k]], errors="coerce").to_numpy(dtype=float)

    acc = np.column_stack([g("ax"), g("ay"), g("az")])
    grav = np.column_stack([g("gx", 0.0), g("gy", 0.0), g("gz", 0.0)])
    if np.isfinite(grav).all() and np.abs(grav).sum() > 0:
        acc = acc - grav  # AndroSensor accelerometer includes gravity
    gyro = np.column_stack([g("wx"), g("wy"), g("wz")])
    mag = np.column_stack([g("mx"), g("my"), g("mz")])
    if not np.isfinite(mag).any():
        mag = None

    t_ms = g("t_ms")
    t = (t_ms - np.nanmin(t_ms)) / 1000.0 if np.isfinite(t_ms).any() \
        else np.arange(len(df)) / FS

    lat, lon = g("lat"), g("lon")
    ok = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(acc).all(1) & np.isfinite(gyro).all(1)
    if ok.sum() < 100:
        raise ValueError(f"{path}: only {ok.sum()} usable rows")
    sl = slice(int(np.argmax(ok)), len(ok) - int(np.argmax(ok[::-1])))

    lat, lon = pd.Series(lat).ffill().bfill().to_numpy()[sl], \
               pd.Series(lon).ffill().bfill().to_numpy()[sl]
    spd = g("gps_speed")[sl] / 3.6                       # km/h -> m/s
    crs = np.radians(g("gps_course", 0.0)[sl])
    nsat = g("n_sat", 0.0)[sl]
    x, y = _enu(lat, lon)

    return Drive(
        name=name or os.path.basename(path).replace(".csv", ""),
        t=t[sl] - t[sl][0], acc=acc[sl], gyro=gyro[sl],
        mag=None if mag is None else mag[sl],
        lat=lat, lon=lon, gps_speed=spd, gps_course=crs, n_sat=nsat, x=x, y=y,
    )


def load_vehicle_speed(path: str) -> np.ndarray:
    """Forward speed (m/s) from a V-*.csv, averaged over the four wheel-speed
    sensors. This is the supervision signal for the velocity head.

    Wheel speeds are rad/s, so they need the rolling radius. The Ford Fiesta
    Titanium used for the "V-" recordings runs 195/55 R16, giving roughly
    0.303 m. Set WHEEL_RADIUS_M if you validate a different figure against the
    indicated vehicle speed column.
    """
    WHEEL_RADIUS_M = 0.303
    df = pd.read_csv(path, low_memory=False, encoding="latin-1")
    col = _resolve(list(df.columns), _V_KEYS)
    ws = [c for k, c in col.items() if k.startswith("ws_")]
    if len(ws) >= 2:
        rad_s = df[ws].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        return np.nanmean(rad_s, axis=1) * WHEEL_RADIUS_M
    if "v_ind" in col:  # fall back to the CAN indicated speed, km/h
        return pd.to_numeric(df[col["v_ind"]], errors="coerce").to_numpy(float) / 3.6
    raise ValueError(f"{path}: no wheel-speed or indicated-speed column found")


def load_gps_outages(path: str) -> set[int]:
    """Row indexes the dataset flags as GPS dropouts. These have no trustworthy
    ground truth and must not be scored."""
    try:
        raw = pd.read_csv(path, header=None).to_numpy().ravel()
    except Exception:
        return set()
    return {int(v) for v in raw if np.isfinite(v)}
