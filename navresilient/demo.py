"""
Synthetic drive generator for IO-VNBD baseline testing.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
from .io_vnbd import FS

HEADER = [
    "LOCATION Latitude : ", "LOCATION Longitude : ", "LOCATION Altitude ( m)",
    "LOCATION Speed ( Kmh)", "LOCATION Accuracy ( m)", "LOCATION ORIENTATION (°)",
    "Satellites in range", "Time since start in ms ", "YYYY-MO-DD HH-MI-SS_SSS",
    "ACCELEROMETER X (m/s²)", "ACCELEROMETER Y (m/s²)", "ACCELEROMETER Z (m/s²)",
    "GRAVITY X (m/s²)", "GRAVITY Y (m/s²)", "GRAVITY Z (m/s²)",
    "GYROSCOPE X (rad/s)", "GYROSCOPE Y (rad/s)", "GYROSCOPE Z (rad/s)",
    "MAGNETIC FIELD X (μT)", "MAGNETIC FIELD Y (μT)", "MAGNETIC FIELD Z (μT)",
    "ORIENTATION Z (azimuth °)", "ORIENTATION X (pitch °)", "ORIENTATION Y (roll °)",
]
G = 9.80665


def synthetic_drive(out_dir: str, minutes: float = 4.0, seed: int = 7) -> str:
    rng = np.random.default_rng(seed)
    n = int(minutes * 60 * FS)
    t = np.arange(n) / FS

    # Plausible drive: accelerate, cruise, stops, curves
    v = 16.7 * (1 - np.exp(-t / 8.0))
    v[int(95 * FS):int(108 * FS)] = 0.0      # junction stop
    v[int(150 * FS):int(162 * FS)] = 0.0    # red light mid-blackout
    v = np.convolve(v, np.ones(10) / 10, "same")
    yaw_rate = 0.28 * np.sin(2 * np.pi * t / 55.0) * (v > 1)
    heading = np.cumsum(yaw_rate) / FS

    a_fwd = np.gradient(v) * FS
    a_lat = v * yaw_rate

    # Sensor corruption: bias, noise, engine vibration, pothole impulses
    bias = np.array([0.049, -0.03, 0.02])
    vib = 0.15 * G * np.column_stack([
        np.sin(2 * np.pi * 24 * t), np.sin(2 * np.pi * 31 * t + 1.1),
        np.sin(2 * np.pi * 27 * t + 2.3)]) * 0.25
    acc = np.column_stack([a_fwd, a_lat, np.zeros(n)]) + bias + vib + rng.normal(0, 0.05, (n, 3))
    acc[:, 2] += G
    grav = np.tile([0.0, 0.0, G], (n, 1))
    gyro = np.column_stack([
        rng.normal(0, 0.01, n), rng.normal(0, 0.01, n),
        -yaw_rate + 0.004 + rng.normal(0, 0.008, n)])

    x, y = np.cumsum(v * np.sin(heading)) / FS, np.cumsum(v * np.cos(heading)) / FS
    R, lat0, lon0 = 6378137.0, 52.4068, -1.5197
    lat = lat0 + np.degrees(y / R)
    lon = lon0 + np.degrees(x / (R * np.cos(np.radians(lat0))))
    hold = (np.arange(n) // int(FS)) * int(FS)
    lat, lon, v_gps = lat[hold], lon[hold], v[hold]

    df = pd.DataFrame({
        HEADER[0]: lat, HEADER[1]: lon, HEADER[2]: 80.0,
        HEADER[3]: v_gps, HEADER[4]: 4.0,
        HEADER[5]: np.degrees(heading[hold]) % 360, HEADER[6]: 9,
        HEADER[7]: t * 1000.0, HEADER[8]: "2026-01-01 00-00-00_000",
        HEADER[9]: acc[:, 0], HEADER[10]: acc[:, 1], HEADER[11]: acc[:, 2],
        HEADER[12]: grav[:, 0], HEADER[13]: grav[:, 1], HEADER[14]: grav[:, 2],
        HEADER[15]: gyro[:, 0], HEADER[16]: gyro[:, 1], HEADER[17]: gyro[:, 2],
        HEADER[18]: 20.0 * np.cos(heading) + rng.normal(0, 0.5, n),
        HEADER[19]: -20.0 * np.sin(heading) + rng.normal(0, 0.5, n),
        HEADER[20]: 42.0 + rng.normal(0, 0.5, n),
        HEADER[21]: np.degrees(heading) % 360, HEADER[22]: -2.0, HEADER[23]: 1.0,
    })
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "S-DEMO-SYNTHETIC.csv")
    df.to_csv(path, index=False)
    return path
