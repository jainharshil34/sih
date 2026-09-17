"""
Sensor Logger / Mobile App Multi-CSV Merger & Synchronizer.

Takes a folder containing separate CSV files (e.g. Accelerometer.csv, Gyroscope.csv, Location.csv)
and resamples/interpolates them onto a uniform time grid (e.g. 10 Hz, 50 Hz, or 100 Hz).

Usage:
    python -m navresilient.merge_phone_sensors --input-dir data/my_recording_folder --output data/synced_walk.csv --fs 10.0
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd


def _find_csv(folder: str, keywords: list[str]) -> Optional[str]:
    """Find a CSV file in folder whose basename contains any of the keywords (case-insensitive)."""
    for fname in os.listdir(folder):
        if not fname.lower().endswith(".csv"):
            continue
        low = fname.lower()
        if any(kw in low for kw in keywords):
            return os.path.join(folder, fname)
    return None


def _get_time_col(df: pd.DataFrame) -> str:
    """Identify the timestamp column in a dataframe."""
    for col in df.columns:
        low = col.lower().strip()
        if any(k in low for k in ["seconds_elapsed", "time_since_start", "time", "timestamp", "epoch"]):
            return col
    return df.columns[0]


def merge_sensor_folder(folder_path: str, output_csv: str, target_fs: float = 10.0) -> str:
    """Merge separate sensor CSVs from a phone recording into a single synchronized CSV."""
    if not os.path.isdir(folder_path):
        raise ValueError(f"Directory not found: {folder_path}")

    print(f"Scanning '{folder_path}' for sensor CSV files...")

    acc_path = _find_csv(folder_path, ["accel", "totalacceleration"])
    gyro_path = _find_csv(folder_path, ["gyro"])
    loc_path = _find_csv(folder_path, ["loc", "gps", "position"])
    grav_path = _find_csv(folder_path, ["grav"])

    if not acc_path:
        raise FileNotFoundError("Could not find Accelerometer CSV file in the folder.")
    if not gyro_path:
        raise FileNotFoundError("Could not find Gyroscope CSV file in the folder.")

    print(f"  Found Accelerometer: {os.path.basename(acc_path)}")
    print(f"  Found Gyroscope:     {os.path.basename(gyro_path)}")
    if loc_path:
        print(f"  Found Location/GPS:  {os.path.basename(loc_path)}")

    # Read Accelerometer
    df_acc = pd.read_csv(acc_path)
    t_acc_col = _get_time_col(df_acc)
    t_acc = df_acc[t_acc_col].values.astype(float)
    if t_acc[0] > 1e11:  # Epoch in nanoseconds/milliseconds -> convert to seconds
        t_acc = (t_acc - t_acc[0]) / (1e9 if t_acc[0] > 1e14 else 1e3)
    else:
        t_acc = t_acc - t_acc[0]

    # Read Gyroscope
    df_gyro = pd.read_csv(gyro_path)
    t_gyro_col = _get_time_col(df_gyro)
    t_gyro = df_gyro[t_gyro_col].values.astype(float)
    if t_gyro[0] > 1e11:
        t_gyro = (t_gyro - t_gyro[0]) / (1e9 if t_gyro[0] > 1e14 else 1e3)
    else:
        t_gyro = t_gyro - t_gyro[0]

    # Common time range
    t_start = max(t_acc[0], t_gyro[0])
    t_end = min(t_acc[-1], t_gyro[-1])

    dt = 1.0 / target_fs
    t_uniform = np.arange(t_start, t_end, dt)
    n_samples = len(t_uniform)

    # Extract & interpolate Accel X/Y/Z
    def _find_xyz_cols(df: pd.DataFrame):
        cols = {c.lower().strip(): c for c in df.columns}
        x_col = next((c for k, c in cols.items() if k in ["x", "accel_x", "acceleration x", "ax"]), None)
        y_col = next((c for k, c in cols.items() if k in ["y", "accel_y", "acceleration y", "ay"]), None)
        z_col = next((c for k, c in cols.items() if k in ["z", "accel_z", "acceleration z", "az"]), None)
        return x_col, y_col, z_col

    ax_c, ay_c, az_c = _find_xyz_cols(df_acc)
    gx_c, gy_c, gz_c = _find_xyz_cols(df_gyro)

    ax_interp = np.interp(t_uniform, t_acc, df_acc[ax_c].values)
    ay_interp = np.interp(t_uniform, t_acc, df_acc[ay_c].values)
    az_interp = np.interp(t_uniform, t_acc, df_acc[az_c].values)

    gx_interp = np.interp(t_uniform, t_gyro, df_gyro[gx_c].values)
    gy_interp = np.interp(t_uniform, t_gyro, df_gyro[gy_c].values)
    gz_interp = np.interp(t_uniform, t_gyro, df_gyro[gz_c].values)

    out_dict = {
        "time": t_uniform,
        "accelerometer x": ax_interp,
        "accelerometer y": ay_interp,
        "accelerometer z": az_interp,
        "gyroscope x": gx_interp,
        "gyroscope y": gy_interp,
        "gyroscope z": gz_interp,
    }

    # If Gravity CSV exists
    if grav_path:
        df_grav = pd.read_csv(grav_path)
        t_grav = df_grav[_get_time_col(df_grav)].values.astype(float)
        if t_grav[0] > 1e11:
            t_grav = (t_grav - t_grav[0]) / (1e9 if t_grav[0] > 1e14 else 1e3)
        else:
            t_grav = t_grav - t_grav[0]
        gx_c, gy_c, gz_c = _find_xyz_cols(df_grav)
        if gx_c and gy_c and gz_c:
            out_dict["gravity x"] = np.interp(t_uniform, t_grav, df_grav[gx_c].values)
            out_dict["gravity y"] = np.interp(t_uniform, t_grav, df_grav[gy_c].values)
            out_dict["gravity z"] = np.interp(t_uniform, t_grav, df_grav[gz_c].values)

    # If Location/GPS CSV exists
    if loc_path:
        df_loc = pd.read_csv(loc_path)
        t_loc = df_loc[_get_time_col(df_loc)].values.astype(float)
        if t_loc[0] > 1e11:
            t_loc = (t_loc - t_loc[0]) / (1e9 if t_loc[0] > 1e14 else 1e3)
        else:
            t_loc = t_loc - t_loc[0]

        lat_c = next((c for c in df_loc.columns if "lat" in c.lower()), None)
        lon_c = next((c for c in df_loc.columns if "lon" in c.lower() or "lng" in c.lower()), None)
        spd_c = next((c for c in df_loc.columns if "speed" in c.lower()), None)

        if lat_c and lon_c:
            out_dict["latitude"] = np.interp(t_uniform, t_loc, df_loc[lat_c].values)
            out_dict["longitude"] = np.interp(t_uniform, t_loc, df_loc[lon_c].values)
            if spd_c:
                out_dict["speed"] = np.interp(t_uniform, t_loc, df_loc[spd_c].values)
            else:
                out_dict["speed"] = np.zeros(n_samples)

    out_df = pd.DataFrame(out_dict)
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    print(f"\nSuccessfully synchronized {n_samples} frames ({n_samples / target_fs:.1f} s) at {target_fs} Hz!")
    print(f"Saved merged dataset to: {output_csv}")
    return output_csv


def main():
    parser = argparse.ArgumentParser(description="Merge multi-sensor CSV folder into single synchronized file")
    parser.add_argument("--input-dir", "-i", required=True, help="Folder containing Accelerometer.csv, Gyroscope.csv, Location.csv")
    parser.add_argument("--output", "-o", default="data/synced_walk.csv", help="Output merged CSV path")
    parser.add_argument("--fs", type=float, default=10.0, help="Target sampling frequency in Hz (default: 10.0)")

    args = parser.parse_args()
    merge_sensor_folder(args.input_dir, args.output, args.fs)


if __name__ == "__main__":
    main()
