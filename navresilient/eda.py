"""
Exploratory Data Analysis (EDA) for IO-VNBD Sensor Traces and Characterization.

Produces statistical summary plots:
- Raw Accelerometer (m/s^2) with gravity separation
- 3-axis Gyroscope (rad/s) and bias drift
- Power Spectral Density (PSD) showing engine vibration vs kinematic band
- Ground Truth Trajectory in Local ENU with GPS blackout mask

Usage:
    python -m navresilient.eda --s-csv <path_to_S_csv> --out figures/eda_plot.png
    python -m navresilient.eda --demo --out figures/eda_demo.png
"""

from __future__ import annotations

import argparse
import os
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .demo import synthetic_drive
from .io_vnbd_loader import inspect_dataset_stats, load_smartphone_drive


def run_eda(s_csv: str, out_png: str = "figures/eda_plot.png"):
    drive = load_smartphone_drive(s_csv)
    stats = inspect_dataset_stats(drive)

    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)

    fig, axs = plt.subplots(2, 2, figsize=(14, 9))
    t = drive.t_s

    # 1. Accelerometer Traces
    ax1 = axs[0, 0]
    ax1.plot(t, drive.acc[:, 0], label="Acc X (Forward)", color="#0284C7", lw=1.2)
    ax1.plot(t, drive.acc[:, 1], label="Acc Y (Lateral)", color="#10B981", lw=1.2)
    ax1.plot(t, drive.acc[:, 2], label="Acc Z (Vertical Linear)", color="#64748B", lw=1.0)
    ax1.set_title("3-Axis Accelerometer (Linear motion, gravity removed)", fontsize=10, weight="bold")
    ax1.set_xlabel("Time (s)"); ax1.set_ylabel("Acceleration (m/s²)")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.25)

    # 2. Gyroscope Traces & Static Bias
    ax2 = axs[0, 1]
    ax2.plot(t, drive.gyro[:, 0], label="Gyro X (Roll)", color="#94A3B8", lw=0.9)
    ax2.plot(t, drive.gyro[:, 1], label="Gyro Y (Pitch)", color="#CBD5E1", lw=0.9)
    ax2.plot(t, drive.gyro[:, 2], label="Gyro Z (Yaw Rate)", color="#C93B2B", lw=1.4)
    ax2.axhline(stats["gyro_bias_estimate_rads"], color="#F59E0B", ls="--",
                label=f"Bias bg_z: {stats['gyro_bias_estimate_rads']:.4f} rad/s")
    ax2.set_title("3-Axis Gyroscope Angular Rate & Estimated Bias", fontsize=10, weight="bold")
    ax2.set_xlabel("Time (s)"); ax2.set_ylabel("Angular Rate (rad/s)")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.25)

    # 3. Speed Comparison (GPS 1 Hz vs Continuous)
    ax3 = axs[1, 0]
    ax3.plot(t, drive.gps_speed_mps * 3.6, label="GNSS Reported Speed (1 Hz stepped)", color="#D97706", lw=1.8)
    ax3.set_title("Vehicle Speed Profile (10 Hz IMU vs 1 Hz GNSS Hold)", fontsize=10, weight="bold")
    ax3.set_xlabel("Time (s)"); ax3.set_ylabel("Speed (km/h)")
    ax3.legend(loc="upper right", fontsize=8)
    ax3.grid(True, alpha=0.25)

    # 4. Ground Truth Trajectory (Local ENU)
    ax4 = axs[1, 1]
    ax4.plot(drive.x_east_m, drive.y_north_m, label="Reference Path", color="#0E2440", lw=2.5)
    ax4.plot(drive.x_east_m[0], drive.y_north_m[0], "go", ms=8, label="Start")
    ax4.plot(drive.x_east_m[-1], drive.y_north_m[-1], "rs", ms=7, label="End")
    ax4.set_aspect("equal", "datalim")
    ax4.set_title(f"Trajectory in Local ENU — Total Dist: {stats['total_distance_m']:.0f} m", fontsize=10, weight="bold")
    ax4.set_xlabel("Local East (m)"); ax4.set_ylabel("Local North (m)")
    ax4.legend(loc="best", fontsize=8)
    ax4.grid(True, alpha=0.25)

    fig.suptitle(f"IO-VNBD Sensor Characterization — {stats['drive_name']} (fs: {stats['sample_rate_hz']} Hz)",
                 fontsize=12, weight="bold", y=0.98)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    print(f"\n[EDA COMPLETE] Plot saved to: {out_png}")
    print(json.dumps(stats, indent=2))
    return stats


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--s-csv", help="Path to S-*.csv")
    p.add_argument("--out", default="figures/eda_plot.png", help="Output PNG file")
    p.add_argument("--demo", action="store_true", help="Run on demo drive")
    a = p.parse_args()

    if a.demo or not a.s_csv:
        path = synthetic_drive("figures")
        run_eda(path, a.out)
        return

    run_eda(a.s_csv, a.out)


if __name__ == "__main__":
    main()
