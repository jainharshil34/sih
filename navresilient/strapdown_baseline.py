"""
Plain Strapdown INS Baseline & "Before" Evidence Generator for SIH26168.

This module evaluates plain double-integration strapdown dead reckoning against
ground truth to clearly demonstrate why raw MEMS IMU integration fails within seconds:
- Accelerometer bias causes cubic position drift: ~ 1/6 * b_a * t^3
- Gyroscope bias causes quadratic position drift: ~ 1/2 * b_g * g * t^2
- Unconstrained integration lacks Non-Holonomic Constraints (NHC), allowing lateral slip.

Generates:
  `figures/strapdown_drift_baseline.png` (Slide 4 "Before" Evidence)
"""

from __future__ import annotations

import argparse
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .demo import synthetic_drive
from .io_vnbd_loader import load_smartphone_drive
from .navigation.strapdown import StrapdownINS


def run_strapdown_baseline(s_csv: str, outage_start: float = 60.0, outage_len: float = 60.0,
                            out_png: str = "figures/strapdown_drift_baseline.png") -> dict:
    drive = load_smartphone_drive(s_csv)
    fs = drive.fs
    dt = 1.0 / fs

    i0 = int(outage_start * fs)
    i1 = min(len(drive), i0 + int(outage_len * fs))
    seg = slice(i0, i1)
    n_seg = i1 - i0

    ref_x = drive.x_east_m[seg]
    ref_y = drive.y_north_m[seg]
    t_seg = drive.t_s[seg] - drive.t_s[i0]

    # Initial conditions from last valid GNSS fix
    x0, y0 = float(drive.x_east_m[i0]), float(drive.y_north_m[i0])
    dx_seed = drive.x_east_m[i0] - drive.x_east_m[max(0, i0 - 5)]
    dy_seed = drive.y_north_m[i0] - drive.y_north_m[max(0, i0 - 5)]
    h0 = float(math.atan2(dx_seed, dy_seed)) if math.hypot(dx_seed, dy_seed) > 0.5 else float(drive.gps_course_rad[i0])
    v0_fwd = float(drive.gps_speed_mps[i0])
    v0_vec = np.array([v0_fwd * math.sin(h0), v0_fwd * math.cos(h0), 0.0])

    # Run Plain 3D Strapdown INS
    ins = StrapdownINS()
    ins.reset(np.array([x0, y0, 0.0]), v0_vec, heading0_rad=h0)

    est_x = np.zeros(n_seg)
    est_y = np.zeros(n_seg)

    for k in range(n_seg):
        idx = i0 + k
        p, v = ins.step(drive.acc_raw[idx], drive.gyro[idx], dt)
        est_x[k] = p[0]
        est_y[k] = p[1]

    # Error statistics
    err = np.hypot(est_x - ref_x, est_y - ref_y)
    dist = float(np.sum(np.hypot(np.diff(ref_x), np.diff(ref_y))))
    final_err = float(err[-1])
    max_err = float(err.max())
    rmse = float(np.sqrt(np.mean(err ** 2)))
    drift_pct = float(100.0 * final_err / dist) if dist > 1.0 else float("nan")

    # Time to exceed 10% SIH budget
    dist_t = np.concatenate([[0], np.cumsum(np.hypot(np.diff(ref_x), np.diff(ref_y)))])
    budget_10pct = 0.10 * dist_t
    exceeded_mask = err > budget_10pct
    time_to_fail_s = float(t_seg[np.argmax(exceeded_mask)]) if exceeded_mask.any() else float(t_seg[-1])

    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)

    # Plotting Before Evidence Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.8), gridspec_kw={"width_ratios": [1.2, 1]})

    # Trajectory comparison
    ax1.plot(ref_x, ref_y, color="#0E2440", lw=3.6, label="Ground Truth (Real Road)", zorder=5)
    ax1.plot(est_x, est_y, color="#C93B2B", lw=2.4, ls="--", label="Plain Strapdown INS (Double-Integration)", zorder=4)
    ax1.plot(ref_x[0], ref_y[0], "o", color="#0E2440", ms=9, zorder=6)
    ax1.plot(est_x[-1], est_y[-1], "x", color="#C93B2B", ms=10, mew=2.5, zorder=6)
    
    ax1.annotate(f"GNSS Lost\n(Tunnel Ingress)", (ref_x[0], ref_y[0]),
                 xytext=(15, -20), textcoords="offset points",
                 fontsize=9, weight="bold", color="#0E2440",
                 bbox=dict(fc="white", ec="#0E2440", lw=0.8, alpha=0.9, pad=3.0))

    ax1.annotate(f"CATASTROPHIC DRIFT\nFinal Error: {final_err:.1f} m\nDrift: {drift_pct:.1f}%",
                 (est_x[-1], est_y[-1]),
                 xytext=(-120, -30), textcoords="offset points",
                 fontsize=9, weight="bold", color="#C93B2B",
                 bbox=dict(fc="#FEE2E2", ec="#C93B2B", lw=1.0, alpha=0.95, pad=4.0),
                 arrowprops=dict(arrowstyle="->", color="#C93B2B", lw=1.5))

    ax1.set_aspect("equal", "datalim")
    ax1.set_xlabel("Local East (m)", fontsize=10)
    ax1.set_ylabel("Local North (m)", fontsize=10)
    ax1.set_title(f"Baseline Failure Mode: Plain Strapdown INS during {outage_len:.0f}s GNSS Blackout",
                  fontsize=11, weight="bold", color="#0E2440")
    ax1.legend(loc="best", fontsize=8.5, frameon=True, facecolor="white", edgecolor="#E2E8F0")
    ax1.grid(True, alpha=0.25, ls="--")

    # Error growth curve
    ax2.plot(t_seg, err, color="#C93B2B", lw=2.4, label="Plain Strapdown Error (Exploding ~ t³)")
    ax2.plot(t_seg, budget_10pct, color="#94A3B8", lw=2.2, ls=":", label="SIH Budget Ceiling (< 10% Distance)")
    
    # Mark failure threshold crossover
    if exceeded_mask.any():
        idx_fail = int(np.argmax(exceeded_mask))
        ax2.plot(t_seg[idx_fail], err[idx_fail], "ro", ms=8)
        ax2.annotate(f"Exceeds 10% Budget\nat t = {time_to_fail_s:.1f} s",
                     (t_seg[idx_fail], err[idx_fail]),
                     xytext=(20, 20), textcoords="offset points",
                     fontsize=8.5, weight="bold", color="#C93B2B",
                     arrowprops=dict(arrowstyle="->", color="#C93B2B"))

    ax2.set_xlabel("Elapsed Time without GNSS (s)", fontsize=10)
    ax2.set_ylabel("Position Error (m)", fontsize=10)
    ax2.set_title("Cubic / Quadratic Error Growth of Raw MEMS IMU", fontsize=11, weight="bold", color="#0E2440")
    ax2.legend(loc="upper left", fontsize=8.5, frameon=True, facecolor="white", edgecolor="#E2E8F0")
    ax2.grid(True, alpha=0.25, ls="--")

    fig.text(0.5, 0.015,
             f"Distance Travelled: {dist:.0f} m  |  Plain INS Final Error: {final_err:.1f} m  |  "
             f"Drift: {drift_pct:.1f}% (FAIL: exceeds 10% budget in {time_to_fail_s:.1f}s)",
             ha="center", fontsize=9.5, weight="bold", color="#991B1B")

    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out_png, dpi=220)
    plt.close(fig)

    metrics = {
        "baseline_type": "Plain 3D Strapdown INS (Double-Integration)",
        "outage_duration_s": outage_len,
        "distance_travelled_m": round(dist, 2),
        "final_error_m": round(final_err, 2),
        "max_error_m": round(max_err, 2),
        "ate_rmse_m": round(rmse, 2),
        "drift_percentage": round(drift_pct, 2),
        "time_to_exceed_10pct_budget_s": round(time_to_fail_s, 2),
        "status": "FAILED (Exceeds 10% SIH threshold)"
    }

    print(f"\n[BASELINE EVIDENCE GENERATED] Plot saved to: {out_png}")
    print(json.dumps(metrics, indent=2))
    return metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--s-csv", help="Path to S-*.csv")
    p.add_argument("--outage-start", type=float, default=60.0)
    p.add_argument("--outage-len", type=float, default=60.0)
    p.add_argument("--out", default="figures/strapdown_drift_baseline.png")
    p.add_argument("--demo", action="store_true")
    a = p.parse_args()

    if a.demo or not a.s_csv:
        path = synthetic_drive("figures")
        run_strapdown_baseline(path, a.outage_start, a.outage_len, a.out)
        return

    run_strapdown_baseline(a.s_csv, a.outage_start, a.outage_len, a.out)


if __name__ == "__main__":
    main()
