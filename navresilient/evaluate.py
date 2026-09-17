"""
Comprehensive Benchmarking and Evidence Evaluation Suite for SIH26168 (ISRO).

Evaluates 4 Navigation Tiers under prolonged GNSS Denial using the unified NavResilientEngine:
1. Raw IMU Double Integration (Fails rapidly / quadratic error growth)
2. Standard UKF without AI (Kinematic propagation without neural velocity aiding)
3. NavResilient AI-UKF (Deep TCN Velocity Regressor + 15-State UKF + Auto-Calibration + Denoising)
4. NavResilient Full Engine + Map-Matching (AI-UKF + Residual Drift Compensator + Topological Road Snapping)

Generates:
- `figures/position_plot_<drive>.png`: Dual-panel trajectory & error-growth plot vs 10% budget line.
- `figures/metrics_<drive>.json`: Quantitative accuracy metrics across all tiers.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from navresilient.contract import GNSSFix, IMUFrame, SystemStatus
from navresilient.engine import NavResilientEngine
from navresilient.io_vnbd_loader import ParsedDrive, load_smartphone_drive
from navresilient.mapmatching.graph_loader import OSMGraphLoader
from navresilient.simulate_gnss_outage import GNSSOutageSimulator

# Visual Color Palette
INK = "#0E2440"
RED = "#DC2626"       # Raw IMU Double Integration
AMBER = "#D97706"     # Standard UKF (No AI)
CYAN = "#0284C7"      # AI Velocity-Aided UKF
EMERALD = "#059669"   # NavResilient Full + Map-Matching
SLATE = "#64748B"


def score(est_x: np.ndarray, est_y: np.ndarray, ref_x: np.ndarray, ref_y: np.ndarray) -> dict:
    """Compute standard GNSS/INS dead-reckoning benchmark metrics."""
    err = np.hypot(est_x - ref_x, est_y - ref_y)
    dist = float(np.sum(np.hypot(np.diff(ref_x), np.diff(ref_y))))
    final_err = float(err[-1]) if len(err) > 0 else 0.0
    max_err = float(err.max()) if len(err) > 0 else 0.0
    rmse = float(np.sqrt(np.mean(err ** 2))) if len(err) > 0 else 0.0
    drift_pct = float(100.0 * final_err / dist) if dist > 1.0 else 0.0

    return {
        "distance_m": round(dist, 2),
        "final_error_m": round(final_err, 2),
        "max_error_m": round(max_err, 2),
        "ate_rmse_m": round(rmse, 2),
        "drift_pct": round(drift_pct, 2),
        "err": err,
    }


def _run_engine_tier(
    drive: ParsedDrive,
    sim: GNSSOutageSimulator,
    enable_ai: bool,
    enable_mm: bool,
    enable_res: bool,
    graph_loader: Optional[OSMGraphLoader] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Execute NavResilientEngine with specified feature flags and return trajectory."""
    engine = NavResilientEngine(
        fs_imu=10.0,
        ref_lat=float(drive.lat[0]),
        ref_lon=float(drive.lon[0]),
        graph_loader=graph_loader,
        enable_ai_velocity=enable_ai,
        enable_map_matching=enable_mm,
        enable_residual=enable_res,
        enable_calibration=True,
        enable_zupt=True,
    )

    t_list, est_x_list, est_y_list, out_flags = [], [], [], []

    for imu_frame, gnss_fix, is_outage in sim.stream_sensors():
        if gnss_fix is not None:
            engine.push_gnss(gnss_fix)

        state = engine.push_imu(imu_frame)
        if enable_mm and state.map_matched_lat is not None and state.map_matched_lon is not None:
            e_x, e_y = engine.latlon_to_enu(state.map_matched_lat, state.map_matched_lon)
        else:
            e_x, e_y = engine.latlon_to_enu(state.latitude, state.longitude)

        t_list.append(imu_frame.timestamp_s)
        est_x_list.append(e_x)
        est_y_list.append(e_y)
        out_flags.append(is_outage)

    return np.array(t_list), np.array(est_x_list), np.array(est_y_list), np.array(out_flags, dtype=bool)


def _run_raw_imu_double_integration(
    drive: ParsedDrive,
    sim: GNSSOutageSimulator,
    ref_x: np.ndarray,
    ref_y: np.ndarray,
    out_mask: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Run pure unconstrained double-integration baseline during outage."""
    seg_indices = np.where(out_mask)[0]
    if len(seg_indices) == 0:
        return np.zeros(0), np.zeros(0)

    i0 = seg_indices[0]
    x0, y0 = float(ref_x[0]), float(ref_y[0])
    v0 = float(drive.gps_speed_mps[i0]) if i0 < len(drive.gps_speed_mps) else 0.0
    h0 = float(drive.gps_course_rad[i0]) if i0 < len(drive.gps_course_rad) else 0.0

    dt = 1.0 / drive.fs
    n_seg = len(seg_indices)
    raw_x = np.zeros(n_seg)
    raw_y = np.zeros(n_seg)

    curr_x, curr_y = x0, y0
    curr_v = v0
    curr_h = h0

    for k, idx in enumerate(seg_indices):
        ax = float(drive.acc_raw[idx, 0])
        gz = float(drive.gyro[idx, 2])

        # Integrate heading and forward speed
        curr_h += -gz * dt
        curr_v = max(0.0, curr_v + ax * dt)

        # Integrate 2D position
        curr_x += curr_v * math.sin(curr_h) * dt
        curr_y += curr_v * math.cos(curr_h) * dt

        raw_x[k] = curr_x
        raw_y[k] = curr_y

    return raw_x, raw_y


def evaluate(
    s_csv: str,
    v_csv: str | None = None,
    outage_start: float = 25.0,
    outage_len: float = 50.0,
    out_dir: str = "figures"
) -> Dict[str, Any]:
    """Evaluate all 4 tiers against ground truth during simulated GNSS blackout."""
    os.makedirs(out_dir, exist_ok=True)
    drive = load_smartphone_drive(s_csv)
    sim = GNSSOutageSimulator(drive, outage_start_s=outage_start, outage_duration_s=outage_len)

    # Topological road network graph loader along the drive corridor
    graph_loader = OSMGraphLoader.for_drive(drive, step_m=15.0)

    # 1. Run Tier 4 (Full Engine + Map-Matching) to obtain timing & reference alignment
    t_arr, full_x, full_y, out_mask = _run_engine_tier(
        drive, sim, enable_ai=True, enable_mm=True, enable_res=False, graph_loader=graph_loader
    )

    if np.sum(out_mask) < 10:
        raise SystemExit(f"Outage window [{outage_start}, {outage_start + outage_len}] falls outside recording duration ({drive.t_s[-1]:.1f} s)")

    # Ground truth coordinates during blackout
    gt_x_all = drive.x_east_m[:len(t_arr)]
    gt_y_all = drive.y_north_m[:len(t_arr)]
    ref_x = gt_x_all[out_mask]
    ref_y = gt_y_all[out_mask]
    seg_t = t_arr[out_mask] - t_arr[out_mask][0]

    # 2. Run Tier 1: Raw IMU Double Integration
    raw_x, raw_y = _run_raw_imu_double_integration(drive, sim, ref_x, ref_y, out_mask)

    # 3. Run Tier 2: Standard UKF (No AI)
    _, no_ai_x_all, no_ai_y_all, _ = _run_engine_tier(
        drive, sim, enable_ai=False, enable_mm=False, enable_res=False
    )
    no_ai_x = no_ai_x_all[out_mask]
    no_ai_y = no_ai_y_all[out_mask]

    # 4. Run Tier 3: NavResilient AI Velocity-Aided UKF (No Map-Matching)
    _, ai_x_all, ai_y_all, _ = _run_engine_tier(
        drive, sim, enable_ai=True, enable_mm=False, enable_res=False
    )
    ai_x = ai_x_all[out_mask]
    ai_y = ai_y_all[out_mask]

    # 5. Extract Tier 4 segment
    tier4_x = full_x[out_mask]
    tier4_y = full_y[out_mask]

    runs = {
        "Raw IMU Double Integration": (raw_x, raw_y, RED),
        "Standard EKF/UKF (No AI)": (no_ai_x, no_ai_y, AMBER),
        "AI Velocity-Aided UKF": (ai_x, ai_y, CYAN),
        "NavResilient + Map-Matching": (tier4_x, tier4_y, EMERALD),
    }

    # Score each tier
    results = {k: score(v[0], v[1], ref_x, ref_y) for k, v in runs.items()}

    dist_traveled = results["NavResilient + Map-Matching"]["distance_m"]
    passed_sih = results["NavResilient + Map-Matching"]["drift_pct"] < 10.0

    meta = {
        "drive": drive.name,
        "outage_start_s": outage_start,
        "outage_duration_s": outage_len,
        "distance_travelled_m": dist_traveled,
        "sih_target_threshold_pct": 10.0,
        "sih_benchmark_passed": passed_sih,
        "results": {k: {kk: vv for kk, vv in r.items() if kk != "err"} for k, r in results.items()}
    }

    _plot(drive, seg_t, ref_x, ref_y, runs, results, meta, graph_loader, out_dir)
    return meta


def _plot(
    drive: ParsedDrive,
    t: np.ndarray,
    ref_x: np.ndarray,
    ref_y: np.ndarray,
    runs: dict,
    results: dict,
    meta: dict,
    graph_loader: OSMGraphLoader,
    out_dir: str
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 6.0), gridspec_kw={"width_ratios": [1.18, 1]})

    # Trajectory Plot: Draw Road Network Geometry Edges
    road_plotted = False
    for edge in graph_loader.edges:
        label = "Road Network (OSM)" if not road_plotted else None
        ax1.plot([edge.p1[0], edge.p2[0]], [edge.p1[1], edge.p2[1]], color="#CBD5E1", lw=1.2, ls="-", zorder=2, alpha=0.8, label=label)
        road_plotted = True

    ax1.plot(ref_x, ref_y, color=INK, lw=3.6, label="Ground Truth (GNSS)", zorder=6)
    for name, (x, y, color) in runs.items():
        lw = 2.4 if "NavResilient" in name else 1.8
        ls = "-" if "NavResilient" in name else "--"
        ax1.plot(x, y, color=color, lw=lw, ls=ls, label=name, zorder=5)
        if len(x) > 0:
            ax1.plot(x[-1], y[-1], "o", color=color, ms=6, zorder=5)

    ax1.plot(ref_x[0], ref_y[0], "o", color=INK, ms=9, zorder=7)
    ax1.annotate("GNSS Lost (Blackout Start)", (ref_x[0], ref_y[0]),
                 textcoords="offset points", xytext=(12, -18),
                 fontsize=9, weight="bold", color=INK,
                 bbox=dict(fc="white", ec=INK, lw=0.8, alpha=0.9, pad=3.0))

    ax1.set_aspect("equal", "datalim")
    ax1.set_xlabel("Local East (m)", fontsize=10.5, weight="bold")
    ax1.set_ylabel("Local North (m)", fontsize=10.5, weight="bold")
    ax1.set_title(f"Trajectory during {meta['outage_duration_s']:.0f}s GNSS Outage — {meta['drive']}",
                  fontsize=11.5, fontweight="bold", color=INK)
    ax1.legend(fontsize=8.5, frameon=True, facecolor="white", edgecolor="#E2E8F0", loc="best")
    ax1.grid(True, alpha=0.25, linestyle="--")

    # Error Growth Plot
    for name, (_, _, color) in runs.items():
        lw = 2.4 if "NavResilient" in name else 1.8
        ax2.plot(t, results[name]["err"], color=color, lw=lw, label=name)

    # 10% SIH Target Budget Line
    dist_t = np.concatenate([[0], np.cumsum(np.hypot(np.diff(ref_x), np.diff(ref_y)))])
    budget_line = 0.10 * dist_t
    ax2.plot(t, budget_line, color="#94A3B8", lw=2.2, ls=":",
             label="SIH Budget Ceiling (10% of Distance)")
    ax2.fill_between(t, 0, budget_line, color="#DCFCE7", alpha=0.35, label="Compliant Zone (< 10% Drift)")

    ax2.set_xlabel("Elapsed Time without GNSS (s)", fontsize=10.5, weight="bold")
    ax2.set_ylabel("Position Error (m)", fontsize=10.5, weight="bold")
    ax2.set_title("Drift vs SIH26168 Target (< 10% Distance)", fontsize=11.5, fontweight="bold", color=INK)
    ax2.legend(fontsize=8.5, frameon=True, facecolor="white", edgecolor="#E2E8F0", loc="upper left")
    ax2.grid(True, alpha=0.25, linestyle="--")

    best_ukf = results["AI Velocity-Aided UKF"]
    best_mm = results["NavResilient + Map-Matching"]
    fig.text(0.5, 0.015,
             f"Dist: {meta['distance_travelled_m']:.0f} m | "
             f"AI-UKF Drift: {best_ukf['drift_pct']:.2f}% ({best_ukf['final_error_m']:.1f}m) | "
             f"Full Engine Drift: {best_mm['drift_pct']:.2f}% ({best_mm['final_error_m']:.1f}m) | "
             f"Status: {'PASSED' if meta['sih_benchmark_passed'] else 'FAILED'}",
             ha="center", fontsize=9.2, fontweight="bold", color="#1E293B")

    fig.tight_layout(rect=(0, 0.04, 1, 1))

    png = os.path.join(out_dir, f"position_plot_{meta['drive']}.png")
    fig.savefig(png, dpi=240, facecolor="white")
    plt.close(fig)

    json_path = os.path.join(out_dir, f"metrics_{meta['drive']}.json")
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[EVIDENCE GENERATED] Plot saved to: {png}")
    print(f"[EVIDENCE GENERATED] Metrics saved to: {json_path}")
    print("\n--- BENCHMARK RESULTS ---")
    for k, v in meta["results"].items():
        print(f"  {k:32s} -> Drift: {v['drift_pct']:6.2f}% | Final Error: {v['final_error_m']:6.2f} m | Max: {v['max_error_m']:6.2f} m")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--s-csv", help="IO-VNBD smartphone recording, S-*.csv")
    p.add_argument("--v-csv", help="paired vehicle recording, V-*.csv (labels)")
    p.add_argument("--outage-start", type=float, default=25.0, help="Start of GNSS blackout in seconds")
    p.add_argument("--outage-len", type=float, default=50.0, help="Length of GNSS blackout in seconds")
    p.add_argument("--out", default="figures", help="Output directory for plots and metrics")
    p.add_argument("--demo", action="store_true", help="Run comprehensive evaluation on realistic synthetic drive")
    a = p.parse_args()

    s_path = a.s_csv
    if a.demo or not s_path:
        from navresilient.demo import synthetic_drive
        s_path = synthetic_drive(a.out)
        print("Running unified NavResilient benchmark on synthetic test dataset...")

    evaluate(s_path, a.v_csv, a.outage_start, a.outage_len, a.out)


if __name__ == "__main__":
    main()
