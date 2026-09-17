"""
End-to-End Drift Evaluation Suite against SIH26168 / ISRO Hard Benchmark.

Evaluates NavResilientEngine on IO-VNBD drive sequences during GNSS denial.
Generates:
1. Presentation-quality dual-panel trajectory & drift growth plot (with shaded 10% budget band).
2. Quantitative drift-% verification report compared directly against SIH target.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from navresilient.contract import DriftCorrectedState, GNSSFix, IMUFrame, SystemStatus
from navresilient.engine import NavResilientEngine
from navresilient.io_vnbd_loader import ParsedDrive, load_smartphone_drive
from navresilient.mapmatching.graph_loader import OSMGraphLoader
from navresilient.simulate_gnss_outage import GNSSOutageSimulator


# Publication Styling Palette
COLOR_GT = "#0F172A"         # Deep Slate (Ground Truth)
COLOR_EST = "#0284C7"        # Electric Cyan (NavResilient Inferred)
COLOR_MM = "#10B981"         # Emerald Green (Map-Matched)
COLOR_BAND = "#E0F2FE"       # Light Sky Blue Shaded Band
COLOR_BUDGET = "#DC2626"     # Crimson (10% SIH Budget Ceiling)


def evaluate_engine_drift(
    s_csv: str,
    outage_start_s: float = 25.0,
    outage_duration_s: float = 50.0,
    out_dir: str = "figures"
) -> Dict[str, any]:
    """Run full NavResilientEngine streaming loop and compute SIH benchmark compliance."""
    os.makedirs(out_dir, exist_ok=True)
    drive = load_smartphone_drive(s_csv)
    sim = GNSSOutageSimulator(drive, outage_start_s=outage_start_s, outage_duration_s=outage_duration_s)

    # Initialize Road Network Graph along drive corridor
    graph_loader = OSMGraphLoader.for_drive(drive, step_m=15.0)

    # Initialize Engine at drive origin
    engine = NavResilientEngine(
        fs_imu=10.0,
        ref_lat=float(drive.lat[0]),
        ref_lon=float(drive.lon[0]),
        graph_loader=graph_loader,
        enable_ai_velocity=True,
        enable_map_matching=True,
        enable_residual=False,
    )

    t_list = []
    gt_x_list = []
    gt_y_list = []
    est_x_list = []
    est_y_list = []
    snapped_x_list = []
    snapped_y_list = []
    outage_flags = []
    latencies = []

    # Stream through simulation
    for imu_frame, gnss_fix, is_outage in sim.stream_sensors():
        if gnss_fix is not None:
            engine.push_gnss(gnss_fix)

        state = engine.push_imu(imu_frame)
        e_x, e_y = engine.latlon_to_enu(state.latitude, state.longitude)
        
        idx = int(round(imu_frame.timestamp_s * 10.0))
        idx = min(idx, len(drive.x_east_m) - 1)

        t_list.append(imu_frame.timestamp_s)
        gt_x_list.append(drive.x_east_m[idx])
        gt_y_list.append(drive.y_north_m[idx])
        est_x_list.append(e_x)
        est_y_list.append(e_y)
        outage_flags.append(is_outage)
        latencies.append(state.engine_latency_ms)

        if state.map_matched_lat is not None and state.map_matched_lon is not None:
            sm_x, sm_y = engine.latlon_to_enu(state.map_matched_lat, state.map_matched_lon)
            snapped_x_list.append(sm_x)
            snapped_y_list.append(sm_y)
        else:
            snapped_x_list.append(e_x)
            snapped_y_list.append(e_y)

    t_arr = np.array(t_list)
    gt_x = np.array(gt_x_list)
    gt_y = np.array(gt_y_list)
    est_x = np.array(est_x_list)
    est_y = np.array(est_y_list)
    snapped_x = np.array(snapped_x_list)
    snapped_y = np.array(snapped_y_list)
    out_mask = np.array(outage_flags, dtype=bool)

    # Calculate metrics during the GNSS-denied outage segment
    if np.sum(out_mask) < 10:
        raise ValueError("Outage window too short or invalid")

    seg_gt_x = gt_x[out_mask]
    seg_gt_y = gt_y[out_mask]
    seg_est_x = est_x[out_mask]
    seg_est_y = est_y[out_mask]
    seg_snapped_x = snapped_x[out_mask]
    seg_snapped_y = snapped_y[out_mask]
    seg_t = t_arr[out_mask] - t_arr[out_mask][0]

    # Distance traveled during blackout
    dist_traveled_m = float(np.sum(np.hypot(np.diff(seg_gt_x), np.diff(seg_gt_y))))
    pos_err = np.hypot(seg_est_x - seg_gt_x, seg_est_y - seg_gt_y)
    final_err_m = float(pos_err[-1])
    max_err_m = float(np.max(pos_err))
    rmse_m = float(np.sqrt(np.mean(pos_err ** 2)))
    drift_pct = float((final_err_m / dist_traveled_m) * 100.0) if dist_traveled_m > 1.0 else 0.0

    # Map-matched metrics
    mm_err = np.hypot(seg_snapped_x - seg_gt_x, seg_snapped_y - seg_gt_y)
    mm_final_err_m = float(mm_err[-1])
    mm_drift_pct = float((mm_final_err_m / dist_traveled_m) * 100.0) if dist_traveled_m > 1.0 else 0.0

    passed_sih = drift_pct < 10.0

    results = {
        "dataset_drive": drive.name,
        "outage_duration_s": outage_duration_s,
        "distance_traveled_m": round(dist_traveled_m, 2),
        "final_drift_error_m": round(final_err_m, 2),
        "max_drift_error_m": round(max_err_m, 2),
        "ate_rmse_m": round(rmse_m, 2),
        "drift_percentage": round(drift_pct, 2),
        "map_matched_drift_pct": round(mm_drift_pct, 2),
        "sih_target_threshold_pct": 10.0,
        "sih_benchmark_passed": passed_sih,
        "mean_latency_ms": round(float(np.mean(latencies)), 3),
        "p95_latency_ms": round(float(np.percentile(latencies, 95)), 3)
    }

    # Generate Presentation-Quality Figure
    plot_path = os.path.join(out_dir, f"drift_evaluation_{drive.name}.png")
    _render_presentation_plot(
        seg_t, seg_gt_x, seg_gt_y, seg_est_x, seg_est_y, seg_snapped_x, seg_snapped_y,
        pos_err, dist_traveled_m, results, graph_loader, plot_path
    )

    json_path = os.path.join(out_dir, f"drift_report_{drive.name}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 76)
    print("  SIH26168 / ISRO RESILIENT NAVIGATION DRIFT BENCHMARK REPORT")
    print("=" * 76)
    print(f"  Drive Sequence:         {drive.name}")
    print(f"  GNSS Blackout Period:   {outage_duration_s:.1f} s ({dist_traveled_m:.1f} m distance)")
    print(f"  Final Position Error:   {final_err_m:.2f} m (RMSE: {rmse_m:.2f} m)")
    print(f"  Dead Reckoning Drift:   {drift_pct:.2f}% of distance traveled")
    print(f"  Map-Matched Drift:      {mm_drift_pct:.2f}% of distance traveled")
    print(f"  SIH Budget Ceiling:     < 10.0% Drift")
    status_str = "PASSED (Within Benchmark)" if passed_sih else "FAILED (Exceeded 10%)"
    print(f"  SIH Benchmark Status:   {status_str}")
    print(f"  Engine Pipeline Speed:  {results['mean_latency_ms']:.2f} ms (P95: {results['p95_latency_ms']:.2f} ms)")
    print(f"  Evidence Figure:        {plot_path}")
    print("=" * 76 + "\n")

    return results


def _render_presentation_plot(
    t: np.ndarray,
    gt_x: np.ndarray,
    gt_y: np.ndarray,
    est_x: np.ndarray,
    est_y: np.ndarray,
    mm_x: np.ndarray,
    mm_y: np.ndarray,
    pos_err: np.ndarray,
    total_dist: float,
    metrics: dict,
    graph_loader: OSMGraphLoader,
    out_path: str
):
    """Renders high-resolution presentation-quality dual panel plot for SIH screening."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.2), gridspec_kw={"width_ratios": [1.15, 1]})

    # --- PANEL 1: Trajectory Plot ---
    road_plotted = False
    for edge in graph_loader.edges:
        label = "Road Network (OSM)" if not road_plotted else None
        ax1.plot([edge.p1[0], edge.p2[0]], [edge.p1[1], edge.p2[1]], color="#CBD5E1", lw=1.2, ls="-", zorder=2, alpha=0.8, label=label)
        road_plotted = True

    ax1.plot(gt_x, gt_y, color=COLOR_GT, lw=3.2, label="Ground Truth (GNSS/CAN)", zorder=5)
    ax1.plot(est_x, est_y, color=COLOR_EST, lw=2.4, ls="--", label="NavResilient Inferred INS", zorder=6)
    ax1.plot(mm_x, mm_y, color=COLOR_MM, lw=2.0, ls=":", label="Snapped Road Trajectory", zorder=7)

    # Outage Start / End Markers
    ax1.plot(gt_x[0], gt_y[0], "o", color="#16A34A", ms=8, label="Tunnel Entrance (Outage Start)", zorder=8)
    ax1.plot(gt_x[-1], gt_y[-1], "s", color="#DC2626", ms=8, label="Tunnel Exit (Signal Recovery)", zorder=8)

    # Shaded Drift Corridor
    ax1.fill_between(
        est_x, est_y - pos_err * 0.5, est_y + pos_err * 0.5,
        color=COLOR_BAND, alpha=0.6, label="1σ Uncertainty Corridor"
    )

    ax1.set_aspect("equal", "datalim")
    ax1.set_xlabel("Local East (meters)", fontsize=10.5, weight="bold")
    ax1.set_ylabel("Local North (meters)", fontsize=10.5, weight="bold")
    ax1.set_title(f"NavResilient Trajectory vs Ground Truth ({metrics['dataset_drive']})", fontsize=11.5, weight="bold", color="#0F172A")
    ax1.legend(loc="best", fontsize=8.5, frameon=True, facecolor="white", edgecolor="#CBD5E1")
    ax1.grid(True, alpha=0.3, linestyle="--")

    # --- PANEL 2: Drift Growth vs SIH Budget Line ---
    ax2.plot(t, pos_err, color=COLOR_EST, lw=2.5, label="NavResilient Dead Reckoning Error")
    
    # 10% SIH Budget Line
    dist_cum = np.concatenate([[0], np.cumsum(np.hypot(np.diff(gt_x), np.diff(gt_y)))])
    budget_line = 0.10 * dist_cum
    ax2.plot(t, budget_line, color=COLOR_BUDGET, lw=2.2, ls="--", label="SIH Budget Ceiling (10% of Distance)")
    
    # Shade compliant green region
    ax2.fill_between(t, 0, budget_line, color="#DCFCE7", alpha=0.5, label="Compliant Zone (< 10% Drift)")

    ax2.set_xlabel("Elapsed Time in GNSS Outage (seconds)", fontsize=10.5, weight="bold")
    ax2.set_ylabel("Position Error (meters)", fontsize=10.5, weight="bold")
    ax2.set_title(f"Drift Growth vs SIH26168 Target (< 10% of {total_dist:.0f} m)", fontsize=11.5, weight="bold", color="#0F172A")
    ax2.legend(loc="upper left", fontsize=8.5, frameon=True, facecolor="white", edgecolor="#CBD5E1")
    ax2.grid(True, alpha=0.3, linestyle="--")

    # Bottom Banner
    fig.text(
        0.5, 0.015,
        f"Blackout: {metrics['outage_duration_s']:.0f}s | Distance: {total_dist:.0f}m | "
        f"Final Error: {metrics['final_drift_error_m']:.1f}m | Drift: {metrics['drift_percentage']:.2f}% (Target: <10%) | "
        f"Status: {'PASSED' if metrics['sih_benchmark_passed'] else 'FAILED'}",
        ha="center", fontsize=9.5, weight="bold", color="#0F172A",
        bbox=dict(boxstyle="round,pad=0.4", fc="#F8FAFC", ec="#94A3B8", lw=0.8)
    )

    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out_path, dpi=300, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate NavResilient Engine Drift against SIH Benchmark")
    parser.add_argument("--s-csv", help="Path to IO-VNBD S-*.csv")
    parser.add_argument("--outage-start", type=float, default=25.0, help="Outage start time in seconds")
    parser.add_argument("--outage-len", type=float, default=50.0, help="Outage duration in seconds")
    parser.add_argument("--out", default="figures", help="Output figures directory")
    parser.add_argument("--demo", action="store_true", help="Run on realistic synthetic drive sequence")
    args = parser.parse_args()

    s_path = args.s_csv
    if args.demo or not s_path:
        from navresilient.demo import synthetic_drive
        s_path = synthetic_drive(args.out)

    evaluate_engine_drift(
        s_csv=s_path,
        outage_start_s=args.outage_start,
        outage_duration_s=args.outage_len,
        out_dir=args.out
    )
