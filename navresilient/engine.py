"""
NavResilient Unified Edge-Deployable Navigation Engine.

Implements the public Integration Contract for Engineer B (App & UI Consumer).
Pipeline:
  1. Calibration (Mount tilt & yaw auto-alignment)
  2. Denoising & Regime Classification (Butterworth <2.5Hz + shock gating + ZUPT)
  3. AI Velocity Estimator (1D-TCN forward speed prediction)
  4. UKF + AI Sensor Fusion (FilterPy UKF + AI Residual Drift Compensator)
  5. Topological HMM Road Map-Matching (OSM graph snapping with off-road resilience)
  -> Emits DriftCorrectedState / PositionEstimate per step.
"""

from __future__ import annotations

import math
import time
from collections import deque
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from navresilient.ai_fusion_residual import AIResidualFusionEngine
from navresilient.calibration import AutoCalibrator
from navresilient.contract import DriftCorrectedState, GNSSFix, IMUFrame, SystemStatus
from navresilient.denoise import HybridIMUDenoiser, MotionRegime
from navresilient.ekf_ukf import NavResilientUKF
from navresilient.mapmatching.graph_loader import OSMGraphLoader
from navresilient.mapmatching.hmm_matcher import HMMMapMatcher
from navresilient.models.tcn_velocity import TCNVelocity
from navresilient.velocity_estimator import LightweightVelocityNet

# Alias for Integration Contract naming flexibility
PositionEstimate = DriftCorrectedState
NavState = DriftCorrectedState
GNSSStatus = SystemStatus


class NavResilientEngine:
    """Unified, edge-deployable navigation inference engine."""

    def __init__(
        self,
        fs_imu: float = 10.0,
        model_path: Optional[str] = "artifacts/models/tcn_velocity.pt",
        residual_model_path: Optional[str] = "artifacts/models/ai_residual_net.pt",
        graph_loader: Optional[OSMGraphLoader] = None,
        ref_lat: float = 12.9716,
        ref_lon: float = 77.5946,
        ref_alt: float = 920.0
    ):
        self.fs_imu = fs_imu
        self.dt_imu = 1.0 / fs_imu

        # Origin reference for local Cartesian ENU coordinates
        self.ref_lat = ref_lat
        self.ref_lon = ref_lon
        self.ref_alt = ref_alt
        self.R_earth = 6378137.0

        # Submodules
        self.calibrator = AutoCalibrator(fs=fs_imu)
        self.denoiser = HybridIMUDenoiser(fs=fs_imu, fc_kinematic=2.5, shock_thresh_g=2.5)
        self.ukf = NavResilientUKF(dt=self.dt_imu)
        self.residual_engine = AIResidualFusionEngine(model_path=residual_model_path)
        self.velocity_model = TCNVelocity(win_s=2.0, epochs=25)
        self.speed_history = deque(maxlen=200)
        self.acc_v_history = deque(maxlen=200)
        self.gyro_v_history = deque(maxlen=200)

        # OSM Road Graph & HMM Matcher
        self.graph_loader = graph_loader or OSMGraphLoader.create_synthetic_grid(
            ref_lat=ref_lat, ref_lon=ref_lon, grid_size_m=1000.0, block_spacing_m=100.0
        )
        self.matcher = HMMMapMatcher(self.graph_loader, sigma_z=6.0, beta=4.0, max_search_radius_m=35.0)

        # Buffers for sliding-window operations
        self.buf_size = int(2.0 * fs_imu)  # 2.0 second window at 10Hz
        self.acc_buf = deque(maxlen=self.buf_size)
        self.gyro_buf = deque(maxlen=self.buf_size)

        # Internal state
        self.status = SystemStatus.INITIALIZING
        self.last_gnss_time = -1.0
        self.last_imu_time = 0.0
        self.total_distance_m = 0.0
        self.outage_start_distance_m = 0.0
        self.outage_start_time_s = 0.0
        self.last_pos_2d = np.zeros(2)
        self.current_estimate: Optional[DriftCorrectedState] = None

    def set_reference_origin(self, lat: float, lon: float, alt: float = 0.0):
        """Set local tangent plane origin."""
        self.ref_lat = lat
        self.ref_lon = lon
        self.ref_alt = alt

    def latlon_to_enu(self, lat: float, lon: float) -> Tuple[float, float]:
        d_lat = math.radians(lat - self.ref_lat)
        d_lon = math.radians(lon - self.ref_lon)
        lat_rad = math.radians(self.ref_lat)
        east = d_lon * math.cos(lat_rad) * self.R_earth
        north = d_lat * self.R_earth
        return east, north

    def enu_to_latlon(self, east: float, north: float) -> Tuple[float, float]:
        lat_rad = math.radians(self.ref_lat)
        d_lat = north / self.R_earth
        d_lon = east / (self.R_earth * math.cos(lat_rad))
        lat = self.ref_lat + math.degrees(d_lat)
        lon = self.ref_lon + math.degrees(d_lon)
        return lat, lon

    def push_gnss(
        self,
        fix: Optional[GNSSFix] = None,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        alt: Optional[float] = None,
        speed_mps: Optional[float] = None,
        heading_deg: Optional[float] = None,
        accuracy_m: Optional[float] = None,
        timestamp: Optional[float] = None
    ):
        """Feed incoming GNSS fix (supports both typed GNSSFix object and raw kwargs)."""
        if fix is not None:
            t = fix.timestamp_s
            lat_v, lon_v, alt_v = fix.latitude, fix.longitude, fix.altitude_m
            spd, crs, acc_m = fix.speed_mps, fix.heading_deg, fix.accuracy_m
        else:
            t = timestamp or 0.0
            lat_v = lat if lat is not None else self.ref_lat
            lon_v = lon if lon is not None else self.ref_lon
            alt_v = alt if alt is not None else self.ref_alt
            spd = speed_mps if speed_mps is not None else 0.0
            crs = heading_deg if heading_deg is not None else 0.0
            acc_m = accuracy_m if accuracy_m is not None else 2.5

        east, north = self.latlon_to_enu(lat_v, lon_v)
        crs_rad = math.radians(crs)

        if self.status == SystemStatus.INITIALIZING:
            self.ukf.initialize(x0=east, y0=north, v0=spd, heading0=crs_rad)
            self.residual_engine.initialize(x0=east, y0=north, v0=spd, heading0=crs_rad)
            self.last_pos_2d = np.array([east, north])
            self.status = SystemStatus.GNSS_LOCKED
        elif self.status == SystemStatus.GNSS_DENIED_INS:
            self.status = SystemStatus.REACQUIRED
            self.ukf.update_gnss(east, north, sigma_pos=acc_m, course_rad=crs_rad)
        else:
            self.status = SystemStatus.GNSS_LOCKED
            self.ukf.update_gnss(east, north, sigma_pos=acc_m, course_rad=crs_rad)

        if spd > 1.5:
            self.ukf.ukf.x[3] = crs_rad

        self.last_gnss_time = t
        self.current_gnss_speed = spd

    def push_imu(
        self,
        frame: Optional[IMUFrame] = None,
        ax: Optional[float] = None,
        ay: Optional[float] = None,
        az: Optional[float] = None,
        gx: Optional[float] = None,
        gy: Optional[float] = None,
        gz: Optional[float] = None,
        timestamp: Optional[float] = None
    ) -> DriftCorrectedState:
        """Process high-rate IMU frame through full navigation pipeline."""
        t_start = time.perf_counter()

        if frame is not None:
            t = frame.timestamp_s
            acc_raw = np.array([frame.ax_mps2, frame.ay_mps2, frame.az_mps2], dtype=float)
            gyro_raw = np.array([frame.gx_rads, frame.gy_rads, frame.gz_rads], dtype=float)
        else:
            t = timestamp or (self.last_imu_time + self.dt_imu)
            acc_raw = np.array([ax or 0.0, ay or 0.0, az if az is not None else 9.81], dtype=float)
            gyro_raw = np.array([gx or 0.0, gy or 0.0, gz or 0.0], dtype=float)

        dt = t - self.last_imu_time if self.last_imu_time > 0 else self.dt_imu
        dt = float(np.clip(dt, 0.001, 0.5))
        self.last_imu_time = t

        # 1. Pipeline Stage 1: Auto-Calibration (Mount tilt/yaw compensation)
        if not self.calibrator.tilt_calibrated and len(self.acc_buf) >= 5:
            self.calibrator.estimate_tilt_from_gravity(np.array(self.acc_buf))

        acc_veh, gyro_veh = self.calibrator.transform_to_vehicle_frame(acc_raw, gyro_raw)

        # 2. Pipeline Stage 2: Denoising & Motion Regime Classification
        acc_filt, is_pothole = self.denoiser.filter_frame(acc_veh)
        self.acc_buf.append(acc_veh)
        self.gyro_buf.append(gyro_veh)
        self.acc_v_history.append(acc_raw)
        self.gyro_v_history.append(gyro_raw)
        if self.status != SystemStatus.GNSS_DENIED_INS:
            self.speed_history.append(getattr(self, "current_gnss_speed", 0.0))

        is_stationary = False
        if len(self.acc_buf) == self.buf_size:
            regime = self.denoiser.classify_motion_regime(
                np.array(self.acc_buf), np.array(self.gyro_buf)
            )
            is_stationary = (regime == MotionRegime.STATIONARY_IDLE)

        # 3. Check for GNSS Outage (> 1.2 seconds without fix)
        if self.last_gnss_time > 0 and (t - self.last_gnss_time) > 1.2:
            if self.status != SystemStatus.GNSS_DENIED_INS:
                self.status = SystemStatus.GNSS_DENIED_INS
                self.outage_start_distance_m = self.total_distance_m
                self.outage_start_time_s = t
                self.ukf.mark_gnss_lost()

                # Calibrate full mounting rotation (Pitch/Roll/Yaw) from raw IMU history
                if len(self.speed_history) >= 20:
                    arr_raw_a = np.array(self.acc_v_history)[:len(self.speed_history)]
                    arr_raw_w = np.array(self.gyro_v_history)[:len(self.speed_history)]
                    arr_spd = np.array(self.speed_history)
                    self.calibrator.calibrate(arr_raw_a, arr_raw_w, arr_spd)

                    # Transform history to calibrated vehicle frame
                    arr_a_veh = np.array([self.calibrator.transform_to_vehicle_frame(a, w)[0] for a, w in zip(arr_raw_a, arr_raw_w)])
                    arr_w_veh = np.array([self.calibrator.transform_to_vehicle_frame(a, w)[1] for a, w in zip(arr_raw_a, arr_raw_w)])
                    
                    # Estimate stationary gyro bias
                    from navresilient.filters.vibration import StopDetector
                    import navresilient.dr as dr
                    stop_det = StopDetector(fs=self.fs_imu)
                    zupt_mask, _ = stop_det.detect(arr_a_veh, arr_w_veh)
                    gbias = dr.estimate_gyro_bias(arr_w_veh[:, 2], zupt_mask)
                    self.ukf.ukf.x[5] = -gbias

                    # Fit online TCN velocity model on recent GNSS-locked drive history
                    if not self.velocity_model.is_fitted:
                        self.velocity_model.fit([(arr_a_veh, arr_w_veh)], [arr_spd], self.fs_imu)

        # 4. Pipeline Stage 3: AI Velocity Estimation
        v_ai = None
        var_ai = 0.20
        if len(self.acc_buf) == self.buf_size and self.velocity_model.is_fitted:
            arr_a = np.array(self.acc_buf)
            arr_w = np.array(self.gyro_buf)
            pred_v, pred_var = self.velocity_model.predict(arr_a, arr_w, self.fs_imu, n_out=len(arr_a))
            v_ai = max(0.0, float(pred_v[-1]))
            var_ai = float(pred_var[-1])

        # 5. Pipeline Stage 4: UKF + AI Sensor Fusion
        acc_fwd = 0.0 if (self.status == SystemStatus.GNSS_DENIED_INS and v_ai is not None) else float(acc_filt[0])
        # Android gyro Z is CCW (+); geographic heading rate is CW (+)
        gyro_z = -float(gyro_veh[2])

        if self.status == SystemStatus.GNSS_DENIED_INS and v_ai is not None:
            self.ukf.ukf.x[2] = v_ai

        self.ukf.predict(acc_fwd=acc_fwd, gyro_z=gyro_z, dt=dt)

        if is_stationary:
            self.ukf.update_zupt(sigma_vel=0.02)
            # Calibrate and clamp static gyro bias during standstill
            self.ukf.ukf.x[5] = -float(gyro_veh[2])
        elif v_ai is not None and self.status == SystemStatus.GNSS_DENIED_INS:
            self.ukf.update_ai_velocity(v_ai=v_ai, var_ai=var_ai)

        east, north = self.ukf.position
        speed = self.ukf.forward_speed
        heading = self.ukf.heading_rad

        # 6. Pipeline Stage 5: Map Matching
        snapped_lat, snapped_lon = None, None
        match_res = self.matcher.match(np.array([east, north]), heading_rad=heading, speed_mps=speed)
        if match_res.is_snapped and not match_res.is_off_road:
            s_lat, s_lon = self.enu_to_latlon(match_res.snapped_pos[0], match_res.snapped_pos[1])
            snapped_lat, snapped_lon = s_lat, s_lon
            # Gently guide filter towards mapped road
            east = 0.8 * east + 0.2 * match_res.snapped_pos[0]
            north = 0.8 * north + 0.2 * match_res.snapped_pos[1]

        # Update cumulative distance
        curr_2d = np.array([east, north])
        step_d = float(np.linalg.norm(curr_2d - self.last_pos_2d))
        self.total_distance_m += step_d
        self.last_pos_2d = curr_2d.copy()

        # Telemetry calculations
        lat, lon = self.enu_to_latlon(east, north)
        heading_deg = float(math.degrees(heading)) % 360.0
        pos_sigma = self.ukf.position_uncertainty

        outage_d = max(1.0, self.total_distance_m - self.outage_start_distance_m)
        drift_pct = (pos_sigma / outage_d) * 100.0 if self.status == SystemStatus.GNSS_DENIED_INS else 0.0

        latency_ms = (time.perf_counter() - t_start) * 1000.0

        estimate = DriftCorrectedState(
            timestamp_s=t,
            latitude=lat,
            longitude=lon,
            altitude_m=self.ref_alt,
            speed_mps=speed,
            speed_kmh=speed * 3.6,
            heading_deg=heading_deg,
            status=self.status,
            drift_pct=round(drift_pct, 2),
            pos_uncertainty_1sigma_m=round(pos_sigma, 2),
            is_stationary=is_stationary,
            pothole_detected=is_pothole,
            map_matched_lat=snapped_lat,
            map_matched_lon=snapped_lon,
            engine_latency_ms=round(latency_ms, 3)
        )
        self.current_estimate = estimate
        return estimate

    def get_state(self) -> Optional[DriftCorrectedState]:
        """Return latest computed state."""
        return self.current_estimate
