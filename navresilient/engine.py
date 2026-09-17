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
from navresilient.heading_fusion import HeadingFusionEngine
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
        ref_alt: float = 920.0,
        enable_ai_velocity: bool = True,
        enable_map_matching: bool = True,
        enable_residual: bool = True,
        enable_zupt: bool = True,
        enable_calibration: bool = True,
    ):
        self.fs_imu = fs_imu
        self.dt_imu = 1.0 / fs_imu
        self.enable_ai_velocity = enable_ai_velocity
        self.enable_map_matching = enable_map_matching
        self.enable_residual = enable_residual
        self.enable_zupt = enable_zupt
        self.enable_calibration = enable_calibration

        # Origin reference for local Cartesian ENU coordinates
        self.ref_lat = ref_lat
        self.ref_lon = ref_lon
        self.ref_alt = ref_alt
        self.R_earth = 6378137.0

        # Submodules
        self.calibrator = AutoCalibrator(fs=fs_imu)
        self.denoiser = HybridIMUDenoiser(fs=fs_imu, fc_kinematic=2.5, shock_thresh_g=2.5)
        self.ukf = NavResilientUKF(dt=self.dt_imu)
        self.heading_filter = HeadingFusionEngine(fs=fs_imu, alpha=0.98)
        self.residual_engine = AIResidualFusionEngine(model_path=residual_model_path)
        self.velocity_model = TCNVelocity(win_s=2.0, epochs=25, model_path=model_path)
        self.speed_history = deque(maxlen=200)
        self.course_history = deque(maxlen=200)
        self.acc_v_history = deque(maxlen=200)
        self.gyro_v_history = deque(maxlen=200)

        # OSM Road Graph & HMM Matcher
        self._custom_graph_provided = graph_loader is not None
        self.graph_loader = graph_loader or OSMGraphLoader(ref_lat=ref_lat, ref_lon=ref_lon)
        self.matcher = HMMMapMatcher(self.graph_loader, sigma_z=6.0, beta=4.0, max_search_radius_m=35.0)
        self.gnss_track_enu: List[np.ndarray] = []
        self.last_graph_node_pos: Optional[np.ndarray] = None
        self.graph_edge_count: int = 0

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
        if self.graph_loader is not None:
            self.graph_loader.ref_lat = lat
            self.graph_loader.ref_lon = lon

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
            self.set_reference_origin(lat_v, lon_v, alt_v)
            east, north = 0.0, 0.0
            self.ukf.initialize(x0=east, y0=north, v0=spd, heading0=crs_rad)
            self.heading_filter.initialize(initial_heading_rad=crs_rad)
            self.residual_engine.initialize(x0=east, y0=north, v0=spd, heading0=crs_rad)
            self.last_pos_2d = np.array([east, north])
            self.status = SystemStatus.GNSS_LOCKED
        elif self.status == SystemStatus.GNSS_DENIED_INS:
            self.status = SystemStatus.REACQUIRED
            self.ukf.update_gnss(east, north, sigma_pos=acc_m, course_rad=crs_rad, speed_mps=spd)
            self.heading_filter.update_gnss(course_rad=crs_rad, speed_mps=spd)
        else:
            self.status = SystemStatus.GNSS_LOCKED
            self.ukf.update_gnss(east, north, sigma_pos=acc_m, course_rad=crs_rad, speed_mps=spd)
            self.heading_filter.update_gnss(course_rad=crs_rad, speed_mps=spd)

        # Dynamic Road Graph Builder: accumulate GNSS waypoints into road graph if no custom graph supplied
        if not self._custom_graph_provided:
            curr_p = np.array([east, north], dtype=float)
            if self.last_graph_node_pos is None:
                self.last_graph_node_pos = curr_p
                self.gnss_track_enu.append(curr_p)
            else:
                d_prev = float(np.linalg.norm(curr_p - self.last_graph_node_pos))
                if d_prev >= 15.0:
                    p_prev = self.last_graph_node_pos
                    u = f"track_{self.graph_edge_count}"
                    v = f"track_{self.graph_edge_count + 1}"
                    self.graph_loader.add_edge_cartesian(
                        u=u, v=v, p1=p_prev, p2=curr_p, one_way=False, name=f"GNSS_Track_{self.graph_edge_count}"
                    )
                    self.graph_edge_count += 1
                    self.last_graph_node_pos = curr_p
                    self.gnss_track_enu.append(curr_p)

        self.last_gnss_time = t
        self.current_gnss_speed = spd
        self.current_gnss_course = crs_rad

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
            mag_raw = np.array([frame.mx_ut or 0.0, frame.my_ut or 0.0, frame.mz_ut or 0.0], dtype=float) if (frame.mx_ut is not None) else None
        else:
            t = timestamp or (self.last_imu_time + self.dt_imu)
            acc_raw = np.array([ax or 0.0, ay or 0.0, az if az is not None else 9.81], dtype=float)
            gyro_raw = np.array([gx or 0.0, gy or 0.0, gz or 0.0], dtype=float)
            mag_raw = None

        dt = t - self.last_imu_time if self.last_imu_time > 0 else self.dt_imu
        dt = float(np.clip(dt, 0.001, 0.5))
        self.last_imu_time = t

        # 1. Pipeline Stage 1: Auto-Calibration (Mount tilt/yaw compensation)
        if self.enable_calibration:
            if not self.calibrator.tilt_calibrated and len(self.acc_buf) >= 5:
                self.calibrator.estimate_tilt_from_gravity(np.array(self.acc_buf))
            acc_veh, gyro_veh = self.calibrator.transform_to_vehicle_frame(acc_raw, gyro_raw)
        else:
            acc_veh, gyro_veh = acc_raw, gyro_raw

        # 2. Pipeline Stage 2: Denoising & Motion Regime Classification
        acc_filt, is_pothole = self.denoiser.filter_frame(acc_veh)
        self.acc_buf.append(acc_veh)
        self.gyro_buf.append(gyro_veh)
        self.acc_v_history.append(acc_raw)
        self.gyro_v_history.append(gyro_raw)
        if self.status != SystemStatus.GNSS_DENIED_INS:
            self.speed_history.append(getattr(self, "current_gnss_speed", 0.0))
            self.course_history.append(getattr(self, "current_gnss_course", 0.0))
            if mag_raw is not None:
                self.heading_filter.calibrate_mag_declination(
                    acc_raw, mag_raw,
                    getattr(self, "current_gnss_course", 0.0),
                    getattr(self, "current_gnss_speed", 0.0)
                )

            # Continuous Pre-Outage Calibration (every 1.0s while GNSS locked)
            if self.enable_calibration and len(self.speed_history) >= 15 and len(self.acc_v_history) % 10 == 0:
                arr_raw_a = np.array(self.acc_v_history)[:len(self.speed_history)]
                arr_raw_w = np.array(self.gyro_v_history)[:len(self.speed_history)]
                arr_spd = np.array(self.speed_history)
                arr_crs = np.array(self.course_history)[:len(self.speed_history)]
                self.calibrator.calibrate(arr_raw_a, arr_raw_w, arr_spd, arr_crs)
                self.heading_filter.gyro_sign = self.calibrator.gyro_sign
                self.heading_filter.gyro_bias = self.calibrator.gyro_bias

        is_stationary = False
        regime = MotionRegime.SMOOTH_CRUISE
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
                self.heading_filter.mark_gnss_lost()

                # Extrapolate road corridor forward along vehicle heading if no custom offline map was provided
                if not self._custom_graph_provided and self.last_graph_node_pos is not None:
                    curr_entry_pos = self.ukf.position
                    if float(np.linalg.norm(curr_entry_pos - self.last_graph_node_pos)) >= 5.0:
                        u = f"track_{self.graph_edge_count}"
                        v = f"track_{self.graph_edge_count + 1}"
                        self.graph_loader.add_edge_cartesian(
                            u=u, v=v, p1=self.last_graph_node_pos, p2=curr_entry_pos, one_way=False, name=f"GNSS_Track_{self.graph_edge_count}"
                        )
                        self.graph_edge_count += 1
                        self.last_graph_node_pos = curr_entry_pos

                    hdg = self.ukf.heading_rad
                    dir_vec = np.array([math.sin(hdg), math.cos(hdg)])
                    p_curr = self.last_graph_node_pos
                    for step_idx in range(1, 101):  # 100 steps of 15m = 1.5 km corridor
                        p_next = p_curr + dir_vec * 15.0
                        u = f"corridor_{step_idx - 1}"
                        v = f"corridor_{step_idx}"
                        self.graph_loader.add_edge_cartesian(
                            u=u, v=v, p1=p_curr, p2=p_next, one_way=False, name=f"Corridor_{step_idx}"
                        )
                        p_curr = p_next

                # Calibrate full mounting rotation (Pitch/Roll/Yaw/Gyro) from raw IMU history
                if self.enable_calibration and len(self.speed_history) >= 15:
                    arr_raw_a = np.array(self.acc_v_history)[:len(self.speed_history)]
                    arr_raw_w = np.array(self.gyro_v_history)[:len(self.speed_history)]
                    arr_spd = np.array(self.speed_history)
                    arr_crs = np.array(self.course_history)[:len(self.speed_history)]
                    self.calibrator.calibrate(arr_raw_a, arr_raw_w, arr_spd, arr_crs)
                    self.heading_filter.gyro_sign = self.calibrator.gyro_sign
                    self.heading_filter.gyro_bias = self.calibrator.gyro_bias

                    # Transform history to calibrated vehicle frame
                    arr_a_veh = np.array([self.calibrator.transform_to_vehicle_frame(a, w)[0] for a, w in zip(arr_raw_a, arr_raw_w)])
                    arr_w_veh = np.array([self.calibrator.transform_to_vehicle_frame(a, w)[1] for a, w in zip(arr_raw_a, arr_raw_w)])
                    
                    # Estimate stationary gyro bias
                    from navresilient.filters.vibration import StopDetector
                    import navresilient.dr as dr
                    stop_det = StopDetector(fs=self.fs_imu)
                    zupt_mask, _ = stop_det.detect(arr_a_veh, arr_w_veh)
                    gbias = dr.estimate_gyro_bias(arr_w_veh[:, 2], zupt_mask)
                    if abs(gbias) > 1e-4:
                        self.heading_filter.gyro_bias = gbias
                        self.ukf.ukf.x[5] = 0.0

                    # Fit online velocity model on recent GNSS-locked drive history
                    if self.enable_ai_velocity and not self.velocity_model.is_fitted:
                        self.velocity_model.fit([(arr_a_veh, arr_w_veh)], [arr_spd], self.fs_imu)

        # 4. Pipeline Stage 3: Smarter AI Velocity Estimation & Gating (Priority 4)
        v_ai = None
        var_ai = 0.20
        if self.enable_ai_velocity and len(self.acc_buf) == self.buf_size and self.velocity_model.is_fitted:
            arr_a = np.array(self.acc_buf)
            arr_w = np.array(self.gyro_buf)
            pred_v, pred_var = self.velocity_model.predict(arr_a, arr_w, self.fs_imu, n_out=len(arr_a))
            raw_v_ai = max(0.0, float(pred_v[-1]))
            raw_var_ai = float(pred_var[-1])

            # Per-Regime Gating & Adaptive Variance Tuning
            if is_stationary or regime == MotionRegime.STATIONARY_IDLE:
                v_ai = 0.0
                var_ai = 0.01
            elif regime == MotionRegime.HARD_BRAKE_OR_ACCEL:
                v_ai = raw_v_ai
                var_ai = raw_var_ai * 3.5  # Soften AI influence during violent braking/accel
            elif regime == MotionRegime.POTHOLE_SHOCK:
                v_ai = raw_v_ai
                var_ai = raw_var_ai * 5.0  # Heavily de-weight AI during road shocks
            elif regime == MotionRegime.DYNAMIC_CORNERING:
                v_ai = raw_v_ai
                var_ai = raw_var_ai * 2.0  # Moderate uncertainty in high lateral G
            else:  # SMOOTH_CRUISE
                v_ai = raw_v_ai
                var_ai = raw_var_ai

            # Speed Sanity & Mahalanobis Residual Gating vs current UKF State
            if not is_stationary and v_ai is not None:
                v_curr_ukf = self.ukf.forward_speed
                # Pedestrian & Slow Mobile Handler: If moving at walking speed but vehicle AI predicts highway speed
                if v_curr_ukf < 3.0 and v_ai > 3.5:
                    step_energy = float(np.std(np.linalg.norm(arr_a, axis=1)))
                    v_ai = float(np.clip(1.15 + step_energy * 0.12, 0.8, 1.6))
                    var_ai = 0.05
                else:
                    v_diff = abs(v_ai - v_curr_ukf)
                    if v_diff > 3.5:
                        # Scale variance quadratically with discrepancy to avoid pulling filter erratically
                        var_ai = var_ai * (1.0 + (v_diff - 3.5) ** 2)
                        if v_diff > 7.0:
                            # Unphysical instantaneous jump -> reject AI velocity update
                            v_ai = None

        # 5. Pipeline Stage 4: Dead-Reckoning & Kinematic Propagation
        is_outage = (self.status == SystemStatus.GNSS_DENIED_INS)
        
        # Step complementary heading filter (Gyro + Tilt-Compensated Mag + GNSS Course Anchor)
        hdg_state = self.heading_filter.step(
            gyro_z_raw=float(gyro_raw[2]),
            acc_raw=acc_raw,
            mag_raw=mag_raw,
            dt=dt,
            is_stationary=is_stationary,
            forward_speed_mps=self.ukf.forward_speed
        )
        gyro_z = hdg_state.yaw_rate_rads
        acc_fwd = 0.0 if is_outage else float(acc_filt[0])

        # Propagate UKF kinematic state (bypassing acc_fwd double integration during outage)
        self.ukf.predict(acc_fwd=acc_fwd, gyro_z=gyro_z, dt=dt)

        if is_stationary and self.enable_zupt:
            self.ukf.update_zupt(sigma_vel=0.02)
            self.ukf.ukf.x[5] = 0.0
        elif v_ai is not None and is_outage and self.enable_ai_velocity:
            self.ukf.update_ai_velocity(v_ai=v_ai, var_ai=var_ai)
        elif is_outage and v_ai is None:
            # Fall back to gentle speed roll-down decay model when AI is uncertain/rejected
            decay_factor = math.exp(-dt / 120.0)
            self.ukf.ukf.x[2] = max(0.0, self.ukf.ukf.x[2] * decay_factor)

        # Apply multi-cue heading fusion during outage
        if is_outage:
            self.ukf.ukf.x[3] = hdg_state.heading_rad

        east, north = self.ukf.position
        speed = self.ukf.forward_speed
        heading = self.ukf.heading_rad

        # 6. Pipeline Stage 5: AI Residual Drift Compensation (Tightened OOD Gating)
        step_dx = float(east - self.last_pos_2d[0]) if self.total_distance_m > 0 else 0.0
        step_dy = float(north - self.last_pos_2d[1]) if self.total_distance_m > 0 else 0.0
        self.residual_engine.update_history(
            step_dx=step_dx,
            step_dy=step_dy,
            v_ukf=speed,
            heading_rad=heading,
            acc_fwd=acc_fwd,
            acc_lat=float(acc_veh[1]),
            gyro_z=gyro_z
        )

        if self.status == SystemStatus.GNSS_DENIED_INS and self.enable_residual and self.residual_engine.is_ready:
            corr_dx, corr_dy, corr_dv, conf = self.residual_engine.predict_step_correction()
            if conf >= 0.30:
                east += corr_dx
                north += corr_dy
                speed = max(0.0, speed + corr_dv)
                self.ukf.ukf.x[0] = east
                self.ukf.ukf.x[1] = north
                self.ukf.ukf.x[2] = speed

        # 7. Pipeline Stage 6: Map Matching
        snapped_lat, snapped_lon = None, None
        if self.enable_map_matching:
            match_res = self.matcher.match(np.array([east, north]), heading_rad=heading, speed_mps=speed)
            if match_res.is_snapped and not match_res.is_off_road:
                s_lat, s_lon = self.enu_to_latlon(match_res.snapped_pos[0], match_res.snapped_pos[1])
                snapped_lat, snapped_lon = s_lat, s_lon
                # Confidence-scaled guidance towards road centerline (stronger during outage)
                if match_res.confidence >= 0.15:
                    if is_outage:
                        blend = float(np.clip(0.70 * match_res.confidence, 0.35, 0.85))
                    else:
                        blend = 0.22 * match_res.confidence
                    east = (1.0 - blend) * east + blend * match_res.snapped_pos[0]
                    north = (1.0 - blend) * north + blend * match_res.snapped_pos[1]
                    self.ukf.ukf.x[0] = east
                    self.ukf.ukf.x[1] = north

                    # Heading constraint from matched road bearing during outage
                    if is_outage and match_res.confidence >= 0.25:
                        d_hdg = (match_res.road_bearing_rad - heading + math.pi) % (2.0 * math.pi) - math.pi
                        if abs(d_hdg) < math.radians(45.0):
                            hdg_blend = 0.18 * match_res.confidence
                            new_hdg = heading + hdg_blend * d_hdg
                            self.ukf.ukf.x[3] = new_hdg
                            self.heading_filter.heading_rad = new_hdg
                            heading = new_hdg

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
