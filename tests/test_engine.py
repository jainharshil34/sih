"""
Unit and Integration Tests for NavResilient Engine & Filters.
"""

import math
import numpy as np
import pytest

from navresilient.engine import NavResilientEngine, GNSSStatus
from navresilient.filters.vibration import VibrationFilter, StopDetector
from navresilient.models.tcn_velocity import TCNVelocityNet, extract_imu_windows
from navresilient.navigation.ukf import VehicleUKF
from navresilient.mapmatching.matcher import RoadNetwork, HMMMapMatcher


class TestVibrationFilter:
    def test_kinematic_separation(self):
        fs = 50.0
        t = np.arange(250) / fs
        # 0.5 Hz vehicle kinematic motion + 20 Hz high frequency engine noise
        clean_motion = 2.0 * np.sin(2 * np.pi * 0.5 * t)
        high_freq_noise = 1.5 * np.sin(2 * np.pi * 20.0 * t)
        sig = clean_motion + high_freq_noise

        vf = VibrationFilter(fs=fs, fc_kinematic=2.5)
        filtered = vf.filter_kinematics(sig)

        # High frequency content should be strongly suppressed
        residual_in_filtered = np.std(filtered[25:-25] - clean_motion[25:-25])
        assert residual_in_filtered < 0.45

    def test_pothole_shock_detection(self):
        vf = VibrationFilter(fs=10.0, shock_thresh_g=1.5)
        normal_acc = np.tile([0.1, 0.0, 9.81], (50, 1))
        # Inject severe pothole spike
        normal_acc[25] = [0.1, 0.0, 9.81 + 30.0]

        shocks = vf.detect_potholes_and_shocks(normal_acc)
        assert bool(shocks[25]) is True
        assert bool(shocks[10]) is False


class TestStopDetector:
    def test_zupt_during_stationary_rest(self):
        fs = 10.0
        n = 40
        acc_v = np.random.normal(0, 0.02, (n, 3))
        acc_v[:, 2] += 9.81
        gyro_v = np.random.normal(0, 0.005, (n, 3))

        detector = StopDetector(fs=fs)
        zupt_mask, metrics = detector.detect(acc_v, gyro_v)
        assert np.mean(zupt_mask[15:]) > 0.8


class TestModelArchitecture:
    def test_tcn_forward_pass_and_export_compliance(self):
        import torch
        net = TCNVelocityNet(in_channels=6, hidden_channels=16, num_layers=2)
        net.eval()

        batch_x = torch.randn(2, 6, 20)
        vel, log_var = net(batch_x)

        assert vel.shape == (2, 1)
        assert log_var.shape == (2, 1)
        assert (vel >= 0).all()  # Velocity must be non-negative


class TestFilterPyUKF:
    def test_ukf_prediction_and_updates(self):
        ukf = VehicleUKF(dt=0.1)
        ukf.initialize(x0=0.0, y0=0.0, v0=15.0, heading0=0.0)

        # Step 1: Predict straight forward
        ukf.predict(raw_acc_fwd=0.0, raw_gyro_z=0.0, dt=0.1)
        pos = ukf.position
        assert pos[1] > 0.0  # North displacement

        # Step 2: Fuse AI velocity update
        ukf.update_ai_velocity(v_ai=15.5, var_ai=0.2)
        assert abs(ukf.velocity - 15.5) < 1.0

        # Step 3: Fuse GNSS position
        ukf.update_gnss(pos_x=0.1, pos_y=1.5, sigma_pos=1.0)
        assert abs(ukf.position[1] - 1.5) < 0.5


class TestCalibration:
    def test_dynamic_mount_calibration(self):
        from navresilient.calibration import DynamicMountCalibrator
        calibrator = DynamicMountCalibrator(fs=10.0)

        # Phone mounted with 15 deg pitch and -5 deg roll
        pitch_true = math.radians(15.0)
        roll_true = math.radians(-5.0)
        g_vec = np.array([
            -math.sin(pitch_true) * 9.81,
            math.sin(roll_true) * math.cos(pitch_true) * 9.81,
            math.cos(roll_true) * math.cos(pitch_true) * 9.81
        ])
        acc_raw = np.tile(g_vec, (50, 1)) + np.random.normal(0, 0.02, (50, 3))
        gps_speed = np.linspace(0, 20, 50)
        gyro_raw = np.zeros((50, 3))

        mount = calibrator.calibrate(acc_raw, gyro_raw, gps_speed)
        assert mount.is_calibrated is True
        assert abs(mount.pitch_deg - 15.0) < 1.0
        assert abs(mount.roll_deg - (-5.0)) < 1.0


class TestDenoiser:
    def test_hybrid_denoiser_and_classifier(self):
        from navresilient.denoise import IMUDenoiser, MotionRegime
        denoiser = IMUDenoiser(fs=10.0)

        # Stationary test
        acc_idle = np.tile([0.0, 0.0, 9.81], (20, 1)) + np.random.normal(0, 0.01, (20, 3))
        gyro_idle = np.random.normal(0, 0.005, (20, 3))
        regime = denoiser.classify_motion_regime(acc_idle, gyro_idle)
        assert regime == MotionRegime.STATIONARY_IDLE

        # Pothole test
        acc_pothole = acc_idle.copy()
        acc_pothole[10, 0] += 30.0  # 30 m/s^2 shock spike
        regime_pothole = denoiser.classify_motion_regime(acc_pothole, gyro_idle)
        assert regime_pothole == MotionRegime.POTHOLE_SHOCK


class TestEngineStreamingIntegration:
    def test_full_engine_outage_resilience(self):
        engine = NavResilientEngine(fs_imu=10.0)

        # Initial GNSS fix at Bangalore
        engine.push_gnss(lat=12.9716, lon=77.5946, alt=920.0,
                         speed_mps=16.0, heading_deg=0.0, accuracy_m=2.0, timestamp=0.0)

        # Drive north with 10Hz IMU for 2 seconds
        t = 0.0
        state = None
        for i in range(20):
            t += 0.1
            state = engine.push_imu(ax=1.0, ay=0.0, az=9.81, gx=0.0, gy=0.0, gz=0.0, timestamp=t)

        assert state is not None
        assert state.status == GNSSStatus.GNSS_LOCKED
        e_x, e_y = engine.latlon_to_enu(state.latitude, state.longitude)
        assert e_y > 0.0 or state.latitude >= 12.9716


class TestEKFUKFAndAIResidual:
    def test_ukf_and_ekf_fusion(self):
        from navresilient.ekf_ukf import NavResilientUKF, NavResilientEKF
        ukf = NavResilientUKF(dt=0.1)
        ukf.initialize(x0=0.0, y0=0.0, v0=10.0, heading0=0.0)
        
        # Predict with forward accel and slight yaw rate
        for _ in range(10):
            ukf.predict(acc_fwd=0.5, gyro_z=0.05)
            
        pos = ukf.position
        assert pos[1] > 5.0  # Moved North
        assert ukf.forward_speed > 10.0

        # Test smooth handoff and GNSS update
        updated = ukf.update_gnss(pos_east=1.0, pos_north=10.0, sigma_pos=2.0)
        assert updated is True

        ekf = NavResilientEKF(dt=0.1)
        ekf.initialize(x0=0.0, y0=0.0, v0=10.0, heading0=0.0)
        ekf.predict(acc_fwd=0.5, gyro_z=0.05)
        assert ekf.position[1] > 0.0

    def test_ai_residual_net_shape(self):
        import torch
        from navresilient.ai_fusion_residual import AIResidualDriftNet
        net = AIResidualDriftNet(in_channels=8, window_size=10)
        dummy_in = torch.randn(2, 8, 10)
        out = net(dummy_in)
        assert out.shape == (2, 3)


class TestHMMMapMatching:
    def test_road_graph_loading_and_snapping(self):
        import numpy as np
        from navresilient.mapmatching import OSMGraphLoader, HMMMapMatcher
        loader = OSMGraphLoader.create_synthetic_grid(grid_size_m=300.0, block_spacing_m=100.0)
        matcher = HMMMapMatcher(loader, sigma_z=5.0, max_search_radius_m=30.0)

        # Vehicle driving along Street_0 (y=0, x from 10 to 50) with slight 3m cross-track noise
        raw_p = np.array([25.0, 3.0])
        res = matcher.match(raw_p, heading_rad=np.pi / 2.0)  # heading East

        assert res.is_snapped is True
        assert res.is_off_road is False
        assert abs(res.snapped_pos[1]) < abs(raw_p[1])  # Snapped closer to road centerline (y=0)
        assert res.confidence > 0.5

    def test_underground_parking_structure_edge_case(self):
        """Verify graceful degradation when vehicle is in an unmapped underground parking basement."""
        import numpy as np
        from navresilient.mapmatching import OSMGraphLoader, HMMMapMatcher
        loader = OSMGraphLoader.create_synthetic_grid(grid_size_m=200.0, block_spacing_m=100.0)
        matcher = HMMMapMatcher(loader, max_search_radius_m=25.0, off_road_dist_threshold_m=30.0)

        # Query point 200m away in an unmapped underground structure
        underground_p = np.array([500.0, 500.0])
        res = matcher.match(underground_p, heading_rad=0.0)

        # Must not crash, must not snap to a distant road, must return raw inertial coordinates
        assert res.is_snapped is False
        assert res.is_off_road is True
        assert np.allclose(res.snapped_pos, underground_p)
        assert res.edge_id == -1
        assert "Parking" in res.road_name or "Off-Road" in res.road_name


