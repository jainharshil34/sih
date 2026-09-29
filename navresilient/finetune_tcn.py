"""
Fine-Tune NavResilient 1D-TCN Velocity Estimator across Highway, Urban, and Smartphone Logs.

Adapts pre-trained TCN model weights to handle:
1. Low-speed stop-and-go urban driving (0 - 15 m/s) and engine-idle vibration.
2. Real smartphone vehicle mount logs (data/my_ride_synced.csv).
3. Pedestrian & micro-mobility movement (data/my_walk_synced.csv).
4. Multi-drive highway benchmarks (IO-VNBD S-Vw12, S-Vta10, S-Vw10, S-Vta12).
"""

from __future__ import annotations

import glob
import math
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from navresilient.calibration import AutoCalibrator
from navresilient.denoise import HybridIMUDenoiser
from navresilient.io_vnbd_loader import load_smartphone_drive, load_vehicle_drive
from navresilient.models.tcn_velocity import TCNVelocityNet, extract_imu_windows
from navresilient.velocity_estimator import prepare_dataset


def collect_multi_regime_training_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Collect calibrated IMU windows and speed targets from all available drive logs."""
    all_X_train, all_Y_train = [], []
    all_X_val, all_Y_val = [], []

    print("Collecting and calibrating multi-regime training datasets...")

    # 1. IO-VNBD Drives
    iovnbd_files = sorted(glob.glob("data/IO-VNBD/S-*.csv"))
    for s_path in iovnbd_files:
        drive_name = os.path.basename(s_path).replace(".csv", "")
        v_path = s_path.replace("S-", "V-")
        try:
            drive = load_smartphone_drive(s_path)
            if os.path.exists(v_path):
                v_speed = load_vehicle_drive(v_path)
                v_speed = np.resize(v_speed, len(drive))
            else:
                v_speed = drive.gps_speed_mps

            calibrator = AutoCalibrator(fs=drive.fs)
            calibrator.calibrate(drive.acc_raw, drive.gyro, drive.gps_speed_mps, drive.gps_course_rad)
            acc_v = np.array([calibrator.transform_to_vehicle_frame(a, w)[0] for a, w in zip(drive.acc_raw, drive.gyro)])
            gyro_v = np.array([calibrator.transform_to_vehicle_frame(a, w)[1] for a, w in zip(drive.acc_raw, drive.gyro)])

            denoiser = HybridIMUDenoiser(fs=drive.fs)
            acc_filt = np.array([denoiser.filter_frame(a)[0] for a in acc_v])

            X, Y = prepare_dataset(acc_filt, gyro_v, v_speed, fs=drive.fs, win_s=2.0, hop_s=0.1)
            if len(X) > 20:
                if "Vw12" in drive_name:
                    all_X_val.append(X)
                    all_Y_val.append(Y)
                else:
                    all_X_train.append(X)
                    all_Y_train.append(Y)
                print(f"  + Added IO-VNBD [{drive_name}]: {len(X)} windows (speeds {v_speed.min():.1f} - {v_speed.max():.1f} m/s)")
        except Exception as e:
            print(f"  - Skipped IO-VNBD [{drive_name}]: {e}")

    # 2. Real Vehicle Smartphone Ride (data/my_ride_synced.csv)
    ride_csv = "data/my_ride_synced.csv"
    if os.path.exists(ride_csv):
        try:
            drive_ride = load_smartphone_drive(ride_csv, name="my_ride_synced")
            calibrator = AutoCalibrator(fs=drive_ride.fs)
            calibrator.calibrate(drive_ride.acc_raw, drive_ride.gyro, drive_ride.gps_speed_mps, drive_ride.gps_course_rad)
            acc_v = np.array([calibrator.transform_to_vehicle_frame(a, w)[0] for a, w in zip(drive_ride.acc_raw, drive_ride.gyro)])
            gyro_v = np.array([calibrator.transform_to_vehicle_frame(a, w)[1] for a, w in zip(drive_ride.acc_raw, drive_ride.gyro)])
            denoiser = HybridIMUDenoiser(fs=drive_ride.fs)
            acc_filt = np.array([denoiser.filter_frame(a)[0] for a in acc_v])

            # Use smoothed vehicle ground-truth speed profile
            # Compute speed from ENU displacement
            dx = np.gradient(drive_ride.x_east_m)
            dy = np.gradient(drive_ride.y_north_m)
            spd_profile = np.hypot(dx, dy) * drive_ride.fs
            spd_profile = np.clip(spd_profile, 0.0, 35.0)

            X_ride, Y_ride = prepare_dataset(acc_filt, gyro_v, spd_profile, fs=drive_ride.fs, win_s=2.0, hop_s=0.05)
            if len(X_ride) > 10:
                # Oversample new urban ride to adapt features effectively
                all_X_train.append(X_ride)
                all_Y_train.append(Y_ride)
                all_X_train.append(X_ride)
                all_Y_train.append(Y_ride)
                print(f"  + Added Real Phone Ride [my_ride_synced]: {len(X_ride)} windows (speeds {spd_profile.min():.1f} - {spd_profile.max():.1f} m/s)")
        except Exception as e:
            print(f"  - Skipped Real Phone Ride: {e}")

    # 3. Real Pedestrian Walk (data/my_walk_synced.csv)
    walk_csv = "data/my_walk_synced.csv"
    if os.path.exists(walk_csv):
        try:
            df_walk = pd.read_csv(walk_csv)
            t_w = df_walk["time"].values
            ax_w = df_walk["accelerometer x"].values
            ay_w = df_walk["accelerometer y"].values
            az_w = df_walk["accelerometer z"].values
            gx_w = df_walk["gyroscope x"].values
            gy_w = df_walk["gyroscope y"].values
            gz_w = df_walk["gyroscope z"].values
            acc_raw_w = np.column_stack([ax_w, ay_w, az_w])
            gyro_w = np.column_stack([gx_w, gy_w, gz_w])

            # Pedestrian walking speed (1.2 m/s avg)
            spd_w = np.full(len(df_walk), 1.25, dtype=np.float32)
            spd_w[:20] = 0.0   # initial rest
            spd_w[-20:] = 0.0  # end rest

            denoiser = HybridIMUDenoiser(fs=10.0)
            acc_filt_w = np.array([denoiser.filter_frame(a)[0] for a in acc_raw_w])
            X_walk, Y_walk = prepare_dataset(acc_filt_w, gyro_w, spd_w, fs=10.0, win_s=2.0, hop_s=0.2)
            if len(X_walk) > 10:
                all_X_train.append(X_walk)
                all_Y_train.append(Y_walk)
                print(f"  + Added Real Pedestrian Walk [my_walk_synced]: {len(X_walk)} windows (walking speed ~1.2 m/s)")
        except Exception as e:
            print(f"  - Skipped Real Pedestrian Walk: {e}")

    # 4. Stationary & Idle Vibration Augmentation
    # Ensures high engine/idle vibration at red lights is learned as 0.0 m/s
    if all_X_train:
        n_stat = 120
        win_len = 20  # 2.0s at 10Hz
        stat_X = []
        stat_Y = np.zeros(n_stat, dtype=np.float32)
        for _ in range(n_stat):
            noise_a = np.random.normal(0, 0.4, size=(3, win_len)).astype(np.float32)
            noise_g = np.random.normal(0, 0.05, size=(3, win_len)).astype(np.float32)
            w = np.vstack([noise_a, noise_g])
            stat_X.append(w)
        all_X_train.append(np.array(stat_X))
        all_Y_train.append(stat_Y)
        print(f"  + Added Stationary/Idle Vibration Augmentation: {n_stat} zero-velocity windows")

    X_train = np.concatenate(all_X_train, axis=0)
    Y_train = np.concatenate(all_Y_train, axis=0)
    X_val = np.concatenate(all_X_val, axis=0) if all_X_val else X_train[:100]
    Y_val = np.concatenate(all_Y_val, axis=0) if all_Y_val else Y_train[:100]

    return X_train, Y_train, X_val, Y_val


def finetune_tcn_model(
    model_path: str = "artifacts/models/tcn_velocity.pt",
    epochs: int = 35,
    lr: float = 1e-3
) -> Dict[str, float]:
    """Fine-tune the TCN model on multi-domain speed regimes."""
    X_train, Y_train, X_val, Y_val = collect_multi_regime_training_data()

    print(f"\nTraining Dataset Size: {len(X_train)} windows | Validation Size: {len(X_val)} windows")
    print(f"Speed Target Stats: min={Y_train.min():.2f} m/s, max={Y_train.max():.2f} m/s, mean={Y_train.mean():.2f} m/s")

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train).unsqueeze(1))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(Y_val).unsqueeze(1))

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    model = TCNVelocityNet(in_channels=6, hidden_channels=32)

    # Load existing pre-trained weights if available for warm start
    if os.path.exists(model_path):
        try:
            state_dict = torch.load(model_path, map_location="cpu")
            model.load_state_dict(state_dict)
            print(f"Loaded existing pre-trained weights from {model_path} for fine-tuning.")
        except Exception as e:
            print(f"Could not load pre-existing weights ({e}), training from initialization.")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    print("\n" + "=" * 75)
    print(f"  FINE-TUNING TCN VELOCITY ESTIMATOR ({sum(p.numel() for p in model.parameters()):,} weights)")
    print("=" * 75)

    best_mae = float("inf")
    best_weights = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            optimizer.zero_grad()
            pv, lv = model(bx)
            # Heteroscedastic negative log-likelihood loss
            loss = 0.5 * (torch.exp(-lv) * (by - pv) ** 2 + lv).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(bx)
        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        all_p, all_t = [], []
        with torch.no_grad():
            for bx, by in val_loader:
                pv, lv = model(bx)
                vloss = 0.5 * (torch.exp(-lv) * (by - pv) ** 2 + lv).mean()
                val_loss += vloss.item() * len(bx)
                all_p.extend(pv.cpu().numpy().ravel())
                all_t.extend(by.cpu().numpy().ravel())

        mae = float(np.mean(np.abs(np.array(all_p) - np.array(all_t))))
        rmse = float(np.sqrt(np.mean((np.array(all_p) - np.array(all_t)) ** 2)))

        if mae < best_mae:
            best_mae = mae
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == epochs or epoch == 1:
            print(f"  Epoch [{epoch:02d}/{epochs}] | Train Loss: {train_loss/len(X_train):.3f} | Val MAE: {mae:.3f} m/s ({mae*3.6:.1f} km/h) | Val RMSE: {rmse:.3f} m/s")

    # Save best fine-tuned weights
    if best_weights is not None:
        model.load_state_dict(best_weights)

    os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"\n[SUCCESS] Fine-tuned weights saved to: {model_path}")

    # Export TorchScript for high-speed edge deployment
    try:
        model.eval()
        example_input = torch.randn(1, 6, 20)
        ts_model = torch.jit.trace(model, example_input)
        ts_path = model_path.replace(".pt", "_torchscript.pt")
        ts_model.save(ts_path)
        print(f"[SUCCESS] Exported TorchScript runtime graph to: {ts_path}")
    except Exception as e:
        print(f"TorchScript export note: {e}")

    return {"best_val_mae_mps": best_mae, "best_val_mae_kmh": best_mae * 3.6}


if __name__ == "__main__":
    finetune_tcn_model()
