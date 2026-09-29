"""
Refine TCN Velocity Neural Network Model directly on Multi-Domain Datasets.

Strictly optimizes PyTorch TCNVelocityNet weights directly on real phone sensor logs
and IO-VNBD benchmarks, ensuring the raw neural model itself regresses forward velocity
accurately without any post-hoc heuristic overrides.
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
from navresilient.models.tcn_velocity import TCNVelocityNet
from navresilient.velocity_estimator import prepare_dataset


def build_training_tensors() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract dense 6-DoF IMU windows paired with true velocity targets."""
    all_X_train, all_Y_train = [], []
    all_X_val, all_Y_val = [], []

    print("[1/3] Building dense training tensors across all drives...")

    # 1. Real Smartphone Vehicle Ride (data/my_ride_synced.csv)
    ride_csv = "data/my_ride_synced.csv"
    if os.path.exists(ride_csv):
        drive_ride = load_smartphone_drive(ride_csv, name="my_ride_synced")
        cal = AutoCalibrator(fs=drive_ride.fs)
        cal.calibrate(drive_ride.acc_raw, drive_ride.gyro, drive_ride.gps_speed_mps, drive_ride.gps_course_rad)
        acc_v = np.array([cal.transform_to_vehicle_frame(a, w)[0] for a, w in zip(drive_ride.acc_raw, drive_ride.gyro)])
        gyro_v = np.array([cal.transform_to_vehicle_frame(a, w)[1] for a, w in zip(drive_ride.acc_raw, drive_ride.gyro)])
        den = HybridIMUDenoiser(fs=drive_ride.fs)
        acc_filt = np.array([den.filter_frame(a)[0] for a in acc_v])

        dx = np.gradient(drive_ride.x_east_m)
        dy = np.gradient(drive_ride.y_north_m)
        spd_ride = np.clip(np.hypot(dx, dy) * drive_ride.fs, 0.0, 30.0)

        # Dense windowing (50ms hop)
        X_r, Y_r = prepare_dataset(acc_filt, gyro_v, spd_ride, fs=drive_ride.fs, win_s=2.0, hop_s=0.05)
        if len(X_r) > 0:
            for _ in range(5):  # Re-weight to ensure vehicle ride is thoroughly learned
                all_X_train.append(X_r)
                all_Y_train.append(Y_r)
            print(f"  + Real Phone Ride: {len(X_r)} dense windows (speeds {spd_ride.min():.1f} - {spd_ride.max():.1f} m/s)")

    # 2. IO-VNBD Drives
    for s_path in sorted(glob.glob("data/IO-VNBD/S-*.csv")):
        drive_id = os.path.basename(s_path).replace(".csv", "")
        v_path = s_path.replace("S-", "V-")
        try:
            drive = load_smartphone_drive(s_path)
            v_speed = load_vehicle_drive(v_path) if os.path.exists(v_path) else drive.gps_speed_mps
            v_speed = np.resize(v_speed, len(drive))

            cal = AutoCalibrator(fs=drive.fs)
            cal.calibrate(drive.acc_raw, drive.gyro, drive.gps_speed_mps, drive.gps_course_rad)
            acc_v = np.array([cal.transform_to_vehicle_frame(a, w)[0] for a, w in zip(drive.acc_raw, drive.gyro)])
            gyro_v = np.array([cal.transform_to_vehicle_frame(a, w)[1] for a, w in zip(drive.acc_raw, drive.gyro)])
            den = HybridIMUDenoiser(fs=drive.fs)
            acc_filt = np.array([den.filter_frame(a)[0] for a in acc_v])

            X, Y = prepare_dataset(acc_filt, gyro_v, v_speed, fs=drive.fs, win_s=2.0, hop_s=0.1)
            if len(X) > 20:
                if "Vw12" in drive_id:
                    all_X_val.append(X)
                    all_Y_val.append(Y)
                else:
                    all_X_train.append(X)
                    all_Y_train.append(Y)
                print(f"  + IO-VNBD [{drive_id}]: {len(X)} windows (speeds {v_speed.min():.1f} - {v_speed.max():.1f} m/s)")
        except Exception as e:
            print(f"  - Skipped {drive_id}: {e}")

    # 3. Real Smartphone Pedestrian Walk
    walk_csv = "data/my_walk_synced.csv"
    if os.path.exists(walk_csv):
        df_walk = pd.read_csv(walk_csv)
        acc_w = df_walk[["accelerometer x", "accelerometer y", "accelerometer z"]].values
        gyro_w = df_walk[["gyroscope x", "gyroscope y", "gyroscope z"]].values
        spd_w = np.full(len(df_walk), 1.25, dtype=np.float32)
        spd_w[:20] = 0.0
        spd_w[-20:] = 0.0
        den = HybridIMUDenoiser(fs=10.0)
        acc_filt_w = np.array([den.filter_frame(a)[0] for a in acc_w])
        X_w, Y_w = prepare_dataset(acc_filt_w, gyro_w, spd_w, fs=10.0, win_s=2.0, hop_s=0.1)
        if len(X_w) > 0:
            all_X_train.append(X_w)
            all_Y_train.append(Y_w)
            print(f"  + Real Pedestrian Walk: {len(X_w)} windows")

    # 4. Stationary & Idle Noise Invariants (0.0 m/s)
    n_stat = 200
    win_len = 20
    stat_X = []
    stat_Y = np.zeros(n_stat, dtype=np.float32)
    for _ in range(n_stat):
        noise_a = np.random.normal(0, np.random.uniform(0.1, 0.6), size=(3, win_len)).astype(np.float32)
        noise_g = np.random.normal(0, np.random.uniform(0.01, 0.08), size=(3, win_len)).astype(np.float32)
        stat_X.append(np.vstack([noise_a, noise_g]))
    all_X_train.append(np.array(stat_X))
    all_Y_train.append(stat_Y)

    X_train = np.concatenate(all_X_train, axis=0)
    Y_train = np.concatenate(all_Y_train, axis=0)
    X_val = np.concatenate(all_X_val, axis=0) if all_X_val else X_train[:200]
    Y_val = np.concatenate(all_Y_val, axis=0) if all_Y_val else Y_train[:200]

    return X_train, Y_train, X_val, Y_val


def refine_model(
    model_path: str = "artifacts/models/tcn_velocity.pt",
    epochs: int = 50,
    lr: float = 1.2e-3
):
    """Directly optimize TCNVelocityNet weights with no post-hoc overrides."""
    X_train, Y_train, X_val, Y_val = build_training_tensors()
    print(f"\n[2/3] Training Dataset: {len(X_train)} windows | Validation: {len(X_val)} windows")

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train).unsqueeze(1))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(Y_val).unsqueeze(1))

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    model = TCNVelocityNet(in_channels=6, hidden_channels=32)

    if os.path.exists(model_path):
        try:
            state_dict = torch.load(model_path, map_location="cpu")
            model.load_state_dict(state_dict)
            print(f"Loaded existing weights from {model_path} for refinement.")
        except Exception as e:
            print(f"Initializing new weights: {e}")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    print("\n" + "=" * 76)
    print(f"  OPTIMIZING PURE TCN NEURAL NETWORK WEIGHTS ({sum(p.numel() for p in model.parameters()):,} parameters)")
    print("=" * 76)

    best_loss = float("inf")
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

        epoch_vloss = val_loss / len(X_val)
        mae = float(np.mean(np.abs(np.array(all_p) - np.array(all_t))))
        rmse = float(np.sqrt(np.mean((np.array(all_p) - np.array(all_t)) ** 2)))

        if epoch_vloss < best_loss:
            best_loss = epoch_vloss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == epochs or epoch == 1:
            print(f"  Epoch [{epoch:02d}/{epochs}] | Train Loss: {train_loss/len(X_train):.3f} | Val Loss: {epoch_vloss:.3f} | Val MAE: {mae:.2f} m/s ({mae*3.6:.1f} km/h)")

    if best_weights is not None:
        model.load_state_dict(best_weights)

    os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"\n[3/3] Successfully saved refined model to: {model_path}")

    # Export TorchScript graph
    try:
        model.eval()
        example_input = torch.randn(1, 6, 20)
        ts_model = torch.jit.trace(model, example_input)
        ts_path = model_path.replace(".pt", "_torchscript.pt")
        ts_model.save(ts_path)
        print(f"Exported TorchScript runtime graph to: {ts_path}")
    except Exception as e:
        print(f"TorchScript export note: {e}")


if __name__ == "__main__":
    refine_model()
