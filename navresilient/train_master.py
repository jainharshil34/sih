"""
Master Multi-Drive AI Model Training & Evaluation Engine for SIH26168 (ISRO).

Trains lightweight on-device neural models on real multi-drive IO-VNBD datasets:
1. TCN Forward Velocity Regressor with heteroscedastic uncertainty (trained against ECU wheel speed).
2. AI Residual Drift Compensator (1D-CNN) trained on multi-drive outage segments.
3. Exports mobile runtime graphs (TorchScript, PyTorch 2.x export, LiteRT metadata).
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import navresilient.dr as dr
from navresilient.ai_fusion_residual import AIResidualDriftNet
from navresilient.io_vnbd_loader import load_smartphone_drive, load_vehicle_drive
from navresilient.models.export import export_model
from navresilient.models.tcn_velocity import TCNVelocityNet
from navresilient.velocity_estimator import LightweightVelocityNet, prepare_dataset


def discover_drive_pairs(dataset_root: str, max_drives: int = 12) -> List[Tuple[str, str, str]]:
    """Find valid paired S-*.csv and V-*.csv files from IO-VNBD dataset."""
    s_dir = os.path.join(dataset_root, "S-Dataset")
    v_dir = os.path.join(dataset_root, "V-Dataset")

    # If flat directory
    if not os.path.exists(s_dir):
        s_dir = dataset_root
        v_dir = dataset_root

    s_files = sorted(glob.glob(os.path.join(s_dir, "S-*.csv")))
    pairs = []

    # Priority drives with diverse dynamics (turns, highway, stop-and-go)
    priority_ids = ["Vw12", "Vta10", "Vtb8", "Vw10", "Vw14a", "Vta14", "Vtb2", "Vw6", "Vta20", "Vtb4", "Vw8", "Vta7"]

    # First add priority drives
    for pid in priority_ids:
        s_candidate = os.path.join(s_dir, f"S-{pid}.csv")
        v_candidate = os.path.join(v_dir, f"V-{pid}.csv")
        if os.path.exists(s_candidate) and os.path.exists(v_candidate):
            pairs.append((s_candidate, v_candidate, pid))

    # Then fill up to max_drives
    for s_path in s_files:
        if len(pairs) >= max_drives:
            break
        name = os.path.basename(s_path).replace("S-", "").replace(".csv", "")
        if any(p[2] == name for p in pairs):
            continue
        v_path = os.path.join(v_dir, f"V-{name}.csv")
        if os.path.exists(v_path) and os.path.getsize(s_path) < 5 * 1024 * 1024:  # < 5MB per drive for fast training
            pairs.append((s_path, v_path, name))

    return pairs


def train_master_velocity_model(
    pairs: List[Tuple[str, str, str]],
    epochs: int = 30,
    lr: float = 2e-3,
    out_path: str = "artifacts/models/tcn_velocity.pt"
) -> Dict[str, float]:
    """Train TCN velocity model across multiple real IO-VNBD drives."""
    all_X_train, all_Y_train = [], []
    all_X_val, all_Y_val = [], []

    print("\n" + "=" * 75)
    print("  STEP 1: PREPARING MULTI-DRIVE IO-VNBD DATASET FOR VELOCITY ESTIMATOR")
    print("=" * 75)

    test_pair = pairs[0]  # S-Vw12 as primary test benchmark
    train_pairs = pairs[1:]

    for s_path, v_path, drive_id in train_pairs:
        try:
            drive = load_smartphone_drive(s_path)
            v_speed = load_vehicle_drive(v_path)
            v_speed = np.resize(v_speed, len(drive))

            # Apply runtime calibration & denoising pipeline on raw phone IMU
            from navresilient.calibration import AutoCalibrator
            from navresilient.denoise import HybridIMUDenoiser
            calibrator = AutoCalibrator(fs=drive.fs)
            calibrator.calibrate(drive.acc_raw, drive.gyro, drive.gps_speed_mps, drive.gps_course_rad)
            acc_v = np.array([calibrator.transform_to_vehicle_frame(a, w)[0] for a, w in zip(drive.acc_raw, drive.gyro)])
            gyro_v = np.array([calibrator.transform_to_vehicle_frame(a, w)[1] for a, w in zip(drive.acc_raw, drive.gyro)])
            
            denoiser = HybridIMUDenoiser(fs=drive.fs)
            acc_filt = np.array([denoiser.filter_frame(a)[0] for a in acc_v])

            X, Y = prepare_dataset(acc_filt, gyro_v, v_speed, fs=drive.fs, win_s=2.0, hop_s=0.2)
            if len(X) > 20:
                all_X_train.append(X)
                all_Y_train.append(Y)
                print(f"  + Loaded Drive S-{drive_id}: {len(X)} windows (Duration: {drive.t_s[-1]:.1f}s)")
        except Exception as e:
            print(f"  - Skipped Drive S-{drive_id}: {e}")

    # Validation on primary test drive (S-Vw12)
    s_test, v_test, test_id = test_pair
    drive_test = load_smartphone_drive(s_test)
    v_test_spd = load_vehicle_drive(v_test)
    v_test_spd = np.resize(v_test_spd, len(drive_test))
    
    cal_test = AutoCalibrator(fs=drive_test.fs)
    cal_test.calibrate(drive_test.acc_raw, drive_test.gyro, drive_test.gps_speed_mps, drive_test.gps_course_rad)
    acc_v_test = np.array([cal_test.transform_to_vehicle_frame(a, w)[0] for a, w in zip(drive_test.acc_raw, drive_test.gyro)])
    gyro_v_test = np.array([cal_test.transform_to_vehicle_frame(a, w)[1] for a, w in zip(drive_test.acc_raw, drive_test.gyro)])
    den_test = HybridIMUDenoiser(fs=drive_test.fs)
    acc_filt_test = np.array([den_test.filter_frame(a)[0] for a in acc_v_test])
    
    X_val, Y_val = prepare_dataset(acc_filt_test, gyro_v_test, v_test_spd, fs=drive_test.fs, win_s=2.0, hop_s=0.2)

    X_train = np.concatenate(all_X_train, axis=0)
    Y_train = np.concatenate(all_Y_train, axis=0)

    print(f"\nTotal Multi-Drive Dataset: {len(X_train)} Train Windows | {len(X_val)} Validation Windows (S-{test_id})")

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train).unsqueeze(1))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(Y_val).unsqueeze(1))

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)

    model = TCNVelocityNet(in_channels=6, hidden_channels=32)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print("\n" + "=" * 75)
    print(f"  TRAINING TCN VELOCITY ESTIMATOR ({sum(p.numel() for p in model.parameters()):,} parameters)")
    print("=" * 75)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            optimizer.zero_grad()
            pv, lv = model(bx)
            loss = 0.5 * (torch.exp(-lv) * (by - pv) ** 2 + lv).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(bx)
        scheduler.step()

        if epoch % 5 == 0 or epoch == epochs:
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
            print(f"  Epoch [{epoch:02d}/{epochs}] | Train Loss: {train_loss/len(X_train):.3f} | Val MAE: {mae:.3f} m/s ({mae*3.6:.2f} km/h) | Val RMSE: {rmse:.3f} m/s")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"\n[OK] Model successfully trained on IO-VNBD & saved to: {out_path}")

    return {"val_mae_mps": mae, "val_rmse_mps": rmse, "val_mae_kmh": mae * 3.6}


def train_master_residual_net(
    pairs: List[Tuple[str, str, str]],
    epochs: int = 25,
    out_path: str = "artifacts/models/ai_residual_net.pt"
):
    """Train AI Residual Drift Net on multi-drive outage segments."""
    from navresilient.ai_fusion_residual import build_residual_training_data

    print("\n" + "=" * 75)
    print("  STEP 2: PREPARING MULTI-DRIVE DATA FOR AI RESIDUAL DRIFT COMPENSATOR")
    print("=" * 75)

    all_feat, all_targ = [], []
    for s_path, v_path, drive_id in pairs[:6]:
        try:
            feat, targ = build_residual_training_data(s_path, v_path, window_len=10, outage_stride=25)
            if len(feat) > 10:
                all_feat.append(feat)
                all_targ.append(targ)
                print(f"  + Extracted Residual Outage Data from S-{drive_id}: {len(feat)} segments")
        except Exception as e:
            print(f"  - Skipped Residual Data for S-{drive_id}: {e}")

    X_res = np.concatenate(all_feat, axis=0)
    Y_res = np.concatenate(all_targ, axis=0)

    # Calculate normalization parameters
    f_mean = np.mean(X_res, axis=(0, 2), keepdims=True)  # (1, 8, 1)
    f_std = np.std(X_res, axis=(0, 2), keepdims=True) + 1e-6
    t_mean = np.mean(Y_res, axis=0, keepdims=True)     # (1, 3)
    t_std = np.std(Y_res, axis=0, keepdims=True) + 1e-6

    X_norm = (X_res - f_mean) / f_std
    Y_norm = (Y_res - t_mean) / t_std

    train_ds = TensorDataset(torch.from_numpy(X_norm.astype(np.float32)), torch.from_numpy(Y_norm.astype(np.float32)))
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

    net = AIResidualDriftNet(in_channels=8, window_size=10)
    optimizer = optim.AdamW(net.parameters(), lr=1.5e-3, weight_decay=1e-4)
    criterion = nn.SmoothL1Loss()

    print(f"\n  TRAINING AI RESIDUAL DRIFT NET ({sum(p.numel() for p in net.parameters()):,} params) on {len(X_res)} samples...")

    for epoch in range(1, epochs + 1):
        net.train()
        total_loss = 0.0
        for bx, by in train_loader:
            optimizer.zero_grad()
            out = net(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(bx)

        if epoch % 5 == 0 or epoch == epochs:
            print(f"  Epoch [{epoch:02d}/{epochs}] | Smooth L1 Loss: {total_loss/len(X_res):.5f}")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    checkpoint = {
        "model_state_dict": net.state_dict(),
        "feat_mean": f_mean.squeeze(),
        "feat_std": f_std.squeeze(),
        "target_mean": t_mean.squeeze(),
        "target_std": t_std.squeeze(),
        "max_z_thresh": 5.0,
    }
    torch.save(checkpoint, out_path)
    print(f"[OK] Residual Drift Net and normalization statistics saved to: {out_path}\n")


def main():
    dataset_root = "IO-VNBD/Synchronised V abd S datasets/Uncategorised IOVNB Dataset"
    if not os.path.exists(dataset_root):
        dataset_root = "data/IO-VNBD"

    pairs = discover_drive_pairs(dataset_root, max_drives=10)
    print(f"Discovered {len(pairs)} synchronized IO-VNBD drive pairs for master training.")

    # 1. Train Velocity Model
    v_metrics = train_master_velocity_model(pairs, epochs=25, out_path="artifacts/models/tcn_velocity.pt")

    # 2. Train AI Residual Model
    train_master_residual_net(pairs, epochs=25, out_path="artifacts/models/ai_residual_net.pt")

    # 3. Export Mobile & Edge Runtime Graphs
    tcn_net = TCNVelocityNet()
    export_model(tcn_net, out_dir="artifacts/models")

    print("\n" + "=" * 75)
    print("  ALL MODELS SUCCESSFULLY TRAINED & EXPORTED FROM REAL IO-VNBD DATASET!")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
