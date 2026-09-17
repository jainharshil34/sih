"""
AI Forward Velocity Estimator (TCN / 1D-CNN) for IO-VNBD.

Problem Statement SIH26168 Requirement:
Ultra-lightweight neural network optimized specifically for on-device mobile/edge latency.
Predicts vehicle forward speed (v_x) and aleatoric measurement uncertainty (sigma_v^2)
from short 6-DoF windowed IMU tensors without requiring OBD-II or wheel speed sensors.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .demo import synthetic_drive
from .io_vnbd_loader import load_smartphone_drive, load_vehicle_drive
from .models.tcn_velocity import TCNVelocityNet

# Canonical unification: LightweightVelocityNet is an alias of TCNVelocityNet
LightweightVelocityNet = TCNVelocityNet


def prepare_dataset(acc_v: np.ndarray, gyro_v: np.ndarray, speed_target: np.ndarray,
                    fs: float = 10.0, win_s: float = 2.0, hop_s: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """Slice IMU streams into overlapping sliding windows (N, 6, L)."""
    win_len = int(win_s * fs)
    hop_len = max(1, int(hop_s * fs))
    imu = np.column_stack([acc_v, gyro_v]).astype(np.float32)

    X_list, Y_list = [], []
    for end_idx in range(win_len, len(imu), hop_len):
        target_v = speed_target[end_idx]
        if np.isfinite(target_v):
            w = imu[end_idx - win_len:end_idx].T  # (6, L)
            X_list.append(w)
            Y_list.append(target_v)

    return np.asarray(X_list, dtype=np.float32), np.asarray(Y_list, dtype=np.float32)


def train_velocity_estimator(s_csv: str, v_csv: Optional[str] = None,
                             epochs: int = 35, lr: float = 1.5e-3,
                             out_model_path: str = "artifacts/models/tcn_velocity.pt") -> Dict[str, float]:
    """Train and validate velocity estimator, reporting MAE and RMSE against ground truth."""
    drive = load_smartphone_drive(s_csv)
    
    if v_csv and os.path.exists(v_csv):
        target_speed = load_vehicle_drive(v_csv)
        target_speed = np.resize(target_speed, len(drive))
        label_name = "Vehicle ECU Wheel Speed"
    else:
        target_speed = drive.gps_speed_mps
        label_name = "GNSS Speed Profile"

    # Extract windowed dataset
    X, Y = prepare_dataset(drive.acc, drive.gyro, target_speed, fs=drive.fs)

    # 80/20 Train/Validation Split
    n_samples = len(X)
    split_idx = int(0.8 * n_samples)
    X_train, Y_train = X[:split_idx], Y[:split_idx]
    X_val, Y_val = X[split_idx:], Y[split_idx:]

    train_ds = torch.utils.data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train).unsqueeze(1))
    val_ds = torch.utils.data.TensorDataset(torch.from_numpy(X_val), torch.from_numpy(Y_val).unsqueeze(1))

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=64, shuffle=False)

    model = LightweightVelocityNet()
    total_params = sum(p.numel() for p in model.parameters())

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print("\n" + "=" * 70)
    print(f"  TRAINING VELOCITY ESTIMATOR ON {drive.name}")
    print(f"  Supervision Label: {label_name}")
    print(f"  Model Parameters:  {total_params:,} ({total_params * 4 / 1024:.1f} KB)")
    print(f"  Training Windows:  {len(X_train)} | Validation: {len(X_val)}")
    print("=" * 70)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for bx, by in train_loader:
            optimizer.zero_grad()
            pred_v, log_var = model(bx)
            # Heteroscedastic Gaussian NLL loss
            loss = 0.5 * (torch.exp(-log_var) * (by - pred_v) ** 2 + log_var).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(bx)
        scheduler.step()

    # Validation Evaluation
    model.eval()
    all_preds, all_vars, all_targets = [], [], []
    with torch.no_grad():
        for bx, by in val_loader:
            pv, lv = model(bx)
            all_preds.extend(pv.cpu().numpy().ravel())
            all_vars.extend(np.exp(lv.cpu().numpy().ravel()))
            all_targets.extend(by.cpu().numpy().ravel())

    preds = np.array(all_preds)
    targets = np.array(all_targets)

    mae = float(np.mean(np.abs(preds - targets)))
    rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    ss_res = np.sum((targets - preds) ** 2)
    r2 = float(1.0 - (ss_res / (ss_tot + 1e-8)))

    os.makedirs(os.path.dirname(os.path.abspath(out_model_path)), exist_ok=True)
    torch.save(model.state_dict(), out_model_path)

    metrics = {
        "dataset": drive.name,
        "supervision_source": label_name,
        "model_parameters": total_params,
        "val_mae_mps": round(mae, 3),
        "val_rmse_mps": round(rmse, 3),
        "val_mae_kmh": round(mae * 3.6, 2),
        "val_rmse_kmh": round(rmse * 3.6, 2),
        "r2_score": round(r2, 3),
        "model_saved_to": out_model_path
    }

    print("\n--- VALIDATION ACCURACY REPORT ---")
    print(f"  Velocity MAE:  {metrics['val_mae_mps']:.3f} m/s ({metrics['val_mae_kmh']:.2f} km/h)")
    print(f"  Velocity RMSE: {metrics['val_rmse_mps']:.3f} m/s ({metrics['val_rmse_kmh']:.2f} km/h)")
    print(f"  R² Score:      {metrics['r2_score']:.3f}")
    print(f"  Model Saved:   {out_model_path}\n")

    return metrics


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--s-csv", help="Path to S-*.csv")
    p.add_argument("--v-csv", help="Path to V-*.csv")
    p.add_argument("--epochs", type=int, default=30)
    a = p.parse_args()

    s_path = a.s_csv if a.s_csv else synthetic_drive("figures")
    train_velocity_estimator(s_path, a.v_csv, epochs=a.epochs)
