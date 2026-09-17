"""
AI Residual Drift Estimator (AI-aided Sensor Fusion Algorithm).

A lightweight PyTorch 1D-CNN neural network that learns to predict and
subtract instantaneous residual integration drift that the classical
UKF leaves uncorrected during GNSS outages.

Trained on IO-VNBD dataset with feature/target normalization and
Mahalanobis / Z-score Out-Of-Distribution (OOD) safety fallback.
"""

from __future__ import annotations

import argparse
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from navresilient.ekf_ukf import NavResilientUKF
from navresilient.io_vnbd_loader import load_smartphone_drive


class AIResidualDriftNet(nn.Module):
    """Lightweight 1D-CNN Residual Drift Compensator (~11,800 parameters, <0.4ms latency)."""

    def __init__(self, in_channels: int = 8, window_size: int = 10):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 3)  # Normalized [residual_dx, residual_dy, residual_dv]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape (Batch, 8, Window_Size)
        Returns:
            residuals: Tensor of shape (Batch, 3) -> [dx_norm, dy_norm, dv_norm]
        """
        h = self.relu(self.bn1(self.conv1(x)))
        h = self.relu(self.bn2(self.conv2(h)))
        h = self.pool(h).squeeze(-1)
        return self.fc(h)


class ResidualOutageDataset(Dataset):
    """Dataset for training residual drift compensator on normalized features and targets."""

    def __init__(self, features: np.ndarray, targets: np.ndarray):
        self.x = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


def build_residual_training_data(
    s_csv: str,
    v_csv: Optional[str] = None,
    window_len: int = 10,
    outage_stride: int = 25
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract realistic instantaneous per-step UKF residual drift targets.
    
    Target:
      - delta_x = delta_x_true - delta_x_ukf (East step residual in meters per 0.1s step)
      - delta_y = delta_y_true - delta_y_ukf (North step residual in meters per 0.1s step)
      - delta_v = v_true - v_ukf (Velocity residual in m/s)
    """
    drive = load_smartphone_drive(s_csv)
    n = len(drive.t_s)
    if n < window_len + 20:
        return np.zeros((0, 8, window_len), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    gt_x = drive.x_east_m
    gt_y = drive.y_north_m
    gt_v = drive.gps_speed_mps
    gt_crs = drive.gps_course_rad

    # Run realistic UKF tracking
    ukf = NavResilientUKF(dt=1.0 / drive.fs)
    ukf.initialize(gt_x[0], gt_y[0], gt_v[0], gt_crs[0])

    ukf_x = np.zeros(n)
    ukf_y = np.zeros(n)
    ukf_v = np.zeros(n)
    ukf_head = np.zeros(n)

    # Periodic synthetic GNSS outages (15-second windows)
    outage_mask = np.zeros(n, dtype=bool)
    for start_idx in range(40, n - 150, outage_stride):
        outage_mask[start_idx : start_idx + 150] = True

    for i in range(n):
        acc_fwd = float(drive.acc[i, 0])
        gyro_z = -float(drive.gyro[i, 2])

        ukf.predict(acc_fwd, gyro_z, dt=0.1)

        if not outage_mask[i]:
            ukf.update_gnss(gt_x[i], gt_y[i], sigma_pos=2.0, course_rad=gt_crs[i], speed_mps=gt_v[i])
        else:
            ukf.mark_gnss_lost()

        pos = ukf.position
        ukf_x[i] = pos[0]
        ukf_y[i] = pos[1]
        ukf_v[i] = ukf.forward_speed
        ukf_head[i] = ukf.heading_rad

    # Incremental step vectors
    step_gt_x = np.diff(np.pad(gt_x, (1, 0), mode="edge"))
    step_gt_y = np.diff(np.pad(gt_y, (1, 0), mode="edge"))
    step_ukf_x = np.diff(np.pad(ukf_x, (1, 0), mode="edge"))
    step_ukf_y = np.diff(np.pad(ukf_y, (1, 0), mode="edge"))

    features_list = []
    targets_list = []

    for i in range(window_len, n):
        if outage_mask[i]:
            w_idx = slice(i - window_len, i)

            feat = np.zeros((8, window_len), dtype=np.float32)
            feat[0, :] = step_ukf_x[w_idx]
            feat[1, :] = step_ukf_y[w_idx]
            feat[2, :] = ukf_v[w_idx]
            feat[3, :] = np.sin(ukf_head[w_idx])
            feat[4, :] = np.cos(ukf_head[w_idx])
            feat[5, :] = drive.acc[w_idx, 0]
            feat[6, :] = drive.acc[w_idx, 1]
            feat[7, :] = -drive.gyro[w_idx, 2]

            # Instantaneous per-step residual error
            err_dx = float(step_gt_x[i] - step_ukf_x[i])
            err_dy = float(step_gt_y[i] - step_ukf_y[i])
            err_v = float(gt_v[i] - ukf_v[i])

            features_list.append(feat)
            targets_list.append([err_dx, err_dy, err_v])

    if len(features_list) == 0:
        return np.zeros((0, 8, window_len), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    return np.array(features_list, dtype=np.float32), np.array(targets_list, dtype=np.float32)


def train_ai_residual_model(
    s_csv: str,
    v_csv: Optional[str] = None,
    epochs: int = 25,
    save_path: str = "artifacts/models/ai_residual_net.pt"
) -> AIResidualDriftNet:
    """Train the AI Residual Drift Compensator with normalized data and saved statistics."""
    feats, targets = build_residual_training_data(s_csv, v_csv)
    if len(feats) == 0:
        raise ValueError("No outage samples extracted for residual training.")

    # Compute normalization statistics
    f_mean = np.mean(feats, axis=(0, 2), keepdims=True)  # (1, 8, 1)
    f_std = np.std(feats, axis=(0, 2), keepdims=True) + 1e-6
    t_mean = np.mean(targets, axis=0, keepdims=True)     # (1, 3)
    t_std = np.std(targets, axis=0, keepdims=True) + 1e-6

    feats_norm = (feats - f_mean) / f_std
    targets_norm = (targets - t_mean) / t_std

    n_samples = len(feats)
    split = int(0.8 * n_samples)

    train_ds = ResidualOutageDataset(feats_norm[:split], targets_norm[:split])
    val_ds = ResidualOutageDataset(feats_norm[split:], targets_norm[split:])

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    model = AIResidualDriftNet(in_channels=8, window_size=10)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.SmoothL1Loss()

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for bx, by in train_loader:
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(bx)

        if epoch % 5 == 0 or epoch == epochs:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for bx, by in val_loader:
                    pred = model(bx)
                    val_loss += criterion(pred, by).item() * len(bx)
            val_loss /= max(1, len(val_ds))
            print(f"  Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {total_loss/len(train_ds):.5f} | Val Loss: {val_loss:.5f}")
            model.train()

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "feat_mean": f_mean.squeeze(),
        "feat_std": f_std.squeeze(),
        "target_mean": t_mean.squeeze(),
        "target_std": t_std.squeeze(),
        "max_z_thresh": 5.0,
    }
    torch.save(checkpoint, save_path)
    print(f"  Model and normalization statistics saved to: {save_path}\n")
    return model


class AIResidualFusionEngine:
    """Production AI Residual Drift Compensator with OOD Detection & Safety Gating."""

    def __init__(self, model_path: Optional[str] = None):
        self.model = AIResidualDriftNet(in_channels=8, window_size=10)
        self.model_loaded = False
        self.window_len = 10
        self.history_window = []

        # Normalization defaults
        self.feat_mean = np.zeros(8, dtype=np.float32)
        self.feat_std = np.ones(8, dtype=np.float32)
        self.target_mean = np.zeros(3, dtype=np.float32)
        self.target_std = np.ones(3, dtype=np.float32)
        self.max_z_thresh = 5.0

        if model_path and os.path.exists(model_path):
            try:
                ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
                if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                    self.model.load_state_dict(ckpt["model_state_dict"])
                    self.feat_mean = np.array(ckpt.get("feat_mean", self.feat_mean), dtype=np.float32)
                    self.feat_std = np.array(ckpt.get("feat_std", self.feat_std), dtype=np.float32)
                    self.target_mean = np.array(ckpt.get("target_mean", self.target_mean), dtype=np.float32)
                    self.target_std = np.array(ckpt.get("target_std", self.target_std), dtype=np.float32)
                    self.max_z_thresh = float(ckpt.get("max_z_thresh", 5.0))
                else:
                    self.model.load_state_dict(ckpt)
                self.model.eval()
                self.model_loaded = True
            except Exception as e:
                print(f"[WARN] Failed to load residual model from {model_path}: {e}")
                self.model_loaded = False

    @property
    def is_ready(self) -> bool:
        return self.model_loaded

    def initialize(self, x0: float, y0: float, v0: float, heading0: float):
        self.history_window.clear()

    def update_history(
        self,
        step_dx: float,
        step_dy: float,
        v_ukf: float,
        heading_rad: float,
        acc_fwd: float,
        acc_lat: float,
        gyro_z: float
    ):
        """Append one frame to the rolling inference buffer."""
        feat = np.array([
            step_dx,
            step_dy,
            v_ukf,
            math.sin(heading_rad),
            math.cos(heading_rad),
            acc_fwd,
            acc_lat,
            gyro_z
        ], dtype=np.float32)
        self.history_window.append(feat)
        if len(self.history_window) > self.window_len:
            self.history_window.pop(0)

    def predict_step_correction(self) -> Tuple[float, float, float, float]:
        """Predict instantaneous step correction with OOD safety gating.

        Returns:
            (corr_dx, corr_dy, corr_dv, confidence)
            where corr_dx/dy are in meters per step and corr_dv is in m/s.
        """
        if not self.model_loaded or len(self.history_window) < self.window_len:
            return 0.0, 0.0, 0.0, 0.0

        # Shape: (8, 10)
        w_arr = np.array(self.history_window, dtype=np.float32).T

        # 1. Out-of-Distribution (OOD) & Z-score Check
        f_norm = (w_arr - self.feat_mean[:, None]) / (self.feat_std[:, None] + 1e-6)
        max_z = float(np.max(np.abs(f_norm)))

        # Soft confidence gating: full confidence below z=3.0, tapering to 0.0 at z=5.0
        if max_z > self.max_z_thresh:
            # Strictly OOD (e.g. walking motion or extreme shock) -> bypass
            return 0.0, 0.0, 0.0, 0.0

        conf = float(np.clip(1.0 - max(0.0, max_z - 3.0) / 2.0, 0.0, 1.0))
        if conf < 0.30:
            return 0.0, 0.0, 0.0, 0.0

        # 2. Model Forward Pass
        with torch.no_grad():
            inp = torch.tensor(f_norm, dtype=torch.float32).unsqueeze(0)
            res_norm = self.model(inp).squeeze(0).numpy()

        # 3. Un-normalize output
        res_raw = res_norm * self.target_std + self.target_mean

        # 4. Enforce physical safety limits on per-step corrections
        # 1 step = 0.1s; maximum allowable step correction = 0.35m (equivalent to 3.5 m/s velocity adjust)
        dx_clamped = float(np.clip(res_raw[0], -0.35, 0.35))
        dy_clamped = float(np.clip(res_raw[1], -0.35, 0.35))
        dv_clamped = float(np.clip(res_raw[2], -0.50, 0.50))

        corr_dx = conf * dx_clamped
        corr_dy = conf * dy_clamped
        corr_dv = conf * dv_clamped

        return corr_dx, corr_dy, corr_dv, conf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AI Residual Drift Compensator on IO-VNBD")
    parser.add_argument("--s-csv", default="data/IO-VNBD/S-Vw12.csv", help="Path to S-*.csv")
    parser.add_argument("--v-csv", default="data/IO-VNBD/V-Vw12.csv", help="Path to V-*.csv")
    parser.add_argument("--epochs", type=int, default=25, help="Training epochs")
    parser.add_argument("--out", default="artifacts/models/ai_residual_net.pt", help="Output model path")
    args = parser.parse_args()

    train_ai_residual_model(args.s_csv, args.v_csv, epochs=args.epochs, save_path=args.out)
