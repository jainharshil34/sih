"""
AI Velocity Estimation Engine using Temporal Convolutional Networks (TCN).

Problem Statement SIH26168 Requirement:
Dead reckoning directly integrating noisy accelerometer data explodes quadratically/cubically.
This module regresses vehicle forward speed (v_x) purely from 6-DoF IMU windows
(a_x, a_y, a_z, w_x, w_y, w_z) without requiring an OBD-II / CAN wheel-speed connection.
Crucially, it also predicts aleatoric uncertainty (sigma_v^2) which dynamically tunes
the measurement covariance R_k in the 15-state ES-EKF.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple, Union

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    class ResidualBlock1D(nn.Module):
        """1D Dilated Temporal Residual Block with Weight Normalization."""
        def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                     dilation: int = 1, dropout: float = 0.1):
            super().__init__()
            padding = (kernel_size - 1) * dilation // 2
            self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                                   padding=padding, dilation=dilation)
            self.bn1 = nn.BatchNorm1d(out_channels)
            self.relu1 = nn.ReLU()
            self.drop1 = nn.Dropout(dropout)

            self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                                   padding=padding, dilation=dilation)
            self.bn2 = nn.BatchNorm1d(out_channels)
            self.relu2 = nn.ReLU()
            self.drop2 = nn.Dropout(dropout)

            self.shortcut = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            res = self.shortcut(x)
            x = self.drop1(self.relu1(self.bn1(self.conv1(x))))
            x = self.drop2(self.relu2(self.bn2(self.conv2(x))))
            return x + res


    class TCNVelocityNet(nn.Module):
        """Temporal Convolutional Network for velocity and uncertainty regression."""
        def __init__(self, in_channels: int = 6, hidden_channels: int = 32,
                     num_layers: int = 3, kernel_size: int = 3,
                     hidden_dim: Optional[int] = None):
            super().__init__()
            if hidden_dim is not None:
                hidden_channels = hidden_dim
            layers = []
            c_in = in_channels
            for i in range(num_layers):
                dilation = 2 ** i
                layers.append(ResidualBlock1D(c_in, hidden_channels, kernel_size=kernel_size,
                                              dilation=dilation, dropout=0.05))
                c_in = hidden_channels
            self.tcn = nn.Sequential(*layers)
            self.pool = nn.AdaptiveAvgPool1d(1)

            # Velocity head (m/s, strictly positive)
            self.head_vel = nn.Sequential(
                nn.Linear(hidden_channels, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Softplus()
            )
            # Uncertainty head (log variance log(sigma^2))
            self.head_var = nn.Sequential(
                nn.Linear(hidden_channels, 16),
                nn.ReLU(),
                nn.Linear(16, 1)
            )

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Args:
                x: (Batch, 6, TimeLength)
            Returns:
                velocity: (Batch, 1) forward speed in m/s
                log_var: (Batch, 1) log variance log(sigma_v^2)
            """
            feat = self.tcn(x)
            pooled = self.pool(feat).squeeze(-1)
            v = self.head_vel(pooled)
            log_var = self.head_var(pooled)
            # Clamp log_var for numerical stability (variance between 0.01 and 25 m^2/s^2)
            log_var = torch.clamp(log_var, min=-4.6, max=3.2)
            return v, log_var


def extract_imu_windows(acc_v: np.ndarray, gyro_v: np.ndarray, fs: float,
                        win_s: float = 2.0, hop_s: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """Create rolling window slices (N, 6, L) and end indices."""
    win_len = int(win_s * fs)
    hop_len = max(1, int(hop_s * fs))
    imu = np.column_stack([acc_v, gyro_v])  # (T, 6)
    n_samples = len(imu)
    
    windows = []
    indices = []
    for end_idx in range(win_len, n_samples + 1, hop_len):
        w = imu[end_idx - win_len:end_idx].T  # (6, L)
        windows.append(w)
        indices.append(end_idx - 1)
    
    if not windows:
        return np.empty((0, 6, win_len)), np.empty(0, dtype=int)
    return np.asarray(windows, dtype=np.float32), np.asarray(indices, dtype=int)


class TCNVelocity:
    """Production TCN & Ridge Velocity Estimator with Heteroscedastic Uncertainty."""

    def __init__(self, win_s: float = 2.0, lr: float = 1e-3, epochs: int = 25,
                 device: str = "cpu", model_path: Optional[str] = None):
        self.win_s = win_s
        self.lr = lr
        self.epochs = epochs
        self.device = torch.device(device) if TORCH_AVAILABLE and torch.cuda.is_available() and device == "cuda" else torch.device("cpu")
        self.model = TCNVelocityNet().to(self.device) if TORCH_AVAILABLE else None
        self.is_fitted = False
        self.is_fitted_torch = False
        self.mean_v = 0.0
        
        # Robust feature scaler & weights for fast numpy edge execution
        self.feature_mu = None
        self.feature_sd = None
        self.ridge_w = None

        if model_path is None:
            # Check default locations for pre-trained weights
            for cand in [
                os.path.join(os.path.dirname(__file__), "../../artifacts/models/tcn_velocity.pt"),
                "artifacts/models/tcn_velocity.pt",
            ]:
                if os.path.exists(cand):
                    model_path = cand
                    break

        if model_path and os.path.exists(model_path) and TORCH_AVAILABLE and self.model is not None:
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.model.eval()
                self.is_fitted = True
                self.is_fitted_torch = True
            except Exception:
                pass

    def fit(self, drives: list[Tuple[np.ndarray, np.ndarray]], targets: list[np.ndarray], fs: float):
        """Train velocity model on paired IMU and velocity targets."""
        all_X, all_Y = [], []
        for (acc_v, gyro_v), y in zip(drives, targets):
            X_w, idx = extract_imu_windows(acc_v, gyro_v, fs, self.win_s)
            if len(idx) == 0:
                continue
            valid = np.isfinite(y[idx])
            all_X.append(X_w[valid])
            all_Y.append(y[idx][valid])

        if not all_X or len(all_X[0]) == 0:
            return self

        X = np.concatenate(all_X, axis=0)
        Y = np.concatenate(all_Y, axis=0).astype(np.float32)
        self.mean_v = float(np.mean(Y))

        # 1. Fit fast baseline weights for edge fallback
        self._fit_numpy_fallback(X, Y, fs)
        self.is_fitted = True

        if not TORCH_AVAILABLE or len(X) < 100:
            return self

        try:
            # 2. Train PyTorch Deep TCN
            dataset = torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(Y).unsqueeze(1))
            loader = torch.utils.data.DataLoader(dataset, batch_size=min(32, len(dataset)), shuffle=True)
            
            optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

            self.model.train()
            for _ in range(self.epochs):
                for batch_x, batch_y in loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    optimizer.zero_grad()
                    pred_v, log_var = self.model(batch_x)
                    loss = 0.5 * (torch.exp(-log_var) * (batch_y - pred_v) ** 2 + log_var).mean()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                scheduler.step()

            self.model.eval()
            self.is_fitted_torch = True
        except Exception:
            pass

        return self

    def _fit_numpy_fallback(self, X_windows: np.ndarray, Y: np.ndarray, fs: float):
        """Fit lightweight statistical features for sub-millisecond edge fallback."""
        feats = []
        for w in X_windows:
            feats.append(np.concatenate([
                w.mean(axis=1),
                w.std(axis=1),
                np.max(np.abs(w), axis=1),
                np.sqrt(np.mean(w ** 2, axis=1)),
                [np.sum(w[0]) / fs],
            ]))
        F = np.asarray(feats, dtype=np.float32)
        self.feature_mu = F.mean(axis=0)
        self.feature_sd = F.std(axis=0) + 1e-6
        Fs = np.column_stack([(F - self.feature_mu) / self.feature_sd, np.ones(len(F))])
        A = Fs.T @ Fs + 0.5 * np.eye(Fs.shape[1])
        self.ridge_w = np.linalg.solve(A, Fs.T @ Y)

    def predict(self, acc_v: np.ndarray, gyro_v: np.ndarray, fs: float,
                n_out: int) -> Tuple[np.ndarray, np.ndarray]:
        """Predict forward speed (m/s) and estimated variance sigma_v^2 (m^2/s^2)."""
        if not self.is_fitted:
            return np.full(n_out, self.mean_v), np.full(n_out, 0.25)

        X_w, idx = extract_imu_windows(acc_v, gyro_v, fs, self.win_s)
        if len(idx) == 0:
            return np.full(n_out, self.mean_v), np.full(n_out, 0.25)

        # 1. Primary path: Deep PyTorch TCN inference
        if self.is_fitted_torch and self.model is not None and TORCH_AVAILABLE:
            try:
                self.model.eval()
                with torch.no_grad():
                    tx = torch.from_numpy(X_w).float().to(self.device)
                    pv, lv = self.model(tx)
                    v_arr = np.maximum(pv.squeeze(-1).cpu().numpy(), 0.0)
                    # Variance = exp(log_var). Increase var when confidence is low to avoid pulling UKF incorrectly
                    var_arr = np.maximum(np.exp(lv.squeeze(-1).cpu().numpy()), 0.05)

                    out_v = np.full(n_out, np.nan)
                    out_var = np.full(n_out, np.nan)
                    out_v[idx] = v_arr
                    out_var[idx] = var_arr

                    out_v = _ffill(out_v, default=self.mean_v)
                    out_var = _ffill(out_var, default=0.20)
                    return np.maximum(out_v, 0.0), np.maximum(out_var, 0.01)
            except Exception:
                pass

        # 2. Fallback path: Ridge statistical regression
        if self.ridge_w is not None and self.feature_mu is not None:
            feats = []
            for w in X_w:
                feats.append(np.concatenate([
                    w.mean(axis=1),
                    w.std(axis=1),
                    np.max(np.abs(w), axis=1),
                    np.sqrt(np.mean(w ** 2, axis=1)),
                    [np.sum(w[0]) / fs],
                ]))
            F = np.asarray(feats, dtype=np.float32)
            Fs = np.column_stack([(F - self.feature_mu) / self.feature_sd, np.ones(len(F))])
            v_arr = np.maximum(Fs @ self.ridge_w, 0.0)
            var_arr = np.full(len(v_arr), 0.20, dtype=np.float32)

            out_v = np.full(n_out, np.nan)
            out_var = np.full(n_out, np.nan)
            out_v[idx] = v_arr
            out_var[idx] = var_arr

            # Forward-fill to cover full time horizon
            out_v = _ffill(out_v, default=self.mean_v)
            out_var = _ffill(out_var, default=0.20)
            return np.maximum(out_v, 0.0), np.maximum(out_var, 0.01)

        return np.full(n_out, self.mean_v), np.full(n_out, 0.25)


def _ffill(a: np.ndarray, default: float = 0.0) -> np.ndarray:
    a = a.copy()
    idx = np.where(np.isfinite(a), np.arange(len(a)), 0)
    np.maximum.accumulate(idx, out=idx)
    a = a[idx]
    return np.nan_to_num(a, nan=default)
