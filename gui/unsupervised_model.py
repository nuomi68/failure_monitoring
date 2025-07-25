import sys
from pathlib import Path
from typing import List, Any, Dict

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PyQt6.QtWidgets import QSlider
from PyQt6.QtCore import Qt, pyqtSignal
import  torch


def train_knn(X, k=5, **params):
    from sklearn.neighbors import NearestNeighbors
    # 过滤掉空参数
    kwargs = {k2: v for k2, v in params.items() if v is not None}
    nbrs = NearestNeighbors(n_neighbors=k, **kwargs)
    nbrs.fit(X)
    dist = nbrs.kneighbors(X)[0][:, -1]        # k-th 距离
    tau = np.quantile(dist, 0.95)              # 默认阈值 95% 分位
    return nbrs, tau, dist

def parse_max_samples(val, n_samples: int):
    """IsolationForest 的 max_samples 允许 int, float, 'auto'"""
    if val == "" or val.lower() == "auto":
        return "auto"
    try:
        if "." in val:
            f = float(val)
            if 0 < f <= 1:
                return f
        i = int(val)
        if i > 0:
            return min(i, n_samples)
    except Exception:
        pass
    return "auto"

def train_iforest(X, n_estimators=100, contamination=0.01, **params):
    from sklearn.ensemble import IsolationForest
    kwargs = {k2: v for k2, v in params.items() if v is not None}
    clf = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        **kwargs
    )
    clf.fit(X)
    score = -clf.decision_function(X)
    tau = np.quantile(score, 0.95)
    return clf, tau, score


# ===== 自编码器实现 =====
class AutoEncoder(torch.nn.Module):
    def __init__(self, input_dim: int, hidden: list[int], latent_dim: int, dropout: float = 0.0):
        super().__init__()
        layers = []
        last = input_dim
        for h in hidden:
            layers += [torch.nn.Linear(last, h), torch.nn.ReLU()]
            if dropout > 0:
                layers += [torch.nn.Dropout(dropout)]
            last = h
        # bottleneck
        layers += [torch.nn.Linear(last, latent_dim), torch.nn.ReLU()]
        # decoder
        dec_layers = []
        last = latent_dim
        for h in reversed(hidden):
            dec_layers += [torch.nn.Linear(last, h), torch.nn.ReLU()]
            last = h
        dec_layers += [torch.nn.Linear(last, input_dim)]
        self.encoder = torch.nn.Sequential(*layers)
        self.decoder = torch.nn.Sequential(*dec_layers)

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out

def train_autoencoder(
    X: np.ndarray,
    hidden=(64, 32),
    latent_dim=16,
    epochs=50,
    batch_size=128,
    lr=1e-3,
    dropout=0.0,
    device=None,
):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    X_tensor = torch.tensor(X, dtype=torch.float32)
    ds = TensorDataset(X_tensor)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model = AutoEncoder(X.shape[1], list(hidden), latent_dim, dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    model.train()
    for ep in range(epochs):
        for (batch,) in dl:
            batch = batch.to(device)
            recon = model(batch)
            loss = loss_fn(recon, batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            print(f"Epoch {ep+1}/{epochs} loss={loss.item():.4f}")

    # 计算重构误差作为分数
    model.eval()
    with torch.no_grad():
        recon = model(X_tensor.to(device)).cpu().numpy()
    mse = np.mean((recon - X) ** 2, axis=1)
    tau = np.quantile(mse, 0.95)
    return model, tau, mse
