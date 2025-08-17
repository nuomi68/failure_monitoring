import math
import logging
from typing import Tuple

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger("failure_monitoring")


def _to_tensor(arr, device):
    return torch.tensor(arr, dtype=torch.float32).to(device)


class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = _to_tensor(X, "cpu").permute(0, 2, 1)  # (N, F, L)
        self.y = _to_tensor(y, "cpu")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class CausalConv1d(nn.Module):
    def __init__(self, in_chan, out_chan, k, d):
        super().__init__()
        self.pad = (k - 1) * d
        self.conv = nn.utils.weight_norm(
            nn.Conv1d(in_chan, out_chan, k, padding=self.pad, dilation=d)
        )

    def forward(self, x):
        out = self.conv(x)
        return out[:, :, :-self.pad] if self.pad else out


class TCNBlock(nn.Module):
    def __init__(self, in_chan, out_chan, k, d, drop):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(in_chan, out_chan, k, d),
            nn.ReLU(),
            nn.Dropout(drop),
            CausalConv1d(out_chan, out_chan, k, d),
            nn.ReLU(),
            nn.Dropout(drop),
        )
        self.down = nn.Conv1d(in_chan, out_chan, 1) if in_chan != out_chan else nn.Identity()

    def forward(self, x):
        return torch.relu(self.net(x) + self.down(x))


class TCNForecaster(nn.Module):
    def __init__(self, num_feat, hid=32, levels=2, k=2, drop=0.2):
        super().__init__()
        channels = [hid] * levels
        layers = []
        in_c = num_feat
        for i, out_c in enumerate(channels):
            layers.append(TCNBlock(in_c, out_c, k, 2 ** i, drop))
            in_c = out_c

        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(in_c, num_feat)

    def forward(self, x):
        y = self.tcn(x)
        last = y[:, :, -1]
        return self.fc(last)


def train_tcn(
    X_train,
    y_train,
    X_val,
    y_val,
    device: str = "cpu",
    lr: float = 1e-3,
    epochs: int = 50,
    batch_size: int = 16,
    hid: int = 32,
    levels: int = 2,
    k: int = 2,
    drop: float = 0.2,
) -> Tuple[nn.Module, float]:
    dtrain = DataLoader(SeqDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    dval = DataLoader(SeqDataset(X_val, y_val), batch_size=batch_size)

    model = TCNForecaster(X_train.shape[-1], hid=hid, levels=levels, k=k, drop=drop).to(device)
    criterion = nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    best = math.inf
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in dtrain:
            xb, yb = xb.to(device), yb.to(device)
            optimiser.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimiser.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(dtrain.dataset)

        model.eval()
        vloss = 0.0
        with torch.no_grad():
            for xb, yb in dval:
                xb, yb = xb.to(device), yb.to(device)
                vloss += criterion(model(xb), yb).item() * xb.size(0)
        vloss /= len(dval.dataset)
        best = min(best, vloss)

        logger.info(
            f"[TCN] epoch {epoch}/{epochs} train_loss={train_loss:.4f} val_loss={vloss:.4f}"
        )

    return model, best


def predict(model: nn.Module, seq, device="cpu"):
    model.eval()
    with torch.no_grad():
        seq = _to_tensor(seq, device).permute(1, 0).unsqueeze(0)  # (1, F, L)
        return model(seq).cpu().numpy().squeeze()
