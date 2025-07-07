import math
from typing import Tuple

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error


def _to_tensor(arr, device):
    return torch.tensor(arr, dtype=torch.float32).to(device)


class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = _to_tensor(X, "cpu")
        self.y = _to_tensor(y, "cpu")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class GRUForecaster(nn.Module):
    def __init__(self, num_feat: int, hidden_size: int = 32, num_layers: int = 2):
        super().__init__()
        self.gru = nn.GRU(num_feat, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size, num_feat)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1])


def train_gru(X_train, y_train, X_val, y_val, device="cpu", lr=1e-3, epochs=50) -> Tuple[nn.Module, float]:
    """Train a GRU forecasting model and return the model and validation MAE."""
    dtrain = DataLoader(SeqDataset(X_train, y_train), batch_size=16, shuffle=True)
    dval = DataLoader(SeqDataset(X_val, y_val), batch_size=16)

    model = GRUForecaster(X_train.shape[-1]).to(device)
    criterion = nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    best = math.inf
    for _ in range(epochs):
        model.train()
        for xb, yb in dtrain:
            xb, yb = xb.to(device), yb.to(device)
            optimiser.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimiser.step()

        model.eval(); vloss = 0.0
        with torch.no_grad():
            for xb, yb in dval:
                xb, yb = xb.to(device), yb.to(device)
                vloss += criterion(model(xb), yb).item() * xb.size(0)
        vloss /= len(dval.dataset)
        best = min(best, vloss)

    return model, best


def predict(model: nn.Module, seq, device="cpu"):
    model.eval()
    with torch.no_grad():
        seq = _to_tensor(seq, device).unsqueeze(0)
        return model(seq).cpu().numpy().squeeze()
