import math
from typing import Tuple

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchtsmixer import TSMixer


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


def build_model(seq_len: int, num_feat: int) -> TSMixer:
    return TSMixer(
        sequence_length=seq_len,
        prediction_length=1,
        input_channels=num_feat,
        output_channels=num_feat,
        num_blocks=4,
        ff_dim=128,
        dropout_rate=0.1,
    )


def train_tsmixer(
    X_train, y_train, X_val, y_val, device="cpu", lr=1e-3, epochs=50
) -> Tuple[TSMixer, float]:
    dtrain = DataLoader(SeqDataset(X_train, y_train), batch_size=16, shuffle=True)
    dval = DataLoader(SeqDataset(X_val, y_val), batch_size=16)

    model = build_model(X_train.shape[1], X_train.shape[2]).to(device)
    criterion = nn.MSELoss()
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr)

    best = math.inf
    for _ in range(epochs):
        model.train()
        for xb, yb in dtrain:
            xb, yb = xb.to(device), yb.to(device)
            optimiser.zero_grad()
            pred = model(xb).squeeze(1)
            loss = criterion(pred, yb)
            loss.backward()
            optimiser.step()

        model.eval(); vloss = 0.0
        with torch.no_grad():
            for xb, yb in dval:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb).squeeze(1)
                vloss += criterion(pred, yb).item() * xb.size(0)
        vloss /= len(dval.dataset)
        best = min(best, vloss)

    return model, best


def predict(model: TSMixer, seq, device="cpu"):
    model.eval()
    with torch.no_grad():
        seq = _to_tensor(seq, device).unsqueeze(0)
        out = model(seq).squeeze(1)
        return out.cpu().numpy().squeeze()
