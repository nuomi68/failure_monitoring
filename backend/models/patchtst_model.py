import math
from typing import Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PatchTSTConfig, PatchTSTForRegression


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


def build_model(seq_len: int, num_feat: int) -> PatchTSTForRegression:
    config = PatchTSTConfig(
        num_input_channels=num_feat,
        num_targets=num_feat,
        context_length=seq_len,
        prediction_length=1,
        patch_length=8,
        patch_stride=8,
        d_model=128,
        num_hidden_layers=4,
        num_attention_heads=8,
        ffn_dim=256,
        dropout=0.1,
        loss="mse",
        channel_attention=False,
    )
    return PatchTSTForRegression(config)


def train_patchtst(X_train, y_train, X_val, y_val, device="cpu", lr=1e-3, epochs=50) -> Tuple[PatchTSTForRegression, float]:
    dtrain = DataLoader(SeqDataset(X_train, y_train), batch_size=32, shuffle=True)
    dval = DataLoader(SeqDataset(X_val, y_val), batch_size=32)

    model = build_model(X_train.shape[1], X_train.shape[2]).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr)

    best = math.inf
    for _ in range(epochs):
        model.train(); tloss = 0.0
        for xb, yb in dtrain:
            xb, yb = xb.to(device), yb.to(device)
            optimiser.zero_grad()
            out = model(past_values=xb, target_values=yb)
            loss = out.loss
            loss.backward(); optimiser.step()
            tloss += loss.item() * xb.size(0)
        tloss /= len(dtrain.dataset)

        model.eval(); vloss = 0.0
        with torch.no_grad():
            for xb, yb in dval:
                xb, yb = xb.to(device), yb.to(device)
                out = model(past_values=xb, target_values=yb)
                vloss += out.loss.item() * xb.size(0)
        vloss /= len(dval.dataset)
        best = min(best, vloss)

    return model, best


def predict(model: PatchTSTForRegression, seq, device="cpu"):
    model.eval()
    with torch.no_grad():
        seq = _to_tensor(seq, device).unsqueeze(0)
        out = model(past_values=seq)
        return out.regression_outputs.cpu().numpy().squeeze()
