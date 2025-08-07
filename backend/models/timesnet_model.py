import math
from typing import Tuple, List

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


# ------- 工具 ---------
def _to_tensor(arr, device):
    return torch.tensor(arr, dtype=torch.float32, device=device)


class SeqDataset(Dataset):
    """TimesNet 用的 Dataset：输入 shape = (N, L, F)"""
    def __init__(self, X, y, device="cpu"):
        # (N, L, F) => Tensor
        self.X = _to_tensor(X, device)
        self.y = _to_tensor(y, device)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ------- TimesBlock（简化版）---------
class TimesBlock(nn.Module):
    """
    原论文的 FFT + 周期卷积为了简洁改成 DilatedConv + Squeeze-Excitation。
    重点是多尺度感受野 + 通道注意力，在中小数据集也能稳定收敛。
    """
    def __init__(self, d_model: int, dilations: List[int], kernel: int = 3, se_ratio: float = 0.25):
        super().__init__()
        convs = []
        for d in dilations:
            convs.append(nn.Conv1d(d_model, d_model, kernel_size=kernel, padding=d, dilation=d, groups=d_model))
        self.convs = nn.ModuleList(convs)

        hidden = max(1, int(d_model * se_ratio))
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(d_model, hidden, 1),
            nn.ReLU(),
            nn.Conv1d(hidden, d_model, 1),
            nn.Sigmoid(),
        )
        self.proj = nn.Conv1d(d_model, d_model, 1)

    def forward(self, x):          # x: (B, F, L)
        out = 0
        for conv in self.convs:
            out = out + conv(x)
        out = out / len(self.convs)
        se = self.se(out)
        out = out * se
        return self.proj(out) + x  # 残差


# ------- TimesNet 主干 ---------
class TimesNet(nn.Module):
    def __init__(self, num_feat: int, d_model: int = 32, num_blocks: int = 3,
                 dilations: List[int] = (1, 2, 4, 8)):
        """
        num_feat : 特征数
        d_model  : 隐藏通道数
        num_blocks : TimesBlock 层数
        dilations  : 每个 block 内的一组扩张卷积
        """
        super().__init__()
        self.embed = nn.Conv1d(num_feat, d_model, 1)   # 通道映射
        blocks = [TimesBlock(d_model, dilations) for _ in range(num_blocks)]
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Conv1d(d_model, num_feat, 1)    # 输出回到原特征维

    def forward(self, x):              # x: (B, L, F)
        x = x.transpose(1, 2)          # -> (B, F, L)
        x = self.embed(x)
        x = self.blocks(x)
        x = self.head(x)
        x = x[:, :, -1]                # 取最后一步
        return x                       # (B, F)


# ------- 训练 / 预测 接口 -------------
def train_timesnet(
    X_train,
    y_train,
    X_val,
    y_val,
    device: str = "cpu",
    lr: float = 1e-3,
    epochs: int = 50,
    batch_size: int = 16,
    d_model: int = 32,
    num_blocks: int = 3,
) -> Tuple[nn.Module, float]:
    """TimesNet 训练，返回模型与验证集 MSE。"""
    train_loader = DataLoader(SeqDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(SeqDataset(X_val, y_val), batch_size=batch_size)

    model = TimesNet(num_feat=X_train.shape[-1], d_model=d_model, num_blocks=num_blocks).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best = math.inf
    for _ in range(epochs):
        # --- train ---
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optim.step()

        # --- validate ---
        model.eval(); vloss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                vloss += criterion(model(xb), yb).item() * xb.size(0)
        vloss /= len(val_loader.dataset)
        best = min(best, vloss)

    return model, best


def predict(model: nn.Module, seq, device: str = "cpu"):
    """seq shape = (look_back, F)"""
    model.eval()
    with torch.no_grad():
        seq = _to_tensor(seq, device).unsqueeze(0)      # (1, L, F)
        return model(seq).cpu().numpy().squeeze()
