# multivariate_lstm_forecast.py
"""多变量时间序列预测对比：GRU (深度)、随机森林、**TimeGAN 数据增强**

脚本输出三组结果：

* **GRU** —— 端到端深度模型（原网络）
* **Random Forest** —— 滑动窗口展平 + 树模型
* **RF + TimeGAN** —— 用 *YData‑Synthetic* TimeGAN 生成等量**合成样本**扩充训练集，再训练随机森林

> 需额外安装：
> ```bash
> pip install ydata-synthetic==1.3.1 tensorflow==2.15.0
> ```
"""

import math
from typing import Tuple
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


# --------------------------------------------------
# ⚙️ 参数配置
# --------------------------------------------------
CSV_PATH   = r"E:\failure_monitoring\data\20230510-20240924_merged.xlsx"
TIME_COL   = "TIME"
LOOK_BACK_DEEP  = 14   # GRU 滑窗
LOOK_BACK_TREE  = 5    # 随机森林滑窗
TEST_SIZE  = 0.2
BATCH_SIZE = 16
HIDDEN_SIZE = 32
NUM_LAYERS = 2
EPOCHS     = 200
LR         = 1e-3
PATIENCE   = 20
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------------------------
# 1️⃣ 读取 & 清洗数据
# --------------------------------------------------

def load_dataframe(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    df.columns = df.columns.str.strip()
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], format="%Y年%m月%d日%H%M")
    return df.sort_values(TIME_COL).reset_index(drop=True)

df_raw = load_dataframe(CSV_PATH)
print("Loaded data →", df_raw.shape)

features = (
    df_raw.drop(columns=[TIME_COL,"值","XE-133","CS-137"])
          .apply(pd.to_numeric, errors="coerce")
          .fillna(method="ffill")
          .fillna(0)
)
feature_names = features.columns.tolist()
print("预测列数:", len(feature_names))

# --------------------------------------------------
# 2️⃣ 标准化
# --------------------------------------------------
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features.astype(np.float32))
if not np.isfinite(features_scaled).all():
    raise ValueError("归一化产生 NaN/Inf，请检查原始数据！")

# --------------------------------------------------
# 3️⃣ 通用滑窗
# --------------------------------------------------

def build_windows(data: np.ndarray, look_back: int):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i : i + look_back])
        y.append(data[i + look_back])
    return np.asarray(X, np.float32), np.asarray(y, np.float32)

# =====================================================================================
# A. GRU 深度模型
# =====================================================================================
# X_deep, y_deep = build_windows(features_scaled, LOOK_BACK_DEEP)
# X_train_d, X_val_d, y_train_d, y_val_d = train_test_split(
#     X_deep, y_deep, test_size=TEST_SIZE, shuffle=False
# )
#
# class SeqDataset(Dataset):
#     def __init__(self, X, y):
#         self.X = torch.tensor(X)
#         self.y = torch.tensor(y)
#     def __len__(self):
#         return len(self.X)
#     def __getitem__(self, idx):
#         return self.X[idx], self.y[idx]
#
# dtrain = DataLoader(SeqDataset(X_train_d, y_train_d), batch_size=BATCH_SIZE, shuffle=True)
# dval   = DataLoader(SeqDataset(X_val_d,   y_val_d),   batch_size=BATCH_SIZE)
#
# class GRUForecaster(nn.Module):
#     def __init__(self, num_feat, hid, layers):
#         super().__init__()
#         self.gru = nn.GRU(num_feat, hid, layers, batch_first=True, dropout=0.3)
#         self.fc  = nn.Linear(hid, num_feat)
#     def forward(self, x):
#         out, _ = self.gru(x)
#         return self.fc(out[:, -1])
#
# model_gru = GRUForecaster(len(feature_names), HIDDEN_SIZE, NUM_LAYERS).to(DEVICE)
# criterion = nn.MSELoss()
# optimiser = torch.optim.Adam(model_gru.parameters(), lr=LR)
#
# print("\n[GRU] 开始训练 …")
# best = math.inf; wait = 0
# for epoch in range(1, EPOCHS + 1):
#     # 训练
#     model_gru.train(); tloss = 0
#     for xb, yb in dtrain:
#         xb, yb = xb.to(DEVICE), yb.to(DEVICE)
#         optimiser.zero_grad()
#         loss = criterion(model_gru(xb), yb)
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model_gru.parameters(), 5.0)
#         optimiser.step()
#         tloss += loss.item() * xb.size(0)
#     tloss /= len(dtrain.dataset)
#     # 验证
#     model_gru.eval(); vloss = 0
#     with torch.no_grad():
#         for xb, yb in dval:
#             xb, yb = xb.to(DEVICE), yb.to(DEVICE)
#             vloss += criterion(model_gru(xb), yb).item() * xb.size(0)
#     vloss /= len(dval.dataset)
#     print(f"Epoch {epoch:03d} | Train={tloss:.4f} | Val={vloss:.4f}")
#     if vloss < best - 1e-6:
#         best = vloss; wait = 0
#         torch.save(model_gru.state_dict(), "best_gru.pt")
#     else:
#         wait += 1
#         if wait >= PATIENCE:
#             print("[GRU] 早停，最佳 Val=", best)
#             break
# model_gru.load_state_dict(torch.load("best_gru.pt"))
#
# with torch.no_grad():
#     preds_g_val = model_gru(torch.tensor(X_val_d).to(DEVICE)).cpu().numpy()
# mae_g = mean_absolute_error(
#     scaler.inverse_transform(y_val_d),
#     scaler.inverse_transform(preds_g_val)
# )
# print(f"[GRU] 验证 MAE={mae_g:.4f}")
#
# # 下一步预测 (GRU)
# with torch.no_grad():
#     next_g_scaled = model_gru(torch.tensor(features_scaled[-LOOK_BACK_DEEP:]).unsqueeze(0).to(DEVICE)).cpu().numpy().squeeze()
# next_g_orig = scaler.inverse_transform(next_g_scaled.reshape(1, -1)).squeeze()
# series_gru = pd.Series(next_g_orig, index=feature_names, name="GRU")
# =====================================================================================
# A. TCN
# =====================================================================================

from torch.nn.utils import weight_norm

# 1) 数据窗口（沿用 LOOK_BACK_DEEP）
X_tcn, y_tcn = build_windows(features_scaled[:-5], LOOK_BACK_DEEP)
X_train_c, X_val_c, y_train_c, y_val_c = train_test_split(
    X_tcn, y_tcn, test_size=TEST_SIZE, shuffle=False
)

class SeqDatasetTCN(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X).permute(0, 2, 1)   # → (N, F, L)
        self.y = torch.tensor(y)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

dtrain_c = DataLoader(SeqDatasetTCN(X_train_c, y_train_c), batch_size=BATCH_SIZE, shuffle=True)
dval_c   = DataLoader(SeqDatasetTCN(X_val_c,   y_val_c), batch_size=BATCH_SIZE)

# ---------- 网络模块 ----------
class CausalConv1d(nn.Module):
    """左侧填充的因果卷积"""
    def __init__(self, in_chan, out_chan, k, d):
        super().__init__()
        self.pad = (k - 1) * d
        self.conv = weight_norm(
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
            nn.Dropout(drop)
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
        self.fc  = nn.Linear(in_c, num_feat)
    def forward(self, x):                 # x: (B, F, L)
        y = self.tcn(x)                   # (B, C, L)
        last = y[:, :, -1]                # (B, C)
        return self.fc(last)              # (B, F)

model_tcn = TCNForecaster(len(feature_names)).to(DEVICE)
criterion = nn.MSELoss()
optimiser = torch.optim.Adam(model_tcn.parameters(), lr=LR)

print("\n[TCN] 开始训练 …")
best = math.inf; wait = 0
for epoch in range(1, EPOCHS + 1):
    # 训练
    model_tcn.train(); tloss = 0
    for xb, yb in dtrain_c:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimiser.zero_grad()
        loss = criterion(model_tcn(xb), yb)
        loss.backward()
        optimiser.step()
        tloss += loss.item() * xb.size(0)
    tloss /= len(dtrain_c.dataset)
    # 验证
    model_tcn.eval(); vloss = 0
    with torch.no_grad():
        for xb, yb in dval_c:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            vloss += criterion(model_tcn(xb), yb).item() * xb.size(0)
    vloss /= len(dval_c.dataset)
    print(f"Epoch {epoch:03d} | Train={tloss:.4f} | Val={vloss:.4f}")
    if vloss < best - 1e-6:
        best = vloss; wait = 0
        torch.save(model_tcn.state_dict(), "best_tcn.pt")
    else:
        wait += 1
        if wait >= PATIENCE:
            print("[TCN] 早停，最佳 Val=", best)
            break
model_tcn.load_state_dict(torch.load("best_tcn.pt"))

# 验证 MAE
with torch.no_grad():
    preds_c_val = model_tcn(torch.tensor(X_val_c).permute(0, 2, 1).to(DEVICE)).cpu().numpy()
mae_c = mean_absolute_error(
    scaler.inverse_transform(y_val_c),
    scaler.inverse_transform(preds_c_val)
)
print(f"[TCN] 验证 MAE={mae_c:.4f}")

# 下一步预测 (TCN)
with torch.no_grad():
    last_seq = torch.tensor(features_scaled[-LOOK_BACK_DEEP-1:-1]).T.unsqueeze(0).to(DEVICE)  # (1, F, L)
    next_c_scaled = model_tcn(last_seq).cpu().numpy().squeeze()
next_c_orig = scaler.inverse_transform(next_c_scaled.reshape(1, -1)).squeeze()
series_tcn = pd.Series(next_c_orig, index=feature_names, name="TCN")
# =====================================================================================
# B. 随机森林  (baseline)
# =====================================================================================
'''
X_tree, y_tree = build_windows(features_scaled, LOOK_BACK_TREE)
cut = int(len(X_tree) * (1 - TEST_SIZE))
X_train_t, X_val_t = X_tree[:cut], X_tree[cut:]
y_train_t, y_val_t = y_tree[:cut], y_tree[cut:]

rf_base = MultiOutputRegressor(RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1))
rf_base.fit(X_train_t.reshape(len(X_train_t), -1), y_train_t)

pred_t_val = rf_base.predict(X_val_t.reshape(len(X_val_t), -1))
mae_t = mean_absolute_error(
    scaler.inverse_transform(y_val_t),
    scaler.inverse_transform(pred_t_val)
)
print(f"[RF]  验证 MAE={mae_t:.4f}")

# 下一步预测 (RF)
last_flat = features_scaled[-LOOK_BACK_TREE-1:-1].flatten().reshape(1, -1)
next_t_scaled = rf_base.predict(last_flat).squeeze()
next_t_orig = scaler.inverse_transform(next_t_scaled.reshape(1, -1)).squeeze()
series_rf = pd.Series(next_t_orig, index=feature_names, name="RandomForest")

from xgboost import XGBRegressor
rf_base = MultiOutputRegressor(
    XGBRegressor(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective='reg:squarederror',
        tree_method='hist'       # GPU: 'gpu_hist'
    )
)
rf_base.fit(X_train_t.reshape(len(X_train_t), -1), y_train_t)
pred_t_val = rf_base.predict(X_val_t.reshape(len(X_val_t), -1))
mae_t = mean_absolute_error(
    scaler.inverse_transform(y_val_t),
    scaler.inverse_transform(pred_t_val)
)
print(f"[XG]  验证 MAE={mae_t:.4f}")

# 下一步预测 (RF)
last_flat = features_scaled[-LOOK_BACK_TREE-1:-1].flatten().reshape(1, -1)
next_t_scaled = rf_base.predict(last_flat).squeeze()
next_t_orig = scaler.inverse_transform(next_t_scaled.reshape(1, -1)).squeeze()
series_xg = pd.Series(next_t_orig, index=feature_names, name="XG")
'''

# =====================================================================================
# D. TimeMixer
# =====================================================================================
# 自动取窗口长度 L 和特征数 F
SEQ_LEN  = X_train_c.shape[1]        # 时间步 L (=14)
NUM_FEAT = X_train_c.shape[2]        # 特征维 F (=26)

# ------ Dataset -----------------------------------------------------------
class SeqDatasetMixer(Dataset):
    """TSMixer 用 (B, L, F)；label 直接 (B, F)"""
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)   # (N, L, F)
        self.y = torch.tensor(y, dtype=torch.float32)   # (N, F)
    def __len__(self):  return len(self.X)
    def __getitem__(self, idx):  return self.X[idx], self.y[idx]

dtrain_m = DataLoader(SeqDatasetMixer(X_train_c, y_train_c),
                      batch_size=BATCH_SIZE, shuffle=True)
dval_m   = DataLoader(SeqDatasetMixer(X_val_c,   y_val_c),
                      batch_size=BATCH_SIZE)
'''
# ------ Model -------------------------------------------------------------
from torchtsmixer import TSMixer
model_mix = TSMixer(
    sequence_length   = SEQ_LEN,      # 14
    prediction_length = 1,            # 只预测下一步
    input_channels    = NUM_FEAT,     # 26
    output_channels   = NUM_FEAT,
    num_blocks        = 4,
    ff_dim            = 128,
    dropout_rate      = 0.1
).to(DEVICE)

criterion = nn.MSELoss()
optimiser  = torch.optim.AdamW(model_mix.parameters(), lr=LR)

# ------ Train / Early-stop -----------------------------------------------
print("\n[TSMixer] 开始训练 …")
best = float("inf"); wait = 0
for epoch in range(1, EPOCHS + 1):
    # training
    model_mix.train(); tloss = 0
    for xb, yb in dtrain_m:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimiser.zero_grad()
        pred = model_mix(xb).squeeze(1)   # (B, F)
        loss = criterion(pred, yb)
        loss.backward(); optimiser.step()
        tloss += loss.item() * xb.size(0)
    tloss /= len(dtrain_m.dataset)

    # validation
    model_mix.eval(); vloss = 0
    with torch.no_grad():
        for xb, yb in dval_m:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred   = model_mix(xb).squeeze(1)
            vloss += criterion(pred, yb).item() * xb.size(0)
    vloss /= len(dval_m.dataset)
    print(f"Epoch {epoch:03d} | Train={tloss:.4f} | Val={vloss:.4f}")

    if vloss < best - 1e-6:
        best = vloss; wait = 0
        torch.save(model_mix.state_dict(), "best_tsmixer.pt")
    else:
        wait += 1
        if wait >= PATIENCE:
            print("[TSMixer] 早停，最佳 Val =", best)
            break
model_mix.load_state_dict(torch.load("best_tsmixer.pt"))

# ------ 验证 MAE ----------------------------------------------------------
with torch.no_grad():
    preds_val = model_mix(torch.tensor(X_val_c).to(DEVICE)).cpu().squeeze(1).numpy()
mae_mix = mean_absolute_error(
    scaler.inverse_transform(y_val_c),
    scaler.inverse_transform(preds_val)
)
print(f"[TSMixer] 验证 MAE = {mae_mix:.4f}")


def compute_relative_errors(compare_df: pd.DataFrame):
    """
    传入一棵 DataFrame，第一列是真实值，后面每一列都是不同模型的预测值。
    返回：
      - rel_err_df: 和 compare_df 同形状的相对误差表 (abs(pred - true) / true)
      - max_err: 每个模型（列）的最大相对误差（Series）
      - mean_err: 每个模型的平均相对误差（Series）
    """
    true = compare_df.iloc[:, 0]
    preds = compare_df.iloc[:, 1:]
    # 避免除以 0
    denom = true.replace(0, np.nan)
    rel_err_df = (preds.sub(true, axis=0).abs()
                        .div(denom, axis=0))
    max_err = rel_err_df.max()
    mean_err = rel_err_df.mean()
    return rel_err_df, max_err, mean_err

# ------ 下一步预测 ---------------------------------------------------------
# 下一步预测 (TCN)
next_series=[]
with torch.no_grad():
    for i in range(1,6):
        last_seq = torch.tensor(features_scaled[-LOOK_BACK_DEEP-i:-i]).T.unsqueeze(0).to(DEVICE)  # (1, F, L)
        next_c_scaled = model_tcn(last_seq).cpu().numpy().squeeze()
        next_c_orig = scaler.inverse_transform(next_c_scaled.reshape(1, -1)).squeeze()
        series_tcn = pd.Series(next_c_orig, index=feature_names, name="TCN")
        last_seq = torch.tensor(features_scaled[-SEQ_LEN - i:-i]).unsqueeze(0).to(DEVICE)  # (1, F, L)
        next_scaled = model_mix(last_seq).cpu().squeeze().numpy()
        next_orig = scaler.inverse_transform(next_scaled.reshape(1, -1)).squeeze()
        series_mix = pd.Series(next_orig, index=feature_names, name="TSMixer")
        next_series=[features.iloc[-i],series_tcn, series_mix]
        compare_df = pd.concat(next_series, axis=1)

        rel_err_df, max_err, mean_err = compute_relative_errors(compare_df)
        conc = pd.concat([compare_df,rel_err_df], axis=1)
        print(f"\n———————— 第{i}步预测结果 ————————")
        conc.columns = ["True", "TCN", "Mix", "TCN_err", "Mix_err"]
        print(compare_df)

        print("\n最大相对误差：")
        print(max_err)
        print("\n平均相对误差：")
        print(mean_err)

'''

from transformers import PatchTSTConfig, PatchTSTForRegression
import torch
from torch.utils.data import DataLoader, Dataset
import math
from sklearn.metrics import mean_absolute_error
import numpy as np

# ---------------------------------------------------------------------
# ⚙️ 超参
# ---------------------------------------------------------------------
PATCH_LEN   = 8          # 每个 patch 时间步
PATCH_STRIDE = 8         # patch 滑动步长 (= PATCH_LEN 通常即可)
D_MODEL     = 128        # Transformer hidden dim
N_LAYERS    = 4          # Transformer blocks
N_HEADS     = 8          # MH‑Attention heads
FF_DIM      = 256        # Feed‑forward dim
DROPOUT     = 0.1
EPOCHS      = 200
LR          = 1e-3
PATIENCE    = 30
BATCH_SIZE  = 32

# ---------------------------------------------------------------------
# 🗂️ 数据集  (B, L, F) ➞ target (B, F)
# ---------------------------------------------------------------------
class SeqDatasetPTST(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)      # (N, L, F)
        self.y = torch.tensor(y, dtype=torch.float32)      # (N, F)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dtrain_p = DataLoader(SeqDatasetPTST(X_train_c, y_train_c), batch_size=BATCH_SIZE, shuffle=True)
dval_p   = DataLoader(SeqDatasetPTST(X_val_c,   y_val_c), batch_size=BATCH_SIZE)

# ---------------------------------------------------------------------
# 🧠 模型
# ---------------------------------------------------------------------
config = PatchTSTConfig(
    num_input_channels = NUM_FEAT,
    num_targets        = NUM_FEAT,      # 多变量回归 — 每列都回归
    context_length     = SEQ_LEN,       # look‑back
    prediction_length  = 1,             # 只预测下一步
    patch_length       = PATCH_LEN,
    patch_stride       = PATCH_STRIDE,
    d_model            = D_MODEL,
    num_hidden_layers  = N_LAYERS,
    num_attention_heads= N_HEADS,
    ffn_dim            = FF_DIM,
    dropout            = DROPOUT,
    loss               = "mse",
    channel_attention  = False          # True = CT‑PatchTST (channels attend across)
)

model_ptst = PatchTSTForRegression(config).to(DEVICE)
optimiser   = torch.optim.AdamW(model_ptst.parameters(), lr=LR)

# ---------------------------------------------------------------------
# 🏃 训练 with early‑stop + LR sched
# ---------------------------------------------------------------------
best = math.inf; wait = 0
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, factor=0.5, patience=10)
print("\n[PatchTST] 开始训练 …")
for epoch in range(1, EPOCHS + 1):
    # Train
    model_ptst.train(); tloss = 0
    for xb, yb in dtrain_p:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimiser.zero_grad()
        out = model_ptst(past_values=xb, target_values=yb)
        loss = out.loss
        loss.backward(); optimiser.step()
        tloss += loss.item() * xb.size(0)
    tloss /= len(dtrain_p.dataset)

    # Val
    model_ptst.eval(); vloss = 0
    with torch.no_grad():
        for xb, yb in dval_p:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            out = model_ptst(past_values=xb, target_values=yb)
            vloss += out.loss.item() * xb.size(0)
    vloss /= len(dval_p.dataset)
    scheduler.step(vloss)

    print(f"Epoch {epoch:03d} | Train={tloss:.4f} | Val={vloss:.4f}")

    # Early stop
    if vloss < best - 1e-6:
        best = vloss; wait = 0
        torch.save(model_ptst.state_dict(), "best_patchtst.pt")
    else:
        wait += 1
        if wait >= PATIENCE:
            print("[PatchTST] 早停，最佳 Val=", best)
            break

# ---------------------------------------------------------------------
# ✅ 评估 MAE
# ---------------------------------------------------------------------
model_ptst.load_state_dict(torch.load("best_patchtst.pt"))
with torch.no_grad():
    preds_val = []
    for xb, _ in dval_p:
        xb = xb.to(DEVICE)
        out = model_ptst(past_values=xb)
        preds_val.append(out.regression_outputs.cpu())
    preds_val = torch.cat(preds_val).numpy()

mae_ptst = mean_absolute_error(
    scaler.inverse_transform(y_val_c),
    scaler.inverse_transform(preds_val)
)
print(f"[PatchTST] 验证 MAE = {mae_ptst:.4f}")

# ---------------------------------------------------------------------
# 🔮 下一步预测
# ---------------------------------------------------------------------
with torch.no_grad():
    last_seq = torch.tensor(features_scaled[-SEQ_LEN-1:-1]).unsqueeze(0).to(DEVICE)  # (1, L, F)
    next_scaled = model_ptst(past_values=last_seq).regression_outputs.cpu().numpy().squeeze()
next_orig = scaler.inverse_transform(next_scaled.reshape(1, -1)).squeeze()
series_ptst = pd.Series(next_orig, index=feature_names, name="PatchTST")

print("下一步预测 (PatchTST):")
print(series_ptst.head())

