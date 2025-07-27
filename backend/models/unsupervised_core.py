
"""
unsupervised_core.py

无监督学习核心适配器：KNN / IsolationForest / AutoEncoder
- 由 ml_interface 注册并调度
- AutoEncoder 的**模型结构与训练循环**在本文件内实现（避免接口层特殊分支）
"""

from __future__ import annotations
from typing import Any, Optional, Literal, Protocol, runtime_checkable
import numpy as np


@runtime_checkable
class AlgoAdapter(Protocol):
    code: str
    kind: Literal["supervised_clf", "supervised_reg", "unsupervised"]
    def build(self, **params) -> Any: ...
    def fit(self, model: Any, X: np.ndarray, y: Optional[np.ndarray] = None) -> Any: ...
    def predict(self, model: Any, X: np.ndarray) -> np.ndarray: ...
    def scores(self, model: Any, X: np.ndarray, *, classes_: Optional[np.ndarray] = None) -> Optional[np.ndarray]: ...
    def default_tau(self, scores: Optional[np.ndarray], *, classes_: Optional[np.ndarray] = None) -> Optional[float]: ...
    def meta_model_type(self) -> str: ...


class _KNNUnAdapter:
    code = "knn"
    kind: Literal["unsupervised"] = "unsupervised"

    def build(self, **params):
        from sklearn.neighbors import NearestNeighbors
        return NearestNeighbors(**params)

    def fit(self, model, X, y=None):
        model.fit(X); return model

    def predict(self, model, X):
        # 无监督不预测离散标签
        return np.array([])

    def scores(self, model, X, *, classes_: Optional[np.ndarray] = None):
        dists = model.kneighbors(X)[0][:, -1]
        return dists

    def default_tau(self, scores, *, classes_: Optional[np.ndarray] = None):
        return float(np.quantile(scores, 0.95)) if scores is not None else None

    def meta_model_type(self) -> str:
        return "knn"


class _IForestAdapter:
    code = "iforest"
    kind: Literal["unsupervised"] = "unsupervised"

    def build(self, **params):
        from sklearn.ensemble import IsolationForest
        return IsolationForest(**params)

    def fit(self, model, X, y=None):
        model.fit(X); return model

    def predict(self, model, X):
        return np.array([])

    def scores(self, model, X, *, classes_: Optional[np.ndarray] = None):
        return -model.decision_function(X)

    def default_tau(self, scores, *, classes_: Optional[np.ndarray] = None):
        return float(np.quantile(scores, 0.95)) if scores is not None else None

    def meta_model_type(self) -> str:
        return "iforest"


class _AutoEncoderAdapter:
    """
    使用 PyTorch 的简单全连接自编码器：
    - scores(X) 返回逐样本 MSE 重构误差
    - default_tau(scores) 取 95% 分位
    需要安装 torch。
    """
    code = "autoencoder"
    kind: Literal["unsupervised"] = "unsupervised"

    def build(self, **params):
        try:
            import torch  # noqa: F401
        except Exception as e:
            raise RuntimeError("Autoencoder 需要安装 PyTorch") from e

        cfg = {
            "hidden": params.get("hidden", [64, 32]),
            "latent_dim": params.get("latent_dim", 16),
            "dropout": params.get("dropout", 0.0),
        }
        # 在 fit() 中会基于输入维度初始化具体层；此处先返回配置占位
        return {"_ae_cfg": cfg, "_ae_train": {
            "epochs": int(params.get("epochs", 50)),
            "batch_size": int(params.get("batch_size", 128)),
            "lr": float(params.get("lr", 1e-3)),
        }}

    def _instantiate(self, input_dim: int, hidden: list[int], latent_dim: int, dropout: float = 0.0):
        import torch
        class _AE(torch.nn.Module):
            def __init__(self, input_dim: int, hidden: list[int], latent_dim: int, dropout: float = 0.0):
                super().__init__()
                enc, last = [], input_dim
                for h in hidden:
                    enc += [torch.nn.Linear(last, h), torch.nn.ReLU()]
                    if dropout > 0: enc += [torch.nn.Dropout(dropout)]
                    last = h
                enc += [torch.nn.Linear(last, latent_dim), torch.nn.ReLU()]

                dec, last = [], latent_dim
                for h in reversed(hidden):
                    dec += [torch.nn.Linear(last, h), torch.nn.ReLU()]; last = h
                dec += [torch.nn.Linear(last, input_dim)]
                self.encoder = torch.nn.Sequential(*enc)
                self.decoder = torch.nn.Sequential(*dec)

            def forward(self, x):
                z = self.encoder(x)
                return self.decoder(z)

        return _AE(input_dim, hidden, latent_dim, dropout)

    def fit(self, model, X, y=None):
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        cfg = model["_ae_cfg"]; train_cfg = model["_ae_train"]
        ae = self._instantiate(X.shape[1], list(cfg["hidden"]), int(cfg["latent_dim"]), float(cfg["dropout"]))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ae = ae.to(device)

        X_tensor = torch.tensor(X, dtype=torch.float32)
        ds = TensorDataset(X_tensor); dl = DataLoader(ds, batch_size=train_cfg["batch_size"], shuffle=True)
        opt = torch.optim.Adam(ae.parameters(), lr=train_cfg["lr"])
        loss_fn = torch.nn.MSELoss()

        ae.train()
        for _ in range(int(train_cfg["epochs"])):
            for (batch,) in dl:
                batch = batch.to(device)
                recon = ae(batch)
                loss = loss_fn(recon, batch)
                opt.zero_grad(); loss.backward(); opt.step()

        ae.eval()
        return ae

    def predict(self, model, X):
        # 无监督不输出离散标签
        return np.array([])

    def scores(self, model, X, *, classes_: Optional[np.ndarray] = None):
        import torch
        device = next(model.parameters()).device if hasattr(model, "parameters") else ("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            Xt = torch.tensor(X, dtype=torch.float32).to(device)
            recon = model(Xt).cpu().numpy()
        return np.mean((recon - X) ** 2, axis=1)

    def default_tau(self, scores, *, classes_: Optional[np.ndarray] = None):
        return float(np.quantile(scores, 0.95)) if scores is not None else None

    def meta_model_type(self) -> str:
        return "autoencoder"


ADAPTERS: list[AlgoAdapter] = [
    _KNNUnAdapter(),
    _IForestAdapter(),
    _AutoEncoderAdapter(),
]
