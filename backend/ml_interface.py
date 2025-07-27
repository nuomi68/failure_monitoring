
"""
ml_interface.py

统一接口层（仅维护一个模型状态）：
- 前端只需下发指令：train / predict / save / load / clear / get_meta
- 新的训练会覆盖（替换）旧模型状态
- 监督与无监督的具体算法实现放在 supervised_core / unsupervised_core
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Protocol, runtime_checkable, Literal

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score,
)
import joblib

# 引入适配器
from supervised_core import ADAPTERS as SUPERVISED_ADAPTERS
from unsupervised_core import ADAPTERS as UNSUPERVISED_ADAPTERS


# ---------------------------- 数据结构 ----------------------------
@dataclass
class ModelArtifact:
    model: Any
    scaler: Optional[Any]
    meta: Dict[str, Any]


@dataclass
class TrainReport:
    y_true: Optional[np.ndarray] = None
    y_pred: Optional[np.ndarray] = None
    scores: Optional[np.ndarray] = None
    metrics_text: str = ""


# ---------------------------- 协议与注册表 ----------------------------
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

_REGISTRY: dict[str, AlgoAdapter] = {}
def register(adapter: AlgoAdapter): _REGISTRY[adapter.code] = adapter
def get_adapter(code: str) -> AlgoAdapter:
    if code not in _REGISTRY:
        raise KeyError(f"Unknown algorithm code: {code}. Registered: {list(_REGISTRY)}")
    return _REGISTRY[code]

for _ad in [*SUPERVISED_ADAPTERS, *UNSUPERVISED_ADAPTERS]:
    register(_ad)


# ---------------------------- 单例模型状态 ----------------------------
class _State:
    current: Optional[ModelArtifact] = None

STATE = _State()


# ---------------------------- 工具函数 ----------------------------
def _scale_features(X: np.ndarray, scaler: Optional[Any] = None):
    scaler = scaler or StandardScaler()
    Xs = scaler.fit_transform(X) if getattr(scaler, "mean_", None) is None else scaler.transform(X)
    return Xs, scaler


# ---------------------------- 训练与预测实现 ----------------------------
def _train_impl(
    alg: str,
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    *,
    params: Optional[Dict[str, Any]] = None,
    scale: bool = True,
    test_size: float = 0.2,
    random_state: int = 0,
    stratify: Optional[np.ndarray] = None,
) -> Tuple[ModelArtifact, TrainReport]:
    params = dict(params or {})
    adapter = get_adapter(alg)

    # 预处理
    if scale: Xs, scaler = _scale_features(X)
    else:     Xs, scaler = X, None

    if adapter.kind.startswith("supervised"):
        if y is None: raise ValueError("监督学习需要提供 y")
        strat = y if (adapter.kind == "supervised_clf" and stratify is None and len(np.unique(y)) > 1) else stratify
        X_tr, X_te, y_tr, y_te = train_test_split(Xs, y, test_size=test_size, random_state=random_state, stratify=strat)

        model = adapter.build(**params)
        model = adapter.fit(model, X_tr, y_tr)

        y_pred = adapter.predict(model, X_te)
        try:
            scores = adapter.scores(model, X_te, classes_=getattr(model, "classes_", None))
        except Exception:
            scores = None

        metrics_text = ""
        if adapter.kind == "supervised_clf":
            parts = []
            if scores is not None and len(np.unique(y)) == 2:
                try:
                    auc = roc_auc_score(y_te, scores); parts.append(f"AUC={auc:.3f}")
                except Exception: pass
            try:
                acc = accuracy_score(y_te, y_pred); parts.append(f"Accuracy={acc:.3f}")
                prec = precision_score(y_te, y_pred, average="binary" if len(np.unique(y_te))==2 else "macro", zero_division=0); parts.append(f"Precision={prec:.3f}")
                rec = recall_score(y_te, y_pred, average="binary" if len(np.unique(y_te))==2 else "macro", zero_division=0); parts.append(f"Recall={rec:.3f}")
                f1  = f1_score(y_te, y_pred, average="binary" if len(np.unique(y_te))==2 else "macro", zero_division=0); parts.append(f"F1={f1:.3f}")
            except Exception: pass
            metrics_text = " | ".join(parts)
        else:
            try:
                mae = mean_absolute_error(y_te, y_pred)
                mse = mean_squared_error(y_te, y_pred)
                r2  = r2_score(y_te, y_pred)
                metrics_text = f"MAE={mae:.3f} | MSE={mse:.3f} | R2={r2:.3f}"
            except Exception: pass

        tau = adapter.default_tau(None) if adapter.kind == "supervised_clf" and len(np.unique(y_tr)) == 2 else None

        meta = {
            "model_type": adapter.meta_model_type(),
            "tau": tau,
            "advanced": params,
            "classes_": getattr(model, "classes_", None),
        }
        art = ModelArtifact(model=model, scaler=scaler, meta=meta)
        rep = TrainReport(y_true=y_te, y_pred=y_pred, scores=scores, metrics_text=metrics_text)
        return art, rep

    # 无监督（包含 AE）
    model = adapter.build(**params)
    model = adapter.fit(model, Xs, None)
    scores = adapter.scores(model, Xs)
    tau = adapter.default_tau(scores)
    meta = {"model_type": adapter.meta_model_type(), "tau": tau, "advanced": params}
    art = ModelArtifact(model=model, scaler=scaler, meta=meta)
    rep = TrainReport(scores=scores)
    return art, rep


def _predict_impl(artifact: ModelArtifact, X: np.ndarray):
    Xs = artifact.scaler.transform(X) if artifact.scaler is not None else X
    mtype = artifact.meta.get("model_type")
    mapping = {"knn_clf":"knn_clf","rf":"rf_clf","knn_reg":"knn_reg","rf_reg":"rf_reg","knn":"knn","iforest":"iforest","autoencoder":"autoencoder"}
    alg = mapping.get(mtype, mtype)
    adapter = get_adapter(alg)

    if adapter.kind == "unsupervised":
        return np.array([]), adapter.scores(artifact.model, Xs)

    y_pred = adapter.predict(artifact.model, Xs)
    try:
        scores = adapter.scores(artifact.model, Xs, classes_=artifact.meta.get("classes_"))
    except Exception:
        scores = None
    return y_pred, scores


# ---------------------------- 对外 API（单例） ----------------------------
class ML:
    """
    单例式接口：不需要在前端传递/管理模型引用。
    - train(...) 训练并替换当前模型
    - predict(X) 在当前模型上预测
    - save(path) / load(path) 保存与加载当前模型
    - run(action=...) 简单指令式调度
    """
    @classmethod
    def train(cls, alg: str, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> TrainReport:
        art, rep = _train_impl(alg=alg, X=X, y=y, **kwargs)
        STATE.current = art  # 覆盖旧模型
        return rep

    @classmethod
    def predict(cls, X: np.ndarray):
        if STATE.current is None:
            raise RuntimeError("当前没有已训练/加载的模型。请先调用 train(...) 或 load(...)。")
        return _predict_impl(STATE.current, X)

    @classmethod
    def get_meta(cls) -> Dict[str, Any]:
        if STATE.current is None: return {}
        return dict(STATE.current.meta)

    @classmethod
    def save(cls, path: str) -> None:
        if STATE.current is None:
            raise RuntimeError("没有可保存的模型。")
        save_artifact(path, STATE.current)

    @classmethod
    def load(cls, path: str) -> Dict[str, Any]:
        art = load_artifact(path)
        STATE.current = art
        return dict(art.meta)

    @classmethod
    def clear(cls) -> None:
        STATE.current = None

    @classmethod
    def run(cls, action: str, **kwargs):
        """
        指令式入口：
        - train: 需要 alg, X[, y], 其余参数同 train(...)
        - predict: 需要 X
        - save: 需要 path
        - load: 需要 path
        - clear: 无参数
        - get_meta: 无参数
        """
        action = action.lower()
        if action == "train":
            return cls.train(**kwargs)
        if action == "predict":
            return cls.predict(**kwargs)
        if action == "save":
            cls.save(**kwargs); return {"ok": True}
        if action == "load":
            meta = cls.load(**kwargs); return {"ok": True, "meta": meta}
        if action == "clear":
            cls.clear(); return {"ok": True}
        if action == "get_meta":
            return cls.get_meta()
        raise ValueError(f"未知指令: {action}")


# ---------------------------- 持久化 ----------------------------
def save_artifact(path: str, artifact: ModelArtifact) -> None:
    obj = {"model": artifact.model, "scaler": artifact.scaler, "meta": artifact.meta, "_interface": "ml_interface.singleton.v1"}
    joblib.dump(obj, str(path))

def load_artifact(path: str) -> ModelArtifact:
    obj = joblib.load(str(path))
    return ModelArtifact(model=obj.get("model"), scaler=obj.get("scaler"), meta=obj.get("meta", {}))
