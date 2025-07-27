"""
ml_interface_singleton_scaled.py

统一接口层（仅维护一个模型状态）+ 可插拔归一化：
- 指令式：train / predict / transform / save / load / clear / get_meta
- 新训练覆盖旧模型
- 支持可选归一化器（standard/minmax/robust/maxabs/power/quantile/normalizer/none），并将拟合后的 scaler 与模型一起保存
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Protocol, runtime_checkable, Literal

import numpy as np
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler,
    PowerTransformer, QuantileTransformer, Normalizer
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score,
)
import joblib

# 引入适配器
from .models.supervised_core import ADAPTERS as SUPERVISED_ADAPTERS
from .models.unsupervised_core import ADAPTERS as UNSUPERVISED_ADAPTERS


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

    def default_tau(self, scores: Optional[np.ndarray], *, classes_: Optional[np.ndarray] = None) -> Optional[
        float]: ...

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


# ---------------------------- 归一化器工厂 ----------------------------
def _make_scaler(spec: Any):
    """
    支持：
      - None / True / "standard"（默认 StandardScaler）
      - "none" / False（不做缩放）
      - 字符串： "minmax" | "robust" | "maxabs" | "power" | "quantile" | "normalizer"
      - 字典： {"name": <同上>, "params": {...}}
    """
    if spec is None or spec is True:
        return StandardScaler()
    if spec is False or (isinstance(spec, str) and spec.lower() == "none"):
        return None

    params: Dict[str, Any] = {}
    if isinstance(spec, str):
        name = spec.lower()
    elif isinstance(spec, dict):
        name = str(spec.get("name", "standard")).lower()
        params = dict(spec.get("params", {}))
    else:
        raise ValueError(f"Unsupported scaler spec: {type(spec)}")

    if name in ("standard", "std", "zscore"):
        return StandardScaler(**params)
    if name in ("minmax", "min_max"):
        return MinMaxScaler(**params)
    if name in ("robust",):
        return RobustScaler(**params)
    if name in ("maxabs", "max_abs"):
        return MaxAbsScaler(**params)
    if name in ("power", "yeojohnson", "boxcox"):
        params = {"method": params.get("method", "yeo-johnson"), **params}
        return PowerTransformer(**params)
    if name in ("quantile", "rank"):
        return QuantileTransformer(**params)
    if name in ("normalizer", "l2", "l1", "max"):
        if name in ("l1", "l2", "max"):
            params = {"norm": name, **params}
        return Normalizer(**params)

    raise ValueError(f"Unknown scaler name: {name}")


def _fit_transform_supervised(X: np.ndarray, y: np.ndarray, scaler_spec: Any, *, test_size: float, random_state: int,
                              stratify: Optional[np.ndarray]):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)
    scaler = _make_scaler(scaler_spec)
    if scaler is not None:
        X_trs = scaler.fit_transform(X_tr)
        X_tes = scaler.transform(X_te)
    else:
        X_trs, X_tes = X_tr, X_te
    return (X_trs, X_tes, y_tr, y_te, scaler)


def _fit_transform_unsupervised(X: np.ndarray, scaler_spec: Any):
    scaler = _make_scaler(scaler_spec)
    if scaler is not None:
        Xs = scaler.fit_transform(X)
    else:
        Xs = X
    return Xs, scaler


# ---------------------------- 训练与预测实现 ----------------------------
def _train_impl(
        alg: str,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        *,
        params: Optional[Dict[str, Any]] = None,
        scaler: Any = None,
        test_size: float = 0.2,
        random_state: int = 0,
        stratify: Optional[np.ndarray] = None,
) -> Tuple[ModelArtifact, TrainReport]:
    params = dict(params or {})
    adapter = get_adapter(alg)

    if adapter.kind.startswith("supervised"):
        if y is None: raise ValueError("监督学习需要提供 y")
        strat = y if (adapter.kind == "supervised_clf" and stratify is None and len(np.unique(y)) > 1) else stratify
        X_tr, X_te, y_tr, y_te, scaler_obj = _fit_transform_supervised(
            X, y, scaler_spec=scaler, test_size=test_size, random_state=random_state, stratify=strat
        )
        # 任务信息
        n_classes = int(len(np.unique(y))) if adapter.kind == "supervised_clf" else None
        is_binary = (adapter.kind == "supervised_clf" and n_classes == 2)
        model = adapter.build(**params)
        model = adapter.fit(model, X_tr, y_tr)

        y_pred = adapter.predict(model, X_te)
        try:
            scores = adapter.scores(model, X_te, classes_=getattr(model, "classes_", None))
        except Exception:
            scores = None
        # 多分类不提供 scores（避免前端将其当“正类概率”使用）

        if adapter.kind == "supervised_clf" and not is_binary:
            scores = None
        metrics_text = ""
        if adapter.kind == "supervised_clf":
            parts = []
            if is_binary and scores is not None:
                try:
                    auc = roc_auc_score(y_te, scores)
                    parts.append(f"AUC={auc:.3f}")
                except Exception:
                    pass
            try:
                acc = accuracy_score(y_te, y_pred)
                parts.append(f"Accuracy={acc:.3f}")
                avg = "binary" if is_binary else "macro"
                prec = precision_score(y_te, y_pred, average=avg, zero_division=0)
                parts.append(f"Precision={prec:.3f}")
                rec = recall_score(y_te, y_pred, average=avg, zero_division=0)
                parts.append(f"Recall={rec:.3f}")
                f1 = f1_score(y_te, y_pred, average=avg, zero_division=0)
                parts.append(f"F1={f1:.3f}")
            except Exception:
                pass
            metrics_text = " | ".join(parts)
        else:
            try:
                mae = mean_absolute_error(y_te, y_pred)
                mse = mean_squared_error(y_te, y_pred)
                r2 = r2_score(y_te, y_pred)
                metrics_text = f"MAE={mae:.3f} | MSE={mse:.3f} | R2={r2:.3f}"
            except Exception:
                pass

        tau = adapter.default_tau(None) if is_binary else None

        meta = {
            "model_type": adapter.meta_model_type(),
            "tau": tau,
            "advanced": params,
            "classes_": getattr(model, "classes_", None),
            "scaler": type(scaler_obj).__name__ if scaler_obj is not None else "None",
        }
        art = ModelArtifact(model=model, scaler=scaler_obj, meta=meta)
        rep = TrainReport(y_true=y_te, y_pred=y_pred, scores=scores, metrics_text=metrics_text)
        return art, rep

    # 无监督
    Xs, scaler_obj = _fit_transform_unsupervised(X, scaler_spec=scaler)
    model = adapter.build(**params)
    model = adapter.fit(model, Xs, None)
    scores = adapter.scores(model, Xs)
    tau = adapter.default_tau(scores)
    meta = {"model_type": adapter.meta_model_type(), "tau": tau, "advanced": params,
            "scaler": type(scaler_obj).__name__ if scaler_obj is not None else "None"}
    art = ModelArtifact(model=model, scaler=scaler_obj, meta=meta)
    rep = TrainReport(scores=scores)
    return art, rep


def _predict_impl(artifact: ModelArtifact, X: np.ndarray):
    Xs = artifact.scaler.transform(X) if artifact.scaler is not None else X
    mtype = artifact.meta.get("model_type")
    mapping = {"knn_clf": "knn_clf", "rf": "rf_clf", "knn_reg": "knn_reg", "rf_reg": "rf_reg", "knn": "knn",
               "iforest": "iforest", "autoencoder": "autoencoder"}
    alg = mapping.get(mtype, mtype)
    adapter = get_adapter(alg)

    if adapter.kind == "unsupervised":
        return np.array([]), adapter.scores(artifact.model, Xs)

    y_pred = adapter.predict(artifact.model, Xs)
    try:
        scores = adapter.scores(artifact.model, Xs, classes_=artifact.meta.get("classes_"))
    except Exception:
        scores = None
    # 多分类不返回 scores

    if artifact.meta.get("task") == "supervised_clf" and not artifact.meta.get("is_binary", False):
        scores = None
    return y_pred, scores


# ---------------------------- 对外 API（单例） ----------------------------
class ML:
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
    def transform(cls, X: np.ndarray):
        """使用当前模型保存的 scaler 对 X 做变换；若无 scaler 则原样返回。"""
        if STATE.current is None:
            raise RuntimeError("当前没有已训练/加载的模型，无法进行归一化变换。")
        sc = STATE.current.scaler
        return (sc.transform(X) if sc is not None else X)

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
        action = action.lower()
        if action == "train":
            return cls.train(**kwargs)
        if action == "predict":
            return cls.predict(**kwargs)
        if action == "transform":
            return cls.transform(**kwargs)
        if action == "save":
            cls.save(**kwargs)
            return {"ok": True}
        if action == "load":
            meta = cls.load(**kwargs)
            return {"ok": True, "meta": meta}
        if action == "clear":
            cls.clear()
            return {"ok": True}
        if action == "get_meta":
            return cls.get_meta()
        raise ValueError(f"未知指令: {action}")


# ---------------------------- 持久化 ----------------------------
def save_artifact(path: str, artifact: ModelArtifact) -> None:
    obj = {"model": artifact.model, "scaler": artifact.scaler, "meta": artifact.meta,
           "_interface": "ml_interface.singleton.scaler.v1"}
    joblib.dump(obj, str(path))


def load_artifact(path: str) -> ModelArtifact:
    obj = joblib.load(str(path))
    return ModelArtifact(model=obj.get("model"), scaler=obj.get("scaler"), meta=obj.get("meta", {}))
