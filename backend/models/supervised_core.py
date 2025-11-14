
"""
supervised_core.py

只包含“监督学习”适配器（KNN/RF 的分类与回归）。
由 ml_interface.py 统一注册和调度。
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


class _KNNClfAdapter:
    code = "knn_clf"
    kind: Literal["supervised_clf"] = "supervised_clf"

    def build(self, **params):
        from sklearn.neighbors import KNeighborsClassifier
        return KNeighborsClassifier(**params)

    def fit(self, model, X, y=None):
        # 确保请求的邻居数量不超过可用的训练样本数量。
        # 如果 n_neighbors 大于拟合时的样本数量，KNeighborsClassifier 会在预测时抛出 ValueError。
        # 在这里进行调整，可以避免当用户指定过大的值时出现运行时错误。
        n_samples = X.shape[0]
        if getattr(model, "n_neighbors", None) is not None and model.n_neighbors > n_samples:
            model.set_params(n_neighbors=n_samples)
        model.fit(X, y)
        return model

    def predict(self, model, X):
        return model.predict(X)

    def scores(self, model, X, *, classes_: Optional[np.ndarray] = None):
        proba = model.predict_proba(X)
        cls = list(model.classes_)
        pos_idx = cls.index(1) if 1 in cls else 1 if len(cls) > 1 else 0
        return proba[:, pos_idx]

    def default_tau(self, scores, *, classes_: Optional[np.ndarray] = None):
        return 0.5

    def meta_model_type(self) -> str:
        return "knn_clf"


class _KNNRegAdapter:
    code = "knn_reg"
    kind: Literal["supervised_reg"] = "supervised_reg"

    def build(self, **params):
        from sklearn.neighbors import KNeighborsRegressor
        return KNeighborsRegressor(**params)

    def fit(self, model, X, y=None):
        # 确保请求的邻居数量不超过可用的训练样本数量。
        # 如果 n_neighbors 大于拟合时的样本数量，KNeighborsClassifier 会在预测时抛出 ValueError。
        # 在这里进行调整，可以避免当用户指定过大的值时出现运行时错误。
        n_samples = X.shape[0]
        if getattr(model, "n_neighbors", None) is not None and model.n_neighbors > n_samples:
            model.set_params(n_neighbors=n_samples)
        model.fit(X, y)
        return model

    def predict(self, model, X):
        return model.predict(X)

    def scores(self, model, X, *, classes_: Optional[np.ndarray] = None):
        return model.predict(X)

    def default_tau(self, scores, *, classes_: Optional[np.ndarray] = None):
        return None

    def meta_model_type(self) -> str:
        return "knn_reg"


class _RFClfAdapter:
    code = "rf_clf"
    kind: Literal["supervised_clf"] = "supervised_clf"

    def build(self, **params):
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(**params)

    def fit(self, model, X, y=None):
        model.fit(X, y); return model

    def predict(self, model, X):
        return model.predict(X)

    def scores(self, model, X, *, classes_: Optional[np.ndarray] = None):
        proba = model.predict_proba(X)
        cls = list(model.classes_)
        pos_idx = cls.index(1) if 1 in cls else 1 if len(cls) > 1 else 0
        return proba[:, pos_idx]

    def default_tau(self, scores, *, classes_: Optional[np.ndarray] = None):
        return 0.5

    def meta_model_type(self) -> str:
        # 兼容 ValidationPage：保持 "rf"
        return "rf"


class _RFRegAdapter:
    code = "rf_reg"
    kind: Literal["supervised_reg"] = "supervised_reg"

    def build(self, **params):
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(**params)

    def fit(self, model, X, y=None):
        model.fit(X, y); return model

    def predict(self, model, X):
        return model.predict(X)

    def scores(self, model, X, *, classes_: Optional[np.ndarray] = None):
        return model.predict(X)

    def default_tau(self, scores, *, classes_: Optional[np.ndarray] = None):
        return None

    def meta_model_type(self) -> str:
        return "rf_reg"


class _XGBClfAdapter:
    code = "xgb_clf"
    kind: Literal["supervised_clf"] = "supervised_clf"

    def build(self, **params):
        try:
            from xgboost import XGBClassifier  # type: ignore
        except ImportError as exc:  # pragma: no cover - import guard
            raise RuntimeError("缺少 xgboost 依赖，请先安装 `pip install xgboost`.") from exc

        cfg = params.copy()
        cfg.setdefault("use_label_encoder", False)
        cfg.setdefault("eval_metric", "logloss")
        if cfg.get("objective") == "auto":  # xgboost 不认识 auto
            cfg.pop("objective")
        return XGBClassifier(**cfg)

    def fit(self, model, X, y=None):
        if y is None:
            raise ValueError("XGBoost 分类需要 y")
        n_classes = len(np.unique(y))
        params = model.get_params()
        objective = params.get("objective")
        if n_classes > 2:
            if objective not in {"multi:softprob", "multi:softmax"}:
                model.set_params(objective="multi:softprob")
            model.set_params(num_class=n_classes)
        else:
            if objective is None:
                model.set_params(objective="binary:logistic")
        model.fit(X, y)
        return model

    def predict(self, model, X):
        return model.predict(X)

    def scores(self, model, X, *, classes_: Optional[np.ndarray] = None):
        if not hasattr(model, "predict_proba"):
            return None
        proba = model.predict_proba(X)
        cls = list(getattr(model, "classes_", []))
        if not cls:
            return None
        pos_cls = 1 if 1 in cls else (cls[1] if len(cls) > 1 else cls[0])
        pos_idx = cls.index(pos_cls)
        return proba[:, pos_idx]

    def default_tau(self, scores, *, classes_: Optional[np.ndarray] = None):
        return 0.5

    def meta_model_type(self) -> str:
        return "xgb_clf"


class _XGBRegAdapter:
    code = "xgb_reg"
    kind: Literal["supervised_reg"] = "supervised_reg"

    def build(self, **params):
        try:
            from xgboost import XGBRegressor  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("缺少 xgboost 依赖，请先安装 `pip install xgboost`.") from exc
        cfg = params.copy()
        cfg.setdefault("eval_metric", "rmse")
        return XGBRegressor(**cfg)

    def fit(self, model, X, y=None):
        if y is None:
            raise ValueError("XGBoost 回归需要 y")
        model.fit(X, y)
        return model

    def predict(self, model, X):
        return model.predict(X)

    def scores(self, model, X, *, classes_: Optional[np.ndarray] = None):
        return model.predict(X)

    def default_tau(self, scores, *, classes_: Optional[np.ndarray] = None):
        return None

    def meta_model_type(self) -> str:
        return "xgb_reg"


ADAPTERS: list[AlgoAdapter] = [
    _KNNClfAdapter(),
    _KNNRegAdapter(),
    _RFClfAdapter(),
    _RFRegAdapter(),
    _XGBClfAdapter(),
    _XGBRegAdapter(),
]
