
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
        # Ensure the requested neighbor count does not exceed the number of
        # available training samples. ``KNeighborsClassifier`` will raise a
        # ``ValueError`` during prediction if ``n_neighbors`` is larger than
        # the fitted sample size.  Adjust it here to avoid runtime errors when
        # users specify an excessively large value.
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
        # ``KNeighborsRegressor`` requires ``n_neighbors`` to be no greater
        # than the number of training samples.  When the user requests a
        # larger value, shrink it to ``len(X)`` to prevent ``ValueError`` on
        # prediction.
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


ADAPTERS: list[AlgoAdapter] = [
    _KNNClfAdapter(),
    _KNNRegAdapter(),
    _RFClfAdapter(),
    _RFRegAdapter(),
]
