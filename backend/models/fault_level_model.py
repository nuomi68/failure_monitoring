from __future__ import annotations

"""
本文件包含多种更稳健的“小样本/基于距离或密度”的分类器实现，
并新增一个简单的工厂方法，便于外部按方法代号创建模型。

可选方法（代号 -> 说明）：
- 'wknn'     -> 距离加权 kNN（自动选 k）
- 'radius'   -> 半径邻域（自适应半径 + 空邻域安全回退）
- 'centroid' -> 最近质心 + 马氏距离（收缩协方差）
- 'kde'      -> 核密度估计（Parzen 窗）
- 'nca_knn'  -> NCA + kNN（先学度量再最近邻）

注：1-NN 基线由 FaultLevelEstimator 内部直接实现，因此这里不提供 '1nn' 的构造。
"""

from typing import Dict

import numpy as np
from numpy.typing import ArrayLike

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold, LeaveOneOut, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from sklearn.neighbors import (
    KNeighborsClassifier as _SKKNN,
    RadiusNeighborsClassifier as _SKRadiusNN,
    NeighborhoodComponentsAnalysis,
    KernelDensity,
)
from sklearn.covariance import LedoitWolf

# ----------------------------
# 常用小工具
# ----------------------------

def _auto_cv(y: ArrayLike, min_splits: int = 5):
    y = np.asarray(y)
    n = len(y)
    if n <= 30:
        return LeaveOneOut()
    return StratifiedKFold(
        n_splits=min(min_splits, np.unique(y, return_counts=True)[1].min()),
        shuffle=True, random_state=42,
    )


def _class_priors(y: ArrayLike, laplace: float = 1.0):
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    priors = (counts + laplace) / (counts.sum() + laplace * len(classes))
    return priors, classes

# ----------------------------
# 1) 距离加权 kNN
# ----------------------------
class WeightedKNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neighbors: int | str = 'auto', metric: str = 'euclidean',
                 scale: bool = True):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.scale = scale

    def fit(self, X: ArrayLike, y: ArrayLike):
        X, y = check_X_y(X, y)
        steps = []
        if self.scale:
            steps.append(('scaler', StandardScaler()))
        steps.append(('knn', _SKKNN(n_neighbors=3, weights='distance', metric=self.metric)))
        pipe = Pipeline(steps)

        if self.n_neighbors == 'auto':
            param_grid = {'knn__n_neighbors': [1, 3, 5, 7]}
            cv = _auto_cv(y)
            gs = GridSearchCV(pipe, param_grid=param_grid, cv=cv, n_jobs=1)
            gs.fit(X, y)
            self.model_ = gs.best_estimator_
            self.best_k_ = gs.best_params_['knn__n_neighbors']
        else:
            pipe.set_params(knn__n_neighbors=int(self.n_neighbors))
            pipe.fit(X, y)
            self.model_ = pipe
            self.best_k_ = int(self.n_neighbors)

        self.classes_ = np.unique(y)
        return self

    def predict(self, X: ArrayLike):
        check_is_fitted(self, 'model_')
        X = check_array(X)
        return self.model_.predict(X)

    def predict_proba(self, X: ArrayLike):
        check_is_fitted(self, 'model_')
        X = check_array(X)
        return self.model_.predict_proba(X)

# ----------------------------
# 2) 半径邻域（安全版）
# ----------------------------
class RadiusNeighborsClassifierPlus(BaseEstimator, ClassifierMixin):
    def __init__(self, radius: float | str = 'auto', metric: str = 'euclidean',
                 outlier_label: str | int | None = 'most_frequent', scale: bool = True,
                 random_state: int = 42):
        self.radius = radius
        self.metric = metric
        self.outlier_label = outlier_label
        self.scale = scale
        self.random_state = random_state

    def _safe_predict(self, rnn, knn_fallback, X):
        neigh_ind = rnn.radius_neighbors(X, return_distance=False)
        empty_mask = np.array([len(ix) == 0 for ix in neigh_ind])
        y_pred = np.empty(X.shape[0], dtype=rnn.classes_.dtype)
        if empty_mask.any():
            y_pred[empty_mask] = knn_fallback.predict(X[empty_mask])
        if (~empty_mask).any():
            y_pred[~empty_mask] = rnn.predict(X[~empty_mask])
        return y_pred

    def fit(self, X: ArrayLike, y: ArrayLike):
        X, y = check_X_y(X, y)
        if self.scale:
            self.scaler_ = StandardScaler().fit(X)
            X_ = self.scaler_.transform(X)
        else:
            self.scaler_ = None
            X_ = X.astype(float)

        self.classes_ = np.unique(y)

        from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
        k_for_scale = min(5, len(X_) - 1) if len(X_) > 1 else 1
        nn = NearestNeighbors(n_neighbors=k_for_scale+1, metric=self.metric)
        nn.fit(X_)
        dists, _ = nn.kneighbors(X_)
        kth = dists[:, k_for_scale]
        base = np.median(kth)
        candidates = np.unique(np.round(base * np.array([0.6, 0.8, 1.0, 1.2, 1.6, 2.2]), 3))

        if self.radius != 'auto':
            self.best_radius_ = float(self.radius)
            self.rnn_ = _SKRadiusNN(radius=self.best_radius_, weights='distance', metric=self.metric,
                                     outlier_label=self.outlier_label)
            self.rnn_.fit(X_, y)
            self.knn_fallback_ = _SKKNN(n_neighbors=1, weights='distance', metric=self.metric).fit(X_, y)
            return self

        cv = _auto_cv(y)
        best_score, best_r = -np.inf, None
        for r in candidates:
            scores = []
            for tr, te in cv.split(X_, y):
                Xtr, ytr = X_[tr], y[tr]
                Xte, yte = X_[te], y[te]
                rnn = _SKRadiusNN(radius=float(r), weights='distance', metric=self.metric,
                                   outlier_label=self.outlier_label).fit(Xtr, ytr)
                knn_fb = _SKKNN(n_neighbors=1, weights='distance', metric=self.metric).fit(Xtr, ytr)
                yhat = self._safe_predict(rnn, knn_fb, Xte)
                scores.append((yhat == yte).mean())
            mean_acc = float(np.mean(scores))
            if mean_acc > best_score + 1e-12:
                best_score, best_r = mean_acc, float(r)

        self.best_radius_ = best_r if best_r is not None else float(candidates[-1])
        self.rnn_ = _SKRadiusNN(radius=self.best_radius_, weights='distance', metric=self.metric,
                                 outlier_label=self.outlier_label).fit(X_, y)
        self.knn_fallback_ = _SKKNN(n_neighbors=1, weights='distance', metric=self.metric).fit(X_, y)
        return self

    def predict(self, X: ArrayLike):
        check_is_fitted(self, 'rnn_')
        X = check_array(X)
        if self.scaler_ is not None:
            X_ = self.scaler_.transform(X)
        else:
            X_ = X.astype(float)
        return self._safe_predict(self.rnn_, self.knn_fallback_, X_)

    def predict_proba(self, X: ArrayLike):
        check_is_fitted(self, 'rnn_')
        X = check_array(X)
        if self.scaler_ is not None:
            X_ = self.scaler_.transform(X)
        else:
            X_ = X.astype(float)
        neigh_ind = self.rnn_.radius_neighbors(X_, return_distance=False)
        empty_mask = np.array([len(ix) == 0 for ix in neigh_ind])
        proba = np.zeros((X_.shape[0], len(self.rnn_.classes_)))
        classes = self.rnn_.classes_
        if (~empty_mask).any():
            proba[~empty_mask] = self.rnn_.predict_proba(X_[~empty_mask])
        if empty_mask.any():
            yfb = self.knn_fallback_.predict(X_[empty_mask])
            for i, lab in enumerate(yfb):
                j = np.where(classes == lab)[0][0]
                proba[(np.where(empty_mask)[0])[i], j] = 1.0
        return proba

# ----------------------------
# 3) 最近质心 + 马氏距离
# ----------------------------
class NearestCentroidMahalanobis(BaseEstimator, ClassifierMixin):
    def __init__(self, use_mahalanobis: bool = True, scale: bool = True):
        self.use_mahalanobis = use_mahalanobis
        self.scale = scale

    def fit(self, X: ArrayLike, y: ArrayLike):
        X, y = check_X_y(X, y)
        if self.scale:
            self.scaler_ = StandardScaler().fit(X)
            X_ = self.scaler_.transform(X)
        else:
            self.scaler_ = None
            X_ = X.astype(float)

        self.classes_, y_idx = np.unique(y, return_inverse=True)
        self.centroids_ = np.vstack([X_[y_idx == i].mean(axis=0) for i in range(len(self.classes_))])

        if self.use_mahalanobis:
            lw = LedoitWolf().fit(X_)
            self.precision_ = lw.precision_
        else:
            self.precision_ = None
        return self

    def _mahalanobis(self, X, means, precision):
        diffs = X[:, None, :] - means[None, :, :]
        d2 = np.einsum('...i,ij,...j->...', diffs, precision, diffs)
        return np.sqrt(np.maximum(d2, 0.0))

    def predict(self, X: ArrayLike):
        check_is_fitted(self, 'centroids_')
        X = check_array(X)
        if self.scaler_ is not None:
            X_ = self.scaler_.transform(X)
        else:
            X_ = X.astype(float)
        if self.precision_ is None:
            dists = np.linalg.norm(X_[:, None, :] - self.centroids_[None, :, :], axis=2)
        else:
            dists = self._mahalanobis(X_, self.centroids_, self.precision_)
        return self.classes_[dists.argmin(axis=1)]

# ----------------------------
# 4) 核密度估计（KDE）
# ----------------------------
class ParzenKDEClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, bandwidth: float | str = 'auto', kernel: str = 'gaussian',
                 scale: bool = True):
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.scale = scale

    def fit(self, X: ArrayLike, y: ArrayLike):
        X, y = check_X_y(X, y)
        if self.scale:
            self.scaler_ = StandardScaler().fit(X)
            X_ = self.scaler_.transform(X)
        else:
            self.scaler_ = None
            X_ = X.astype(float)

        priors, classes = _class_priors(y)
        self.classes_ = classes
        self.class_priors_ = priors

        self.kdes_: list[KernelDensity] = []
        self.class_indices_: list[int] = []

        if self.bandwidth == 'auto':
            candidates = np.round(np.logspace(-1, np.log10(2.0), 9), 3)
            cv = _auto_cv(y)
            best_bw, best_score = None, -np.inf
            for bw in candidates:
                scores = []
                for tr, te in cv.split(X_, y):
                    Xtr, ytr = X_[tr], y[tr]
                    Xte, yte = X_[te], y[te]
                    models = {}
                    for c in self.classes_:
                        kde = KernelDensity(kernel=self.kernel, bandwidth=bw).fit(Xtr[ytr == c])
                        models[c] = kde
                    s = 0.0
                    for c in self.classes_:
                        m = (yte == c)
                        if m.sum() == 0:
                            continue
                        logprob = models[c].score_samples(Xte[m])
                        s += (logprob + np.log(priors[self.classes_ == c][0])).mean()
                    scores.append(s)
                mean_score = float(np.mean(scores))
                if mean_score > best_score:
                    best_score, best_bw = mean_score, float(bw)
            self.bandwidth_ = best_bw
        else:
            self.bandwidth_ = float(self.bandwidth)

        for c in self.classes_:
            kde = KernelDensity(kernel=self.kernel, bandwidth=self.bandwidth_).fit(X_[y == c])
            self.kdes_.append(kde)
            self.class_indices_.append(c)
        return self

    def _log_joint(self, X_):
        n, C = X_.shape[0], len(self.classes_)
        out = np.empty((n, C), dtype=float)
        for j, c in enumerate(self.class_indices_):
            log_like = self.kdes_[j].score_samples(X_)
            out[:, j] = log_like + np.log(self.class_priors_[self.classes_ == c][0])
        return out

    def predict(self, X: ArrayLike):
        check_is_fitted(self, 'kdes_')
        X = check_array(X)
        X_ = self.scaler_.transform(X) if self.scaler_ is not None else X.astype(float)
        log_joint = self._log_joint(X_)
        return self.classes_[log_joint.argmax(axis=1)]

    def predict_proba(self, X: ArrayLike):
        check_is_fitted(self, 'kdes_')
        X = check_array(X)
        X_ = self.scaler_.transform(X) if self.scaler_ is not None else X.astype(float)
        z = self._log_joint(X_)
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        return p

# ----------------------------
# 5) NCA + kNN
# ----------------------------
class NCAKNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_components: int | None = None, n_neighbors: int | str = 'auto',
                 max_iter: int = 200, tol: float = 1e-5, scale: bool = True,
                 random_state: int = 42):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.max_iter = max_iter
        self.tol = tol
        self.scale = scale
        self.random_state = random_state

    def fit(self, X: ArrayLike, y: ArrayLike):
        X, y = check_X_y(X, y)
        steps = []
        if self.scale:
            steps.append(('scaler', StandardScaler()))
        steps.append(('nca', NeighborhoodComponentsAnalysis(
            n_components=self.n_components,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state
        )))
        steps.append(('knn', _SKKNN(n_neighbors=3, weights='distance')))
        pipe = Pipeline(steps)

        if self.n_neighbors == 'auto':
            param_grid = {'knn__n_neighbors': [1, 3, 5]}
            cv = _auto_cv(y)
            gs = GridSearchCV(pipe, param_grid=param_grid, cv=cv, n_jobs=1)
            gs.fit(X, y)
            self.model_ = gs.best_estimator_
            self.best_k_ = gs.best_params_['knn__n_neighbors']
        else:
            pipe.set_params(knn__n_neighbors=int(self.n_neighbors))
            pipe.fit(X, y)
            self.model_ = pipe
            self.best_k_ = int(self.n_neighbors)

        self.classes_ = np.unique(y)
        return self

    def predict(self, X: ArrayLike):
        check_is_fitted(self, 'model_')
        X = check_array(X)
        return self.model_.predict(X)

    def predict_proba(self, X: ArrayLike):
        check_is_fitted(self, 'model_')
        X = check_array(X)
        return self.model_.predict_proba(X)

# ----------------------------
# 工厂方法
# ----------------------------
METHODS_REGISTRY: Dict[str, type] = {
    'wknn': WeightedKNNClassifier,
    'radius': RadiusNeighborsClassifierPlus,
    'centroid': NearestCentroidMahalanobis,
    'kde': ParzenKDEClassifier,
    'nca_knn': NCAKNNClassifier,
}


def build_model(method: str, **kwargs):
    """按代号创建模型（不含 '1nn'）。"""
    method = method.lower()
    if method not in METHODS_REGISTRY:
        raise ValueError(f"未知方法: {method}. 可选: {list(METHODS_REGISTRY)}")
    cls = METHODS_REGISTRY[method]
    return cls(**kwargs)
