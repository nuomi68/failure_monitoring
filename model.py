from __future__ import annotations

"""
小样本分类器实现。

设计目标：
- 仍然以“距离/相似度”为核心思想，但比“最近邻（1-NN）”更稳健；
- 统一 sklearn 风格接口（fit / predict / predict_proba），便于与现有代码衔接；
- 自带常见稳健化手段：标准化、参数自动选择（在样本很少时优先 LOOCV）、
  协方差收缩、带先验的密度估计等；

包含的分类器：
1) WeightedKNNClassifier   —— 距离加权的 kNN（支持自动选择 k），比 1-NN 抗噪声；
2) RadiusNeighborsClassifierPlus —— 半径邻域分类器（可自动选半径），对类内密度差异更稳健；
3) NearestCentroidMahalanobis —— 最近质心/马氏距离分类器（Ledoit-Wolf 收缩协方差）；
4) ParzenKDEClassifier     —— 基于核密度估计（Parzen窗）的生成式分类器，带类别先验；
5) NCAKNNClassifier        —— 先用 NCA 学习度量，再做 kNN（小样本下常比直接欧氏距离稳健）。

使用方式见文件底部 __main__ 示例。

注意：这些实现依赖 scikit-learn。
"""

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, List

import numpy as np
from numpy.typing import ArrayLike

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold, LeaveOneOut, GridSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from sklearn.neighbors import (
    KNeighborsClassifier as _SKKNN,
    RadiusNeighborsClassifier as _SKRadiusNN,
    NearestCentroid as _SKNearestCentroid,
    NeighborhoodComponentsAnalysis,
    KernelDensity,
)
from sklearn.covariance import LedoitWolf


# ----------------------------
# 常用小工具
# ----------------------------

def _auto_cv(y: ArrayLike, min_splits: int = 5):
    """根据标签数量自动选择交叉验证方案。
    - 样本极少时（每类 ≥2 即可），使用 Leave-One-Out（LOOCV），
      这样每次验证都尽量利用最大训练集；
    - 样本稍多时，使用 5 折分层交叉验证。
    """
    y = np.asarray(y)
    n = len(y)
    # 若样本数较少，用 LOOCV 更稳健（但会慢一些）
    if n <= 30:
        return LeaveOneOut()
    # 否则用 5 折分层 CV
    return StratifiedKFold(n_splits=min(min_splits, np.unique(y, return_counts=True)[1].min()),
                           shuffle=True, random_state=42)


def _class_priors(y: ArrayLike, laplace: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """计算类别先验及类别列表（带 Laplace 平滑）。
    返回：priors, classes
    """
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    priors = (counts + laplace) / (counts.sum() + laplace * len(classes))
    return priors, classes


# ----------------------------
# 1) 距离加权 kNN
# ----------------------------
class WeightedKNNClassifier(BaseEstimator, ClassifierMixin):
    """距离加权的 kNN 分类器。

    关键点：
    - 使用 weights='distance'，远处邻居权重小，抗噪声能力优于 1-NN；
    - 支持 n_neighbors='auto'，通过 CV 自动选 k；
    - 默认内置 StandardScaler，避免特征量纲影响距离计算。
    """

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
        # 先放个占位 K 值，若 'auto' 会在下方 GridSearch 中替换
        steps.append(('knn', _SKKNN(n_neighbors=3, weights='distance', metric=self.metric)))
        pipe = Pipeline(steps)

        if self.n_neighbors == 'auto':
            # 在小样本下，网格不宜过大；常用奇数 K 更稳定
            param_grid = {'knn__n_neighbors': [1, 3, 5, 7]}
            cv = _auto_cv(y)
            gs = GridSearchCV(pipe, param_grid=param_grid, cv=cv, n_jobs=None)
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

    def predict(self, X: ArrayLike) -> np.ndarray:
        check_is_fitted(self, 'model_')
        X = check_array(X)
        return self.model_.predict(X)

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        check_is_fitted(self, 'model_')
        X = check_array(X)
        # KNN 自带概率（基于邻居比例/权重）
        return self.model_.predict_proba(X)


# ----------------------------
# 2) 半径邻域分类器（自动选半径）
# ----------------------------
class RadiusNeighborsClassifierPlus(BaseEstimator, ClassifierMixin):
    """半径邻域分类器（安全版）。

    关键点：
    - 用半径 r 来界定邻域，邻居数量可变，适合密度不均的情况；
    - 支持 r='auto'，通过**稳健的手写 CV**选择（避免 GridSearch + predict_proba 在空邻域时报错）；
    - 预测阶段对“空邻域”做**安全回退**：若某样本半径内没有邻居，则回退到 1-NN 的预测；
    - 对类内密度差异、局部孤点相对更稳健。
    """

    def __init__(self, radius: float | str = 'auto', metric: str = 'euclidean',
                 outlier_label: Optional[str | int] = 'most_frequent', scale: bool = True,
                 random_state: int = 42):
        self.radius = radius
        self.metric = metric
        self.outlier_label = outlier_label  # 'most_frequent' 可减少极端情况误判
        self.scale = scale
        self.random_state = random_state

    def _safe_predict(self, rnn, knn_fallback, X):
        """对每个样本：若半径邻域为空，则使用 1-NN 回退；否则使用 RNN 预测。
        这样可以避免 sklearn 在 predict_proba 内部因空邻域直接报错。"""
        # 先找出哪些点邻域为空
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
        rng = np.random.RandomState(self.random_state)

        # 预处理：标准化（避免量纲影响距离）
        if self.scale:
            self.scaler_ = StandardScaler().fit(X)
            X_ = self.scaler_.transform(X)
        else:
            self.scaler_ = None
            X_ = X.astype(float)

        self.classes_ = np.unique(y)

        # 计算“典型距离尺度”：每个点到其第 k 个最近邻（含异类）的距离，取中位数
        # 用它来生成候选半径，能减少过小半径导致的空邻域
        from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
        k_for_scale = min(5, len(X_) - 1) if len(X_) > 1 else 1
        nn = NearestNeighbors(n_neighbors=k_for_scale+1, metric=self.metric)
        nn.fit(X_)
        dists, _ = nn.kneighbors(X_)
        # 排除自身（第 1 列为 0）后取第 k 个邻居距离
        kth = dists[:, k_for_scale]
        base = np.median(kth)
        # 在 base 附近做几何级数候选；再加一个更大的兜底值
        candidates = np.unique(np.round(base * np.array([0.6, 0.8, 1.0, 1.2, 1.6, 2.2]), 3))

        # 若用户给定具体半径则直接用
        if self.radius != 'auto':
            self.best_radius_ = float(self.radius)
            # 在全量数据上拟合最终模型与回退 1-NN
            self.rnn_ = _SKRadiusNN(radius=self.best_radius_, weights='distance', metric=self.metric,
                                     outlier_label=self.outlier_label)
            self.rnn_.fit(X_, y)
            self.knn_fallback_ = _SKKNN(n_neighbors=1, weights='distance', metric=self.metric).fit(X_, y)
            return self

        # 自动半径选择：使用分层 K 折或 LOOCV，手写评估逻辑，避免 sklearn 在内部调用 predict_proba
        cv = _auto_cv(y)
        best_score = -np.inf
        best_r = None
        for r in candidates:
            fold_scores = []
            for train_idx, test_idx in cv.split(X_, y):
                Xtr, Xte = X_[train_idx], X_[test_idx]
                ytr, yte = y[train_idx], y[test_idx]
                rnn = _SKRadiusNN(radius=float(r), weights='distance', metric=self.metric,
                                   outlier_label=self.outlier_label)
                rnn.fit(Xtr, ytr)
                knn_fb = _SKKNN(n_neighbors=1, weights='distance', metric=self.metric).fit(Xtr, ytr)
                # 使用安全预测
                yhat = self._safe_predict(rnn, knn_fb, Xte)
                acc = (yhat == yte).mean() if len(yte) else 0.0
                fold_scores.append(acc)
            score = float(np.mean(fold_scores))
            if score > best_score + 1e-12:
                best_score = score
                best_r = float(r)

        self.best_radius_ = best_r if best_r is not None else float(candidates[-1])
        # 在全量数据上拟合最终模型与回退 1-NN
        self.rnn_ = _SKRadiusNN(radius=self.best_radius_, weights='distance', metric=self.metric,
                                 outlier_label=self.outlier_label)
        self.rnn_.fit(X_, y)
        self.knn_fallback_ = _SKKNN(n_neighbors=1, weights='distance', metric=self.metric).fit(X_, y)
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        check_is_fitted(self, 'rnn_')
        X = check_array(X)
        if self.scaler_ is not None:
            X_ = self.scaler_.transform(X)
        else:
            X_ = X.astype(float)
        return self._safe_predict(self.rnn_, self.knn_fallback_, X_)

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """给出近似概率：
        - 对非空邻域样本，调用 RNN 的 predict_proba；
        - 对空邻域样本，使用 1-NN 回退并返回 one-hot 概率（近似）。
        """
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
            # one-hot 近似
            for i, lab in enumerate(yfb):
                proba_idx = np.where(classes == lab)[0][0]
                proba[i if empty_mask.sum()==1 else np.where(empty_mask)[0][i], proba_idx] = 1.0
        return proba

# ----------------------------
# 3) 最近质心 + 马氏距离（带收缩协方差）
# ----------------------------
class NearestCentroidMahalanobis(BaseEstimator, ClassifierMixin):
    """最近质心/马氏距离分类器。

    思路：
    - 先对每个类别计算类均值（质心 μ_c）；
    - 计算全局协方差 Σ 并用 Ledoit-Wolf 收缩估计 Σ^(-1)；
    - 预测时计算马氏距离 d_c(x) = sqrt( (x-μ_c)^T Σ^(-1) (x-μ_c) )，取距离最小的类。

    说明：
    - 当使用单位协方差时退化为“最近质心（欧氏距离）”；
    - 收缩协方差在小样本/高维时更稳健，避免 Σ 奇异或病态。
    """

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
        # 计算每个类的质心
        self.centroids_ = np.vstack([X_[y_idx == i].mean(axis=0) for i in range(len(self.classes_))])

        if self.use_mahalanobis:
            # 使用 Ledoit-Wolf 收缩估计协方差逆
            lw = LedoitWolf().fit(X_)
            self.precision_ = lw.precision_  # Σ^(-1)
        else:
            self.precision_ = None
        return self

    def _mahalanobis(self, X: np.ndarray, means: np.ndarray, precision: np.ndarray) -> np.ndarray:
        # 计算所有样本到各质心的马氏距离矩阵，返回形状 (n_samples, n_classes)
        diffs = X[:, None, :] - means[None, :, :]
        # einsum 形式：d^2 = (x-m)^T P (x-m)
        d2 = np.einsum('...i,ij,...j->...', diffs, precision, diffs)
        return np.sqrt(np.maximum(d2, 0.0))

    def predict(self, X: ArrayLike) -> np.ndarray:
        check_is_fitted(self, 'centroids_')
        X = check_array(X)
        if self.scaler_ is not None:
            X_ = self.scaler_.transform(X)
        else:
            X_ = X.astype(float)

        if self.precision_ is None:
            # 欧氏距离到质心
            diffs = X_[:, None, :] - self.centroids_[None, :, :]
            dists = np.linalg.norm(diffs, axis=2)
        else:
            dists = self._mahalanobis(X_, self.centroids_, self.precision_)
        # 取距离最小的类别索引
        idx = dists.argmin(axis=1)
        return self.classes_[idx]


# ----------------------------
# 4) 基于核密度估计（Parzen 窗）的生成式分类器
# ----------------------------
class ParzenKDEClassifier(BaseEstimator, ClassifierMixin):
    """每个类别拟合一个 Kernel Density（高斯核），
    预测时选取后验概率最大的类别。

    特点：
    - 对类别分布非球形、非均匀的情况更灵活；
    - 使用类别先验（带 Laplace 平滑）修正类不平衡；
    - 带简单的带宽（bandwidth）自动选择。
    """

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

        # 分别保存每个类别的 KDE 模型
        self.kdes_: List[KernelDensity] = []
        self.class_indices_ = []

        # 自动选择带宽：在一组候选值中，用 CV 最大化“真类对数似然”
        if self.bandwidth == 'auto':
            # 候选带宽（标准化后通常 0.1~2.0 比较常见）
            candidates = np.round(np.logspace(-1, np.log10(2.0), 9), 3)
            cv = _auto_cv(y)
            best_bw = None
            best_score = -np.inf
            for bw in candidates:
                # CV 评估该带宽下的平均真类 log-likelihood
                scores = []
                for train_idx, test_idx in cv.split(X_, y):
                    Xtr, ytr = X_[train_idx], y[train_idx]
                    Xte, yte = X_[test_idx], y[test_idx]
                    # 拟合每类 KDE
                    models = {}
                    for c in self.classes_:
                        kde = KernelDensity(kernel=self.kernel, bandwidth=bw)
                        kde.fit(Xtr[ytr == c])
                        models[c] = kde
                    # 计算测试样本在其真类密度下的 logprob（加上 log 先验）
                    fold_score = 0.0
                    for c in self.classes_:
                        mask = (yte == c)
                        if mask.sum() == 0:
                            continue
                        logprob = models[c].score_samples(Xte[mask])
                        fold_score += (logprob + np.log(priors[self.classes_ == c][0])).mean()
                    scores.append(fold_score)
                mean_score = np.mean(scores)
                if mean_score > best_score:
                    best_score = mean_score
                    best_bw = float(bw)
            self.bandwidth_ = best_bw
        else:
            self.bandwidth_ = float(self.bandwidth)

        # 用最优带宽在全数据上拟合每类 KDE
        for c in self.classes_:
            kde = KernelDensity(kernel=self.kernel, bandwidth=self.bandwidth_)
            kde.fit(X_[y == c])
            self.kdes_.append(kde)
            self.class_indices_.append(c)
        return self

    def _log_joint(self, X_: np.ndarray) -> np.ndarray:
        """计算每个样本在各类别下的 log p(x|c) + log p(c)，返回 (n_samples, n_classes)"""
        n = X_.shape[0]
        C = len(self.classes_)
        out = np.empty((n, C), dtype=float)
        for j, c in enumerate(self.class_indices_):
            log_like = self.kdes_[j].score_samples(X_)
            out[:, j] = log_like + np.log(self.class_priors_[self.classes_ == c][0])
        return out

    def predict(self, X: ArrayLike) -> np.ndarray:
        check_is_fitted(self, 'kdes_')
        X = check_array(X)
        if self.scaler_ is not None:
            X_ = self.scaler_.transform(X)
        else:
            X_ = X.astype(float)
        log_joint = self._log_joint(X_)
        idx = log_joint.argmax(axis=1)
        return self.classes_[idx]

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        # 通过 softmax(log_joint) 得到后验概率
        check_is_fitted(self, 'kdes_')
        X = check_array(X)
        if self.scaler_ is not None:
            X_ = self.scaler_.transform(X)
        else:
            X_ = X.astype(float)
        log_joint = self._log_joint(X_)
        # 数值稳定的 softmax
        z = log_joint - log_joint.max(axis=1, keepdims=True)
        prob = np.exp(z)
        prob /= prob.sum(axis=1, keepdims=True)
        return prob


# ----------------------------
# 5) NCA + kNN：先学度量再最近邻
# ----------------------------
class NCAKNNClassifier(BaseEstimator, ClassifierMixin):
    """Neighborhood Components Analysis (NCA) + kNN。

    思路：
    - NCA 学习一个线性投影，使得同类样本在投影空间更紧密，异类更分离；
    - 再在该投影空间里执行 kNN（通常 k 较小）。

    小样本建议：
    - 适当限制投影维度（n_components）和迭代次数（max_iter），防止过拟合；
    - 使用 LOOCV/StratifiedKFold 为 k 做一个小网格搜索。
    """

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
        # 先放占位 K 值
        steps.append(('knn', _SKKNN(n_neighbors=3, weights='distance')))
        pipe = Pipeline(steps)

        if self.n_neighbors == 'auto':
            param_grid = {'knn__n_neighbors': [1, 3, 5]}
            cv = _auto_cv(y)
            gs = GridSearchCV(pipe, param_grid=param_grid, cv=cv)
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

    def predict(self, X: ArrayLike) -> np.ndarray:
        check_is_fitted(self, 'model_')
        X = check_array(X)
        return self.model_.predict(X)

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        check_is_fitted(self, 'model_')
        X = check_array(X)
        return self.model_.predict_proba(X)


# ----------------------------
# 示例：与现有 1-NN 思路保持风格一致的用法
# ----------------------------
if __name__ == "__main__":
    # 演示用合成数据（两类、少量样本）
    rng = np.random.default_rng(0)
    n_per_class = 8
    X0 = rng.normal(loc=[0, 0], scale=[1.0, 1.0], size=(n_per_class, 2))
    X1 = rng.normal(loc=[2, 2], scale=[1.0, 1.0], size=(n_per_class, 2))
    X = np.vstack([X0, X1])
    y = np.array([0]*n_per_class + [1]*n_per_class)

    X_test = np.array([[3.0, -0.1], [3.1, 1.2], [1.0, 2.8]])

    models = [
        WeightedKNNClassifier(n_neighbors='auto'),
        RadiusNeighborsClassifierPlus(radius='auto'),
        NearestCentroidMahalanobis(use_mahalanobis=True),
        ParzenKDEClassifier(bandwidth='auto'),
        NCAKNNClassifier(n_neighbors='auto'),
    ]

    for m in models:
        m.fit(X, y)
        pred = m.predict(X_test)
        print(m.__class__.__name__, '=>', pred)
