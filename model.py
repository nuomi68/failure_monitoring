import os
import sys
import logging
from typing import Tuple, Any, Dict, Optional
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from tools import  logger
# ---------- KNN 模型 ----------

# ---------------------------------------------------------------------------
# 模型训练
# ---------------------------------------------------------------------------

def train_knn(
    X: np.ndarray,
    k: int = 5,
    quantile: float | None = None,
) -> Tuple[NearestNeighbors, float]:
    """训练 k‑NN 并返回阈值 ``tau``。

    ``tau`` 为样本 *k‑距离* 的 ``quantile`` 分位数；若 `quantile` 为 `None`，则使用最大值。"""
    logger.info("训练 k‑NN (k=%d)…", k)
    knn = NearestNeighbors(n_neighbors=k).fit(X)
    dists, _ = knn.kneighbors(X)
    k_dist = dists[:, -1]
    tau = np.quantile(k_dist, quantile) if quantile else float(k_dist.max())
    logger.info("k‑NN 阈值 τ=%.4f", tau)
    return knn, tau


def train_iforest(
    X: np.ndarray,
    n_estimators: int = 100,
    contamination: float = 0.05,
    random_state: int | None = 42,
) -> Tuple[IsolationForest, float]:
    """训练 IsolationForest 并返回阈值 ``tau``。"""
    logger.info(
        "训练 IsolationForest (n_estimators=%d, contamination=%.2f)…",
        n_estimators,
        contamination,
    )
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
    ).fit(X)
    # decision_function 越负越异常；取相反数使分数越大越异常
    scores = -model.decision_function(X)
    tau = np.quantile(scores, 1 - contamination)
    logger.info("IsolationForest 阈值 τ=%.4f", tau)
    return model, tau


def score_model(
    model: Any,
    X: np.ndarray,
    scaler: StandardScaler,
    metadata: Dict[str, Any],
    model_type: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    对新数据进行评分，返回分数和异常标签。

    参数:
        model: 训练好的模型或 (knn, tau) 元组。
        X: 原始特征矩阵。
        scaler: 标准化器。
        metadata: 模型元数据，包含阈值等。
        model_type: 模型类型 'knn', 'rf', or 'bp'.

    返回:
        score: 对应的异常分数或预测结果。
        labels: 二值异常标签数组（True 表示异常）。
    """
    X_scaled = scaler.transform(X)
    if model_type == "knn":
        knn, tau = model
        dists, _ = knn.kneighbors(X_scaled)
        score = dists[:, -1]
        labels = score > tau
    else:
        # 对于 rf/bp，直接使用预测结果
        preds = model.predict(X_scaled)
        score = preds
        labels = preds.astype(bool)
    return score, labels

# ---------- BP 神经网络模型 ----------

# def train_bp(
#     X: np.ndarray,
#     y: np.ndarray,
#     hidden_layer_sizes: Tuple[int, ...]
# ) -> MLPClassifier:
#     """训练 MLP 神经网络，返回已训练模型"""
#     mlp = MLPClassifier(
#         hidden_layer_sizes=hidden_layer_sizes,
#         random_state=0,
#         max_iter=500
#     )
#     mlp.fit(X, y)
#     logger.info("BP 神经网络训练完成，最终 loss=%.4f" % mlp.loss_curve_[-1])
#     return mlp


# def test_bp(
#     model: MLPClassifier,
#     scaler: StandardScaler,
#     X: np.ndarray
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """对新数据进行 MLP 预测，返回概率和标签"""
#     Xs = transform_data(scaler, X)
#     probs = model.predict_proba(Xs)[:, 1]
#     labels = model.predict(Xs).astype(bool)
#     return probs, labels