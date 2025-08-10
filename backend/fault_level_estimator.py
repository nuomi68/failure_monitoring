from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Callable, Any, Dict, List

import numpy as np
import joblib
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

# 引入更稳健的多种分类方法（已在 fault_level_model.py 中实现）
from backend.models.fault_level_model import (
    WeightedKNNClassifier,
    RadiusNeighborsClassifierPlus,
    NearestCentroidMahalanobis,
    ParzenKDEClassifier,
    NCAKNNClassifier,
)

# 供前端显示的方法清单（key 为内部代号，value 为人类可读名称）
METHODS_DISPLAY: Dict[str, str] = {
    "1nn": "1-NN 最近邻（基线）",
    "wknn": "距离加权 kNN（自动选 k）",
    "radius": "半径邻域（自适应半径）",
    "centroid": "最近质心 + 马氏距离",
    "kde": "核密度估计（KDE）",
    "nca_knn": "NCA + kNN（度量学习）",
}


@dataclass
class FaultLevelEstimator:
    """统一的故障等级估计器封装。

    目标：
    - 与现有页面保持一致的使用方式（无需显式 .fit），但支持多种更稳健的方法；
    - 提供统一的 save/load 持久化；
    - 前端可以选择算法方法；

    参数
    ----
    labelled_X : (n_samples, n_features) 的 numpy 数组
        已标注样本的特征矩阵。
    labels : (n_samples,) 的 numpy 数组
        对应的故障等级标签。
    method : str
        算法方法代号，见 METHODS_DISPLAY 的 key。默认 '1nn'。
    metric : str or callable
        距离度量（用于 1-NN 与部分方法）。
    scaler : 可选
        仅对 method='1nn' 起作用；若提供，则在距离计算前对数据做 transform。
    scale : bool
        对除 '1nn' 外的方法是否在内部启用 StandardScaler（多数情况下推荐 True）。
    feature_names : 可选 list[str]
        特征列名（用于保存/加载后进行列对齐）。
    """

    labelled_X: np.ndarray
    labels: np.ndarray
    method: str = "1nn"
    metric: str | Callable[[np.ndarray, np.ndarray], float] = "euclidean"
    scaler: Optional[Any] = None
    scale: bool = True
    feature_names: Optional[List[str]] = None

    # 运行期属性
    _labelled_scaled: np.ndarray | None = field(init=False, default=None)
    model_: Any | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        if self.labelled_X.shape[0] != self.labels.shape[0]:
            raise ValueError("labelled_X 和 labels 行数必须一致")
        self.method = self.method.lower()
        if self.method not in METHODS_DISPLAY:
            raise ValueError(f"未知方法: {self.method}. 可选: {list(METHODS_DISPLAY)}")

        # 1) 兼容旧的 1-NN 基线（无需显式 fit）
        if self.method == "1nn":
            if self.scaler is not None:
                try:
                    self._labelled_scaled = self.scaler.transform(self.labelled_X)
                except Exception as e:
                    raise ValueError(f"应用 scaler 失败: {e}") from e
            else:
                # 若未显式提供 scaler，但希望标准化，则内部拟合一个
                if self.scale:
                    self.scaler = StandardScaler().fit(self.labelled_X)
                    self._labelled_scaled = self.scaler.transform(self.labelled_X)
                else:
                    self._labelled_scaled = self.labelled_X.astype(float)
        else:
            # 2) 其它方法：延迟拟合（在第一次 predict 时拟合），以保持与旧用法一致
            self._labelled_scaled = None
            self.model_ = self._build_model()

    # ------------------------------------------------------------------
    # 内部：根据方法构建模型（ sklearn 风格，已在 fault_level_model.py 内实现 ）
    # ------------------------------------------------------------------
    def _build_model(self):
        if self.method == "wknn":
            return WeightedKNNClassifier(n_neighbors="auto", metric=self.metric, scale=self.scale)
        if self.method == "radius":
            return RadiusNeighborsClassifierPlus(radius="auto", metric=self.metric, scale=self.scale)
        if self.method == "centroid":
            return NearestCentroidMahalanobis(use_mahalanobis=True, scale=self.scale)
        if self.method == "kde":
            return ParzenKDEClassifier(bandwidth="auto", scale=self.scale)
        if self.method == "nca_knn":
            return NCAKNNClassifier(n_neighbors="auto", scale=self.scale, random_state=42)
        raise RuntimeError("未实现的方法")

    def _ensure_model_fit(self):
        """惰性拟合：第一次使用时再对所选模型进行 fit。"""
        if self.method == "1nn":
            return
        if self.model_ is None:
            self.model_ = self._build_model()
        # 通过是否存在 classes_ 等属性判断是否已经 fit
        if not hasattr(self.model_, "classes_"):
            self.model_.fit(self.labelled_X.astype(float), self.labels)

    # ------------------------------------------------------------------
    # 预测接口
    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray, *, metric: Optional[str | Callable[[np.ndarray, np.ndarray], float]] = None) -> np.ndarray:
        """对 X 进行故障等级预测。"""
        if metric is None:
            metric = self.metric

        if self.method == "1nn":
            # 1-NN 路径：直接基于存储的样本计算最近邻
            if self.scaler is not None:
                try:
                    X_scaled = self.scaler.transform(X)
                except Exception as e:
                    raise ValueError(f"对输入 X 应用 scaler 失败: {e}") from e
            else:
                X_scaled = X.astype(float)
            dists = pairwise_distances(X_scaled, self._labelled_scaled, metric=metric)
            nearest_idx = np.argmin(dists, axis=1)
            return self.labels[nearest_idx]
        else:
            # 其它方法：交给内部模型
            self._ensure_model_fit()
            return self.model_.predict(X.astype(float))

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """保存模型到磁盘。包含：训练样本、标签、方法、尺度信息、特征名；
        对于非 1-NN，还会保存已拟合的 sklearn 模型。"""
        data: Dict[str, Any] = {
            "labelled_X": self.labelled_X,
            "labels": self.labels,
            "method": self.method,
            "metric": self.metric,
            "scale": self.scale,
            "feature_names": self.feature_names,
        }
        if self.method == "1nn":
            data["scaler"] = self.scaler
        else:
            # 确保已拟合后保存
            self._ensure_model_fit()
            data["model"] = self.model_
        joblib.dump(data, path)

    @classmethod
    def load(cls, path: str) -> "FaultLevelEstimator":
        """从磁盘加载模型。"""
        data = joblib.load(path)
        method = data.get("method", "1nn")
        obj = cls(
            labelled_X=data["labelled_X"],
            labels=data["labels"],
            method=method,
            metric=data.get("metric", "euclidean"),
            scaler=data.get("scaler", None),
            scale=data.get("scale", True),
            feature_names=data.get("feature_names", None),
        )
        if method != "1nn":
            obj.model_ = data.get("model", None)
        return obj

    # ------------------------------------------------------------------
    # 工具：返回可选方法（给前端下拉使用）
    # ------------------------------------------------------------------
    @staticmethod
    def available_methods() -> Dict[str, str]:
        return METHODS_DISPLAY.copy()
