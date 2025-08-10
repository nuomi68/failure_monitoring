from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Callable, Any, Dict, List

import numpy as np
import joblib
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import LabelEncoder
from .tools import make_scaler, SCALERS_DISPLAY

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
        缩放器规格，与 ml_interface 一致，可为：
        - None / "none" / False：不做缩放
        - True / "standard"（默认）：StandardScaler
        - 字符串："minmax"、"robust"、"maxabs"、"power"、"quantile"、"normalizer"
        - 字典：{"name": <上述之一>, "params": {...}}
        - 已拟合的缩放器对象（需实现 transform）
    feature_names : 可选 list[str]
        特征列名（用于保存/加载后进行列对齐）。
    """

    labelled_X: np.ndarray
    labels: np.ndarray
    method: str = "1nn"
    metric: str | Callable[[np.ndarray, np.ndarray], float] = "euclidean"
    scaler: Any = "standard"
    feature_names: Optional[List[str]] = None

    # 运行期属性
    _labelled_scaled: np.ndarray | None = field(init=False, default=None)
    model_: Any | None = field(init=False, default=None)
    scaler_spec: Any | None = field(init=False, default=None)
    label_encoder: LabelEncoder | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        if self.labelled_X.shape[0] != self.labels.shape[0]:
            raise ValueError("labelled_X 和 labels 行数必须一致")
        self.method = self.method.lower()
        if self.method not in METHODS_DISPLAY:
            raise ValueError(f"未知方法: {self.method}. 可选: {list(METHODS_DISPLAY)}")

        # 若标签为非数值，则先做编码
        if not np.issubdtype(self.labels.dtype, np.number):
            le = LabelEncoder()
            self.labels = le.fit_transform(self.labels.astype(str))
            self.label_encoder = le

        # 统一处理缩放
        self.scaler_spec = self.scaler
        self.scaler, self._labelled_scaled = self._init_scaler(self.scaler_spec)

        if self.method != "1nn":
            # 其它方法：延迟拟合（在第一次 predict 时拟合），以保持与旧用法一致
            self.model_ = self._build_model()

    # ------------------------------------------------------------------
    # 内部：根据方法构建模型（ sklearn 风格，已在 fault_level_model.py 内实现 ）
    # ------------------------------------------------------------------
    def _build_model(self):
        if self.method == "wknn":
            return WeightedKNNClassifier(n_neighbors="auto", metric=self.metric, scale=False)
        if self.method == "radius":
            return RadiusNeighborsClassifierPlus(radius="auto", metric=self.metric, scale=False)
        if self.method == "centroid":
            return NearestCentroidMahalanobis(use_mahalanobis=True, scale=False)
        if self.method == "kde":
            return ParzenKDEClassifier(bandwidth="auto", scale=False)
        if self.method == "nca_knn":
            return NCAKNNClassifier(n_neighbors="auto", scale=False, random_state=42)
        raise RuntimeError("未实现的方法")

    def _init_scaler(self, spec: Any):
        if spec is None or spec is False or (isinstance(spec, str) and spec.lower() == "none"):
            return None, self.labelled_X.astype(float)
        scaler = make_scaler(spec)
        if hasattr(scaler, "fit") and not hasattr(scaler, "n_features_in_"):
            scaler.fit(self.labelled_X)
        Xs = scaler.transform(self.labelled_X)
        return scaler, Xs

    def _ensure_model_fit(self):
        """惰性拟合：第一次使用时再对所选模型进行 fit。"""
        if self.method == "1nn":
            return
        if self.model_ is None:
            self.model_ = self._build_model()
        # 通过是否存在 classes_ 等属性判断是否已经 fit
        if not hasattr(self.model_, "classes_"):
            self.model_.fit(self._labelled_scaled, self.labels)

    # ------------------------------------------------------------------
    # 预测接口
    # ------------------------------------------------------------------
    def predict(
        self,
        X: np.ndarray,
        *,
        metric: Optional[str | Callable[[np.ndarray, np.ndarray], float]] = None,
        decode: bool = True,
    ) -> np.ndarray:
        """对 X 进行故障等级预测。

        Parameters
        ----------
        decode : bool, default True
            若为 True 且存在 label_encoder，则输出原始标签；否则输出内部编码后的整数。
        """
        if metric is None:
            metric = self.metric

        X_scaled = X.astype(float)
        if self.scaler is not None:
            try:
                X_scaled = self.scaler.transform(X_scaled)
            except Exception as e:
                raise ValueError(f"对输入 X 应用 scaler 失败: {e}") from e

        if self.method == "1nn":
            dists = pairwise_distances(X_scaled, self._labelled_scaled, metric=metric)
            nearest_idx = np.argmin(dists, axis=1)
            preds = self.labels[nearest_idx]
        else:
            self._ensure_model_fit()
            preds = self.model_.predict(X_scaled)
        if decode and self.label_encoder is not None:
            try:
                preds = self.label_encoder.inverse_transform(preds.astype(int))
            except Exception:
                pass
        return preds

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
            "feature_names": self.feature_names,
            "scaler_spec": self.scaler_spec,
            "scaler": self.scaler,
            "label_encoder": self.label_encoder,
        }
        if self.method != "1nn":
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
            feature_names=data.get("feature_names", None),
        )
        obj.scaler_spec = data.get("scaler_spec", obj.scaler_spec)
        obj.label_encoder = data.get("label_encoder", None)
        if method != "1nn":
            obj.model_ = data.get("model", None)
        return obj

    # ------------------------------------------------------------------
    # 工具：返回可选方法（给前端下拉使用）
    # ------------------------------------------------------------------
    @staticmethod
    def available_methods() -> Dict[str, str]:
        return METHODS_DISPLAY.copy()

    @staticmethod
    def available_scalers():
        return SCALERS_DISPLAY.copy()
