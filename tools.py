from sklearn.preprocessing import StandardScaler
import numpy as np
import logging
from typing import Tuple, Any, Dict
import matplotlib.pyplot as plt
from pathlib import Path
from joblib import dump, load
# ---------------------------------------------------------------------------
# 日志设置
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="[%(asctime)s] %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ---------- 预处理 ----------

def scale_features(X: np.ndarray) -> Tuple[np.ndarray, StandardScaler]:
    """对特征做标准化，返回 ``(X_scaled, scaler)``。"""
    scaler = StandardScaler().fit(X)
    return scaler.transform(X), scaler

def plot_scores(times, scores, threshold=None, title="模型性能指标"):
    """
    通用绘图函数，用于展示训练指标折线图或阈值水平线。
    """
    plt.figure(figsize=(10, 4))
    plt.plot(times, scores, marker='o')
    if threshold is not None:
        plt.axhline(threshold, ls='--', label=f'阈值 {threshold:.2f}')
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.tight_layout()
    plt.show()

def save_model(path: Path, model: Any, scaler: StandardScaler, metadata: Dict[str, Any]):
    """
    将模型、标准化器和元数据保存到文件。

    参数:
        path: 输出文件路径 (.joblib)。
        model: 训练好的模型对象。
        scaler: StandardScaler 对象。
        metadata: 模型参数和其他元数据信息。
    """
    dump((model, scaler, metadata), path)


def load_model(path: Path) -> Tuple[Any, StandardScaler, Dict[str, Any]]:
    """
    从文件中加载模型、标准化器和元数据。

    参数:
        path: 模型文件路径 (.joblib)。

    返回:
        model: 模型对象。
        scaler: StandardScaler 对象。
        metadata: 模型参数和元数据。
    """
    return load(path)