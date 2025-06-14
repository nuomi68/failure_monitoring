
import pandas as pd
from pathlib import Path
from typing import Tuple
import numpy as np


def load_dataframe(path: Path) -> pd.DataFrame:
    """
    加载 Excel 或 CSV 文件。
    解析 TIME 列，将其设置为索引，并返回仅包含数值列的 DataFrame。

    参数:
        path: 文件路径，支持 .xlsx/.xls/.csv。

    返回:
        pd.DataFrame: 以 TIME 为索引，所有列均为数值类型。
    """
    df = pd.read_excel("./data/20230510-20240924.xlsx", sheet_name="数据单")
    # if path.suffix.lower() in {".xlsx", ".xls"}:
    #     df = pd.read_excel(path, engine="openpyxl", sheet_name="数据单")
    # else:
    #     df = pd.read_csv(path)
    # if "TIME" not in df.columns:
    #     # 如果没有 TIME 列，则将第一列重命名为 TIME
    #     df.rename(columns={df.columns[0]: "TIME"}, inplace=True)
    # 转换为时间类型
    df["TIME"] = pd.to_datetime(df['TIME'], format='%Y年%m月%d日%H%M', errors='coerce')
    # 设置索引为 TIME
    df = df.set_index("TIME")
    # 将所有列转换为数值类型，无法转换的为 NaN
    numeric = df.apply(pd.to_numeric, errors="coerce")
    # 丢弃包含 NaN 的行
    return numeric.dropna(axis=0, how="any")

################################################################################
# 核心功能：训练、评分、生成噪声样本                                              #
################################################################################

def generate_noisy_test(
    X: np.ndarray,
    small_sigma: float = 0.01,
    large_sigma: float = 0.1,
    n_per_class: int = 1000,
    random_state: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """基于 *X* 生成带噪声的测试集。

    *   **无异常样本**：对原始样本添加 *small_sigma* 量级的高斯噪声。
    *   **有异常样本**：对原始样本添加 *large_sigma* 量级的高斯噪声。

    返回:
        X_test : (2*n_per_class, n_features) 测试特征
        y_test : (2*n_per_class,) 布尔标签，1 表示异常
    """
    rng = np.random.default_rng(random_state)
    n_features = X.shape[1]
    std = X.std(axis=0, keepdims=True)

    idx_small = rng.choice(len(X), size=n_per_class, replace=True)
    idx_large = rng.choice(len(X), size=n_per_class, replace=True)

    X_small = X[idx_small] + rng.normal(0, small_sigma, size=(n_per_class, n_features)) * std
    noise = np.abs(rng.normal(0, large_sigma, size=(n_per_class, n_features)))
    X_large = X[idx_large] + (noise*10+0.1) * std

    X_test = np.vstack([X_small, X_large]).astype(np.float32)
    y_test = np.hstack([np.zeros(n_per_class, dtype=bool), np.ones(n_per_class, dtype=bool)])

    return X_test, y_test