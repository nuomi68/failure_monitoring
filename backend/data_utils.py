import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

TIME_COL = "TIME"


def load_dataset(path: str, time_format: str | None = None):
    """Load and preprocess the dataset used by all models.

    Parameters
    ----------
    path:
        Path to the CSV/Excel file.
    time_format:
        Optional ``strftime`` format string that describes how the ``TIME``
        column is formatted in ``path``. If omitted, pandas will attempt to
        infer the format automatically.
    """

    if path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    df.columns = df.columns.str.strip()
    if time_format:
        df[TIME_COL] = pd.to_datetime(df[TIME_COL], format=time_format)
    else:
        df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    df = df.sort_values(TIME_COL).reset_index(drop=True)

    features = (
        df.drop(columns=[TIME_COL, "值", "XE-133", "CS-137", "KR-89"], errors="ignore")
        .apply(pd.to_numeric, errors="coerce")
        .ffill()
        .fillna(0)
    )

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features.astype(np.float32))
    return features_scaled, features.columns.tolist(), scaler


def build_windows(data: np.ndarray, look_back: int):
    """Create sliding windows for sequence models."""
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i : i + look_back])
        y.append(data[i + look_back])
    return np.asarray(X, np.float32), np.asarray(y, np.float32)


def compute_relative_errors(compare_df: pd.DataFrame, max_thresh: float = 1000.0):
    """Compute relative errors for predicted values, excluding large outliers from the mean.

    Parameters
    ----------
    compare_df : pd.DataFrame
        第一列是真实值，后续列是各模型预测值。
    max_thresh : float
        超过该阈值的相对误差不计入平均值。

    Returns
    -------
    tuple[pd.DataFrame, pd.Series, pd.Series]
        - rel_err_df: 原始相对误差 DataFrame
        - max_err: 每个模型的最大相对误差
        - mean_err: 排除 > max_thresh 后的平均相对误差
    """
    true = compare_df.iloc[:, 0]
    preds = compare_df.iloc[:, 1:]

    # 计算原始相对误差
    denom = true.replace(0, np.nan)
    rel_err_df = preds.sub(true, axis=0).abs().div(denom, axis=0)

    # 最大误差不变
    max_err = rel_err_df.max()

    # 把大于阈值的项替换为 NaN，再计算平均（skipna=True）
    rel_err_capped = rel_err_df.mask(abs(rel_err_df) > max_thresh, np.nan)
    mean_err = rel_err_capped.mean()

    return rel_err_df, max_err, mean_err
