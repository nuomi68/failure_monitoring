from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler,
    PowerTransformer, QuantileTransformer, Normalizer, LabelEncoder,
)
from typing import Any, Dict, Tuple
import pandas as pd
import numpy as np

# 供前端显示的缩放器选项
SCALERS_DISPLAY = [("标准化 (Z‑score)", "standard"),
                   ("MinMax [0,1]", "minmax"),
                   ("Robust (IQR)", "robust"),
                   ("MaxAbs", "maxabs"),
                   ("PowerTransformer", "power"),
                   ("QuantileTransformer", "quantile"),
                   ("Normalizer (L2)", "normalizer"),
                   ("不做缩放", "none")]


def make_scaler(spec: Any):
    """按照 ml_interface 的规则构造缩放器。"""
    if spec is None or spec is False:
        return None
    if isinstance(spec, str):
        name = spec.lower()
        params: Dict[str, Any] = {}
    elif isinstance(spec, dict):
        name = str(spec.get("name", "standard")).lower()
        params = dict(spec.get("params", {}))
    else:
        # 若传入已有对象（可能已拟合），直接返回
        if hasattr(spec, "transform"):
            return spec
        raise ValueError(f"Unsupported scaler spec: {type(spec)}")

    if name in ("standard", "std", "zscore"):
        return StandardScaler(**params)
    if name in ("minmax", "min_max"):
        return MinMaxScaler(**params)
    if name in ("robust",):
        return RobustScaler(**params)
    if name in ("maxabs", "max_abs"):
        return MaxAbsScaler(**params)
    if name in ("power", "yeojohnson", "boxcox"):
        params = {"method": params.get("method", "yeo-johnson"), **params}
        return PowerTransformer(**params)
    if name in ("quantile", "rank"):
        return QuantileTransformer(**params)
    if name in ("normalizer", "l2", "l1", "max"):
        if name in ("l1", "l2", "max"):
            params = {"norm": name, **params}
        return Normalizer(**params)
    if name in ("none",):
        return None
    raise ValueError(f"Unknown scaler name: {name}")


def encode_and_scale(
    df: pd.DataFrame, scaler_spec: Any = "standard"
) -> Tuple[np.ndarray, Any, Dict[str, LabelEncoder]]:
    """Encode categorical columns and apply scaling in one step.

    Parameters
    ----------
    df : pd.DataFrame
        Raw feature table from front-end.
    scaler_spec : Any, default "standard"
        Specification passed to :func:`make_scaler`.

    Returns
    -------
    tuple
        ``(data_scaled, scaler_obj, encoders)`` where ``data_scaled`` is a
        ``numpy.ndarray`` after encoding and scaling, ``scaler_obj`` is the
        fitted scaler (or ``None``) and ``encoders`` maps column names to
        their ``LabelEncoder``.
    """

    work = df.copy()
    encoders: Dict[str, LabelEncoder] = {}
    for col in work.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        work[col] = le.fit_transform(work[col].astype(str))
        encoders[col] = le

    scaler = make_scaler(scaler_spec)
    arr = work.to_numpy(dtype=float)
    if scaler is not None:
        arr = scaler.fit_transform(arr)

    return arr, scaler, encoders