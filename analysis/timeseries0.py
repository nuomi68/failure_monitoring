from __future__ import annotations

import json
import os
import random
import re
import time
from importlib import import_module
from pathlib import Path
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.data_utils import build_windows

DATA_DIR = ROOT_DIR / "data"
REPORTS_DIR = ROOT_DIR / "reports"

# ================================
# Config (edit values in this block)
# ================================

# Saved artifacts (models/scalers/splits) root directory.
MODELS_DIR = ROOT_DIR / "models_saved" / "timeseries0"

# Public dataset (datasetsforecast.LongHorizon) settings.
LH_DIRECTORY = DATA_DIR  # download/cache directory
LH_GROUPS: Tuple[str, ...] = ()#("ETTh1", "Weather")  # set () to disable
LH_COLUMNS = ""  # optional: comma-separated series names (empty = all)
LH_MAX_SERIES = 0  # optional: cap #series after pivot (0 = no limit)

# Local Excel datasets (set () to disable).
LOCAL_FILES: Tuple[Path, ...] = (DATA_DIR / "平均功率.xlsx",)
LOCAL_TIME_COL = "时间"
LOCAL_FEATURE_COLS: Tuple[str, ...] = (
    "一回路平均温度",
    "硼浓度（硼表）",
    "平均核功率",
    "燃耗",
)
LOCAL_TARGET_COLS: Tuple[str, ...] = ("平均核功率",)
# Local Excel evaluation mode:
# - "full": evaluate full dataset only (default behaviour)
# - "halves": split into two time-ordered parts and evaluate each part
# - "both": evaluate full dataset + both halves
LOCAL_EXCEL_EVAL_MODE = "halves"
LOCAL_EXCEL_SPLIT_STRATEGY = "auto_zero_run"  # "auto_zero_run" | "ratio"
LOCAL_EXCEL_SPLIT_RATIO = 0.5  # used when strategy="ratio" or auto fallback
LOCAL_EXCEL_ZERO_THRESHOLD = 1.0  # used when strategy="auto_zero_run"
LOCAL_EXCEL_MIN_ZERO_RUN = 500  # used when strategy="auto_zero_run"

# Device: "auto" / "cpu" / "cuda" / "cuda:0" ...
DEVICE_SETTING = "auto"

# Reproducibility.
SEED = 42

# Relative error reporting (for targets).
# Note: when true values are near zero, percentage errors can blow up; denom_floor prevents that.
REL_ERR_THRESHOLD = 0.05  # 5%
REL_ERR_DENOM_FLOOR = 1.0
REPORT_SCALED_METRICS = (os.getenv("TS0_REPORT_SCALED_METRICS", "1") or "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}

# Split ratio (time-ordered; must sum to 1.0).
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

# Common training baseline params.
TRAIN_LOOK_BACK = 96
TRAIN_EPOCHS = int(os.getenv("TS0_EPOCHS", "60"))
TRAIN_BATCH_SIZE = 32
TRAIN_LR = 1e-3
TRAIN_DROPOUT = 0.1
TCN_HID = int(os.getenv("TS0_TCN_HID", "32"))
TCN_LEVELS = int(os.getenv("TS0_TCN_LEVELS", "3"))
TCN_K = int(os.getenv("TS0_TCN_K", "3"))
TCN_BATCH_SIZE = int(os.getenv("TS0_TCN_BATCH_SIZE", str(max(TRAIN_BATCH_SIZE, 256))))
_patience_env = os.getenv("TS0_PATIENCE")
if _patience_env is None:
    TRAIN_PATIENCE: Optional[int] = 8  # set None to disable early-stopping
else:
    _patience_env = _patience_env.strip()
    TRAIN_PATIENCE = None if not _patience_env or _patience_env.lower() in {"none", "null"} else int(_patience_env)

# Models to evaluate (in order). Override with `TS0_MODELS="gru,tsmixer,timesnet,tcn"` if needed.
# Note: a 1-item tuple must have a trailing comma, e.g. ("tcn",)
MODEL_ORDER: Tuple[str, ...] = ("tcn","gru","tsmixer","timesnet")

# TimesNet tuning knobs (kept consistent across datasets/parts).
TIMESNET_LOOK_BACK = TRAIN_LOOK_BACK
TIMESNET_EPOCHS = TRAIN_EPOCHS
TIMESNET_BATCH_SIZE = TRAIN_BATCH_SIZE
TIMESNET_LR = 5e-4
TIMESNET_D_MODEL = 64
TIMESNET_NUM_BLOCKS = 3
TIMESNET_PATIENCE: Optional[int] = TRAIN_PATIENCE
TIMESNET_RESIDUAL = True

def _resolve_device(setting: str) -> str:
    raw = (setting or "").strip()
    if not raw:
        raw = "auto"

    lower = raw.lower()
    if lower in {"auto", "default"}:
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    if lower in {"gpu", "cuda"}:
        candidate = "cuda"
    elif lower.startswith("cuda:"):
        candidate = lower
    elif lower.startswith("cuda"):
        candidate = "cuda"
    else:
        candidate = "cpu"

    if candidate.startswith("cuda"):
        try:
            import torch

            if not torch.cuda.is_available():
                return "cpu"
        except Exception:
            return "cpu"

    return candidate


DEVICE = _resolve_device(DEVICE_SETTING)

DEFAULT_LOOKBACK = {
    "gru": TRAIN_LOOK_BACK,
    "tcn": TRAIN_LOOK_BACK,
    "tsmixer": TRAIN_LOOK_BACK,
    "rf": TRAIN_LOOK_BACK,
    "xgb": TRAIN_LOOK_BACK,
    "timesnet": TRAIN_LOOK_BACK,
}

MODEL_MODULES: Dict[str, Tuple[str, str, str]] = {
    "gru": ("backend.models.gru_model", "train_gru", "predict"),
    "tcn": ("backend.models.tcn_model", "train_tcn", "predict"),
    "tsmixer": ("backend.models.tsmixer_model", "train_tsmixer", "predict"),
    "rf": ("backend.models.random_forest_model", "train_rf", "predict"),
    "xgb": ("backend.models.xgboost_model", "train_xgb", "predict"),
    "timesnet": ("backend.models.timesnet_model", "train_timesnet", "predict"),
}

COMMON_TORCH_PARAMS: Dict[str, Any] = {
    "look_back": TRAIN_LOOK_BACK,
    "epochs": TRAIN_EPOCHS,
    "batch_size": TRAIN_BATCH_SIZE,
    "lr": TRAIN_LR,
    "patience": TRAIN_PATIENCE,
    "device": DEVICE,
}

MODEL_PARAM_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "gru": {
        **COMMON_TORCH_PARAMS,
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": TRAIN_DROPOUT,
    },
    "tcn": {
        **COMMON_TORCH_PARAMS,
        "batch_size": TCN_BATCH_SIZE,
        "hid": TCN_HID,
        "levels": TCN_LEVELS,
        "k": TCN_K,
        "drop": TRAIN_DROPOUT,
    },
    "tsmixer": {
        **COMMON_TORCH_PARAMS,
        "num_blocks": 4,
        "ff_dim": 256,
        "dropout": TRAIN_DROPOUT,
    },
    "rf": {
        "look_back": TRAIN_LOOK_BACK,
        "n_estimators": 600,
        "min_samples_leaf": 2,
        "random_state": SEED,
        "n_jobs": -1,
    },
    "xgb": {
        "look_back": TRAIN_LOOK_BACK,
        "n_estimators": 600,
        "learning_rate": 0.05,
        "max_depth": 4,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "random_state": SEED,
        "n_jobs": 1,
    },
    "timesnet": {
        **COMMON_TORCH_PARAMS,
        "look_back": TIMESNET_LOOK_BACK,
        "epochs": TIMESNET_EPOCHS,
        "batch_size": TIMESNET_BATCH_SIZE,
        "lr": TIMESNET_LR,
        "patience": TIMESNET_PATIENCE,
        "d_model": TIMESNET_D_MODEL,
        "num_blocks": TIMESNET_NUM_BLOCKS,
        "residual": TIMESNET_RESIDUAL,
    },
}


def _set_global_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        try:
            torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
            torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
        except Exception:
            pass
    except Exception:
        pass


def _safe_slug(text: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff._-]+", "_", (text or "").strip())
    return cleaned.strip("._-") or "artifact"


def _normalize_model_order(value: Any) -> List[str]:
    if isinstance(value, str):
        parsed = [m.strip() for m in value.split(",") if m.strip()]
        return parsed
    try:
        return [str(m).strip() for m in value if str(m).strip()]
    except TypeError:
        text = str(value).strip()
        return [text] if text else []


def _load_clean_features(
    group: str,
    *,
    min_look_back: int,
) -> Tuple[np.ndarray, List[str], StandardScaler, Dict[str, int]]:
    try:
        from datasetsforecast.long_horizon import LongHorizon  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Missing dependency 'datasetsforecast'. Install via: pip install datasetsforecast"
        ) from exc

    y_df, *_ = LongHorizon.load(directory=str(LH_DIRECTORY), group=group)
    required = {"unique_id", "ds", "y"}
    if not required.issubset(set(y_df.columns)):
        raise RuntimeError(
            f"Unexpected LongHorizon schema for group={group!r}; "
            f"expected columns {sorted(required)}, got {sorted(y_df.columns)}"
        )

    y_df = y_df.copy()
    y_df["ds"] = pd.to_datetime(y_df["ds"])

    wide = (
        y_df.pivot(index="ds", columns="unique_id", values="y")
        .sort_index()
        .ffill()
        .bfill()
    )

    if LH_COLUMNS:
        requested = [c.strip() for c in LH_COLUMNS.split(",") if c.strip()]
        missing = [c for c in requested if c not in wide.columns]
        if missing:
            raise RuntimeError(
                f"Requested series not found in LongHorizon/{group}: {', '.join(missing)}"
            )
        wide = wide[requested]
    elif LH_MAX_SERIES > 0 and wide.shape[1] > LH_MAX_SERIES:
        wide = wide.iloc[:, :LH_MAX_SERIES]

    if wide.shape[0] < 10 or wide.shape[1] < 1:
        raise RuntimeError(
            f"LongHorizon/{group} did not produce enough data after pivot: "
            f"{wide.shape[0]} rows, {wide.shape[1]} series."
        )

    n = int(wide.shape[0])
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))
    train_end = max(2, min(train_end, n - 2))
    val_end = max(train_end + 1, min(val_end, n - 1))

    if train_end < min_look_back + 1:
        raise RuntimeError(
            f"Train split too small for look_back={min_look_back} on LongHorizon/{group}: "
            f"train_end={train_end}, n={n}."
        )
    if val_end <= train_end:
        raise RuntimeError(f"Invalid split for LongHorizon/{group}: train_end={train_end}, val_end={val_end}")

    scaler = StandardScaler().fit(wide.iloc[:train_end].astype(float).values)
    data_scaled = scaler.transform(wide.astype(float).values)
    feature_cols = [str(c) for c in wide.columns]
    return (
        data_scaled.astype(np.float32),
        feature_cols,
        scaler,
        {"n": n, "train_end": train_end, "val_end": val_end},
    )


def _load_local_excel_features(
    path: Path,
    *,
    min_look_back: int,
) -> Tuple[np.ndarray, List[str], StandardScaler, Dict[str, int], Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(str(path))

    df = pd.read_excel(path)
    if df.empty:
        raise RuntimeError(f"Empty dataset: {path}")

    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    time_col = LOCAL_TIME_COL
    missing = [c for c in (time_col, *LOCAL_FEATURE_COLS) if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"Local Excel header mismatch for {path.name}. Missing columns: {', '.join(missing)}"
        )

    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
    if df.empty:
        raise RuntimeError(f"No valid timestamp rows in dataset: {path}")

    feature_cols = list(LOCAL_FEATURE_COLS)
    feat_df = df[feature_cols].apply(pd.to_numeric, errors="coerce").ffill().bfill()
    if feat_df.isna().any().any():
        feat_df = feat_df.fillna(0.0)

    n = int(len(feat_df))
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))
    train_end = max(2, min(train_end, n - 2))
    val_end = max(train_end + 1, min(val_end, n - 1))

    if train_end < min_look_back + 1:
        raise RuntimeError(
            f"Train split too small for look_back={min_look_back} on {path.name}: train_end={train_end}, n={n}."
        )
    if val_end <= train_end:
        raise RuntimeError(f"Invalid split for {path.name}: train_end={train_end}, val_end={val_end}")

    scaler = StandardScaler().fit(feat_df.iloc[:train_end].astype(float).values)
    data_scaled = scaler.transform(feat_df.astype(float).values)

    meta: Dict[str, Any] = {
        "path": str(path),
        "time_col": str(time_col),
        "num_rows_raw": int(df.shape[0]),
    }
    return (
        data_scaled.astype(np.float32),
        [str(c) for c in feature_cols],
        scaler,
        {"n": n, "train_end": train_end, "val_end": val_end},
        meta,
    )


def _compute_split_points(
    n: int,
    *,
    min_look_back: int,
    label: str,
) -> Dict[str, int]:
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))
    train_end = max(2, min(train_end, n - 2))
    val_end = max(train_end + 1, min(val_end, n - 1))

    if train_end < min_look_back + 1:
        raise RuntimeError(
            f"Train split too small for look_back={min_look_back} on {label}: train_end={train_end}, n={n}."
        )
    if val_end <= train_end:
        raise RuntimeError(f"Invalid split for {label}: train_end={train_end}, val_end={val_end}")

    return {"n": int(n), "train_end": int(train_end), "val_end": int(val_end)}


def _load_local_excel_raw_features(
    path: Path,
) -> Tuple[pd.DataFrame, pd.Series, List[str], Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(str(path))

    df = pd.read_excel(path)
    if df.empty:
        raise RuntimeError(f"Empty dataset: {path}")

    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    time_col = LOCAL_TIME_COL
    missing = [c for c in (time_col, *LOCAL_FEATURE_COLS) if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"Local Excel header mismatch for {path.name}. Missing columns: {', '.join(missing)}"
        )

    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
    if df.empty:
        raise RuntimeError(f"No valid timestamp rows in dataset: {path}")

    time_values = df[time_col].copy()

    feature_cols = list(LOCAL_FEATURE_COLS)
    feat_df = df[feature_cols].apply(pd.to_numeric, errors="coerce").ffill().bfill()
    if feat_df.isna().any().any():
        feat_df = feat_df.fillna(0.0)

    meta: Dict[str, Any] = {
        "path": str(path),
        "time_col": str(time_col),
        "num_rows_raw": int(df.shape[0]),
        "time_start": str(df[time_col].iloc[0]),
        "time_end": str(df[time_col].iloc[-1]),
    }
    return feat_df, time_values, [str(c) for c in feature_cols], meta


def _load_model_adapters() -> Tuple[
    Dict[str, Dict[str, Callable[..., Any]]],
    Dict[str, str],
]:
    registry: Dict[str, Dict[str, Callable[..., Any]]] = {}
    failures: Dict[str, str] = {}

    for name, (module_path, train_attr, predict_attr) in MODEL_MODULES.items():
        try:
            module = import_module(module_path)
            registry[name] = {
                "train": getattr(module, train_attr),
                "predict": getattr(module, predict_attr),
            }
        except Exception as exc:
            failures[name] = f"{type(exc).__name__}: {exc}"
    return registry, failures


def _split_windows_7_2_1(
    data_scaled: np.ndarray,
    look_back: int,
    *,
    train_end: int,
    val_end: int,
    target_indices: Optional[List[int]] = None,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    if look_back < 1:
        raise ValueError("look_back must be >= 1")
    X_all, y_all = build_windows(data_scaled, look_back)
    if len(X_all) == 0:
        raise RuntimeError("Not enough windows to build train/val/test splits.")
    if target_indices is not None:
        y_all = y_all[:, list(target_indices)]
    targets = np.arange(look_back, len(data_scaled))
    if len(targets) != len(X_all):
        raise RuntimeError("Internal window alignment mismatch.")

    train_mask = targets < train_end
    val_mask = (targets >= train_end) & (targets < val_end)
    test_mask = targets >= val_end

    X_train, y_train = X_all[train_mask], y_all[train_mask]
    X_val, y_val = X_all[val_mask], y_all[val_mask]
    X_test, y_test = X_all[test_mask], y_all[test_mask]

    if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
        raise RuntimeError(
            "Split produced empty windows: "
            f"train={len(X_train)}, val={len(X_val)}, test={len(X_test)} "
            f"(look_back={look_back}, n={len(data_scaled)}, train_end={train_end}, val_end={val_end})."
        )

    return X_train, X_val, X_test, y_train, y_val, y_test


def _inverse_transform_selected(
    arr: np.ndarray,
    scaler: StandardScaler,
    *,
    target_indices: Optional[List[int]],
) -> np.ndarray:
    if arr.size == 0:
        return arr
    if target_indices is None:
        return scaler.inverse_transform(arr)

    idx = np.asarray(list(target_indices), dtype=int)
    mean = np.asarray(scaler.mean_, dtype=np.float32)[idx]
    scale = np.asarray(scaler.scale_, dtype=np.float32)[idx]
    arr2d = arr if arr.ndim == 2 else arr.reshape(-1, 1)
    return arr2d * scale + mean


def _regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scaler: StandardScaler,
    *,
    target_indices: Optional[List[int]] = None,
) -> Dict[str, float]:
    if y_true.size == 0:
        return {}
    y_true_inv = _inverse_transform_selected(y_true, scaler, target_indices=target_indices)
    y_pred_inv = _inverse_transform_selected(y_pred, scaler, target_indices=target_indices)
    denom_floor = float(REL_ERR_DENOM_FLOOR)
    denom = np.maximum(np.abs(y_true_inv), denom_floor)
    ape = np.abs(y_pred_inv - y_true_inv) / denom
    ape_flat = np.asarray(ape, dtype=np.float32).reshape(-1)
    return {
        "mae": float(mean_absolute_error(y_true_inv, y_pred_inv)),
        "rmse": float(np.sqrt(mean_squared_error(y_true_inv, y_pred_inv))),
        "r2": float(r2_score(y_true_inv, y_pred_inv)),
        "mape_pct": float(np.mean(ape_flat) * 100.0),
        "p95_ape_pct": float(np.quantile(ape_flat, 0.95) * 100.0),
        "within_5pct": float(np.mean(ape_flat <= float(REL_ERR_THRESHOLD)) * 100.0),
        "small_true_pct": float(np.mean(np.abs(y_true_inv).reshape(-1) < denom_floor) * 100.0),
    }


def _regression_metrics_scaled(
    y_true_scaled: np.ndarray,
    y_pred_scaled: np.ndarray,
) -> Dict[str, float]:
    if y_true_scaled.size == 0:
        return {}
    y_true = np.asarray(y_true_scaled, dtype=np.float32)
    y_pred = np.asarray(y_pred_scaled, dtype=np.float32)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _series_summary(values: pd.Series) -> Dict[str, float]:
    arr = pd.to_numeric(values, errors="coerce").dropna()
    if arr.empty:
        return {}
    q25 = float(arr.quantile(0.25))
    q50 = float(arr.quantile(0.5))
    q75 = float(arr.quantile(0.75))
    return {
        "count": int(len(arr)),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "p25": q25,
        "p50": q50,
        "p75": q75,
        "iqr": float(q75 - q25),
        "max": float(arr.max()),
    }


def _persistence_baseline_metrics(
    data_scaled: np.ndarray,
    scaler: StandardScaler,
    *,
    look_back: int,
    split: Dict[str, int],
    target_indices: Optional[List[int]] = None,
) -> Dict[str, Dict[str, float]]:
    n = int(split["n"])
    train_end = int(split["train_end"])
    val_end = int(split["val_end"])

    def _select(arr: np.ndarray) -> np.ndarray:
        if target_indices is None:
            return arr
        return arr[:, list(target_indices)]

    def _metrics_for_targets(targets: np.ndarray) -> Dict[str, float]:
        if len(targets) == 0:
            return {}
        y_true = _select(data_scaled[targets])
        y_pred = _select(data_scaled[targets - 1])
        return _regression_metrics(y_true, y_pred, scaler, target_indices=target_indices)

    train_targets = np.arange(int(look_back), int(train_end))
    val_targets = np.arange(max(int(look_back), int(train_end)), int(val_end))
    test_targets = np.arange(max(int(look_back), int(val_end)), int(n))

    return {
        "train": _metrics_for_targets(train_targets),
        "validation": _metrics_for_targets(val_targets),
        "test": _metrics_for_targets(test_targets),
    }


def _persistence_baseline_metrics_scaled(
    data_scaled: np.ndarray,
    *,
    look_back: int,
    split: Dict[str, int],
    target_indices: Optional[List[int]] = None,
) -> Dict[str, Dict[str, float]]:
    n = int(split["n"])
    train_end = int(split["train_end"])
    val_end = int(split["val_end"])

    def _select(arr: np.ndarray) -> np.ndarray:
        if target_indices is None:
            return arr
        return arr[:, list(target_indices)]

    def _metrics_for_targets(targets: np.ndarray) -> Dict[str, float]:
        if len(targets) == 0:
            return {}
        y_true = _select(data_scaled[targets])
        y_pred = _select(data_scaled[targets - 1])
        return _regression_metrics_scaled(y_true, y_pred)

    train_targets = np.arange(int(look_back), int(train_end))
    val_targets = np.arange(max(int(look_back), int(train_end)), int(val_end))
    test_targets = np.arange(max(int(look_back), int(val_end)), int(n))

    return {
        "train": _metrics_for_targets(train_targets),
        "validation": _metrics_for_targets(val_targets),
        "test": _metrics_for_targets(test_targets),
    }


def _window_time_sanity(
    times: pd.Series,
    *,
    look_back: int,
    split: Dict[str, int],
    sample_count: int = 5,
) -> Dict[str, Any]:
    n = int(split["n"])
    train_end = int(split["train_end"])
    val_end = int(split["val_end"])

    def _bounds(start: int, end: int) -> Dict[str, Any]:
        if end < start or start < 0 or end >= n:
            return {}
        return {
            "target_index_start": int(start),
            "target_time_start": str(times.iloc[start]),
            "target_index_end": int(end),
            "target_time_end": str(times.iloc[end]),
            "first_input_index_start": int(start - look_back),
            "first_input_time_start": str(times.iloc[start - look_back]),
            "first_input_index_end": int(start - 1),
            "first_input_time_end": str(times.iloc[start - 1]),
            "last_input_index_start": int(end - look_back),
            "last_input_time_start": str(times.iloc[end - look_back]),
            "last_input_index_end": int(end - 1),
            "last_input_time_end": str(times.iloc[end - 1]),
        }

    train_start = int(look_back)
    train_end_idx = int(train_end - 1)
    val_start = int(max(look_back, train_end))
    val_end_idx = int(val_end - 1)
    test_start = int(max(look_back, val_end))
    test_end_idx = int(n - 1)

    sanity: Dict[str, Any] = {
        "look_back": int(look_back),
        "bounds": {
            "train": _bounds(train_start, train_end_idx),
            "validation": _bounds(val_start, val_end_idx),
            "test": _bounds(test_start, test_end_idx),
        },
        "samples": {"test": []},
    }

    test_targets = np.arange(test_start, n, dtype=int)
    if len(test_targets) and sample_count > 0:
        rng = np.random.default_rng(int(SEED))
        chosen = rng.choice(test_targets, size=min(int(sample_count), len(test_targets)), replace=False)
        chosen = np.sort(chosen)
        samples = []
        for t in chosen.tolist():
            t = int(t)
            samples.append(
                {
                    "target_index": t,
                    "target_time": str(times.iloc[t]),
                    "input_index_start": int(t - look_back),
                    "input_time_start": str(times.iloc[t - look_back]),
                    "input_index_end": int(t - 1),
                    "input_time_end": str(times.iloc[t - 1]),
                }
            )
        sanity["samples"]["test"] = samples

    return sanity


METRIC_DISPLAY_ORDER: Tuple[str, ...] = (
    "mae",
    "rmse",
    "r2",
    "mape_pct",
    "within_5pct",
    "p95_ape_pct",
    "small_true_pct",
)


def _format_metrics(metrics: Dict[str, float], prefix: str) -> str:
    parts: List[str] = []
    for key in METRIC_DISPLAY_ORDER:
        if key not in metrics:
            continue
        value = float(metrics[key])
        if key.endswith("_pct") or key.endswith("5pct"):
            parts.append(f"{prefix}_{key}={value:.2f}%")
        else:
            parts.append(f"{prefix}_{key}={value:.6e}")
    return ", ".join(parts)


def evaluate_model(
    name: str,
    adapters: Dict[str, Dict[str, Callable[..., Any]]],
    data_scaled: np.ndarray,
    scaler: StandardScaler,
    *,
    dataset_label: str,
    split: Dict[str, int],
    out_dir: Path,
    target_indices: Optional[List[int]] = None,
) -> Dict[str, Any]:
    if name not in adapters:
        raise KeyError(f"Model '{name}' is not available (dependency missing).")

    params = MODEL_PARAM_OVERRIDES.get(name, {}).copy()
    if name == "timesnet" and target_indices is not None and len(target_indices) == 1:
        params.setdefault("target_index", int(target_indices[0]))
    look_back = int(params.pop("look_back", DEFAULT_LOOKBACK.get(name, 14)))
    X_train, X_val, X_test, y_train, y_val, y_test = _split_windows_7_2_1(
        data_scaled,
        look_back,
        train_end=int(split["train_end"]),
        val_end=int(split["val_end"]),
        target_indices=target_indices,
    )

    logs: List[str] = []

    def _log(msg: str) -> None:
        print(f"[{dataset_label}][{name}] {msg}", flush=True)
        logs.append(msg)

    _log(f"seed={SEED} look_back={look_back} train/val/test={len(X_train)}/{len(X_val)}/{len(X_test)}")
    start = time.time()
    _set_global_seeds(SEED)
    train_fn = adapters[name]["train"]
    if name in {"rf", "xgb"}:
        result = train_fn(X_train, y_train, X_val, y_val, log_callback=_log, **params)
    else:
        result = train_fn(
            X_train,
            y_train,
            X_val,
            y_val,
            log_callback=_log,
            **params,
        )
    duration = time.time() - start

    model_obj = result[0] if isinstance(result, tuple) else result
    predict_fn = adapters[name]["predict"]
    out_dim = int(y_train.shape[1]) if y_train.ndim == 2 else 1
    batch_predict_warned = False

    def _predict_batch(samples: np.ndarray) -> np.ndarray:
        nonlocal batch_predict_warned
        if len(samples) == 0:
            return np.empty((0, out_dim), dtype=np.float32)

        if name in {"rf", "xgb"}:
            flat = samples.reshape(len(samples), -1)
            arr = np.asarray(model_obj.predict(flat), dtype=np.float32)
            return arr.reshape(len(samples), out_dim)

        if name in {"gru", "tcn", "tsmixer", "timesnet"}:
            try:
                import torch

                model_obj.eval()
                try:
                    device = str(next(model_obj.parameters()).device)
                except StopIteration:
                    device = "cpu"

                batch_size = int(params.get("batch_size", TRAIN_BATCH_SIZE))
                outputs: List[np.ndarray] = []
                with torch.no_grad():
                    for start in range(0, len(samples), batch_size):
                        xb = torch.tensor(
                            samples[start : start + batch_size], dtype=torch.float32, device=device
                        )
                        if name == "tcn":
                            xb = xb.permute(0, 2, 1)  # (B, F, L)
                        out = model_obj(xb)
                        if isinstance(out, (tuple, list)):
                            out = out[0]
                        if getattr(out, "ndim", 0) == 3 and out.shape[1] == 1:
                            out = out.squeeze(1)
                        outputs.append(out.detach().cpu().numpy())

                arr = np.concatenate(outputs, axis=0)
                return np.asarray(arr, dtype=np.float32).reshape(len(samples), out_dim)
            except Exception as exc:
                if not batch_predict_warned:
                    _log(
                        f"Warning: batched predict failed ({type(exc).__name__}: {exc}); falling back to per-sample predict."
                    )
                    batch_predict_warned = True

        preds = [predict_fn(model_obj, seq) for seq in samples]
        arr = np.asarray(preds, dtype=np.float32)
        return arr.reshape(len(preds), out_dim)

    train_preds = _predict_batch(X_train)
    val_preds = _predict_batch(X_val)
    test_preds = _predict_batch(X_test)

    metrics = {
        "train": _regression_metrics(y_train, train_preds, scaler, target_indices=target_indices),
        "validation": _regression_metrics(y_val, val_preds, scaler, target_indices=target_indices),
        "test": _regression_metrics(y_test, test_preds, scaler, target_indices=target_indices),
    }
    metrics_scaled = {
        "train": _regression_metrics_scaled(y_train, train_preds),
        "validation": _regression_metrics_scaled(y_val, val_preds),
        "test": _regression_metrics_scaled(y_test, test_preds),
    }

    artifacts: Dict[str, str] = {}
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "dataset": dataset_label,
            "seed": int(SEED),
            "device": DEVICE,
            "model": name,
            "params": {"look_back": int(look_back), **params},
        }
        if target_indices is not None:
            payload["target_indices"] = list(target_indices)
        meta_path = out_dir / f"{_safe_slug(name)}_meta.json"
        meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        artifacts["meta_path"] = str(meta_path)

        if name in {"gru", "tcn", "tsmixer", "timesnet"}:
            import torch

            state_dict = {k: v.detach().cpu() for k, v in model_obj.state_dict().items()}
            model_path = out_dir / f"{_safe_slug(name)}.pt"
            torch.save({"state_dict": state_dict, "meta": payload}, model_path)
            artifacts["model_path"] = str(model_path)
        else:
            model_path = out_dir / f"{_safe_slug(name)}.joblib"
            joblib.dump(model_obj, model_path)
            artifacts["model_path"] = str(model_path)
    except Exception as exc:
        _log(f"Warning: failed to save model artifacts ({type(exc).__name__}: {exc})")

    return {
        "status": "ok",
        "look_back": look_back,
        "params": {"look_back": int(look_back), **params},
        "train_samples": int(len(X_train)),
        "val_samples": int(len(X_val)),
        "test_samples": int(len(X_test)),
        "metrics": metrics,
        "metrics_scaled": metrics_scaled,
        "logs": logs,
        "train_time_sec": float(duration),
        "artifacts": artifacts,
    }


def main() -> None:
    adapters, failures = _load_model_adapters()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    requested = (DEVICE_SETTING or "").strip().lower()
    if DEVICE == "cpu" and (requested in {"cuda", "gpu"} or requested.startswith("cuda")):
        print("Warning: DEVICE_SETTING requested CUDA but CUDA is not available; falling back to CPU.", flush=True)

    max_look_back = max(
        *DEFAULT_LOOKBACK.values(),
        *(
            int(params.get("look_back", 0))
            for params in MODEL_PARAM_OVERRIDES.values()
            if isinstance(params, dict)
        ),
    )

    local_files = [p for p in LOCAL_FILES if p.exists()]
    missing_local_files = [p for p in LOCAL_FILES if not p.exists()]
    if missing_local_files:
        missing_display = ", ".join(str(p) for p in missing_local_files[:5])
        suffix = "..." if len(missing_local_files) > 5 else ""
        print(f"Warning: local dataset file(s) not found; skipping: {missing_display}{suffix}", flush=True)

    local_eval_mode = (LOCAL_EXCEL_EVAL_MODE or "full").strip().lower()
    if local_eval_mode not in {"full", "halves", "both"}:
        raise ValueError(
            f"Invalid LOCAL_EXCEL_EVAL_MODE={LOCAL_EXCEL_EVAL_MODE!r}; expected 'full', 'halves', or 'both'."
        )

    model_order = _normalize_model_order(MODEL_ORDER)
    raw_model_order = (os.getenv("TS0_MODELS") or "").strip()
    if raw_model_order:
        requested = [m.strip() for m in raw_model_order.split(",") if m.strip()]
        invalid = [m for m in requested if m not in MODEL_MODULES]
        if invalid:
            raise ValueError(
                f"Invalid TS0_MODELS={raw_model_order!r}; unknown model(s): {', '.join(invalid)}"
            )
        model_order = requested

    dataset_labels: List[str] = [f"LongHorizon/{g}" for g in LH_GROUPS]
    for p in local_files:
        base = f"LocalExcel/{p.name}"
        if local_eval_mode in {"full", "both"}:
            dataset_labels.append(base)
        if local_eval_mode in {"halves", "both"}:
            dataset_labels.append(f"{base}::part1")
            dataset_labels.append(f"{base}::part2")

    run_meta = {
        "seed": int(SEED),
        "device": DEVICE,
        "split_ratio": {"train": TRAIN_RATIO, "val": VAL_RATIO, "test": TEST_RATIO},
        "local_excel_eval_mode": local_eval_mode,
        "local_target_cols": list(LOCAL_TARGET_COLS),
        "relative_error": {
            "threshold": float(REL_ERR_THRESHOLD),
            "denom_floor": float(REL_ERR_DENOM_FLOOR),
        },
        "timesnet_params": {
            "look_back": int(TIMESNET_LOOK_BACK),
            "epochs": int(TIMESNET_EPOCHS),
            "batch_size": int(TIMESNET_BATCH_SIZE),
            "lr": float(TIMESNET_LR),
            "d_model": int(TIMESNET_D_MODEL),
            "num_blocks": int(TIMESNET_NUM_BLOCKS),
            "patience": TIMESNET_PATIENCE,
            "residual": bool(TIMESNET_RESIDUAL),
        },
        "train_params": {
            "look_back": int(TRAIN_LOOK_BACK),
            "epochs": int(TRAIN_EPOCHS),
            "batch_size": int(TRAIN_BATCH_SIZE),
            "lr": float(TRAIN_LR),
            "dropout": float(TRAIN_DROPOUT),
            "patience": TRAIN_PATIENCE,
        },
        "model_order": list(model_order),
        "datasets": dataset_labels,
        "save_dir": str(MODELS_DIR),
    }

    summary: Dict[str, Any] = {
        "run": run_meta,
        "datasets": {},
        "failures": failures,
    }

    lines = [
        "Multi-model Time-Series Forecast Benchmark (7:2:1)",
        "=" * 52,
        f"Datasets: {', '.join(dataset_labels)}",
        f"Device: {DEVICE} | Seed: {SEED}",
        f"Split: train/val/test = {TRAIN_RATIO:.1f}/{VAL_RATIO:.1f}/{TEST_RATIO:.1f}",
        f"Relative error: threshold={REL_ERR_THRESHOLD * 100:.1f}%, denom_floor={REL_ERR_DENOM_FLOOR:g}",
        "Metrics: original(inverse_transform)"
        + (" + scaled(StandardScaler space)" if REPORT_SCALED_METRICS else ""),
        f"TimesNet params: look_back={TIMESNET_LOOK_BACK}, epochs={TIMESNET_EPOCHS}, batch_size={TIMESNET_BATCH_SIZE}, lr={TIMESNET_LR:g}, d_model={TIMESNET_D_MODEL}, num_blocks={TIMESNET_NUM_BLOCKS}, residual={TIMESNET_RESIDUAL}, patience={TIMESNET_PATIENCE}",
        f"Train params: look_back={TRAIN_LOOK_BACK}, epochs={TRAIN_EPOCHS}, batch_size={TRAIN_BATCH_SIZE}, lr={TRAIN_LR:g}, dropout={TRAIN_DROPOUT:g}, patience={TRAIN_PATIENCE}",
        f"Save dir: {MODELS_DIR}",
        "",
    ]

    for group in LH_GROUPS:
        dataset_label = f"LongHorizon/{group}"
        lines.append("")
        lines.append(f"Dataset: {dataset_label}")
        lines.append("-" * 52)

        try:
            data_scaled, feature_cols, scaler, split = _load_clean_features(
                group, min_look_back=max_look_back
            )
            n = int(split["n"])
            train_end = int(split["train_end"])
            val_end = int(split["val_end"])
            split_sizes = {
                "train": int(train_end),
                "val": int(val_end - train_end),
                "test": int(n - val_end),
            }

            dataset_dir = MODELS_DIR / f"{_safe_slug(dataset_label)}_seed{SEED}"
            model_dir = dataset_dir / "models"
            dataset_dir.mkdir(parents=True, exist_ok=True)
            model_dir.mkdir(parents=True, exist_ok=True)

            scaler_path = dataset_dir / "scaler.joblib"
            joblib.dump(scaler, scaler_path)
            feature_path = dataset_dir / "feature_cols.json"
            feature_path.write_text(json.dumps(feature_cols, indent=2), encoding="utf-8")
            split_path = dataset_dir / "split.json"
            split_path.write_text(
                json.dumps(
                    {
                        **split,
                        "ratio": {
                            "train": TRAIN_RATIO,
                            "val": VAL_RATIO,
                            "test": TEST_RATIO,
                        },
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            dataset_summary: Dict[str, Any] = {
                "status": "ok",
                "dataset": {
                    "source": "datasetsforecast.LongHorizon",
                    "group": group,
                    "directory": str(LH_DIRECTORY),
                    "device": DEVICE,
                    "seed": int(SEED),
                    "columns": LH_COLUMNS if LH_COLUMNS else None,
                    "max_series": int(LH_MAX_SERIES) if LH_MAX_SERIES > 0 else None,
                    "num_rows": int(n),
                    "num_features": int(len(feature_cols)),
                },
                "split": {**split, "sizes": split_sizes},
                "artifacts": {
                    "dataset_dir": str(dataset_dir),
                    "scaler_path": str(scaler_path),
                    "feature_cols_path": str(feature_path),
                    "split_path": str(split_path),
                },
                "models": {},
            }

            summary["datasets"][dataset_label] = dataset_summary

            lines.append(f"Samples: {n}, features: {len(feature_cols)}")
            lines.append(
                f"Split points: train_end={train_end}, val_end={val_end} "
                f"(sizes train/val/test={split_sizes['train']}/{split_sizes['val']}/{split_sizes['test']})"
            )
            lines.append("")

            for name in model_order:
                if name in failures:
                    dataset_summary["models"][name] = {"status": "unavailable", "reason": failures[name]}
                    lines.append(f"{name}: SKIPPED ({failures[name]})")
                    continue
                if name not in adapters:
                    lines.append(f"{name}: SKIPPED (adapter missing)")
                    dataset_summary["models"][name] = {"status": "unavailable", "reason": "adapter missing"}
                    continue
                try:
                    result = evaluate_model(
                        name,
                        adapters,
                        data_scaled,
                        scaler,
                        dataset_label=dataset_label,
                        split=split,
                        out_dir=model_dir,
                    )
                    dataset_summary["models"][name] = result
                    val_metrics = result["metrics"].get("validation", {})
                    test_metrics = result["metrics"].get("test", {})
                    val_str = _format_metrics(val_metrics, "val")
                    test_str = _format_metrics(test_metrics, "test")
                    line = (
                        f"{name}: samples(train/val/test)={result['train_samples']}/"
                        f"{result['val_samples']}/{result['test_samples']} | "
                        f"{val_str}; {test_str}"
                    )
                    if REPORT_SCALED_METRICS:
                        val_scaled = result.get("metrics_scaled", {}).get("validation", {})
                        test_scaled = result.get("metrics_scaled", {}).get("test", {})
                        if val_scaled and test_scaled:
                            val_scaled_str = _format_metrics(val_scaled, "val_scaled")
                            test_scaled_str = _format_metrics(test_scaled, "test_scaled")
                            line = f"{line} | scaled: {val_scaled_str}; {test_scaled_str}"
                    lines.append(line)
                except Exception as exc:
                    reason = f"{type(exc).__name__}: {exc}"
                    dataset_summary["models"][name] = {"status": "failed", "reason": reason}
                    lines.append(f"{name}: FAILED ({reason})")
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            summary["datasets"][dataset_label] = {"status": "failed", "reason": reason}
            lines.append(f"FAILED ({reason})")

    for local_path in local_files:
        base_label = f"LocalExcel/{local_path.name}"
        variant_labels: List[str] = []
        if local_eval_mode in {"full", "both"}:
            variant_labels.append(base_label)
        if local_eval_mode in {"halves", "both"}:
            variant_labels.append(f"{base_label}::part1")
            variant_labels.append(f"{base_label}::part2")

        try:
            feat_df_full, time_values_full, feature_cols, local_meta = _load_local_excel_raw_features(local_path)
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            for dataset_label in variant_labels:
                lines.append("")
                lines.append(f"Dataset: {dataset_label}")
                lines.append("-" * 52)
                summary["datasets"][dataset_label] = {"status": "failed", "reason": reason}
                lines.append(f"FAILED ({reason})")
            continue

        target_cols = [c for c in LOCAL_TARGET_COLS if str(c).strip()]
        target_indices: Optional[List[int]] = None
        if target_cols:
            missing_targets = [c for c in target_cols if c not in feature_cols]
            if missing_targets:
                reason = (
                    f"Local target column mismatch for {local_path.name}. Missing targets: {', '.join(missing_targets)}"
                )
                for dataset_label in variant_labels:
                    lines.append("")
                    lines.append(f"Dataset: {dataset_label}")
                    lines.append("-" * 52)
                    summary["datasets"][dataset_label] = {"status": "failed", "reason": reason}
                    lines.append(f"FAILED ({reason})")
                continue
            target_indices = [feature_cols.index(c) for c in target_cols]

        n_total = int(len(feat_df_full))

        split_strategy = (LOCAL_EXCEL_SPLIT_STRATEGY or "ratio").strip().lower()
        if split_strategy not in {"auto_zero_run", "ratio"}:
            raise ValueError(
                f"Invalid LOCAL_EXCEL_SPLIT_STRATEGY={LOCAL_EXCEL_SPLIT_STRATEGY!r}; "
                "expected 'auto_zero_run' or 'ratio'."
            )

        variants: List[Tuple[str, pd.DataFrame, Dict[str, Any]]] = []
        if local_eval_mode in {"full", "both"}:
            variants.append(
                (
                    base_label,
                    feat_df_full.reset_index(drop=True),
                    {**local_meta, "segment": "full", "segment_row_start": 0, "segment_row_end": n_total},
                )
            )
        if local_eval_mode in {"halves", "both"}:
            min_len = int(max_look_back) + 2
            min_len = max(32, min_len)
            split_idx: Optional[int] = None
            split_debug: Dict[str, Any] = {
                "strategy": split_strategy,
                "ratio_config": float(LOCAL_EXCEL_SPLIT_RATIO),
                "zero_threshold": float(LOCAL_EXCEL_ZERO_THRESHOLD),
                "min_zero_run": int(LOCAL_EXCEL_MIN_ZERO_RUN),
                "min_segment_len": int(min_len),
            }

            if split_strategy == "auto_zero_run" and target_cols:
                try:
                    threshold = float(LOCAL_EXCEL_ZERO_THRESHOLD)
                    mask = feat_df_full[target_cols[0]].to_numpy() < threshold
                    idx = np.where(mask)[0]
                    if len(idx):
                        runs: List[Tuple[int, int]] = []
                        start = int(idx[0])
                        prev = int(idx[0])
                        for raw_i in idx[1:]:
                            i = int(raw_i)
                            if i == prev + 1:
                                prev = i
                            else:
                                runs.append((start, prev))
                                start = i
                                prev = i
                        runs.append((start, prev))
                        best_start, best_end = max(runs, key=lambda t: t[1] - t[0])
                        best_len = int(best_end - best_start + 1)
                        split_debug["auto_zero_run"] = {
                            "start": int(best_start),
                            "end": int(best_end),
                            "len": int(best_len),
                        }
                        if best_len >= int(LOCAL_EXCEL_MIN_ZERO_RUN):
                            candidate = int(best_start)
                            if min_len <= candidate <= n_total - min_len:
                                split_idx = candidate
                except Exception as exc:
                    split_debug["auto_zero_run_error"] = f"{type(exc).__name__}: {exc}"

            if split_idx is None:
                ratio = float(LOCAL_EXCEL_SPLIT_RATIO)
                if not (0.0 < ratio < 1.0):
                    raise ValueError(
                        f"Invalid LOCAL_EXCEL_SPLIT_RATIO={LOCAL_EXCEL_SPLIT_RATIO!r}; expected a float in (0, 1)."
                    )
                candidate = int(round(n_total * ratio))
                split_idx = max(min_len, min(candidate, n_total - min_len))
                split_debug["fallback"] = {"strategy": "ratio", "ratio_used": float(ratio)}

            split_idx = int(split_idx)
            split_debug["split_idx"] = int(split_idx)
            split_debug["split_ratio"] = float(split_idx / n_total) if n_total else None

            variants.append(
                (
                    f"{base_label}::part1",
                    feat_df_full.iloc[:split_idx].reset_index(drop=True),
                    {
                        **local_meta,
                        "segment": "part1",
                        "segment_row_start": 0,
                        "segment_row_end": split_idx,
                        "two_part_split": split_debug,
                    },
                )
            )
            variants.append(
                (
                    f"{base_label}::part2",
                    feat_df_full.iloc[split_idx:].reset_index(drop=True),
                    {
                        **local_meta,
                        "segment": "part2",
                        "segment_row_start": split_idx,
                        "segment_row_end": n_total,
                        "two_part_split": split_debug,
                    },
                )
            )

        for dataset_label, feat_df, seg_meta in variants:
            lines.append("")
            lines.append(f"Dataset: {dataset_label}")
            lines.append("-" * 52)

            try:
                split = _compute_split_points(len(feat_df), min_look_back=max_look_back, label=dataset_label)
                n = int(split["n"])
                train_end = int(split["train_end"])
                val_end = int(split["val_end"])
                split_sizes = {
                    "train": int(train_end),
                    "val": int(val_end - train_end),
                    "test": int(n - val_end),
                }

                scaler = StandardScaler().fit(feat_df.iloc[:train_end].astype(float).values)
                data_scaled = scaler.transform(feat_df.astype(float).values).astype(np.float32)

                dataset_dir = MODELS_DIR / f"{_safe_slug(dataset_label)}_seed{SEED}"
                model_dir = dataset_dir / "models"
                dataset_dir.mkdir(parents=True, exist_ok=True)
                model_dir.mkdir(parents=True, exist_ok=True)

                scaler_path = dataset_dir / "scaler.joblib"
                joblib.dump(scaler, scaler_path)
                feature_path = dataset_dir / "feature_cols.json"
                feature_path.write_text(json.dumps(feature_cols, indent=2), encoding="utf-8")
                split_path = dataset_dir / "split.json"
                split_path.write_text(
                    json.dumps(
                        {
                            **split,
                            "ratio": {
                                "train": TRAIN_RATIO,
                                "val": VAL_RATIO,
                                "test": TEST_RATIO,
                            },
                            "target_cols": target_cols if target_cols else None,
                            "target_indices": target_indices,
                            "segment": seg_meta.get("segment"),
                            "segment_row_start": seg_meta.get("segment_row_start"),
                            "segment_row_end": seg_meta.get("segment_row_end"),
                            "two_part_split": seg_meta.get("two_part_split"),
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )

                dataset_summary = {
                    "status": "ok",
                    "dataset": {
                        "source": "local_excel",
                        "path": str(local_path),
                        "time_col": seg_meta.get("time_col"),
                        "device": DEVICE,
                        "seed": int(SEED),
                        "segment": seg_meta.get("segment"),
                        "segment_row_start": seg_meta.get("segment_row_start"),
                        "segment_row_end": seg_meta.get("segment_row_end"),
                        "num_rows": int(n),
                        "num_features": int(len(feature_cols)),
                        "target_cols": target_cols if target_cols else None,
                        "target_indices": target_indices,
                    },
                    "split": {**split, "sizes": split_sizes},
                    "artifacts": {
                        "dataset_dir": str(dataset_dir),
                        "scaler_path": str(scaler_path),
                        "feature_cols_path": str(feature_path),
                        "split_path": str(split_path),
                    },
                    "meta": seg_meta,
                    "target_stats": {},
                    "baselines": {},
                    "sanity": {},
                    "models": {},
                }

                summary["datasets"][dataset_label] = dataset_summary

                lines.append(f"Samples: {n}, features: {len(feature_cols)}")
                if target_cols:
                    lines.append(f"Targets: {', '.join(target_cols)}")
                seg_name = seg_meta.get("segment")
                if seg_name and seg_name != "full":
                    seg_start = seg_meta.get("segment_row_start")
                    seg_end = seg_meta.get("segment_row_end")
                    if seg_start is not None and seg_end is not None:
                        lines.append(f"Segment (full rows): [{seg_start}, {seg_end})")
                if seg_name == "part1":
                    split_info = seg_meta.get("two_part_split")
                    if isinstance(split_info, dict):
                        split_idx = split_info.get("split_idx")
                        split_ratio = split_info.get("split_ratio")
                        strategy = split_info.get("strategy")
                        if split_idx is not None and split_ratio is not None:
                            lines.append(
                                f"Two-part split: strategy={strategy} split_idx={split_idx} ratio={float(split_ratio):.4f}"
                            )
                lines.append(
                    f"Split points: train_end={train_end}, val_end={val_end} "
                    f"(sizes train/val/test={split_sizes['train']}/{split_sizes['val']}/{split_sizes['test']})"
                )

                if target_cols:
                    all_stats: Dict[str, Dict[str, Dict[str, float]]] = {}
                    for col in target_cols:
                        all_stats[str(col)] = {
                            "train": _series_summary(feat_df[col].iloc[:train_end]),
                            "validation": _series_summary(feat_df[col].iloc[train_end:val_end]),
                            "test": _series_summary(feat_df[col].iloc[val_end:]),
                        }
                    dataset_summary["target_stats"] = all_stats

                    col0 = target_cols[0]
                    train_stats = all_stats.get(str(col0), {}).get("train", {})
                    val_stats = all_stats.get(str(col0), {}).get("validation", {})
                    test_stats = all_stats.get(str(col0), {}).get("test", {})
                    if train_stats and val_stats and test_stats:
                        lines.append(
                            f"{col0} std(train/val/test)={train_stats.get('std', 0.0):.6g}/"
                            f"{val_stats.get('std', 0.0):.6g}/{test_stats.get('std', 0.0):.6g}"
                        )

                baseline = _persistence_baseline_metrics(
                    data_scaled,
                    scaler,
                    look_back=int(TRAIN_LOOK_BACK),
                    split=split,
                    target_indices=target_indices,
                )
                baseline_scaled: Dict[str, Dict[str, float]] = {}
                if REPORT_SCALED_METRICS:
                    baseline_scaled = _persistence_baseline_metrics_scaled(
                        data_scaled,
                        look_back=int(TRAIN_LOOK_BACK),
                        split=split,
                        target_indices=target_indices,
                    )
                dataset_summary["baselines"]["persistence"] = {
                    "look_back": int(TRAIN_LOOK_BACK),
                    "metrics": baseline,
                }
                if REPORT_SCALED_METRICS and baseline_scaled:
                    dataset_summary["baselines"]["persistence"]["metrics_scaled"] = baseline_scaled
                baseline_val = baseline.get("validation", {})
                baseline_test = baseline.get("test", {})
                if baseline_val and baseline_test:
                    val_str = _format_metrics(baseline_val, "val")
                    test_str = _format_metrics(baseline_test, "test")
                    line = f"baseline(persistence, look_back={TRAIN_LOOK_BACK}): {val_str}; {test_str}"
                    if REPORT_SCALED_METRICS:
                        baseline_val_scaled = baseline_scaled.get("validation", {})
                        baseline_test_scaled = baseline_scaled.get("test", {})
                        if baseline_val_scaled and baseline_test_scaled:
                            val_scaled_str = _format_metrics(baseline_val_scaled, "val_scaled")
                            test_scaled_str = _format_metrics(baseline_test_scaled, "test_scaled")
                            line = f"{line} | scaled: {val_scaled_str}; {test_scaled_str}"
                    lines.append(line)

                seg_start_full = seg_meta.get("segment_row_start")
                seg_end_full = seg_meta.get("segment_row_end")
                try:
                    seg_start_i = int(seg_start_full) if seg_start_full is not None else 0
                    seg_end_i = int(seg_end_full) if seg_end_full is not None else seg_start_i + int(len(feat_df))
                    times_seg = time_values_full.iloc[seg_start_i:seg_end_i].reset_index(drop=True)
                    dataset_summary["sanity"] = _window_time_sanity(
                        times_seg,
                        look_back=int(TRAIN_LOOK_BACK),
                        split=split,
                    )
                except Exception as exc:
                    dataset_summary["sanity"] = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}

                lines.append("")

                for name in model_order:
                    if name in failures:
                        dataset_summary["models"][name] = {"status": "unavailable", "reason": failures[name]}
                        lines.append(f"{name}: SKIPPED ({failures[name]})")
                        continue
                    if name not in adapters:
                        lines.append(f"{name}: SKIPPED (adapter missing)")
                        dataset_summary["models"][name] = {"status": "unavailable", "reason": "adapter missing"}
                        continue
                    try:
                        result = evaluate_model(
                            name,
                            adapters,
                            data_scaled,
                            scaler,
                            dataset_label=dataset_label,
                            split=split,
                            out_dir=model_dir,
                            target_indices=target_indices,
                        )
                        dataset_summary["models"][name] = result
                        val_metrics = result["metrics"].get("validation", {})
                        test_metrics = result["metrics"].get("test", {})
                        val_str = _format_metrics(val_metrics, "val")
                        test_str = _format_metrics(test_metrics, "test")
                        line = (
                            f"{name}: samples(train/val/test)={result['train_samples']}/"
                            f"{result['val_samples']}/{result['test_samples']} | "
                            f"{val_str}; {test_str}"
                        )
                        if REPORT_SCALED_METRICS:
                            val_scaled = result.get("metrics_scaled", {}).get("validation", {})
                            test_scaled = result.get("metrics_scaled", {}).get("test", {})
                            if val_scaled and test_scaled:
                                val_scaled_str = _format_metrics(val_scaled, "val_scaled")
                                test_scaled_str = _format_metrics(test_scaled, "test_scaled")
                                line = f"{line} | scaled: {val_scaled_str}; {test_scaled_str}"
                        lines.append(line)
                    except Exception as exc:
                        reason = f"{type(exc).__name__}: {exc}"
                        dataset_summary["models"][name] = {"status": "failed", "reason": reason}
                        lines.append(f"{name}: FAILED ({reason})")
            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}"
                summary["datasets"][dataset_label] = {"status": "failed", "reason": reason}
                lines.append(f"FAILED ({reason})")

    (REPORTS_DIR / "timeseries_forecast_report.txt").write_text(
        "\n".join(lines), encoding="utf-8-sig"
    )
    (REPORTS_DIR / "timeseries_forecast_metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("Time-series benchmark complete. See reports directory for outputs.")


if __name__ == "__main__":
    main()
