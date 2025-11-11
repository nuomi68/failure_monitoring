from __future__ import annotations

import json
import os
import random
import time
from importlib import import_module
from pathlib import Path
import sys
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.data_utils import build_windows
from noisy_sample_generator import NON_FEATURE_COLUMNS, load_single

DATA_DIR = ROOT_DIR / "data"
REPORTS_DIR = ROOT_DIR / "reports"
SOURCE_FILE = "20230510-20240924.xlsx"

DEFAULT_LOOKBACK = {
    "gru": 32,
    "tcn": 14,
    "tsmixer": 32,
    "rf": 5,
    "xgb": 14,
    "timesnet": 32,
}

MODEL_MODULES: Dict[str, Tuple[str, str, str]] = {
    # "gru": ("backend.models.gru_model", "train_gru", "predict"),
    # "tcn": ("backend.models.tcn_model", "train_tcn", "predict"),
    "tsmixer": ("backend.models.tsmixer_model", "train_tsmixer", "predict"),
    # "rf": ("backend.models.random_forest_model", "train_rf", "predict"),
    # "xgb": ("backend.models.xgboost_model", "train_xgb", "predict"),
    # "timesnet": ("backend.models.timesnet_model", "train_timesnet", "predict"),
}

MODEL_PARAM_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "gru": {"epochs": 10, "batch_size": 32, "hidden_size": 32, "num_layers": 1, "device": "cpu"},
    "tcn": {
        "look_back": 18,
        "epochs": 32,
        "batch_size": 32,
        "hid": 128,
        "levels": 2,
        "k": 3,
        "drop": 0.1,
        "lr": 5e-4,
        "device": "cpu",
    },
    "tsmixer": {
        "look_back": 28,
        "epochs": 16,
        "batch_size": 32,
        "num_blocks": 3,
        "ff_dim": 96,
        "dropout": 0.1,
        "lr": 7e-4,
        "device": "cpu",
    },
    "rf": {"n_estimators": 400, "random_state": 42, "n_jobs": -1},
    "xgb": {"n_estimators": 400, "learning_rate": 0.05, "max_depth": 4, "reg_lambda": 1.0},
    "timesnet": {"epochs": 6, "batch_size": 16, "device": "cpu"},
}

MODEL_ORDER_DEFAULT = ("rf", "xgb", "gru", "tcn", "tsmixer", "timesnet")

_env_models = os.environ.get("TS_BENCH_MODELS", "").strip()
if _env_models:
    MODEL_ORDER = tuple(
        model.strip() for model in _env_models.split(",") if model.strip()
    )
else:
    MODEL_ORDER = MODEL_ORDER_DEFAULT

VAL_RATIO = 0.2
HOLDOUT = 30


def _set_global_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _load_clean_features() -> Tuple[np.ndarray, List[str], StandardScaler]:
    df = load_single()
    feature_cols = [
        c
        for c in df.columns
        if c not in NON_FEATURE_COLUMNS and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not feature_cols:
        raise RuntimeError("No numeric features available for forecasting.")
    feat_df = df[feature_cols].astype(float)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(feat_df.values)
    return data_scaled.astype(np.float32), feature_cols, scaler


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


def _train_val_split(
    data: np.ndarray, look_back: int, holdout: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    effective_holdout = min(max(holdout, 5), max(len(data) - look_back - 1, 5))
    X, y = build_windows(data[:-effective_holdout], look_back)
    if len(X) < 4:
        raise RuntimeError("Not enough windows to train/validate.")
    split_idx = max(1, int(len(X) * (1 - VAL_RATIO)))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    if len(X_val) == 0:
        X_val = X_train[-1:]
        y_val = y_train[-1:]
    return X_train, X_val, y_train, y_val


def _build_holdout(data: np.ndarray, look_back: int, holdout: int) -> Tuple[np.ndarray, np.ndarray]:
    seqs, targets = [], []
    start_idx = max(look_back, len(data) - holdout)
    for idx in range(start_idx, len(data)):
        seqs.append(data[idx - look_back : idx])
        targets.append(data[idx])
    if not seqs:
        return np.empty((0, look_back, data.shape[1])), np.empty((0, data.shape[1]))
    return np.asarray(seqs, dtype=np.float32), np.asarray(targets, dtype=np.float32)


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, scaler: StandardScaler) -> Dict[str, float]:
    if y_true.size == 0:
        return {}
    y_true_inv = scaler.inverse_transform(y_true)
    y_pred_inv = scaler.inverse_transform(y_pred)
    return {
        "mae": float(mean_absolute_error(y_true_inv, y_pred_inv)),
        "rmse": float(np.sqrt(mean_squared_error(y_true_inv, y_pred_inv))),
        "r2": float(r2_score(y_true_inv, y_pred_inv)),
    }


def evaluate_model(
    name: str,
    adapters: Dict[str, Dict[str, Callable[..., Any]]],
    data_scaled: np.ndarray,
    scaler: StandardScaler,
) -> Dict[str, Any]:
    if name not in adapters:
        raise KeyError(f"Model '{name}' is not available (dependency missing).")

    params = MODEL_PARAM_OVERRIDES.get(name, {}).copy()
    look_back = int(params.pop("look_back", DEFAULT_LOOKBACK.get(name, 14)))
    X_train, X_val, y_train, y_val = _train_val_split(data_scaled, look_back, HOLDOUT)
    X_hold, y_hold = _build_holdout(data_scaled, look_back, HOLDOUT)

    logs: List[str] = []

    def _log(msg: str) -> None:
        logs.append(msg)

    start = time.time()
    _set_global_seeds(42)
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

    n_features = data_scaled.shape[1]

    def _predict_batch(samples: np.ndarray) -> np.ndarray:
        if len(samples) == 0:
            return np.empty((0, n_features), dtype=np.float32)
        preds = [predict_fn(model_obj, seq) for seq in samples]
        arr = np.asarray(preds, dtype=np.float32)
        return arr.reshape(len(preds), n_features)

    train_preds = _predict_batch(X_train)
    val_preds = _predict_batch(X_val)
    hold_preds = _predict_batch(X_hold)

    metrics = {
        "train": _regression_metrics(y_train, train_preds, scaler),
        "validation": _regression_metrics(y_val, val_preds, scaler),
        "holdout": _regression_metrics(y_hold, hold_preds, scaler),
    }

    return {
        "status": "ok",
        "look_back": look_back,
        "train_samples": int(len(X_train)),
        "val_samples": int(len(X_val)),
        "holdout_samples": int(len(X_hold)),
        "metrics": metrics,
        "logs": logs,
        "train_time_sec": float(duration),
    }


def main() -> None:
    data_scaled, feature_cols, scaler = _load_clean_features()
    adapters, failures = _load_model_adapters()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {
        "dataset": {
            "source_file": SOURCE_FILE,
            "num_rows": int(data_scaled.shape[0]),
            "num_features": len(feature_cols),
        },
        "models": {},
        "failures": failures,
    }

    lines = [
        "Multi-model Time-Series Forecast Benchmark",
        "=" * 44,
        f"Source file: {SOURCE_FILE}",
        f"Samples: {data_scaled.shape[0]}, features: {len(feature_cols)}",
        "",
    ]

    for name in MODEL_ORDER:
        if name in failures:
            summary["models"][name] = {"status": "unavailable", "reason": failures[name]}
            lines.append(f"{name}: SKIPPED ({failures[name]})")
            continue
        if name not in adapters:
            lines.append(f"{name}: SKIPPED (adapter missing)")
            summary["models"][name] = {"status": "unavailable", "reason": "adapter missing"}
            continue
        try:
            result = evaluate_model(name, adapters, data_scaled, scaler)
            summary["models"][name] = result
            val_metrics = result["metrics"].get("validation", {})
            hold_metrics = result["metrics"].get("holdout", {})
            val_str = ", ".join(f"val_{k}={v:.4f}" for k, v in val_metrics.items())
            hold_str = ", ".join(f"hold_{k}={v:.4f}" for k, v in hold_metrics.items())
            lines.append(
                f"{name}: samples(train/val/hold)={result['train_samples']}/"
                f"{result['val_samples']}/{result['holdout_samples']} | "
                f"{val_str}; {hold_str or 'holdout=N/A'}"
            )
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            summary["models"][name] = {"status": "failed", "reason": reason}
            lines.append(f"{name}: FAILED ({reason})")

    (REPORTS_DIR / "timeseries_forecast_report.txt").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    (REPORTS_DIR / "timeseries_forecast_metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("Time-series benchmark complete. See reports directory for outputs.")


if __name__ == "__main__":
    main()
