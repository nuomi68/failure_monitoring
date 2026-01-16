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
FIGURES_DIR = REPORTS_DIR / "figures"
SOURCE_FILE = "20200803-20210504.xlsx"
TARGET_FEATURES = [
    "I-131",
    "I-132",
    "I-133",
    "I-134",
    "I-135",
    "KR-85M",
    "KR-87",
    "KR-88",
    "XE-133",
    "XE-135",
]
PLOT_FEATURES = ["I-131", "I-133"]

DEVICE = (os.environ.get("TS_DEVICE", "cpu") or "cpu").strip()
_device_lower = DEVICE.lower()
if _device_lower in {"gpu", "cuda"}:
    DEVICE = "cuda"
elif _device_lower.startswith("cuda:"):
    DEVICE = _device_lower
elif _device_lower.startswith("cuda"):
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

TRAIN_LOOK_BACK = int(os.environ.get("TS_LOOK_BACK", "32"))
TRAIN_EPOCHS = int(os.environ.get("TS_EPOCHS", "20"))
TRAIN_BATCH_SIZE = int(os.environ.get("TS_BATCH_SIZE", "32"))
TRAIN_LR = float(os.environ.get("TS_LR", "1e-3"))
TRAIN_DROPOUT = float(os.environ.get("TS_DROPOUT", "0.1"))

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
        "hid": 64,
        "levels": 3,
        "k": 3,
        "drop": TRAIN_DROPOUT,
    },
    "tsmixer": {
        **COMMON_TORCH_PARAMS,
        "num_blocks": 3,
        "ff_dim": 128,
        "dropout": TRAIN_DROPOUT,
    },
    "rf": {
        "look_back": TRAIN_LOOK_BACK,
        "n_estimators": 400,
        "random_state": 42,
        "n_jobs": -1,
    },
    "xgb": {
        "look_back": TRAIN_LOOK_BACK,
        "n_estimators": 400,
        "learning_rate": 0.05,
        "max_depth": 4,
        "reg_lambda": 1.0,
    },
    "timesnet": {
        **COMMON_TORCH_PARAMS,
        "d_model": 32,
        "num_blocks": 3,
    },
}

MODEL_ORDER_DEFAULT = ("gru", "tcn", "tsmixer", "timesnet")

_env_models = os.environ.get("TS_BENCH_MODELS", "").strip()
if _env_models:
    MODEL_ORDER = tuple(
        model.strip() for model in _env_models.split(",") if model.strip()
    )
else:
    MODEL_ORDER = MODEL_ORDER_DEFAULT

VAL_RATIO = 0.2
HOLDOUT = 30
TEST_RATIO = 0.2
SPLIT_MODE = os.environ.get("TS_SPLIT_MODE", "random").strip().lower()
SPLIT_SEED = int(os.environ.get("TS_SPLIT_SEED", "42"))


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
    feature_cols = [c for c in TARGET_FEATURES if c in df.columns]
    if not feature_cols:
        raise RuntimeError("No target features found in the dataset.")
    missing = [c for c in TARGET_FEATURES if c not in feature_cols]
    if missing:
        print(f"Warning: missing target features skipped: {', '.join(missing)}")
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


def _split_windows(
    data: np.ndarray,
    look_back: int,
    *,
    split_mode: str,
    holdout: int,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (X_train, X_val, X_test, y_train, y_val, y_test)."""
    split_mode = (split_mode or "random").strip().lower()
    if split_mode not in {"random", "temporal"}:
        raise ValueError(f"Unknown split_mode: {split_mode!r} (use 'random' or 'temporal').")

    if split_mode == "temporal":
        X_train, X_val, y_train, y_val = _train_val_split(data, look_back, holdout)
        effective_holdout = min(max(holdout, 5), max(len(data) - look_back - 1, 5))
        X_test, y_test = _build_holdout(data, look_back, effective_holdout)
        return X_train, X_val, X_test, y_train, y_val, y_test

    X_all, y_all = build_windows(data, look_back)
    if len(X_all) < 6:
        raise RuntimeError("Not enough windows to train/validate/test.")

    idx = np.arange(len(X_all))
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)

    test_n = int(round(len(idx) * float(test_ratio)))
    test_n = max(1, min(test_n, len(idx) - 2))
    test_idx = idx[:test_n]
    train_val_idx = idx[test_n:]

    val_n = int(round(len(train_val_idx) * float(val_ratio)))
    val_n = max(1, min(val_n, len(train_val_idx) - 1))
    val_idx = train_val_idx[:val_n]
    train_idx = train_val_idx[val_n:]

    return (
        X_all[train_idx],
        X_all[val_idx],
        X_all[test_idx],
        y_all[train_idx],
        y_all[val_idx],
        y_all[test_idx],
    )


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


def _relative_mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scaler: StandardScaler,
    feature_cols: List[str],
    *,
    eps: float = 1e-12,
) -> Dict[str, float]:
    """Relative MAE per feature: MAE / mean(|true|) on original scale."""
    if y_true.size == 0:
        return {}
    true_inv = scaler.inverse_transform(y_true)
    pred_inv = scaler.inverse_transform(y_pred)
    mae = np.mean(np.abs(pred_inv - true_inv), axis=0)
    denom = np.mean(np.abs(true_inv), axis=0)
    denom = np.where(denom < eps, np.nan, denom)
    rel = mae / denom
    return {feature_cols[i]: float(rel[i]) for i in range(min(len(feature_cols), len(rel)))}


def _plot_holdout_error(
    model_name: str,
    split_label: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scaler: StandardScaler,
    feature_cols: List[str],
    save_dir: Path,
    targets: List[str] | None = None,
    log_fn: Callable[[str], None] | None = None,
) -> Dict[str, str]:
    """Plot ground truth + prediction for selected features with absolute error bars."""
    if y_true.size == 0:
        return {}
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:
        if log_fn:
            log_fn(f"Skip error plot (matplotlib unavailable): {exc}")
        return {}

    save_dir.mkdir(parents=True, exist_ok=True)
    true_inv = scaler.inverse_transform(y_true)
    pred_inv = scaler.inverse_transform(y_pred)
    target_names = targets or (feature_cols[:1] if feature_cols else [])
    plotted: Dict[str, str] = {}

    for feature_name in target_names:
        if feature_name not in feature_cols:
            if log_fn:
                log_fn(f"Skip plot for {feature_name} (not in feature columns).")
            continue
        feature_idx = feature_cols.index(feature_name)
        steps = np.arange(len(true_inv))
        err = np.abs(true_inv[:, feature_idx] - pred_inv[:, feature_idx])

        fig, ax1 = plt.subplots(figsize=(9, 4))
        ax1.plot(steps, true_inv[:, feature_idx], label=f"{feature_name} truth", color="tab:blue")
        ax1.plot(
            steps,
            pred_inv[:, feature_idx],
            label=f"{feature_name} pred",
            color="tab:green",
            linestyle="--",
            alpha=0.85,
        )
        ax1.set_xlabel(f"{split_label} step")
        ax1.set_ylabel(f"{feature_name} (original units)")
        ax1.set_title(f"{model_name} {split_label}: value + error")

        ax2 = ax1.twinx()
        ax2.bar(steps, err, color="tab:red", alpha=0.25, label="Abs error")
        ax2.set_ylabel(f"|error| ({feature_name})")

        handles, labels = [], []
        for ax in (ax1, ax2):
            h, l = ax.get_legend_handles_labels()
            handles.extend(h)
            labels.extend(l)
        ax1.legend(handles, labels, loc="upper left")

        safe_feat = feature_name.replace("-", "")
        path = save_dir / f"{model_name}_{split_label}_error_{safe_feat}.png"
        plt.tight_layout()
        plt.savefig(path, dpi=300)
        plt.close()
        plotted[feature_name] = str(path)
    return plotted


def evaluate_model(
    name: str,
    adapters: Dict[str, Dict[str, Callable[..., Any]]],
    data_scaled: np.ndarray,
    scaler: StandardScaler,
    feature_cols: List[str],
    *,
    param_overrides: Dict[str, Any] | None = None,
    generate_plots: bool = True,
    capture_logs: bool = True,
) -> Dict[str, Any]:
    if name not in adapters:
        raise KeyError(f"Model '{name}' is not available (dependency missing).")

    params = MODEL_PARAM_OVERRIDES.get(name, {}).copy()
    if param_overrides:
        params.update(param_overrides)
    look_back = int(params.pop("look_back", DEFAULT_LOOKBACK.get(name, 14)))
    X_train, X_val, X_test, y_train, y_val, y_test = _split_windows(
        data_scaled,
        look_back,
        split_mode=SPLIT_MODE,
        holdout=HOLDOUT,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        seed=SPLIT_SEED,
    )

    logs: List[str] = []

    def _log(msg: str) -> None:
        logs.append(msg)

    start = time.time()
    _set_global_seeds(42)
    train_fn = adapters[name]["train"]
    log_cb = _log if capture_logs else None
    if name in {"rf", "xgb"}:
        result = train_fn(X_train, y_train, X_val, y_val, log_callback=log_cb, **params)
    else:
        result = train_fn(
            X_train,
            y_train,
            X_val,
            y_val,
            log_callback=log_cb,
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
    test_preds = _predict_batch(X_test)

    metrics = {
        "train": _regression_metrics(y_train, train_preds, scaler),
        "validation": _regression_metrics(y_val, val_preds, scaler),
        "test": _regression_metrics(y_test, test_preds, scaler),
    }

    rel_err_test = _relative_mae(y_test, test_preds, scaler, feature_cols)
    rel_vals = np.asarray(list(rel_err_test.values()), dtype=float) if rel_err_test else np.asarray([])
    rel_err_mean = float(np.nanmean(rel_vals)) if rel_vals.size and np.isfinite(rel_vals).any() else float("nan")

    plots: Dict[str, str] = {}
    if generate_plots:
        plots = _plot_holdout_error(
            name,
            "test",
            y_test,
            test_preds,
            scaler,
            feature_cols,
            FIGURES_DIR,
            targets=PLOT_FEATURES,
            log_fn=_log if capture_logs else None,
        )

    return {
        "status": "ok",
        "look_back": look_back,
        "params": {"look_back": look_back, **params},
        "train_samples": int(len(X_train)),
        "val_samples": int(len(X_val)),
        "test_samples": int(len(X_test)),
        "metrics": metrics,
        "relative_error_test": rel_err_test,
        "relative_error_test_mean": rel_err_mean,
        "logs": logs,
        "train_time_sec": float(duration),
        "holdout_error_plots": plots,
    }


def main() -> None:
    data_scaled, feature_cols, scaler = _load_clean_features()
    adapters, failures = _load_model_adapters()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {
        "dataset": {
            "source_file": SOURCE_FILE,
            "device": DEVICE,
            "num_rows": int(data_scaled.shape[0]),
            "num_features": len(feature_cols),
        },
        "split": {
            "mode": SPLIT_MODE,
            "val_ratio": float(VAL_RATIO),
            "test_ratio": float(TEST_RATIO),
            "seed": int(SPLIT_SEED),
        },
        "models": {},
        "failures": failures,
    }

    lines = [
        "Multi-model Time-Series Forecast Benchmark",
        "=" * 44,
        f"Split mode: {SPLIT_MODE} (windows shuffled)" if SPLIT_MODE == "random" else f"Split mode: {SPLIT_MODE}",
        f"Source file: {SOURCE_FILE}",
        f"Device: {DEVICE}",
        f"Samples: {data_scaled.shape[0]}, features: {len(feature_cols)}",
        "",
    ]

    table_rows: List[Dict[str, Any]] = []

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
            result = evaluate_model(name, adapters, data_scaled, scaler, feature_cols)
            summary["models"][name] = result
            val_metrics = result["metrics"].get("validation", {})
            test_metrics = result["metrics"].get("test", {})
            val_str = ", ".join(f"val_{k}={v:.4f}" for k, v in val_metrics.items())
            test_str = ", ".join(f"test_{k}={v:.4f}" for k, v in test_metrics.items())
            mean_rel = result.get("relative_error_test_mean")
            lines.append(
                f"{name}: samples(train/val/test)={result['train_samples']}/"
                f"{result['val_samples']}/{result['test_samples']} | "
                f"{val_str}; {test_str or 'test=N/A'} | rel_mean={mean_rel:.4f}"
            )

            row = {
                "model": name,
                "look_back": int(result.get("look_back", 0)),
                "train_samples": int(result.get("train_samples", 0)),
                "val_samples": int(result.get("val_samples", 0)),
                "test_samples": int(result.get("test_samples", 0)),
                "rel_mean": float(mean_rel) if mean_rel is not None else float("nan"),
            }
            for feat in feature_cols:
                row[feat] = result.get("relative_error_test", {}).get(feat, float("nan"))
            table_rows.append(row)
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            summary["models"][name] = {"status": "failed", "reason": reason}
            lines.append(f"{name}: FAILED ({reason})")

    if table_rows:
        table_df = pd.DataFrame(table_rows)
        out_csv = REPORTS_DIR / "timeseries_relative_error_table.csv"
        table_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

        # Pretty table figure (percent formatting for relative errors).
        try:
            import matplotlib.pyplot as plt  # type: ignore

            display_df = table_df.copy()
            for feat in feature_cols + ["rel_mean"]:
                if feat in display_df.columns:
                    display_df[feat] = (display_df[feat].astype(float) * 100.0).round(2)
            fig_w = max(12.0, 0.9 * len(display_df.columns))
            fig_h = max(2.6, 0.55 * (len(display_df) + 1))
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))
            ax.axis("off")
            tbl = ax.table(
                cellText=display_df.values,
                colLabels=list(display_df.columns),
                cellLoc="center",
                loc="center",
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(8)
            tbl.scale(1, 1.25)
            fig.tight_layout()
            fig_path = FIGURES_DIR / "timeseries_relative_error_table.png"
            fig.savefig(fig_path, dpi=300)
            plt.close(fig)
            lines.append("")
            lines.append(f"Relative error table: {out_csv.name} (and figures/{fig_path.name})")
        except Exception as exc:
            lines.append("")
            lines.append(f"Relative error table: {out_csv.name} (figure skipped: {type(exc).__name__})")

    (REPORTS_DIR / "timeseries_forecast_report.txt").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    (REPORTS_DIR / "timeseries_forecast_metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("Time-series benchmark complete. See reports directory for outputs.")


if __name__ == "__main__":
    main()
