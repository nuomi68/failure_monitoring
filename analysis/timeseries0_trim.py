from __future__ import annotations

import json
import os
import random
import re
import sys
import time
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.data_utils import build_windows

DATA_DIR = ROOT_DIR / "data"
REPORTS_DIR = ROOT_DIR / "reports"

# ------------------------
# Minimal config (trimmed)
# ------------------------
#, "_seg2.xlsx"
DATASET_SUFFIXES: Tuple[str, ...] = ( "_seg2.xlsx")
TIME_COL_IDX = 0
FEATURE_COL_IDXS: Tuple[int, ...] = (1, 2, 4, 5)
TARGET_FEATURE_IDX = 2

SEED = 42

REL_ERR_THRESHOLD = 0.05  # 5%
REL_ERR_DENOM_FLOOR = 1.0
PLOT_SPLIT_SERIES = (os.getenv("TS0_PLOT_SPLIT_SERIES", "1") or "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
PLOT_SPLIT_DPI = int(os.getenv("TS0_PLOT_SPLIT_DPI", "160"))
_FONT_CONFIGURED = False

TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

TRAIN_LOOK_BACK = 96
TRAIN_EPOCHS = int(os.getenv("TS0_EPOCHS", "60"))
TRAIN_BATCH_SIZE = 32
TRAIN_LR = 1e-3
TRAIN_DROPOUT = 0.1
MIN_EARLY_STOP_EPOCHS = int(os.getenv("TS0_MIN_EARLY_STOP_EPOCHS", "20"))
_patience_raw = int(os.getenv("TS0_PATIENCE", "8"))
if _patience_raw > 0:
    TRAIN_PATIENCE = max(_patience_raw, MIN_EARLY_STOP_EPOCHS)
else:
    TRAIN_PATIENCE = _patience_raw

TCN_HID = int(os.getenv("TS0_TCN_HID", "32"))
TCN_LEVELS = int(os.getenv("TS0_TCN_LEVELS", "3"))
TCN_K = int(os.getenv("TS0_TCN_K", "3"))
TCN_BATCH_SIZE = int(os.getenv("TS0_TCN_BATCH_SIZE", str(max(TRAIN_BATCH_SIZE, 64))))

TIMESNET_LOOK_BACK = TRAIN_LOOK_BACK
TIMESNET_EPOCHS = TRAIN_EPOCHS
TIMESNET_BATCH_SIZE = TRAIN_BATCH_SIZE
TIMESNET_LR = 5e-4
TIMESNET_D_MODEL = 64
TIMESNET_NUM_BLOCKS = 3
TIMESNET_PATIENCE: Optional[int] = TRAIN_PATIENCE
TIMESNET_RESIDUAL = True
#"gru", "tcn", "tsmixer", "timesnet"
MODEL_ORDER: Tuple[str, ...] = ("tcn",)

MODEL_MODULES: Dict[str, Tuple[str, str, str]] = {
    "gru": ("backend.models.gru_model", "train_gru", "predict"),
    "tcn": ("backend.models.tcn_model", "train_tcn", "predict"),
    "tsmixer": ("backend.models.tsmixer_model", "train_tsmixer", "predict"),
    "timesnet": ("backend.models.timesnet_model", "train_timesnet", "predict"),
}


def _resolve_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


DEVICE = _resolve_device()

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
    "timesnet": {
        "look_back": TIMESNET_LOOK_BACK,
        "epochs": TIMESNET_EPOCHS,
        "batch_size": TIMESNET_BATCH_SIZE,
        "lr": TIMESNET_LR,
        "d_model": TIMESNET_D_MODEL,
        "num_blocks": TIMESNET_NUM_BLOCKS,
        "residual": TIMESNET_RESIDUAL,
        "patience": TIMESNET_PATIENCE,
        "device": DEVICE,
    },
}


def _set_global_seeds(seed: int) -> None:
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
    cleaned = re.sub(r"[^0-9A-Za-z._-]+", "_", (text or "").strip())
    return cleaned.strip("._-") or "artifact"


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


def _load_excel_features(path: Path) -> Tuple[pd.DataFrame, List[str], pd.Series]:
    df = pd.read_excel(path)
    df.columns = [str(c).strip() for c in df.columns]
    if len(df.columns) <= max(FEATURE_COL_IDXS):
        raise RuntimeError(f"Unexpected columns in {path.name}.")

    time_col = df.columns[TIME_COL_IDX]
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
    if df.empty:
        raise RuntimeError(f"No valid time rows in {path.name}.")
    time_values = df[time_col].copy()

    feature_cols = [df.columns[i] for i in FEATURE_COL_IDXS]
    feat_df = df[feature_cols].apply(pd.to_numeric, errors="coerce").ffill().bfill()
    if feat_df.isna().any().any():
        feat_df = feat_df.fillna(0.0)
    return feat_df, [str(c) for c in feature_cols], time_values


def _compute_split_points(n: int, *, min_look_back: int) -> Dict[str, int]:
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))
    train_end = max(2, min(train_end, n - 2))
    val_end = max(train_end + 1, min(val_end, n - 1))

    if train_end < min_look_back + 1:
        raise RuntimeError(
            f"Train split too small for look_back={min_look_back}: train_end={train_end}, n={n}."
        )
    if val_end <= train_end:
        raise RuntimeError(f"Invalid split: train_end={train_end}, val_end={val_end}")
    return {"n": int(n), "train_end": int(train_end), "val_end": int(val_end)}


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
    }


def _day_axis_values(times: pd.Series) -> np.ndarray:
    if times.empty:
        return np.zeros(0, dtype=np.float32)
    dt = pd.to_datetime(times, errors="coerce")
    if dt.isna().all():
        return np.arange(len(times), dtype=np.float32) + 1.0
    base = dt.iloc[0]
    delta_days = (dt - base).dt.total_seconds().fillna(0.0) / 86400.0
    return (delta_days + 1.0).to_numpy(dtype=np.float32)


def _configure_matplotlib_fonts() -> None:
    global _FONT_CONFIGURED
    if _FONT_CONFIGURED:
        return
    _FONT_CONFIGURED = True
    try:
        from matplotlib import font_manager, rcParams

        candidates = [
            "Microsoft YaHei",
            "Microsoft YaHei UI",
            "SimHei",
            "Noto Sans CJK SC",
            "Source Han Sans SC",
            "PingFang SC",
            "WenQuanYi Zen Hei",
            "Arial Unicode MS",
        ]
        available = {f.name for f in font_manager.fontManager.ttflist}
        for name in candidates:
            if name in available:
                rcParams["font.sans-serif"] = [name] + rcParams.get("font.sans-serif", [])
                break
        rcParams["axes.unicode_minus"] = False
    except Exception:
        pass


def _plot_series_by_day(
    times: pd.Series,
    values: pd.Series,
    *,
    title: str,
    file_tag: str,
    out_dir: Path,
) -> Optional[str]:
    if not PLOT_SPLIT_SERIES:
        return None
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception:
        return None

    _configure_matplotlib_fonts()

    x = _day_axis_values(times)
    y = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float32)
    if len(x) == 0 or len(y) == 0:
        return None
    mask = np.isfinite(x) & np.isfinite(y)
    if not np.any(mask):
        return None
    x = x[mask]
    y = y[mask]

    fig, ax = plt.subplots(figsize=(10, 4), dpi=PLOT_SPLIT_DPI)
    ax.plot(x, y, linewidth=0.8)
    max_day = int(np.ceil(float(np.nanmax(x)))) if len(x) else 1
    max_day = max(1, max_day)
    if max_day <= 15:
        ticks = np.arange(1, max_day + 1, dtype=int)
    else:
        step = max(1, int(np.ceil(max_day / 10)))
        ticks = np.arange(1, max_day + 1, step, dtype=int)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"Day {int(t)}" for t in ticks])
    ax.set_xlabel("Day")
    ax.set_ylabel("value")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{_safe_slug(file_tag)}.png"
    fig.savefig(out_path)
    plt.close(fig)
    return str(out_path)


METRIC_DISPLAY_ORDER: Tuple[str, ...] = (
    "mae",
    "rmse",
    "r2",
    "mape_pct",
    "within_5pct",
    "p95_ape_pct",
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
    target_indices: Optional[List[int]] = None,
) -> Dict[str, Any]:
    if name not in adapters:
        raise KeyError(f"Model '{name}' is not available (dependency missing).")

    params = MODEL_PARAM_OVERRIDES.get(name, {}).copy()
    if name == "timesnet" and target_indices is not None and len(target_indices) == 1:
        params.setdefault("target_index", int(target_indices[0]))
    look_back = int(params.pop("look_back", TRAIN_LOOK_BACK))
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
                for start_idx in range(0, len(samples), batch_size):
                    xb = torch.tensor(samples[start_idx : start_idx + batch_size], dtype=torch.float32, device=device)
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

    return {
        "status": "ok",
        "look_back": look_back,
        "params": {"look_back": int(look_back), **params},
        "train_samples": int(len(X_train)),
        "val_samples": int(len(X_val)),
        "test_samples": int(len(X_test)),
        "metrics": metrics,
        "logs": logs,
        "train_time_sec": float(duration),
    }


def _load_datasets() -> List[Path]:
    candidates = [p for p in DATA_DIR.iterdir() if p.is_file() and p.suffix.lower() == ".xlsx"]
    matches = [p for p in candidates if p.name.endswith(DATASET_SUFFIXES)]
    return sorted(matches, key=lambda p: p.name.lower())


def main() -> None:
    adapters, failures = _load_model_adapters()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    datasets = _load_datasets()
    if not datasets:
        raise RuntimeError(f"No datasets found with suffixes {DATASET_SUFFIXES!r} in {DATA_DIR}.")

    lines = [
        "Trimmed Time-Series Forecast Benchmark (7:2:1)",
        "=" * 52,
        f"Datasets: {', '.join(p.name for p in datasets)}",
        f"Device: {DEVICE} | Seed: {SEED}",
        f"Split: train/val/test = {TRAIN_RATIO:.1f}/{VAL_RATIO:.1f}/{TEST_RATIO:.1f}",
        f"Relative error: threshold={REL_ERR_THRESHOLD * 100:.1f}%, denom_floor={REL_ERR_DENOM_FLOOR:g}",
        f"Train params: look_back={TRAIN_LOOK_BACK}, epochs={TRAIN_EPOCHS}, batch_size={TRAIN_BATCH_SIZE}, lr={TRAIN_LR:g}, dropout={TRAIN_DROPOUT:g}, patience={TRAIN_PATIENCE}",
        "",
    ]

    summary: Dict[str, Any] = {"datasets": {}, "failures": failures}

    target_indices = [int(TARGET_FEATURE_IDX)] if TARGET_FEATURE_IDX is not None else None

    for path in datasets:
        dataset_label = path.name
        lines.append("")
        lines.append(f"Dataset: {dataset_label}")
        lines.append("-" * 52)

        try:
            feat_df, feature_cols, time_values = _load_excel_features(path)
            split = _compute_split_points(len(feat_df), min_look_back=TRAIN_LOOK_BACK)
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

            for idx, col in enumerate(feature_cols):
                _plot_series_by_day(
                    time_values,
                    feat_df.iloc[:, idx],
                    title=f"{path.stem} {col}",
                    file_tag=f"{path.stem}_feature{idx + 1}",
                    out_dir=REPORTS_DIR,
                )

            dataset_summary: Dict[str, Any] = {
                "status": "ok",
                "dataset": {
                    "path": str(path),
                    "num_rows": int(n),
                    "num_features": int(len(feature_cols)),
                    "feature_cols": feature_cols,
                    "target_indices": target_indices,
                },
                "split": {**split, "sizes": split_sizes},
                "models": {},
            }
            summary["datasets"][dataset_label] = dataset_summary

            lines.append(f"Samples: {n}, features: {len(feature_cols)}")
            lines.append(
                f"Split points: train_end={train_end}, val_end={val_end} "
                f"(sizes train/val/test={split_sizes['train']}/{split_sizes['val']}/{split_sizes['test']})"
            )
            lines.append("")

            for name in MODEL_ORDER:
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
                        target_indices=target_indices,
                    )
                    dataset_summary["models"][name] = result
                    val_metrics = result["metrics"].get("validation", {})
                    test_metrics = result["metrics"].get("test", {})
                    val_str = _format_metrics(val_metrics, "val")
                    test_str = _format_metrics(test_metrics, "test")
                    lines.append(
                        f"{name}: samples(train/val/test)={result['train_samples']}/"
                        f"{result['val_samples']}/{result['test_samples']} | "
                        f"{val_str}; {test_str}"
                    )
                except Exception as exc:
                    reason = f"{type(exc).__name__}: {exc}"
                    dataset_summary["models"][name] = {"status": "failed", "reason": reason}
                    lines.append(f"{name}: FAILED ({reason})")
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            summary["datasets"][dataset_label] = {"status": "failed", "reason": reason}
            lines.append(f"FAILED ({reason})")

    (REPORTS_DIR / "timeseries_forecast_report_trim.txt").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    (REPORTS_DIR / "timeseries_forecast_metrics_trim.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("Trimmed benchmark complete. See reports directory for outputs.")


if __name__ == "__main__":
    main()
