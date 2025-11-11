from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.models import unsupervised_core
from noisy_sample_generator import NON_FEATURE_COLUMNS, load_single

REPORTS_DIR = ROOT_DIR / "reports"
DATA_SUBDIR = REPORTS_DIR / "noisy_samples"
NOISE_TYPES = ("gaussian", "drift")
TOP_RATIO = 0.1

MODEL_SPECS: Dict[str, Dict] = {
    "iforest": {
        "params": {"n_estimators": 400, "contamination": 0.05, "random_state": 42},
    },
    "knn": {
        "params": {"n_neighbors": 35},
    },
    "autoencoder": {
        "params": {
            "hidden": [64, 32],
            "latent_dim": 16,
            "epochs": 10,
            "batch_size": 128,
            "lr": 1e-3,
            "dropout": 0.1,
        }
    },
}


def _require_dataset(noise_type: str) -> Path:
    path = DATA_SUBDIR / f"{noise_type}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Please run noisy_sample_generator.py first."
        )
    return path


def _load_noise_dataset(noise_type: str) -> pd.DataFrame:
    df = pd.read_csv(_require_dataset(noise_type))
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
    if "label" not in df.columns:
        raise ValueError(f"'label' column missing in {noise_type} dataset")
    df["label"] = df["label"].fillna(0).astype(int)
    return df


def _select_feature_columns(df: pd.DataFrame) -> list[str]:
    feature_cols = [
        c
        for c in df.columns
        if c not in NON_FEATURE_COLUMNS and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not feature_cols:
        raise RuntimeError("No numeric columns available for unsupervised evaluation.")
    return feature_cols


def _prepare_training_basis() -> Tuple[list[str], StandardScaler, np.ndarray]:
    clean_df = load_single()
    feature_cols = _select_feature_columns(clean_df)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(clean_df[feature_cols].to_numpy(dtype=float))
    return feature_cols, scaler, X_train


def _adapter_map() -> Dict[str, unsupervised_core.AlgoAdapter]:
    mapping = {}
    for adapter in unsupervised_core.ADAPTERS:
        mapping[adapter.meta_model_type()] = adapter
        mapping[adapter.code] = adapter
    return mapping


def _fit_models(X_train: np.ndarray) -> Tuple[Dict[str, Tuple[unsupervised_core.AlgoAdapter, object]], Dict[str, str]]:
    adapter_lookup = _adapter_map()
    models: Dict[str, Tuple[unsupervised_core.AlgoAdapter, object]] = {}
    failures: Dict[str, str] = {}

    for model_name, spec in MODEL_SPECS.items():
        adapter = adapter_lookup.get(model_name) or adapter_lookup.get(spec.get("adapter", ""))
        if adapter is None:
            failures[model_name] = "Adapter not available"
            continue
        try:
            model = adapter.build(**spec.get("params", {}))
            fitted = adapter.fit(model, X_train)
            models[model_name] = (adapter, fitted)
        except Exception as exc:
            failures[model_name] = f"{type(exc).__name__}: {exc}"
    return models, failures


def _threshold_outputs(scores: np.ndarray, top_ratio: float = TOP_RATIO) -> np.ndarray:
    if len(scores) == 0:
        return np.array([])
    cutoff = np.percentile(scores, 100 * (1 - top_ratio))
    return (scores >= cutoff).astype(int)


def _score_dataset(
    models: Dict[str, Tuple[unsupervised_core.AlgoAdapter, object]],
    X: np.ndarray,
    y: np.ndarray,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, np.ndarray]]:
    metrics: Dict[str, Dict[str, float]] = {}
    raw_scores: Dict[str, np.ndarray] = {}
    label_variety = len(np.unique(y)) > 1

    for name, (adapter, model) in models.items():
        scores = adapter.scores(model, X)
        if scores is None:
            continue
        if not isinstance(scores, np.ndarray):
            scores = np.asarray(scores, dtype=float)
        tau = adapter.default_tau(scores)
        if tau is None:
            preds = _threshold_outputs(scores)
        else:
            preds = (scores >= tau).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y, preds, average="binary", zero_division=0
        )
        auc = roc_auc_score(y, scores) if label_variety else 0.0
        ap = average_precision_score(y, scores) if label_variety else 0.0
        metrics[name] = {
            "roc_auc": float(auc),
            "average_precision": float(ap),
            "precision@10%": float(precision),
            "recall@10%": float(recall),
            "f1@10%": float(f1),
            "threshold": float(tau) if tau is not None else None,
        }
        raw_scores[name] = scores
    return metrics, raw_scores


def _persist_scores(
    noise_type: str, score_map: Dict[str, np.ndarray], labels: np.ndarray
) -> None:
    out_dir = DATA_SUBDIR / "unsupervised_scores"
    out_dir.mkdir(parents=True, exist_ok=True)
    for detector_name, scores in score_map.items():
        df_out = pd.DataFrame({"score": scores, "label": labels})
        df_out.to_csv(
            out_dir / f"{noise_type}_{detector_name}.csv",
            index=False,
            encoding="utf-8-sig",
        )


def evaluate_noise_type(
    noise_type: str,
    feature_cols: list[str],
    scaler: StandardScaler,
    models: Dict[str, Tuple[unsupervised_core.AlgoAdapter, object]],
) -> Dict[str, Dict[str, float]]:
    df = _load_noise_dataset(noise_type)
    X = scaler.transform(df[feature_cols].to_numpy(dtype=float))
    y = df["label"].to_numpy(dtype=int)
    metrics, scores = _score_dataset(models, X, y)
    _persist_scores(noise_type, scores, y)
    return metrics


def save_report(
    results: Dict[str, Dict[str, Dict[str, float]]],
    model_failures: Dict[str, str],
) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "Unsupervised Evaluation (train on clean, test on noisy windows)",
        "=" * 58,
        "",
    ]
    for noise_type, detectors in results.items():
        lines.append(f"Noise: {noise_type}")
        for detector_name, stats in detectors.items():
            metric_line = ", ".join(f"{k}={v:.4f}" for k, v in stats.items())
            lines.append(f"  {detector_name}: {metric_line}")
        lines.append("")
    if model_failures:
        lines.append("Skipped models:")
        for name, reason in model_failures.items():
            lines.append(f"  {name}: {reason}")
        lines.append("")
    (REPORTS_DIR / "noisy_unsupervised_report.txt").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    (REPORTS_DIR / "noisy_unsupervised_report.json").write_text(
        json.dumps({"results": results, "failures": model_failures}, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    feature_cols, scaler, X_train = _prepare_training_basis()
    models, failures = _fit_models(X_train)
    if not models:
        raise RuntimeError("No unsupervised models could be initialized.")
    overall: Dict[str, Dict[str, Dict[str, float]]] = {}
    for noise_type in NOISE_TYPES:
        overall[noise_type] = evaluate_noise_type(
            noise_type, feature_cols, scaler, models
        )
    save_report(overall, failures)
    print("Unsupervised evaluation complete. See reports directory for details.")


if __name__ == "__main__":
    main()
