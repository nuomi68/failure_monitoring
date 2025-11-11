from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.models import supervised_core
from noisy_sample_generator import NON_FEATURE_COLUMNS, load_single

REPORTS_DIR = ROOT_DIR / "reports"
DATA_SUBDIR = REPORTS_DIR / "noisy_samples"
NOISE_TYPES = ("gaussian", "drift")

MODEL_SPECS: Dict[str, Dict] = {
    "rf": {
        "adapter": "rf",
        "params": {"n_estimators": 400, "class_weight": "balanced", "random_state": 42},
    },
    "knn_clf": {
        "adapter": "knn_clf",
        "params": {"n_neighbors": 15, "weights": "distance"},
    },
}


def _require_dataset(noise_type: str) -> Path:
    path = DATA_SUBDIR / f"{noise_type}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Please run noisy_sample_generator.py first."
        )
    return path


def _load_positive_samples(noise_type: str) -> pd.DataFrame:
    df = pd.read_csv(_require_dataset(noise_type))
    if "label" not in df.columns:
        raise ValueError(f"'label' column missing in {noise_type} dataset")
    damaged = df[df["label"] == 1].copy()
    damaged.drop(columns=["label"], inplace=True)
    damaged["target"] = 1
    damaged["noise_type"] = noise_type
    return damaged


def _build_combined_dataset(
    noise_types: Tuple[str, ...],
    clean_df: pd.DataFrame | None = None,
) -> Tuple[pd.DataFrame, List[str]]:
    clean_df = clean_df.copy() if clean_df is not None else load_single().copy()
    clean_df["target"] = 0
    clean_df["noise_type"] = "clean"

    positives = [_load_positive_samples(noise) for noise in noise_types]
    positive_df = (
        pd.concat(positives, ignore_index=True, sort=False) if positives else pd.DataFrame()
    )
    combined = pd.concat([clean_df, positive_df], ignore_index=True, sort=False)

    numeric_cols = [
        c for c in combined.columns if pd.api.types.is_numeric_dtype(combined[c])
    ]
    combined[numeric_cols] = combined[numeric_cols].apply(
        pd.to_numeric, errors="coerce"
    )
    combined[numeric_cols] = combined[numeric_cols].interpolate(
        limit_direction="both"
    )
    combined[numeric_cols] = combined[numeric_cols].fillna(
        combined[numeric_cols].median()
    )

    feature_cols = [
        c
        for c in combined.columns
        if c not in NON_FEATURE_COLUMNS.union({"target", "noise_type"})
        and pd.api.types.is_numeric_dtype(combined[c])
    ]
    if not feature_cols:
        raise RuntimeError("No numeric columns available for supervised training.")
    return combined, feature_cols


def _adapter_map() -> Dict[str, supervised_core.AlgoAdapter]:
    mapping = {}
    for adapter in supervised_core.ADAPTERS:
        mapping[adapter.meta_model_type()] = adapter
        mapping[adapter.code] = adapter
    return mapping


def _train_models(
    df: pd.DataFrame, feature_cols: List[str]
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, str]]:
    X = df[feature_cols].to_numpy(dtype=float)
    y = df["target"].to_numpy(dtype=int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    adapter_lookup = _adapter_map()
    metrics: Dict[str, Dict[str, float]] = {}
    failures: Dict[str, str] = {}

    for model_name, spec in MODEL_SPECS.items():
        adapter = adapter_lookup.get(spec["adapter"] or model_name)
        if adapter is None:
            failures[model_name] = "Adapter not available"
            continue
        try:
            model = adapter.build(**spec.get("params", {}))
            fitted = adapter.fit(model, X_train_scaled, y_train)
            y_pred = adapter.predict(fitted, X_test_scaled)
            scores = adapter.scores(fitted, X_test_scaled, classes_=None)
            if scores is None:
                scores = y_pred
            report = classification_report(
                y_test, y_pred, output_dict=True, zero_division=0
            )
            metrics[model_name] = {
                "accuracy": float(report.get("accuracy", 0.0)),
                "precision_1": float(report["1"]["precision"]),
                "recall_1": float(report["1"]["recall"]),
                "f1_1": float(report["1"]["f1-score"]),
                "roc_auc": float(roc_auc_score(y_test, scores)),
            }
        except Exception as exc:
            failures[model_name] = f"{type(exc).__name__}: {exc}"
    return metrics, failures


def save_report(
    datasets: Dict[
        str, Dict[str, Dict[str, float] | Dict[str, Dict[str, float]] | Dict[str, str]]
    ]
) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "Supervised Binary Evaluation (clean vs. noisy damage segments)",
        "=" * 62,
    ]
    for dataset_name, payload in datasets.items():
        info = payload["info"]
        metrics = payload["metrics"]
        failures = payload["failures"]
        lines.extend(
            [
                "",
                f"Dataset: {dataset_name}",
                f"  Clean samples: {int(info['clean_count'])}",
                f"  Damaged samples: {int(info['damage_count'])}",
                f"  Damage ratio: {info['damage_ratio']:.3f}",
            ]
        )
        for model_name, stats in metrics.items():
            metric_line = ", ".join(f"{k}={v:.4f}" for k, v in stats.items())
            lines.append(f"  {model_name}: {metric_line}")
        if failures:
            lines.append("  Skipped models:")
            for name, reason in failures.items():
                lines.append(f"    {name}: {reason}")
    (REPORTS_DIR / "noisy_supervised_report.txt").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    (REPORTS_DIR / "noisy_supervised_report.json").write_text(
        json.dumps(datasets, indent=2), encoding="utf-8"
    )


def main() -> None:
    base_clean = load_single()
    dataset_configs: Dict[str, Tuple[str, ...]] = {
        "gaussian": ("gaussian",),
        "drift": ("drift",),
        "gaussian+drift": ("gaussian", "drift"),
    }
    results: Dict[str, Dict] = {}

    for name, noises in dataset_configs.items():
        df, feature_cols = _build_combined_dataset(noises, clean_df=base_clean)
        dataset_info = {
            "clean_count": float((df["target"] == 0).sum()),
            "damage_count": float((df["target"] == 1).sum()),
            "damage_ratio": float((df["target"] == 1).mean()),
        }
        metrics, failures = _train_models(df, feature_cols)
        if not metrics and failures:
            raise RuntimeError(f"No supervised models evaluated for dataset '{name}'.")
        results[name] = {"info": dataset_info, "metrics": metrics, "failures": failures}

    save_report(results)
    print("Supervised evaluation complete. See reports directory for outputs.")


if __name__ == "__main__":
    main()
