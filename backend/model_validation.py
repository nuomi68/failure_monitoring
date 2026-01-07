from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    recall_score,
    r2_score,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder

from .ml_interface import ML, TrainReport

__all__ = [
    "ValidationResult",
    "DEFAULT_DROP_COLUMNS",
    "load_supervised_dataset",
    "run_supervised_validation",
    "print_validation_report",
]

# Columns that are typically excluded from model training in this project.
DEFAULT_DROP_COLUMNS: tuple[str, ...] = ("XE-133", "CS-137", "KR-89")


@dataclass
class ValidationResult:
    """Container summarising a single validation run."""

    alg: str
    task: str
    metrics: Dict[str, float]
    meta: Dict[str, Any]
    n_total: int
    n_train: int
    n_test: int
    y_true: Optional[np.ndarray] = None
    y_pred: Optional[np.ndarray] = None
    scores: Optional[np.ndarray] = None
    confusion_matrix: Optional[np.ndarray] = None
    labels: Optional[list[str]] = None
    extra: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation."""
        data: Dict[str, Any] = {
            "alg": self.alg,
            "task": self.task,
            "metrics": dict(self.metrics),
            "meta": dict(self.meta),
            "labels": list(self.labels) if self.labels is not None else None,
            "n_total": int(self.n_total),
            "n_train": int(self.n_train),
            "n_test": int(self.n_test),
            "extra": dict(self.extra or {}),
        }
        if self.y_true is not None:
            data["y_true"] = np.asarray(self.y_true).tolist()
        if self.y_pred is not None:
            data["y_pred"] = np.asarray(self.y_pred).tolist()
        if self.scores is not None:
            data["scores"] = np.asarray(self.scores).tolist()
        if self.confusion_matrix is not None:
            data["confusion_matrix"] = np.asarray(self.confusion_matrix).tolist()
        return data


def _prepare_target(series: pd.Series) -> np.ndarray:
    """Render the target column into a numeric or categorical numpy array."""
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().all():
        filled = series.ffill().bfill().fillna("missing").astype(str)
        if filled.isna().any():
            raise ValueError("Target column contains all-NaN values after filling.")
        return filled.to_numpy()
    numeric = numeric.ffill().bfill()
    if numeric.isna().all():
        numeric = numeric.fillna(0.0)
    else:
        numeric = numeric.fillna(numeric.median())
    if numeric.isna().any():
        raise ValueError("Target column still contains NaN after preprocessing.")
    return numeric.to_numpy(dtype=np.float64)


def load_supervised_dataset(
    data_path: str | Path,
    *,
    target_column: str = "value",
    time_column: str = "TIME",
    datetime_format: Optional[str] = None,
    drop_columns: Optional[Sequence[str]] = None,
    skip_rows: int = 0,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load a tabular dataset and return ``(X, y, feature_names)`` ready for ``ML.train``.

    Parameters
    ----------
    data_path:
        CSV or Excel file containing the dataset.
    target_column:
        Name of the column to predict.
    time_column:
        Timestamp column to drop from features after optional parsing.
    datetime_format:
        Optional explicit datetime format string for ``time_column``.
    drop_columns:
        Additional feature columns to exclude.
    skip_rows:
        Skip the first ``skip_rows`` rows (useful if the sheet contains headers above the data).
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    if skip_rows:
        df = df.iloc[int(skip_rows):].reset_index(drop=True)

    df.columns = df.columns.str.strip()

    if target_column not in df.columns:
        raise KeyError(f"Target column '{target_column}' not found in dataset.")

    if time_column and time_column in df.columns:
        if datetime_format:
            df[time_column] = pd.to_datetime(df[time_column], format=datetime_format, errors="coerce")
        else:
            df[time_column] = pd.to_datetime(df[time_column], errors="coerce")

    y = _prepare_target(df[target_column])

    drop_set = set(DEFAULT_DROP_COLUMNS) if drop_columns is None else set(drop_columns)
    drop_set.add(target_column)
    if time_column:
        drop_set.add(time_column)

    feature_df = df.drop(columns=list(drop_set), errors="ignore").copy()
    if feature_df.empty:
        raise ValueError("No feature columns remain after dropping exclusions.")

    processed: Dict[str, pd.Series] = {}
    for col in feature_df.columns:
        series = feature_df[col]
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.isna().all():
            filled = series.ffill().bfill().fillna("missing").astype(str)
            processed[col] = filled
            continue
        numeric = numeric.ffill().bfill().fillna(0.0)
        processed[col] = numeric.astype(float)

    feature_df = pd.DataFrame(processed)
    feature_names = feature_df.columns.tolist()
    X = feature_df.to_numpy()
    return X, y, feature_names


def load_unsupervised_dataset(
    data_path: str | Path,
    *,
    time_column: str = "TIME",
    datetime_format: Optional[str] = None,
    drop_columns: Optional[Sequence[str]] = None,
    label_column: Optional[str] = None,
    positive_label: Any = 1,
    skip_rows: int = 0,
) -> tuple[np.ndarray, list[str], Optional[np.ndarray]]:
    """
    Load dataset for unsupervised validation.

    Returns ``(X, feature_names, labels)`` where labels is optional and
    indicates anomaly ground truth (1 => anomaly) if ``label_column`` is provided.
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    if skip_rows:
        df = df.iloc[int(skip_rows):].reset_index(drop=True)

    df.columns = df.columns.str.strip()

    labels: Optional[np.ndarray] = None
    if label_column:
        if label_column not in df.columns:
            raise KeyError(f"Label column '{label_column}' not found in dataset.")
        raw = df[label_column]
        if positive_label is not None:
            if isinstance(positive_label, str):
                tgt = str(positive_label).strip().lower()
                labels = raw.astype(str).str.strip().str.lower().eq(tgt).astype(int).to_numpy()
            else:
                labels = (raw == positive_label).astype(int).to_numpy()
        else:
            numeric = pd.to_numeric(raw, errors="coerce")
            if numeric.isna().all():
                labels = raw.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "anomaly"}).astype(int).to_numpy()
            else:
                labels = (numeric.fillna(0) > 0).astype(int).to_numpy()

    if time_column and time_column in df.columns:
        if datetime_format:
            df[time_column] = pd.to_datetime(df[time_column], format=datetime_format, errors="coerce")
        else:
            df[time_column] = pd.to_datetime(df[time_column], errors="coerce")

    drop_set = set(DEFAULT_DROP_COLUMNS) if drop_columns is None else set(drop_columns)
    if time_column:
        drop_set.add(time_column)
    if label_column:
        drop_set.add(label_column)

    feature_df = df.drop(columns=list(drop_set), errors="ignore").copy()
    if feature_df.empty:
        raise ValueError("No feature columns remain after dropping exclusions.")

    processed: Dict[str, pd.Series] = {}
    for col in feature_df.columns:
        series = feature_df[col]
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.isna().all():
            filled = series.ffill().bfill().fillna("missing").astype(str)
            processed[col] = filled
            continue
        numeric = numeric.ffill().bfill().fillna(0.0)
        processed[col] = numeric.astype(float)

    feature_df = pd.DataFrame(processed)
    feature_names = feature_df.columns.tolist()
    X = feature_df.to_numpy()
    return X, feature_names, labels


def _compute_classification_metrics(report: TrainReport, meta: Dict[str, Any], alg: str, n_total: int) -> ValidationResult:
    if report.y_true is None or report.y_pred is None:
        raise ValueError("Train report does not contain validation predictions.")

    y_true = np.asarray(report.y_true)
    y_pred = np.asarray(report.y_pred)
    scores = None if report.scores is None else np.asarray(report.scores).ravel()

    le = LabelEncoder()
    y_true_enc = le.fit_transform(y_true)
    y_pred_enc = le.transform(y_pred)

    average = "binary" if meta.get("is_binary") else "macro"
    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true_enc, y_pred_enc)),
        "precision": float(precision_score(y_true_enc, y_pred_enc, average=average, zero_division=0)),
        "recall": float(recall_score(y_true_enc, y_pred_enc, average=average, zero_division=0)),
        "f1": float(f1_score(y_true_enc, y_pred_enc, average=average, zero_division=0)),
    }
    if meta.get("is_binary") and scores is not None and len(np.unique(y_true_enc)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true_enc, scores))

    conf = confusion_matrix(y_true_enc, y_pred_enc)
    n_test = len(y_true_enc)
    n_train = max(n_total - n_test, 0)

    return ValidationResult(
        alg=alg,
        task="supervised_clf",
        metrics=metrics,
        meta=dict(meta),
        n_total=n_total,
        n_train=n_train,
        n_test=n_test,
        y_true=y_true,
        y_pred=y_pred,
        scores=scores,
        confusion_matrix=conf,
        labels=le.classes_.tolist(),
    )


def _compute_regression_metrics(report: TrainReport, meta: Dict[str, Any], alg: str, n_total: int) -> ValidationResult:
    if report.y_true is None or report.y_pred is None:
        raise ValueError("Train report does not contain validation predictions.")

    y_true = np.asarray(report.y_true, dtype=float)
    y_pred = np.asarray(report.y_pred, dtype=float)
    scores = None if report.scores is None else np.asarray(report.scores).ravel()

    mse = mean_squared_error(y_true, y_pred)
    metrics: Dict[str, float] = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)),
    }
    if np.any(np.isclose(y_true, 0)):
        metrics["mape"] = float("nan")
    else:
        metrics["mape"] = float(mean_absolute_percentage_error(y_true, y_pred))

    n_test = len(y_true)
    n_train = max(n_total - n_test, 0)

    return ValidationResult(
        alg=alg,
        task="supervised_reg",
        metrics=metrics,
        meta=dict(meta),
        n_total=n_total,
        n_train=n_train,
        n_test=n_test,
        y_true=y_true,
        y_pred=y_pred,
        scores=scores,
    )


def run_unsupervised_validation(
    alg: str,
    data_path: str | Path,
    *,
    time_column: str = "TIME",
    datetime_format: Optional[str] = None,
    drop_columns: Optional[Sequence[str]] = None,
    label_column: Optional[str] = None,
    positive_label: Any = 1,
    skip_rows: int = 0,
    scaler: Any = "standard",
    params: Optional[Dict[str, Any]] = None,
    calc_recipes: Optional[list[dict]] = None,
) -> ValidationResult:
    """
    Train an unsupervised detector via :mod:`ml_interface` and evaluate score statistics.

    If ``label_column`` is provided, anomaly detection precision/recall will also be reported.
    """
    X, feature_names, labels = load_unsupervised_dataset(
        data_path,
        time_column=time_column,
        datetime_format=datetime_format,
        drop_columns=drop_columns,
        label_column=label_column,
        positive_label=positive_label,
        skip_rows=skip_rows,
    )

    report = ML.train(
        alg=alg,
        X=X,
        feature_names=feature_names,
        params=dict(params or {}),
        scaler=scaler,
        calc_recipes=calc_recipes,
    )
    meta = ML.get_meta()
    scores = report.scores
    if scores is None:
        ML.clear()
        raise RuntimeError("Unsupervised training did not return score outputs.")
    scores_arr = np.asarray(scores, dtype=float).ravel()
    if scores_arr.size != len(X):
        scores_arr = np.resize(scores_arr, len(X))

    tau = float(meta.get("tau", 0.5) or 0.5)
    if np.isnan(tau):
        tau = 0.5
    preds = (scores_arr >= tau).astype(int)

    metrics: Dict[str, float] = {
        "score_mean": float(np.mean(scores_arr)),
        "score_std": float(np.std(scores_arr, ddof=0)),
        "score_p90": float(np.quantile(scores_arr, 0.9)),
        "score_p95": float(np.quantile(scores_arr, 0.95)),
        "tau": float(tau),
        "ratio_over_tau": float(preds.sum() / len(preds)) if len(preds) else 0.0,
    }

    conf = None
    y_true = None
    if labels is not None:
        y_true = np.asarray(labels, dtype=int).ravel()
        precision = precision_score(y_true, preds, zero_division=0)
        recall = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)
        support = int(np.sum(y_true))
        metrics.update(
            {
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "support_anomaly": float(support),
            }
        )
        conf = confusion_matrix(y_true, preds)
        if len(np.unique(y_true)) > 1:
            try:
                auc = roc_auc_score(y_true, scores_arr)
                metrics["roc_auc"] = float(auc)
            except Exception:
                pass

    result = ValidationResult(
        alg=alg,
        task="unsupervised",
        metrics=metrics,
        meta=dict(meta),
        n_total=len(X),
        n_train=len(X),
        n_test=0,
        y_true=y_true,
        y_pred=preds,
        scores=scores_arr,
        confusion_matrix=conf,
        labels=["normal", "anomaly"],
        extra={"tau": float(tau)},
    )
    ML.clear()
    return result


def run_supervised_validation(
    alg: str,
    data_path: str | Path,
    *,
    target_column: str = "value",
    time_column: str = "TIME",
    datetime_format: Optional[str] = None,
    drop_columns: Optional[Sequence[str]] = None,
    skip_rows: int = 0,
    test_size: float = 0.2,
    random_state: int = 0,
    stratify: bool | Sequence[Any] | None = None,
    scaler: Any = "standard",
    params: Optional[Dict[str, Any]] = None,
    calc_recipes: Optional[list[dict]] = None,
) -> ValidationResult:
    """
    Train the specified algorithm via :mod:`ml_interface` and evaluate on a hold-out split.
    """
    X, y, feature_names = load_supervised_dataset(
        data_path,
        target_column=target_column,
        time_column=time_column,
        datetime_format=datetime_format,
        drop_columns=drop_columns,
        skip_rows=skip_rows,
    )

    if stratify is True:
        stratify_vec: Optional[np.ndarray] = np.asarray(y)
    elif stratify is None or stratify is False:
        stratify_vec = None
    else:
        strat_arr = np.asarray(stratify)
        if strat_arr.shape[0] != len(y):
            raise ValueError("Provided stratify array must match the number of samples.")
        stratify_vec = strat_arr

    params_dict = dict(params or {})

    report = ML.train(
        alg=alg,
        X=X,
        y=y,
        feature_names=feature_names,
        params=params_dict,
        scaler=scaler,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_vec,
        calc_recipes=calc_recipes,
    )
    meta = ML.get_meta()
    task = meta.get("task")
    n_total = len(y)

    if task == "supervised_clf":
        result = _compute_classification_metrics(report, meta, alg, n_total)
    elif task == "supervised_reg":
        result = _compute_regression_metrics(report, meta, alg, n_total)
    else:
        raise ValueError(f"Unsupported task type returned by ML interface: {task}")

    ML.clear()
    return result


def print_validation_report(result: ValidationResult) -> None:
    """Pretty-print a validation result for quick manual inspection."""
    header = (
        f"[{result.alg}] task={result.task} | "
        f"test_samples={result.n_test}/{result.n_total} | "
        f"train_samples={result.n_train}"
    )
    print(header)
    for key, value in result.metrics.items():
        if isinstance(value, float):
            print(f"  {key:>12}: {value:.4f}")
        else:
            print(f"  {key:>12}: {value}")
    if result.confusion_matrix is not None:
        print("  Confusion matrix:")
        rows = np.asarray(result.confusion_matrix, dtype=int)
        for row in rows:
            row_str = " ".join(f"{int(v):4d}" for v in row)
            print(f"    {row_str}")
        if result.labels:
            print(f"  Labels: {', '.join(map(str, result.labels))}")
    if result.extra:
        print("  Extra info:")
        for key, value in result.extra.items():
            if isinstance(value, float):
                print(f"    {key:>12}: {value:.4f}")
            else:
                print(f"    {key:>12}: {value}")
