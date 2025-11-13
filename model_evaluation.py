from __future__ import annotations

import argparse
import json
import math
import sys
import textwrap
import math
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from backend.model_validation import (
    ValidationResult,
    print_validation_report,
    run_supervised_validation,
    run_unsupervised_validation,
)
from backend.timeseries_interface import ModelManager as TimeSeriesModelManager
from backend.fault_level_estimator import FaultLevelEstimator
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split

DEFAULT_DATA_PATH = "data/20230510-20240924_test.xlsx"
TMP_DIR = Path(".model_eval_tmp")


def _module_available(name: str) -> bool:
    try:
        import_module(name)
        return True
    except Exception:
        return False


def _slugify(text: str) -> str:
    safe = "".join(ch if ch.isalnum() else "_" for ch in text)
    safe = "_".join(part for part in safe.split("_") if part)
    return safe.lower() or "model"


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def _write_table(df: pd.DataFrame, path: Path) -> None:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df.to_excel(path, index=False)
    else:
        df.to_csv(path, index=False)


def _derive_label(df: pd.DataFrame, spec: Dict[str, Any]) -> tuple[pd.DataFrame, str]:
    src = spec.get("source_column")
    if not src or src not in df.columns:
        raise KeyError(f"derived_label.source_column '{src}' not found in dataframe columns")

    series = pd.to_numeric(df[src], errors="coerce")
    if series.isna().all():
        raise ValueError(f"column '{src}' cannot be converted to numeric for label derivation")

    positive = spec.get("positive_label", 1)
    negative = spec.get("negative_label", 0)
    method = spec.get("method", "threshold").lower()
    output_column = spec.get("output_column") or spec.get("target_column") or "label"

    if method == "threshold":
        threshold = spec.get("threshold", "median")
        if isinstance(threshold, dict) and "percentile" in threshold:
            pct = float(threshold["percentile"])
            thr = float(series.quantile(pct))
        elif isinstance(threshold, str):
            name = threshold.lower()
            if name == "median":
                thr = float(series.median())
            elif name == "mean":
                thr = float(series.mean())
            else:
                raise ValueError(f"Unsupported threshold keyword '{threshold}'")
        else:
            thr = float(threshold)
        labels = np.where(series >= thr, positive, negative)
    elif method == "top_k":
        k = int(spec.get("k", max(1, int(len(series) * 0.05))))
        idx = series.nlargest(k).index
        labels = np.full(len(series), negative, dtype=object)
        labels[idx] = positive
    else:
        raise ValueError(f"Unsupported derived_label.method '{method}'")

    df = df.copy()
    df[output_column] = labels
    return df, output_column


def _prepare_dataset(entry: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
    derived = entry.get("derived_label")
    if not derived:
        return entry["data_path"], entry

    source_path = Path(entry["data_path"])
    df = _read_table(source_path)
    df.columns = df.columns.str.strip()
    df_prepared, output_col = _derive_label(df, derived)

    TMP_DIR.mkdir(parents=True, exist_ok=True)
    target_name = f"{_slugify(entry.get('name', entry['alg']))}_{source_path.name}"
    tmp_path = TMP_DIR / target_name
    _write_table(df_prepared, tmp_path)

    updated_entry = dict(entry)
    if entry.get("task", "supervised").startswith("supervised"):
        updated_entry["target_column"] = output_col
    else:
        updated_entry["label_column"] = output_col
    updated_entry["data_path"] = str(tmp_path)
    return str(tmp_path), updated_entry


def _validate_entry(entry: Dict[str, Any]) -> None:
    if "alg" not in entry:
        raise KeyError("Each configuration entry must include 'alg'.")
    if "data_path" not in entry:
        raise KeyError(f"Configuration for '{entry['alg']}' is missing 'data_path'.")


def _evaluate_model(entry: Dict[str, Any], quiet: bool = False) -> Optional[ValidationResult]:
    _validate_entry(entry)
    requires = entry.get("requires", [])
    for module_name in requires:
        if not _module_available(module_name):
            print(f"[SKIP] {entry.get('name', entry['alg'])}: missing dependency '{module_name}'")
            return None

    prepared_path, prepared_entry = _prepare_dataset(entry)
    task = prepared_entry.get("task", "supervised")
    name = prepared_entry.get("name", prepared_entry["alg"])
    if not quiet:
        print(f"[RUN ] {name} -> {prepared_path}")

    if task.startswith("supervised"):
        result = run_supervised_validation(
            alg=prepared_entry["alg"],
            data_path=prepared_path,
            target_column=prepared_entry.get("target_column", "value"),
            time_column=prepared_entry.get("time_column", "TIME"),
            datetime_format=prepared_entry.get("datetime_format"),
            drop_columns=prepared_entry.get("drop_columns"),
            skip_rows=int(prepared_entry.get("skip_rows", 0)),
            test_size=float(prepared_entry.get("test_size", 0.2)),
            random_state=int(prepared_entry.get("random_state", 0)),
            stratify=prepared_entry.get("stratify"),
            scaler=prepared_entry.get("scaler", "standard"),
            params=prepared_entry.get("params"),
            calc_recipes=prepared_entry.get("calc_recipes"),
        )
    elif task == "timeseries":
        result = _evaluate_timeseries(prepared_entry)
    elif task == "fault_level":
        result = _evaluate_fault_level(prepared_entry)
    else:
        result = run_unsupervised_validation(
            alg=prepared_entry["alg"],
            data_path=prepared_path,
            time_column=prepared_entry.get("time_column", "TIME"),
            datetime_format=prepared_entry.get("datetime_format"),
            drop_columns=prepared_entry.get("drop_columns"),
            label_column=prepared_entry.get("label_column"),
            positive_label=prepared_entry.get("positive_label", 1),
            skip_rows=int(prepared_entry.get("skip_rows", 0)),
            scaler=prepared_entry.get("scaler", "standard"),
            params=prepared_entry.get("params"),
            calc_recipes=prepared_entry.get("calc_recipes"),
        )

    if not quiet:
        print_validation_report(result)
        print("-" * 80)
    return result


def _load_config(path: Optional[str]) -> List[Dict[str, Any]]:
    if not path:
        return DEFAULT_CONFIG.copy()

    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    if cfg_path.suffix.lower() in {".json"}:
        return json.loads(cfg_path.read_text(encoding="utf-8"))

    try:
        import yaml  # type: ignore

        with cfg_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        if not isinstance(data, list):
            raise ValueError("YAML configuration must contain a list at the top level.")
        return data
    except ImportError:
        raise RuntimeError("PyYAML is required to load non-JSON configuration files.")


def _summarise(results: Iterable[ValidationResult]) -> List[Dict[str, Any]]:
    summary = []
    for res in results:
        row = {
            "alg": res.alg,
            "task": res.task,
            "n_total": res.n_total,
            "n_test": res.n_test,
        }
        for key, value in res.metrics.items():
            if isinstance(value, float) and math.isnan(value):
                row[key] = None
            else:
                row[key] = value
        summary.append(row)
    return summary


def _print_summary(summary: List[Dict[str, Any]]) -> None:
    if not summary:
        print("No models were evaluated.")
        return

    all_keys = {"alg", "task", "n_total", "n_test"}
    for row in summary:
        all_keys.update(row.keys())

    columns = [k for k in ["alg", "task", "n_total", "n_test"] if k in all_keys]
    metric_cols = sorted(all_keys - set(columns))
    columns.extend(metric_cols)

    col_widths = {col: max(len(col), *(len(f"{row.get(col, '')}") for row in summary)) for col in columns}

    def fmt(row: Dict[str, Any], key: str) -> str:
        val = row.get(key, "")
        if isinstance(val, float):
            if math.isnan(val):
                return "nan"
            return f"{val:.4f}"
        return str(val)

    header = " | ".join(f"{col:<{col_widths[col]}}" for col in columns)
    print(header)
    print("-" * len(header))
    for row in summary:
        print(" | ".join(f"{fmt(row, col):<{col_widths[col]}}" for col in columns))


def _save_output(summary: List[Dict[str, Any]], json_path: Optional[str], csv_path: Optional[str]) -> None:
    if json_path:
        Path(json_path).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"JSON summary saved to {json_path}")
    if csv_path:
        pd.DataFrame(summary).to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"CSV summary saved to {csv_path}")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="批量评估多个模型，输出统一的误差指标。",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--config",
        help=textwrap.dedent(
            """\
            配置文件路径（JSON 或 YAML）。若不提供，则使用内置示例配置：
              - data/20230510-20240924_test.xlsx 做回归、分类与异常检测
              - 分类标签默认使用 value 列的中位数阈值生成
              - 异常标签默认使用 value 的 98% 分位数生成
            """
        ),
    )
    parser.add_argument("--output-json", help="将评估汇总写入指定 JSON 文件。")
    parser.add_argument("--output-csv", help="将评估汇总写入指定 CSV 文件。")
    parser.add_argument("--quiet", action="store_true", help="仅输出最终汇总表，隐藏逐模型详情。")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    config = _load_config(args.config)
    results: List[ValidationResult] = []

    for entry in config:
        if not entry.get("enabled", True):
            print(f"[SKIP] {entry.get('name', entry.get('alg'))}: disabled in config.")
            continue
        res = _evaluate_model(entry, quiet=args.quiet)
        if res is not None:
            results.append(res)
            if args.quiet:
                continue
    summary = _summarise(results)
    print("\n汇总指标：")
    _print_summary(summary)
    _save_output(summary, args.output_json, args.output_csv)
    _write_task_reports(results)
    return 0


def _evaluate_timeseries(entry: Dict[str, Any]) -> ValidationResult:
    data_path = Path(entry["data_path"])
    df = _read_table(data_path)
    df.columns = df.columns.str.strip()

    time_col = entry.get("time_column", "TIME")
    time_format = entry.get("datetime_format") or ""
    manager = TimeSeriesModelManager()

    manifest = manager.register_dataset(
        df,
        time_col=time_col,
        time_format=time_format,
        source_name=data_path.name,
    )
    dataset_id = manifest["dataset_id"]

    model_type = entry.get("model_type") or entry.get("alg")
    params = dict(entry.get("params") or {})
    holdout = int(params.get("holdout", 5))

    response = manager.train(
        dataset_id=dataset_id,
        model_type=model_type,
        params=params,
    )
    metrics = response.get("metrics", {})
    extra = response.get("extra", {})
    look_back = int(extra.get("look_back", params.get("look_back", 14)))

    total_rows = len(df)
    usable = max(total_rows - holdout - look_back, 0)
    n_test = int(np.ceil(usable * 0.2))
    n_train = max(usable - n_test, 0)

    result = ValidationResult(
        alg=model_type,
        task="timeseries",
        metrics={k: float(v) for k, v in metrics.items()},
        meta={"model_type": model_type, "params": params},
        n_total=total_rows,
        n_train=n_train,
        n_test=n_test,
        extra={"look_back": look_back, "holdout": holdout},
    )
    return result


def _evaluate_fault_level(entry: Dict[str, Any]) -> ValidationResult:
    data_path = Path(entry["data_path"])
    df = _read_table(data_path)
    df.columns = df.columns.str.strip()

    label_col = entry.get("label_column", "level")
    feature_cols = [c for c in df.columns if c != label_col]
    if not feature_cols:
        raise ValueError("Fault-level dataset must contain feature columns.")

    test_size = float(entry.get("test_size", 0.25))
    random_state = int(entry.get("random_state", 42))

    counts = df[label_col].value_counts()
    use_synthetic = counts.min() < 2 or len(df) < 20

    if use_synthetic:
        df_train = df
        rng = np.random.default_rng(random_state)
        base = df[feature_cols].to_numpy(dtype=float)
        perturb = base * (1 + rng.normal(0.0, 0.08, size=base.shape))
        X_test = np.clip(perturb, a_min=0.0, a_max=None)
        y_true = df[label_col].astype(str).to_numpy()
    else:
        stratify_labels = df[label_col] if counts.min() >= 2 else None
        df_train, df_test = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_labels,
        )
        X_test = df_test[feature_cols].to_numpy(dtype=float)
        y_true = df_test[label_col].astype(str).to_numpy()

    estimator = FaultLevelEstimator(
        labelled_X=df_train[feature_cols].to_numpy(dtype=float),
        labels=df_train[label_col].astype(str).to_numpy(),
        method=entry.get("method", "wknn"),
        scaler=entry.get("scaler", "standard"),
        feature_names=feature_cols,
    )

    y_pred = estimator.predict(X_test)

    acc = float(accuracy_score(y_true, y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    conf = confusion_matrix(y_true, y_pred)

    samples = []
    for idx, (yt, yp) in enumerate(zip(y_true[:5], y_pred[:5])):
        sample = {
            "true": yt,
            "pred": yp,
            "features": {
                feature_cols[i]: float(X_test[idx, i]) for i in range(len(feature_cols))
            },
        }
        samples.append(sample)

    result = ValidationResult(
        alg=entry.get("name", entry["alg"]),
        task="fault_level",
        metrics={
            "accuracy": acc,
            "precision_macro": float(precision),
            "recall_macro": float(recall),
            "f1_macro": float(f1),
        },
        meta={
            "method": entry.get("method", "wknn"),
            "scaler": entry.get("scaler", "standard"),
        },
        n_total=len(df),
        n_train=len(df_train),
        n_test=len(y_true),
        y_true=y_true,
        y_pred=y_pred,
        confusion_matrix=conf,
        extra={"samples": samples},
    )
    return result


def _write_task_reports(results: List[ValidationResult]) -> None:
    task_map = {
        "timeseries": Path("reports/model_eval_timeseries_report.txt"),
        "fault_level": Path("reports/model_eval_fault_report.txt"),
    }

    grouped: Dict[str, List[ValidationResult]] = {}
    for res in results:
        grouped.setdefault(res.task, []).append(res)

    for task, path in task_map.items():
        if task not in grouped:
            continue
        lines = [f"{task.replace('_', ' ').title()} Evaluation", "=" * 32]
        for res in grouped[task]:
            lines.append(f"Model: {res.alg}")
            for key, value in res.metrics.items():
                if isinstance(value, float):
                    lines.append(f"  {key}: {value:.4f}")
                else:
                    lines.append(f"  {key}: {value}")
            if res.confusion_matrix is not None:
                lines.append("  Confusion Matrix:")
                for row in np.asarray(res.confusion_matrix, dtype=int):
                    row_str = " ".join(f"{int(v):4d}" for v in row)
                    lines.append(f"    {row_str}")
            if res.extra and "samples" in res.extra:
                lines.append("  Sample Predictions:")
                for sample in res.extra["samples"]:
                    lines.append(
                        f"    true={sample['true']} pred={sample['pred']} "
                        f"features={sample['features']}"
                    )
            lines.append("")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines), encoding="utf-8")


DEFAULT_CONFIG: List[Dict[str, Any]] = [
    {
        "name": "RandomForest Regression",
        "alg": "rf_reg",
        "task": "supervised_reg",
        "data_path": DEFAULT_DATA_PATH,
        "target_column": "value",
        "test_size": 0.2,
        "random_state": 42,
        "scaler": "standard",
        "params": {"n_estimators": 200, "random_state": 42},
    },
    {
        "name": "KNN Regression",
        "alg": "knn_reg",
        "task": "supervised_reg",
        "data_path": DEFAULT_DATA_PATH,
        "target_column": "value",
        "test_size": 0.25,
        "random_state": 7,
        "scaler": "standard",
        "params": {"n_neighbors": 5, "weights": "distance"},
    },
    {
        "name": "RandomForest Classification (value ≥ median)",
        "alg": "rf_clf",
        "task": "supervised_clf",
        "data_path": DEFAULT_DATA_PATH,
        "target_column": "value_label",
        "test_size": 0.3,
        "random_state": 99,
        "scaler": "standard",
        "stratify": True,
        "params": {"n_estimators": 150, "random_state": 99},
        "derived_label": {
            "source_column": "value",
            "output_column": "value_label",
            "method": "threshold",
            "threshold": "median",
            "positive_label": "fault",
            "negative_label": "normal",
        },
    },
    {
        "name": "KNN Classification (value ≥ median)",
        "alg": "knn_clf",
        "task": "supervised_clf",
        "data_path": DEFAULT_DATA_PATH,
        "target_column": "value_label",
        "test_size": 0.3,
        "random_state": 13,
        "scaler": "standard",
        "stratify": True,
        "params": {"n_neighbors": 7},
        "derived_label": {
            "source_column": "value",
            "output_column": "value_label",
            "method": "threshold",
            "threshold": "median",
            "positive_label": "fault",
            "negative_label": "normal",
        },
    },
    {
        "name": "KNN Detector (value ≥ 98% quantile)",
        "alg": "knn",
        "task": "unsupervised",
        "data_path": DEFAULT_DATA_PATH,
        "label_column": "anomaly_label",
        "scaler": "standard",
        "params": {"n_neighbors": 20},
        "derived_label": {
            "source_column": "value",
            "output_column": "anomaly_label",
            "method": "threshold",
            "threshold": {"percentile": 0.98},
            "positive_label": 1,
            "negative_label": 0,
        },
    },
    {
        "name": "IsolationForest Detector (value ≥ 98% quantile)",
        "alg": "iforest",
        "task": "unsupervised",
        "data_path": DEFAULT_DATA_PATH,
        "label_column": "anomaly_label",
        "scaler": "standard",
        "params": {"n_estimators": 256, "contamination": 0.02, "random_state": 42},
        "derived_label": {
            "source_column": "value",
            "output_column": "anomaly_label",
            "method": "threshold",
            "threshold": {"percentile": 0.98},
            "positive_label": 1,
            "negative_label": 0,
        },
    },
    {
        "name": "AutoEncoder Detector (value ≥ 98% quantile)",
        "alg": "autoencoder",
        "task": "unsupervised",
        "data_path": DEFAULT_DATA_PATH,
        "label_column": "anomaly_label",
        "scaler": "standard",
        "requires": ["torch"],
        "params": {
            "hidden": [64, 32],
            "latent_dim": 16,
            "epochs": 15,
            "batch_size": 128,
            "lr": 1e-3,
        },
        "derived_label": {
            "source_column": "value",
            "output_column": "anomaly_label",
            "method": "threshold",
            "threshold": {"percentile": 0.98},
            "positive_label": 1,
            "negative_label": 0,
        },
    },
    {
        "name": "TimeSeries RandomForest",
        "alg": "ts_rf",
        "task": "timeseries",
        "model_type": "rf",
        "data_path": DEFAULT_DATA_PATH,
        "time_column": "TIME",
        "params": {
            "look_back": 6,
            "holdout": 14,
        },
    },
    {
        "name": "TimeSeries TCN",
        "alg": "ts_tcn",
        "task": "timeseries",
        "model_type": "tcn",
        "data_path": DEFAULT_DATA_PATH,
        "time_column": "TIME",
        "requires": ["torch"],
        "params": {
            "look_back": 14,
            "holdout": 14,
            "epochs": 3,
            "batch_size": 32,
        },
    },
    {
        "name": "Fault Level Estimation (wKNN)",
        "alg": "fault_wknn",
        "task": "fault_level",
        "data_path": "data/break_level.xlsx",
        "label_column": "level",
        "method": "wknn",
        "scaler": "standard",
        "test_size": 0.3,
        "random_state": 7,
    },
    {
        "name": "Fault Level Estimation (Radius)",
        "alg": "fault_radius",
        "task": "fault_level",
        "data_path": "data/break_level.xlsx",
        "label_column": "level",
        "method": "radius",
        "scaler": "standard",
        "test_size": 0.3,
        "random_state": 21,
    },
]


if __name__ == "__main__":
    sys.exit(main())
