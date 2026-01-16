"""
Generate publication-ready metrics + figures for the paper.

This script evaluates the tabular dataset `data/部分破损.xlsx` only, using the backend
supervised models: KNN / RF / XGB / LightGBM, plus unsupervised detectors:
IsolationForest / KNN / AutoEncoder.

Outputs:
- `reports/figures1/` (figures)
- `reports/paper_figures_summary.json` (metrics summary)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn import metrics  # noqa: E402


ROOT_DIR = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT_DIR / "reports"
FIG_DIR = REPORTS_DIR / "figures1"
DATA_DIR = ROOT_DIR / "data"
FILE_PARTIAL = DATA_DIR / "\u90e8\u5206\u7834\u635f.xlsx"  # 部分破损.xlsx
TARGET_COLUMN = "\u662f\u5426\u7834\u635f"  # 是否破损

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.model_validation import run_supervised_validation  # noqa: E402
from backend.model_validation import run_unsupervised_validation  # noqa: E402
from matplotlib import font_manager, rcParams
SUP_TEST_SIZE = 0.3
SUP_RANDOM_STATES = (0, 1, 2, 3, 4)
UNSUP_TOP_RATIO = 0.15


def _ensure_dirs() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def _plot_points(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: str | None,
    title: str,
    path: Path,
) -> None:
    plt.figure(figsize=(8, 4.2))
    order = list(df.sort_values(y, ascending=False)[x].unique())
    if hue:
        sns.pointplot(
            data=df,
            x=x,
            y=y,
            hue=hue,
            dodge=0.3,
            markers="o",
            linestyles="-",
            order=order,
            palette="crest",
        )
    else:
        sns.pointplot(
            data=df,
            x=x,
            y=y,
            dodge=False,
            markers="o",
            linestyles="-",
            order=order,
            color=sns.color_palette("crest", 1)[0],
        )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def _plot_heatmap(data: pd.DataFrame, title: str, path: Path, fmt: str = ".3f") -> None:
    csv_path = path.with_name(path.stem + f"{title}_matrix.csv")
    json_path = path.with_name(path.stem +f"{title}_matrix.json")

    # CSV：通用、肉眼可读；utf-8-sig 方便 Excel 打开不乱码
    data.to_csv(csv_path, encoding="utf-8-sig")

    # JSON：保留 index/columns 结构更稳（不怕逗号/特殊字符）
    data.to_json(json_path, force_ascii=False, orient="split")

    # 同时存一份标题/格式等元信息（可选）
    meta_path = path.with_name(path.stem + "_meta.json")
    meta = {"title": title, "fmt": fmt, "rows": list(data.index), "cols": list(data.columns)}
    Path(meta_path).write_text(pd.Series(meta).to_json(force_ascii=False), encoding="utf-8")

    print("Saved matrix:", csv_path)
    print("Saved matrix:", json_path)
    # 1) 选字体文件（比按 name 更稳）
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
    chosen = next((n for n in candidates if n in available), None)

    # 2) 构造 FontProperties
    fp = font_manager.FontProperties(family=chosen) if chosen else None

    plt.figure(figsize=(6.5, 3.5))
    ax = sns.heatmap(
        data,
        annot=True,
        fmt=fmt,
        cmap="viridis",
        cbar_kws={"shrink": 0.8},
        annot_kws={"fontproperties": fp} if fp else None,  # ← 关键
    )

    # 3) 标题/坐标轴也显式设字体（避免被主题影响）
    if fp:
        ax.set_title(title, fontproperties=fp)
        ax.set_xlabel("指标", fontproperties=fp)
        ax.set_ylabel("模型", fontproperties=fp)
        for tick in ax.get_xticklabels():
            tick.set_fontproperties(fp)
        for tick in ax.get_yticklabels():
            tick.set_fontproperties(fp)
    else:
        ax.set_title(title)
        ax.set_xlabel("指标")
        ax.set_ylabel("模型")

    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def _binary_truth(y_true: np.ndarray, *, classes: List[Any] | None) -> Tuple[np.ndarray, Any | None]:
    """
    Convert the original y_true labels into a 0/1 vector aligned with ML's score output,
    where class index 1 is treated as the positive class.
    """
    if classes and len(classes) >= 2:
        positive = classes[1]
    else:
        uniq = list(pd.unique(y_true))
        if len(uniq) < 2:
            return np.zeros(len(y_true), dtype=int), None
        positive = sorted(uniq, key=lambda v: str(v))[1]
    y_bin = (np.asarray(y_true) == positive).astype(int)
    return y_bin, positive


def _dataset_info(path: Path, target_column: str) -> Dict[str, Any]:
    df = pd.read_excel(path)
    df.columns = df.columns.astype(str).str.strip()
    if target_column not in df.columns:
        raise KeyError(f"Target column '{target_column}' not found in {path.name}")
    y = df[target_column].astype(str).str.strip()
    counts = y.value_counts(dropna=False).to_dict()
    info: Dict[str, Any] = {
        "path": str(path),
        "n_samples": int(len(df)),
        "target": target_column,
        "label_counts": {str(k): int(v) for k, v in counts.items()},
    }
    if len(counts) == 2:
        sorted_labels = sorted(map(str, counts.keys()))
        pos_label = sorted_labels[1]
        info["positive_label"] = pos_label
        info["positive_rate"] = float((y == pos_label).mean())
    return info


def run_supervised_block() -> Dict[str, Any]:
    print("[1/2] Supervised (部分破损) block")
    if not FILE_PARTIAL.exists():
        raise FileNotFoundError(FILE_PARTIAL)

    data_info = _dataset_info(FILE_PARTIAL, TARGET_COLUMN)

    model_specs: Dict[str, Tuple[str, Dict[str, Any]]] = {
        "KNN": ("knn_clf", {"n_neighbors": 15, "weights": "distance"}),
        "RF": (
            "rf_clf",
            {
                "n_estimators": 300,
                "max_depth": None,
                "class_weight": "balanced_subsample",
                "n_jobs": -1,
                "random_state": 42,
            },
        ),
        "XGB": (
            "xgb_clf",
            {
                "n_estimators": 300,
                "learning_rate": 0.05,
                "max_depth": 4,
                "subsample": 0.85,
                "colsample_bytree": 0.9,
                "reg_lambda": 1.0,
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "tree_method": "hist",
                "random_state": 42,
                "verbosity": 0,
            },
        ),
        "LGBM": (
            "lgbm_clf",
            {
                "n_estimators": 300,
                "learning_rate": 0.05,
                "max_depth": -1,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "reg_lambda": 1.0,
                "objective": "binary",
                "metric": "binary_logloss",
                "random_state": 42,
                "n_jobs": -1,
                "verbose": -1,
            },
        ),
    }

    per_split_rows: List[Dict[str, Any]] = []
    roc_pool: Dict[str, Dict[str, List[np.ndarray]]] = {}
    failures: Dict[str, str] = {}

    for display_name, (alg, params) in model_specs.items():
        for split_id, seed in enumerate(SUP_RANDOM_STATES):
            try:
                result = run_supervised_validation(
                    alg,
                    FILE_PARTIAL,
                    target_column=TARGET_COLUMN,
                    drop_columns=(),
                    test_size=SUP_TEST_SIZE,
                    random_state=seed,
                    stratify=True,
                    scaler="standard",
                    params=params,
                )
            except Exception as exc:
                failures[display_name] = f"{type(exc).__name__}: {exc}"
                break

            auc = float(result.metrics.get("roc_auc", float("nan")))
            per_split_rows.append(
                {
                    "model": display_name,
                    "split_id": int(split_id),
                    "seed": int(seed),
                    "accuracy": float(result.metrics.get("accuracy", 0.0)),
                    "precision": float(result.metrics.get("precision", 0.0)),
                    "recall": float(result.metrics.get("recall", 0.0)),
                    "f1": float(result.metrics.get("f1", 0.0)),
                    "auc": auc,
                }
            )

            if result.scores is None or result.y_true is None:
                continue
            y_bin, _ = _binary_truth(
                np.asarray(result.y_true),
                classes=(result.meta or {}).get("classes_"),
            )
            if len(np.unique(y_bin)) < 2:
                continue
            roc_pool.setdefault(display_name, {"y": [], "s": []})
            roc_pool[display_name]["y"].append(np.asarray(y_bin, dtype=int))
            roc_pool[display_name]["s"].append(np.asarray(result.scores, dtype=float).ravel())

    metrics_mean: List[Dict[str, float]] = []
    metrics_std: List[Dict[str, float]] = []
    roc_payload: List[Tuple[str, np.ndarray, np.ndarray, float]] = []
    if per_split_rows:
        df = pd.DataFrame(per_split_rows)
        for model_name in df["model"].unique():
            sub = df[df["model"] == model_name]
            mean_row = {
                "model": model_name,
                "accuracy": float(sub["accuracy"].mean()),
                "precision": float(sub["precision"].mean()),
                "recall": float(sub["recall"].mean()),
                "f1": float(sub["f1"].mean()),
                "auc": float(sub["auc"].mean(skipna=True)),
            }
            std_row = {
                "model": model_name,
                "accuracy": float(sub["accuracy"].std(ddof=0)),
                "precision": float(sub["precision"].std(ddof=0)),
                "recall": float(sub["recall"].std(ddof=0)),
                "f1": float(sub["f1"].std(ddof=0)),
                "auc": float(sub["auc"].std(ddof=0)),
            }
            metrics_mean.append(mean_row)
            metrics_std.append(std_row)

        # Pooled ROC across splits (concatenate scores/labels)
        for model_name, payload in roc_pool.items():
            y_all = np.concatenate(payload["y"], axis=0) if payload["y"] else np.array([])
            s_all = np.concatenate(payload["s"], axis=0) if payload["s"] else np.array([])
            if y_all.size == 0 or s_all.size == 0 or len(np.unique(y_all)) < 2:
                continue
            fpr, tpr, _ = metrics.roc_curve(y_all, s_all)
            auc = float(metrics.roc_auc_score(y_all, s_all))
            roc_payload.append((model_name, fpr, tpr, auc))

    if metrics_mean:
        plot_df = pd.DataFrame(metrics_mean)
        heat_df = plot_df.set_index("model")[["accuracy", "precision", "recall", "f1", "auc"]]
        _plot_heatmap(
            heat_df,
            f"监督学习破损状态判别",
            FIG_DIR / "clf_heatmap.png",
            fmt=".4f",
        )
        _plot_points(
            plot_df[["model", "auc"]],
            "model",
            "auc",
            None,
            f"Supervised AUC mean (splits={len(SUP_RANDOM_STATES)})",
            FIG_DIR / "clf_auc.png",
        )

    if roc_payload:
        plt.figure(figsize=(6.5, 5))
        for name, fpr, tpr, auc in roc_payload:
            plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
        plt.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.title("ROC - damage detection (pooled splits)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIG_DIR / "clf_roc.png", dpi=300)
        plt.close()

    return {
        "dataset": data_info,
        "split_info": {"test_size": SUP_TEST_SIZE, "seeds": list(SUP_RANDOM_STATES)},
        "metrics": metrics_mean,
        "metrics_std": metrics_std,
        "metrics_by_split": per_split_rows,
        "failures": failures,
    }


def run_unsupervised_block() -> Dict[str, Any]:
    print("[2/2] Unsupervised (部分破损) block")
    if not FILE_PARTIAL.exists():
        raise FileNotFoundError(FILE_PARTIAL)

    data_info = _dataset_info(FILE_PARTIAL, TARGET_COLUMN)
    positive_label = data_info.get("positive_label")
    if positive_label is None:
        raise ValueError("Unsupervised evaluation requires a binary label column.")

    model_specs: Dict[str, Tuple[str, Dict[str, Any]]] = {
        "IForest": (
            "iforest",
            {"n_estimators": 400, "contamination": 0.05, "random_state": 42, "n_jobs": -1},
        ),
        "KNN": ("knn", {"n_neighbors": 35}),
        "AutoEncoder": (
            "autoencoder",
            {
                "hidden": [64, 32],
                "latent_dim": 16,
                "epochs": 10,
                "batch_size": 128,
                "lr": 1e-3,
                "dropout": 0.1,
            },
        ),
    }

    metrics_rows: List[Dict[str, float]] = []
    roc_payload: List[Tuple[str, np.ndarray, np.ndarray, float]] = []
    failures: Dict[str, str] = {}

    for display_name, (alg, params) in model_specs.items():
        try:
            result = run_unsupervised_validation(
                alg,
                FILE_PARTIAL,
                label_column=TARGET_COLUMN,
                positive_label=positive_label,
                drop_columns=(),
                scaler="standard",
                params=params,
            )
        except Exception as exc:
            failures[display_name] = f"{type(exc).__name__}: {exc}"
            continue

        if result.scores is None or result.y_true is None:
            failures[display_name] = "missing scores or labels"
            continue

        scores = np.asarray(result.scores, dtype=float).ravel()
        y_true = np.asarray(result.y_true, dtype=int).ravel()
        if scores.size == 0 or y_true.size == 0:
            failures[display_name] = "empty scores/labels"
            continue

        cutoff = float(np.quantile(scores, 1.0 - UNSUP_TOP_RATIO))
        y_pred = (scores >= cutoff).astype(int)

        acc = metrics.accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = metrics.precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        auc = (
            float(metrics.roc_auc_score(y_true, scores))
            if len(np.unique(y_true)) > 1
            else float("nan")
        )
        metrics_rows.append(
            {
                "model": display_name,
                "accuracy": float(acc),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "auc": float(auc),
            }
        )

        if len(np.unique(y_true)) > 1:
            fpr, tpr, _ = metrics.roc_curve(y_true, scores)
            roc_payload.append((display_name, fpr, tpr, float(auc)))

    if metrics_rows:
        plot_df = pd.DataFrame(metrics_rows)
        heat_df = plot_df.set_index("model")[["accuracy", "precision", "recall", "f1", "auc"]]
        _plot_heatmap(
            heat_df,
            f"无监督学习破损检测",
            FIG_DIR / "unsup_heatmap.png",
        )
        _plot_points(
            plot_df[["model", "auc"]],
            "model",
            "auc",
            None,
            "Unsupervised AUC",
            FIG_DIR / "unsup_auc.png",
        )

    if roc_payload:
        plt.figure(figsize=(6.5, 5))
        for name, fpr, tpr, auc in roc_payload:
            plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
        plt.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.title("ROC - unsupervised detectors")
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIG_DIR / "unsup_roc.png", dpi=300)
        plt.close()

    return {"dataset": data_info, "metrics": metrics_rows, "failures": failures}


def main() -> None:
    _ensure_dirs()
    summary = {
        "supervised": run_supervised_block(),
        "unsupervised": run_unsupervised_block(),
    }
    (REPORTS_DIR / "paper_figures_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved figures to: {FIG_DIR}")


if __name__ == "__main__":
    main()
