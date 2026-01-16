"""
Label-tuning experiment on clean vs mixed activity datasets.

Data:
- FILE_CLEAN: all healthy
- FILE_MIXED: partially damaged

Steps:
1) Build pseudo labels on mixed set via robust z-score thresholds.
2) Grid-search label parameters; pick the setting maximizing supervised F1 (RF).
3) Evaluate supervised (RF/XGB/KNN/LGBM*) and unsupervised (IForest/KNN-dist/Autoencoder*).
4) Save metrics and heatmaps under reports/.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
REPORTS_DIR = ROOT_DIR / "reports"
FIG_DIR = REPORTS_DIR / "figures"

FILE_CLEAN = DATA_DIR / "20230510-20240924.xlsx"
FILE_MIXED = DATA_DIR / "20200803-20210504.xlsx"


# ----------------------------- IO helpers -----------------------------
def parse_time(series: pd.Series) -> pd.Series:
    raw = series.astype(str).str.replace(r"\s+", "", regex=True)
    digits = raw.str.replace(r"\D", "", regex=True)
    dt = pd.to_datetime(digits, format="%Y%m%d%H%M", errors="coerce")
    dt = dt.fillna(pd.to_datetime(digits.str[:8], format="%Y%m%d", errors="coerce"))
    return dt


def load_activity(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
    if "TIMESTAMP" not in df.columns and "TIME" in df.columns:
        df["TIMESTAMP"] = parse_time(df["TIME"])
    elif "TIMESTAMP" not in df.columns:
        df["TIMESTAMP"] = pd.RangeIndex(len(df), name="t")
    num_cols = [c for c in df.columns if c not in {"TIME", "TIMESTAMP"}]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values("TIMESTAMP").reset_index(drop=True)
    df[num_cols] = df[num_cols].interpolate(limit_direction="both")
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    return df


def feature_columns(df: pd.DataFrame) -> List[str]:
    return [
        c
        for c in df.columns
        if c not in {"TIME", "TIMESTAMP"} and pd.api.types.is_numeric_dtype(df[c])
    ]


def align_features(clean_df: pd.DataFrame, mixed_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    clean_cols = feature_columns(clean_df)
    mixed_cols = feature_columns(mixed_df)
    cols = [c for c in clean_cols if c in mixed_cols]
    return clean_df[cols].astype(float), mixed_df[cols].astype(float), cols


# ----------------------------- labeling -----------------------------
def label_with_params(
    mixed_feat: pd.DataFrame,
    clean_feat: pd.DataFrame,
    z_thresh: float,
    min_features: int,
    target_ratio: float,
) -> np.ndarray:
    """Pseudo labels using robust MAD score and a target positive ratio."""
    med = clean_feat.median()
    mad = (clean_feat - med).abs().median().replace(0, 1e-6)
    scale = 1.4826 * mad
    z = (mixed_feat - med) / scale
    score = z.abs().median(axis=1) + 0.5 * z.abs().max(axis=1)
    base = (z.abs() >= z_thresh).sum(axis=1) >= min_features
    base_rate = float(base.mean())
    quantile_cut = score.quantile(1 - target_ratio)
    labels = base | (score >= quantile_cut)
    rate = float(labels.mean())
    if rate > 0.6:
        labels = score >= quantile_cut  # drop base if too many positives
        rate = float(labels.mean())
    if rate > 0.6:
        tighter_cut = score.quantile(0.8)
        labels = score >= tighter_cut
    if rate < 0.05:
        looser_cut = score.quantile(0.95)
        labels = score >= looser_cut
    return labels.astype(int).to_numpy()


def grid_search_labels(
    clean_feat: pd.DataFrame,
    mixed_feat: pd.DataFrame,
    cols: List[str],
    grid: Dict[str, List[float]],
) -> Tuple[np.ndarray, Dict[str, float]]:
    best_f1, best_labels, best_params = -1.0, None, {}
    candidates: list[tuple[np.ndarray, Dict[str, float]]] = []
    X_full = pd.concat([clean_feat, mixed_feat], axis=0)
    y_clean = np.zeros(len(clean_feat), dtype=int)
    for z in grid["z_thresh"]:
        for m in grid["min_features"]:
            for r in grid["target_ratio"]:
                y_mixed = label_with_params(mixed_feat, clean_feat, z, m, r)
                rate = float(y_mixed.mean())
                y_all = np.concatenate([y_clean, y_mixed])
                f1 = _quick_f1(X_full.to_numpy(dtype=float), y_all)
                # enforce reasonable positive rate; keep fallback if nothing fits
                params_cur = {"z_thresh": z, "min_features": m, "target_ratio": r, "f1_ref": f1, "pos_rate": rate}
                candidates.append((y_mixed, params_cur))
                if 0.05 <= rate <= 0.6 and f1 > best_f1:
                    best_f1 = f1
                    best_labels = y_mixed
                    best_params = params_cur
    if best_labels is None:
        # pick candidate whose pos_rate is closest to 0.3 (avoid all-positive)
        candidates.sort(key=lambda t: abs(t[1]["pos_rate"] - 0.3))
        if not candidates:
            raise RuntimeError("Label grid search failed.")
        best_labels, best_params = candidates[0]
    return best_labels, best_params


def _quick_f1(X: np.ndarray, y: np.ndarray) -> float:
    # simple RF baseline to rank label quality
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    clf = Pipeline(
        [("scaler", StandardScaler()), ("clf", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))]
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return float(metrics.f1_score(y_test, y_pred))


# ----------------------------- evaluation -----------------------------
def eval_supervised(X: np.ndarray, y: np.ndarray) -> Tuple[List[Dict], Dict[str, str]]:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    specs = {
        "RF": RandomForestClassifier(n_estimators=600, max_depth=None, class_weight="balanced_subsample", random_state=42, n_jobs=-1),
        "XGB": _build_xgb(),
        "KNN": KNeighborsClassifier(n_neighbors=15, weights="distance"),
        "LGBM": _build_lgbm(),
    }
    metrics_rows: List[Dict] = []
    failures: Dict[str, str] = {}
    for name, obj in specs.items():
        if isinstance(obj, tuple):
            model, reason = obj
        else:
            model, reason = obj, None
        if model is None:
            failures[name] = reason or "unavailable"
            continue
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", model)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_prob = _proba(pipe, X_test)
        report = metrics.classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        auc = metrics.roc_auc_score(y_test, y_prob) if y_prob is not None else 0.0
        metrics_rows.append(
            {
                "model": name,
                "accuracy": float(report.get("accuracy", 0.0)),
                "precision": float(report["1"]["precision"]),
                "recall": float(report["1"]["recall"]),
                "f1": float(report["1"]["f1-score"]),
                "auc": float(auc),
            }
        )
    return metrics_rows, failures


def eval_unsupervised(clean_feat: np.ndarray, mixed_feat: np.ndarray, y_true: np.ndarray) -> Tuple[List[Dict], Dict[str, str]]:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(clean_feat)
    X_test = scaler.transform(mixed_feat)
    rows: List[Dict] = []
    failures: Dict[str, str] = {}

    # Isolation Forest
    try:
        det = IsolationForest(n_estimators=400, contamination=0.1, random_state=42, n_jobs=-1)
        det.fit(X_train)
        scores = -det.score_samples(X_test)
        rows.append(_score_unsup("IForest", scores, y_true))
    except Exception as exc:
        failures["IForest"] = f"{type(exc).__name__}: {exc}"

    # KNN distance
    try:
        knn = NearestNeighbors(n_neighbors=10)
        knn.fit(X_train)
        dists, _ = knn.kneighbors(X_test, n_neighbors=min(10, len(X_train)))
        scores = dists.mean(axis=1)
        rows.append(_score_unsup("KNN-dist", scores, y_true))
    except Exception as exc:
        failures["KNN-dist"] = f"{type(exc).__name__}: {exc}"

    # Autoencoder (if torch available)
    try:
        scores = _ae_scores(X_train, X_test)
        if scores is None:
            failures["Autoencoder"] = "PyTorch not installed"
        else:
            rows.append(_score_unsup("Autoencoder", scores, y_true))
    except Exception as exc:
        failures["Autoencoder"] = f"{type(exc).__name__}: {exc}"

    return rows, failures


def _score_unsup(name: str, scores_raw: np.ndarray, y_true: np.ndarray) -> Dict:
    scores = np.asarray(scores_raw, dtype=float)
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    cutoff = np.quantile(scores, 0.9)
    preds = (scores >= cutoff).astype(int)
    acc = metrics.accuracy_score(y_true, preds)
    precision, recall, f1, _ = metrics.precision_recall_fscore_support(y_true, preds, average="binary", zero_division=0)
    auc = metrics.roc_auc_score(y_true, scores) if len(np.unique(y_true)) > 1 else 0.0
    return {
        "model": name,
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(auc),
    }


# ----------------------------- model builders -----------------------------
def _proba(model, X):
    if not hasattr(model, "predict_proba"):
        return None
    try:
        proba = model.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] > 1:
            return proba[:, 1]
        return proba.ravel()
    except Exception:
        return None


def _build_xgb():
    try:
        from xgboost import XGBClassifier
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    return (
        XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.85,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=42,
        ),
        None,
    )


def _build_lgbm():
    try:
        from lightgbm import LGBMClassifier
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    return (
        LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=-1,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective="binary",
            metric="binary_logloss",
            random_state=42,
        ),
        None,
    )


def _ae_scores(X_train: np.ndarray, X_test: np.ndarray, epochs: int = 8) -> np.ndarray | None:
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except Exception:
        return None

    input_dim = X_train.shape[1]

    class AE(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU())
            self.decoder = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, d))

        def forward(self, x):
            return self.decoder(self.encoder(x))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AE(input_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    dl = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32)), batch_size=128, shuffle=True)
    model.train()
    for _ in range(epochs):
        for (batch,) in dl:
            batch = batch.to(device)
            recon = model(batch)
            loss = loss_fn(recon, batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        Xt = torch.tensor(X_test, dtype=torch.float32).to(device)
        recon = model(Xt).cpu().numpy()
    return np.mean((recon - X_test) ** 2, axis=1)


# ----------------------------- plotting -----------------------------
def plot_heatmap(df: pd.DataFrame, title: str, path: Path) -> None:
    plt.figure(figsize=(7, 3.5))
    sns.heatmap(df, annot=True, fmt=".3f", cmap="viridis", cbar_kws={"shrink": 0.8})
    plt.title(title)
    plt.xlabel("Metric")
    plt.ylabel("Model")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


# ----------------------------- main -----------------------------
def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    clean_df = load_activity(FILE_CLEAN)
    mixed_df = load_activity(FILE_MIXED)
    clean_feat, mixed_feat, cols = align_features(clean_df, mixed_df)

    grid = {"z_thresh": [3.0, 3.5, 4.0], "min_features": [3, 4, 5], "target_ratio": [0.1, 0.2, 0.3, 0.4]}
    labels_mixed, params = grid_search_labels(clean_feat, mixed_feat, cols, grid)
    labels_all = np.concatenate([np.zeros(len(clean_feat), dtype=int), labels_mixed])

    # Supervised eval
    X_all = pd.concat([clean_feat, mixed_feat], axis=0).to_numpy(dtype=float)
    sup_metrics, sup_fail = eval_supervised(X_all, labels_all)
    if sup_metrics:
        sup_df = pd.DataFrame(sup_metrics).set_index("model")
        plot_heatmap(sup_df[["accuracy", "precision", "recall", "f1", "auc"]], "Supervised metrics", FIG_DIR / "tuning_supervised_heatmap.png")

    # Unsupervised eval
    unsup_metrics, unsup_fail = eval_unsupervised(clean_feat.to_numpy(dtype=float), mixed_feat.to_numpy(dtype=float), labels_mixed)
    if unsup_metrics:
        unsup_df = pd.DataFrame(unsup_metrics).set_index("model")
        plot_heatmap(unsup_df[["accuracy", "precision", "recall", "f1", "auc"]], "Unsupervised metrics", FIG_DIR / "tuning_unsupervised_heatmap.png")

    summary = {
        "label_params": params,
        "label_positive_rate": float(labels_mixed.mean()),
        "supervised": {"metrics": sup_metrics, "failures": sup_fail},
        "unsupervised": {"metrics": unsup_metrics, "failures": unsup_fail},
    }
    (REPORTS_DIR / "tuning_experiment.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Done. Results saved to reports/ and figures.")


if __name__ == "__main__":
    main()
