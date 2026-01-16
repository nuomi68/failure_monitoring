"""
Closed-loop clustering + distance-model agreement for fault-level pseudo labeling.

Workflow:
1) KMeans(20) to get cluster ids.
2) Map 20 clusters -> 3 coarse levels (manual or default pattern) to form pseudo labels L1.
3) Train distance-based models on L1 and measure L2 vs L1 agreement via CV.
4) Greedy tweak of the 20->3 mapping to maximize agreement.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
FILE_FAULT = DATA_DIR / "break_level2c.xlsx"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend.models.fault_level_model import (  # noqa: E402
    NCAKNNClassifier,
    NearestCentroidMahalanobis,
    ParzenKDEClassifier,
    RadiusNeighborsClassifierPlus,
    WeightedKNNClassifier,
)
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402

DEFAULT_CLUSTER_TO_LEVEL = np.array(
    [
        0, 1, 2, 0, 1,
        2, 0, 1, 2, 0,
        1, 2, 0, 1, 2,
        0, 1, 2, 0, 1,
    ],
    dtype=int,
)


def _maybe_log(msg: str) -> None:
    print(msg)


def _load_fault_level() -> Tuple[np.ndarray, np.ndarray, list[str], Dict[str, str]]:
    if not FILE_FAULT.exists():
        raise FileNotFoundError(FILE_FAULT)
    df = pd.read_excel(FILE_FAULT)
    df.columns = [str(c).strip() for c in df.columns]
    if "level" not in df.columns:
        raise ValueError("level column missing in break_level2c.xlsx")
    feature_cols = [c for c in df.columns if c != "level"]
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").ffill().fillna(0).to_numpy(dtype=float)
    raw_labels = df["level"].astype(str).to_numpy()
    label_map: Dict[str, str] = {}
    cleaned = []
    for i, lbl in enumerate(pd.unique(raw_labels)):
        alias = f"C{i+1}"
        label_map[lbl] = alias
    for lbl in raw_labels:
        cleaned.append(label_map[str(lbl)])
    return X, np.asarray(cleaned), feature_cols, label_map


def kmeans20_label_samples(
    X: np.ndarray,
    n_clusters: int = 20,
    cluster_to_level: Optional[np.ndarray] = None,
    random_state: int = 42,
    n_init: str = "auto",
    scale: bool = True,
    level_names: Tuple[str, str, str] = ("L1", "L2", "L3"),
):
    """
    Fit KMeans and map each cluster to a coarse level.
    Returns: fitted pipeline, cluster ids, first labels, centers.
    """
    X = np.asarray(X, dtype=float)
    steps = []
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("kmeans", KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)))
    pipe = Pipeline(steps).fit(X)

    kmeans = pipe.named_steps["kmeans"]
    cluster_ids = kmeans.labels_
    centers = kmeans.cluster_centers_

    if cluster_to_level is None:
        raise ValueError("cluster_to_level must be provided (length 20, values in {0,1,2})")

    cluster_to_level = np.asarray(cluster_to_level, dtype=int)
    if cluster_to_level.shape[0] != n_clusters:
        raise ValueError(f"cluster_to_level length should be {n_clusters}, got {cluster_to_level.shape[0]}")
    if len(set(cluster_to_level.tolist())) < 3:
        raise ValueError("cluster_to_level must cover all three classes (0,1,2)")

    L1_num = cluster_to_level[cluster_ids]
    L1 = np.array([level_names[i] for i in L1_num], dtype=object)
    return pipe, cluster_ids, L1, centers


def cv_agreement_with_pseudo_labels(
    X: np.ndarray,
    L1: np.ndarray,
    model_builders: Dict[str, object],
    n_splits: int = 4,
    scale_for_models: bool = True,
    random_state: int = 42,
):
    """Cross-validated agreement between models trained on L1 and the pseudo labels themselves."""
    X = np.asarray(X, dtype=float)
    L1 = np.asarray(L1)
    min_count = np.unique(L1, return_counts=True)[1].min()
    splits = min(n_splits, int(min_count)) if min_count >= 2 else 2
    skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=random_state)

    results = {}
    for name, base_model in model_builders.items():
        fold_acc, fold_bacc, fold_f1 = [], [], []
        for tr, va in skf.split(X, L1):
            model = clone(base_model)
            if scale_for_models:
                clf = Pipeline([("scaler", StandardScaler()), ("clf", model)])
            else:
                clf = model
            clf.fit(X[tr], L1[tr])
            pred = clf.predict(X[va])
            fold_acc.append(metrics.accuracy_score(L1[va], pred))
            fold_bacc.append(metrics.balanced_accuracy_score(L1[va], pred))
            fold_f1.append(metrics.f1_score(L1[va], pred, average="macro"))
        results[name] = {
            "acc(L2 vs L1)": float(np.mean(fold_acc)),
            "bacc(L2 vs L1)": float(np.mean(fold_bacc)),
            "macro_f1(L2 vs L1)": float(np.mean(fold_f1)),
            "folds": splits,
        }
    return results


def greedy_optimize_cluster_mapping(
    X: np.ndarray,
    cluster_ids: np.ndarray,
    cluster_to_level: np.ndarray,
    base_model,
    max_iter: int = 30,
    n_splits: int = 4,
    random_state: int = 42,
):
    """
    Fix cluster assignments and adjust the 20->3 mapping to maximize L2 vs L1 balanced accuracy.
    """
    cluster_ids = np.asarray(cluster_ids, dtype=int)
    mapping = np.asarray(cluster_to_level, dtype=int).copy()
    n_clusters = mapping.shape[0]

    def _score(m: np.ndarray) -> float:
        L1_num = m[cluster_ids]
        L1 = np.array([f"L{i+1}" for i in L1_num], dtype=object)
        res = cv_agreement_with_pseudo_labels(
            X, L1, {"_": base_model}, n_splits=n_splits, scale_for_models=True, random_state=random_state
        )
        return res["_"]["bacc(L2 vs L1)"]

    best = _score(mapping)

    for _ in range(max_iter):
        improved = False
        for c in range(n_clusters):
            cur = mapping[c]
            for new_level in (0, 1, 2):
                if new_level == cur:
                    continue
                trial = mapping.copy()
                trial[c] = new_level
                if len(set(trial.tolist())) < 3:
                    continue
                s = _score(trial)
                if s > best + 1e-6:
                    mapping, best = trial, s
                    improved = True
        if not improved:
            break

    return mapping, best


def _build_models() -> Dict[str, object]:
    return {
        "1NN": KNeighborsClassifier(n_neighbors=1, weights="distance"),
        "W-KNN": WeightedKNNClassifier(n_neighbors="auto", metric="euclidean", scale=False),
        "Radius": RadiusNeighborsClassifierPlus(radius="auto", metric="euclidean", scale=False),
        "Centroid-M": NearestCentroidMahalanobis(use_mahalanobis=True, scale=False),
        "Parzen": ParzenKDEClassifier(bandwidth="auto", scale=False),
        "NCA-KNN": NCAKNNClassifier(n_neighbors="auto", scale=False, random_state=42),
    }


def run_fault_labeling(
    cluster_to_level: Optional[np.ndarray] = None,
    base_model_name: str = "Centroid-M",
    max_iter: int = 30,
    n_splits: int = 4,
    random_state: int = 42,
) -> Dict:
    """
    Full loop: initial mapping -> agreement -> greedy mapping tweak -> agreement after tweak.
    """
    _maybe_log("[fault-label] Loading data")
    X, y_raw, feature_cols, label_map = _load_fault_level()
    models = _build_models()
    if cluster_to_level is None:
        cluster_to_level = DEFAULT_CLUSTER_TO_LEVEL

    _maybe_log("[fault-label] KMeans(20) and first-pass labels (L1)")
    pipe, cluster_ids, L1, centers = kmeans20_label_samples(
        X,
        n_clusters=20,
        cluster_to_level=cluster_to_level,
        random_state=random_state,
        n_init="auto",
    )

    _maybe_log("[fault-label] CV agreement before mapping tweak")
    agreement_before = cv_agreement_with_pseudo_labels(X, L1, models, n_splits=n_splits)

    base_model = models.get(base_model_name)
    if base_model is None:
        raise ValueError(f"Unknown base_model_name: {base_model_name}")

    _maybe_log("[fault-label] Greedy mapping optimization")
    best_mapping, best_bacc = greedy_optimize_cluster_mapping(
        X,
        cluster_ids,
        cluster_to_level,
        base_model=base_model,
        max_iter=max_iter,
        n_splits=n_splits,
        random_state=random_state,
    )
    L1_opt = np.array([f"L{i+1}" for i in best_mapping[cluster_ids]], dtype=object)

    _maybe_log("[fault-label] CV agreement after mapping tweak")
    agreement_after = cv_agreement_with_pseudo_labels(X, L1_opt, models, n_splits=n_splits)

    return {
        "dataset": {"features": feature_cols, "label_map": label_map, "num_samples": int(len(X))},
        "kmeans": {
            "n_clusters": 20,
            "cluster_to_level_before": np.asarray(cluster_to_level, dtype=int).tolist(),
            "cluster_to_level_after": best_mapping.tolist(),
            "best_bacc_base_model": float(best_bacc),
        },
        "agreement_before": agreement_before,
        "agreement_after": agreement_after,
    }


if __name__ == "__main__":
    result = run_fault_labeling()
    print(json.dumps(result, indent=2, ensure_ascii=False))
