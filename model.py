from typing import Tuple
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import IsolationForest


def train_knn(X: np.ndarray, k: int = 5, quantile: float | None = 0.95, metric: str = "euclidean") -> Tuple[NearestNeighbors, float]:
    knn = NearestNeighbors(n_neighbors=k, metric=metric)
    knn.fit(X)
    dists, _ = knn.kneighbors(X)
    dk = dists[:, -1]
    if quantile is not None:
        tau = np.quantile(dk, quantile)
    else:
        tau = float(dk.max())
    return knn, float(tau)


def train_iforest(
    X: np.ndarray,
    n_estimators: int = 100,
    contamination: float = 0.01,
    random_state: int = 42,
) -> Tuple[IsolationForest, float]:
    clf = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
    )
    clf.fit(X)
    scores = -clf.decision_function(X)
    tau = np.quantile(scores, 1 - contamination)
    return clf, float(tau)
