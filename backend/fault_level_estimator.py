

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Callable, Any

import numpy as np
import joblib
from sklearn.metrics import pairwise_distances


@dataclass
class FaultLevelEstimator:
    """Nearest‑neighbor fault level estimator.

    Parameters
    ----------
    labelled_X : ndarray of shape (n_samples, n_features)
        Feature matrix for the labelled examples.  Each row corresponds
        to one example with a known fault level.
    labels : ndarray of shape (n_samples,)
        Array of fault level labels corresponding to ``labelled_X``.
    scaler : optional
        A fitted scaler with ``transform`` method.  When provided, both
        the labelled and query samples will be passed through the scaler
        before distance computation.  This ensures a consistent feature
        space between training and inference.  If ``None``, no scaling
        is applied.
    metric : str or callable, optional
        Distance metric to use.  If a string, it should be a valid
        scikit‑learn metric name (default is ``'euclidean'``).  If a
        callable, it must take two 1‑D arrays and return a scalar
        distance.
    """

    labelled_X: np.ndarray
    labels: np.ndarray
    scaler: Optional[Any] = None
    metric: str | Callable[[np.ndarray, np.ndarray], float] = 'euclidean'

    def __post_init__(self) -> None:
        if self.labelled_X.shape[0] != self.labels.shape[0]:
            raise ValueError("labelled_X and labels must have the same number of rows")
        # If a scaler is provided, transform the labelled data once up front.
        if self.scaler is not None:
            try:
                self._labelled_scaled = self.scaler.transform(self.labelled_X)
            except Exception as e:
                raise ValueError(f"Failed to apply scaler: {e}") from e
        else:
            self._labelled_scaled = self.labelled_X.astype(float)

    def predict(self, X: np.ndarray, *, metric: Optional[str | Callable[[np.ndarray, np.ndarray], float]] = None) -> np.ndarray:
        """Predict fault levels for each sample in ``X``.

        Parameters
        ----------
        X : ndarray of shape (m_samples, n_features)
            Query samples to classify.  They should be in the same
            feature order as the labelled examples.  If ``scaler`` was
            provided at construction, ``X`` will be transformed before
            distance computation.
        metric : optional
            Override the distance metric for this call.  When ``None``,
            the estimator's default ``metric`` is used.  See
            ``sklearn.metrics.pairwise_distances`` for valid values.

        Returns
        -------
        ndarray of shape (m_samples,)
            Predicted fault level for each input sample.
        """
        if metric is None:
            metric = self.metric
        # Scale the query samples if necessary
        if self.scaler is not None:
            try:
                X_scaled = self.scaler.transform(X)
            except Exception as e:
                raise ValueError(f"Failed to apply scaler to input X: {e}") from e
        else:
            X_scaled = X.astype(float)
        # Compute pairwise distances between each query and all labelled examples
        # ``pairwise_distances`` supports both string and callable metrics
        dists = pairwise_distances(X_scaled, self._labelled_scaled, metric=metric)
        # Index of the nearest labelled example for each query
        nearest_idx = np.argmin(dists, axis=1)
        # Map to corresponding labels
        return self.labels[nearest_idx]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Persist the estimator to disk.

        The saved object includes the raw labelled samples, labels, and
        (if available) the scaler.  When reloaded, the estimator will
        reconstruct its internal state and be ready for prediction.

        Parameters
        ----------
        path : str
            Destination file path.  The parent directory should exist.
        """
        data: dict[str, Any] = {
            "labelled_X": self.labelled_X,
            "labels": self.labels,
            "scaler": self.scaler,
            "metric": self.metric,
        }
        joblib.dump(data, path)

    @classmethod
    def load(cls, path: str) -> "FaultLevelEstimator":
        """Load a previously saved estimator.

        Parameters
        ----------
        path : str
            Path to the file produced by ``save``.

        Returns
        -------
        FaultLevelEstimator
            Restored estimator instance.
        """
        data = joblib.load(path)
        return cls(
            labelled_X=data["labelled_X"],
            labels=data["labels"],
            scaler=data.get("scaler", None),
            metric=data.get("metric", 'euclidean'),
        )