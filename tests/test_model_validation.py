import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

sys.path.append(str(Path(__file__).resolve().parents[1]))

from backend.model_validation import (
    print_validation_report,
    run_supervised_validation,
    run_unsupervised_validation,
)


def _regression_frame(n_samples: int = 260, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(n_samples, 4))
    coefs = np.array([0.8, -1.5, 2.2, 0.5])
    noise = rng.normal(scale=0.01, size=n_samples)
    target = features @ coefs + noise

    df = pd.DataFrame(features, columns=["KR-85M", "KR-87", "KR-88", "I-135"])
    df.insert(0, "TIME", pd.date_range("2024-01-01", periods=n_samples, freq="h"))
    df["value"] = target
    return df


def _classification_frame(n_samples: int = 320, seed: int = 42) -> pd.DataFrame:
    X, y = make_classification(
        n_samples=n_samples,
        n_features=5,
        n_informative=4,
        n_redundant=0,
        n_clusters_per_class=1,
        class_sep=1.5,
        random_state=seed,
    )
    labels = np.where(y == 1, "fault", "normal")
    cols = ["KR-85M", "KR-87", "KR-88", "I-135", "TGAS"]
    df = pd.DataFrame(X, columns=cols)
    df.insert(0, "TIME", pd.date_range("2024-01-01", periods=n_samples, freq="h"))
    df["label"] = labels
    return df


def _unsupervised_frame(n_normal: int = 360, n_anomaly: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    normal = rng.normal(loc=0.0, scale=1.0, size=(n_normal, 3))
    anomaly = rng.uniform(low=6.0, high=10.0, size=(n_anomaly, 3))
    X = np.vstack([normal, anomaly])
    labels = np.concatenate([np.zeros(n_normal, dtype=int), np.ones(n_anomaly, dtype=int)])
    order = rng.permutation(len(X))
    X = X[order]
    labels = labels[order]

    cols = ["KR-85M", "KR-87", "TGAS"]
    df = pd.DataFrame(X, columns=cols)
    df.insert(0, "TIME", pd.date_range("2024-02-01", periods=len(df), freq="h"))
    df["anomaly"] = labels
    return df


def _write_dataset(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
    else:
        df.to_excel(path, index=False)
    return path


def _assert_regression_metrics(result, min_r2: float) -> None:
    print_validation_report(result)
    assert result.task == "supervised_reg"
    assert set(result.metrics) >= {"mae", "mse", "rmse", "mape", "r2"}
    assert result.metrics["r2"] >= min_r2
    assert result.n_test > 0 and result.n_train > 0


def _assert_classification_metrics(result, min_accuracy: float) -> None:
    print_validation_report(result)
    assert result.task == "supervised_clf"
    for key in ("accuracy", "precision", "recall", "f1"):
        assert key in result.metrics
        assert 0.0 <= result.metrics[key] <= 1.0
    assert result.metrics["accuracy"] >= min_accuracy
    assert result.confusion_matrix is not None
    assert np.asarray(result.confusion_matrix).shape[0] >= 2
    assert set(result.labels or []) == {"fault", "normal"}


def _assert_unsupervised_metrics(
    result,
    min_recall: float,
    max_ratio: float = 0.5,
    min_auc: float = 0.8,
) -> None:
    print_validation_report(result)
    assert result.task == "unsupervised"
    required = {"score_mean", "score_std", "tau", "ratio_over_tau"}
    assert required.issubset(result.metrics)
    assert 0.0 <= result.metrics["ratio_over_tau"] <= max_ratio
    if "recall" in result.metrics:
        assert result.metrics["recall"] >= min_recall
    if "roc_auc" in result.metrics:
        assert result.metrics["roc_auc"] >= min_auc
    if result.confusion_matrix is not None:
        assert np.asarray(result.confusion_matrix).shape == (2, 2)
    assert result.extra and "tau" in result.extra


def test_rf_regression_validation(tmp_path):
    dataset = _write_dataset(_regression_frame(seed=0), tmp_path / "rf_reg.csv")
    result = run_supervised_validation(
        alg="rf_reg",
        data_path=dataset,
        target_column="value",
        test_size=0.25,
        random_state=7,
        scaler="standard",
        params={"n_estimators": 200, "random_state": 7},
    )
    _assert_regression_metrics(result, min_r2=0.88)


def test_knn_regression_validation(tmp_path):
    dataset = _write_dataset(_regression_frame(seed=2024), tmp_path / "knn_reg.csv")
    result = run_supervised_validation(
        alg="knn_reg",
        data_path=dataset,
        target_column="value",
        test_size=0.2,
        random_state=0,
        scaler="standard",
        params={"n_neighbors": 5, "weights": "distance"},
    )
    _assert_regression_metrics(result, min_r2=0.85)


def test_rf_classifier_validation(tmp_path):
    dataset = _write_dataset(_classification_frame(seed=21), tmp_path / "rf_clf.csv")
    result = run_supervised_validation(
        alg="rf_clf",
        data_path=dataset,
        target_column="label",
        test_size=0.3,
        random_state=99,
        scaler="standard",
        params={"n_estimators": 150, "random_state": 99},
        stratify=True,
    )
    _assert_classification_metrics(result, min_accuracy=0.9)


def test_knn_classifier_validation(tmp_path):
    dataset = _write_dataset(_classification_frame(seed=17), tmp_path / "knn_clf.csv")
    result = run_supervised_validation(
        alg="knn_clf",
        data_path=dataset,
        target_column="label",
        test_size=0.25,
        random_state=13,
        scaler="standard",
        params={"n_neighbors": 7},
        stratify=True,
    )
    _assert_classification_metrics(result, min_accuracy=0.85)


def test_unsupervised_knn_validation(tmp_path):
    dataset = _write_dataset(_unsupervised_frame(seed=0), tmp_path / "unsup_knn.csv")
    result = run_unsupervised_validation(
        alg="knn",
        data_path=dataset,
        label_column="anomaly",
        scaler=None,
        params={"n_neighbors": 10},
    )
    _assert_unsupervised_metrics(result, min_recall=0.4, min_auc=0.9)


def test_unsupervised_iforest_validation(tmp_path):
    dataset = _write_dataset(_unsupervised_frame(seed=11), tmp_path / "unsup_iforest.csv")
    result = run_unsupervised_validation(
        alg="iforest",
        data_path=dataset,
        label_column="anomaly",
        scaler="standard",
        params={"n_estimators": 128, "contamination": 0.1, "random_state": 42},
    )
    _assert_unsupervised_metrics(result, min_recall=0.3, min_auc=0.9)


def test_unsupervised_autoencoder_validation(tmp_path):
    pytest.importorskip("torch", reason="Autoencoder test requires PyTorch")
    dataset = _write_dataset(_unsupervised_frame(seed=5), tmp_path / "unsup_autoencoder.csv")
    result = run_unsupervised_validation(
        alg="autoencoder",
        data_path=dataset,
        label_column="anomaly",
        scaler="standard",
        params={
            "hidden": [32, 16],
            "latent_dim": 8,
            "epochs": 10,
            "batch_size": 64,
            "lr": 1e-3,
        },
    )
    _assert_unsupervised_metrics(result, min_recall=0.3, max_ratio=0.7, min_auc=0.85)
