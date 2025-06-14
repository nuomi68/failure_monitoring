from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def load_dataframe(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path, engine="openpyxl")
    else:
        df = pd.read_csv(path)
    if "TIME" in df.columns:
        df["TIME"] = pd.to_datetime(df["TIME"], errors="coerce")
        df = df.set_index("TIME")
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="any")
    return df


def generate_noisy_test(
    X: np.ndarray,
    small_sigma: float,
    large_sigma: float,
    n_per_class: int,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    normal = X[:n_per_class] + rng.normal(scale=small_sigma, size=(n_per_class, X.shape[1]))
    abnormal = X[:n_per_class] + rng.normal(scale=large_sigma, size=(n_per_class, X.shape[1]))
    X_test = np.vstack([normal, abnormal])
    y_test = np.array([0] * n_per_class + [1] * n_per_class)
    return X_test, y_test