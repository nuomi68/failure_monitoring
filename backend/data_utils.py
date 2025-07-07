import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

TIME_COL = "TIME"


def load_dataset(path: str):
    """Load and preprocess the dataset used by all models."""
    df = pd.read_excel(path)
    df.columns = df.columns.str.strip()
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], format="%Y年%m月%d日%H%M")
    df = df.sort_values(TIME_COL).reset_index(drop=True)

    features = (
        df.drop(columns=[TIME_COL, "值", "XE-133", "CS-137"], errors="ignore")
        .apply(pd.to_numeric, errors="coerce")
        .fillna(method="ffill")
        .fillna(0)
    )

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features.astype(np.float32))
    return features_scaled, features.columns.tolist(), scaler


def build_windows(data: np.ndarray, look_back: int):
    """Create sliding windows for sequence models."""
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i : i + look_back])
        y.append(data[i + look_back])
    return np.asarray(X, np.float32), np.asarray(y, np.float32)
