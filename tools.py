import logging
from pathlib import Path
from typing import Tuple, Any, Dict

import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# simple logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def scale_features(X: np.ndarray) -> Tuple[np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

def plot_scores(index, scores, threshold: float, title: str = ""):
    plt.figure(figsize=(12, 4))
    plt.plot(index, scores, label="score")
    plt.axhline(threshold, color="r", linestyle="--", label=f"tau={threshold:.4f}")
    plt.title(title)
    plt.xlabel("index")
    plt.ylabel("score")
    plt.legend()
    plt.tight_layout()
    plt.show()

def save_model(path: Path, model: Any, scaler: StandardScaler, meta: Dict[str, Any]):
    joblib.dump((model, scaler, meta), path)


def load_model(path: Path):
    return joblib.load(path)
