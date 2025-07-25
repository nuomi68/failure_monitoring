from sklearn.preprocessing import StandardScaler
import numpy as np
from typing import Tuple, Any, Dict
import matplotlib.pyplot as plt
from pathlib import Path
from joblib import dump, load
import logging

logger = logging.getLogger("failure_monitoring")
if not logger.handlers:
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"
    )

    file_handler = logging.FileHandler("log.txt", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.propagate = False

# ================= 工具函数 =================
def scale_features(X: np.ndarray):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    return Xs, scaler

def save_model(path: Path, model, scaler, meta):
    if meta.get("model_type") == "autoencoder":
        import torch
        torch.save({
            "state_dict": model.state_dict(),
            "scaler": scaler,
            "meta": meta,
        }, path)
    else:
        import joblib
        joblib.dump({"model": model, "scaler": scaler, "meta": meta}, path)