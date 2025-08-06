"""Simple front/back interface for time series forecasting.

This module provides two things:

* ``handle_uploaded_file`` – copy an uploaded dataset to the local
  ``data`` directory and trigger model training on it.
* ``ModelManager`` – a lightweight singleton that stores the latest
  trained models and associated scalers so the GUI can query training
  status and prediction results.
"""

from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any, List

from .controller import run_all_models, predict_next_steps


def handle_uploaded_file(file_path: str, time_format: str | None = None) -> dict[str, Any]:
    """Save ``file_path`` and train models on the new dataset.

    Parameters
    ----------
    file_path:
        Path to the uploaded Excel/CSV file.
    time_format:
        Optional ``strftime`` format string describing the ``TIME`` column
        format in ``file_path``.

    Returns
    -------
    dict
        Dictionary with trained models, scaled data, feature names,
        fitted scaler and initial prediction results.
    """
    upload_dir = Path(__file__).resolve().parents[1] / "data"
    upload_dir.mkdir(exist_ok=True)
    saved_path = upload_dir / Path(file_path).name
    shutil.copy(file_path, saved_path)

    models, data_scaled, feature_names, scaler, preds = run_all_models(
        str(saved_path), time_format
    )
    return {
        "models": models,
        "data": data_scaled,
        "feature_names": feature_names,
        "scaler": scaler,
        "predictions": preds,
    }


class ModelManager:
    """Singleton used by the GUI to keep track of model state."""

    _instance: "ModelManager" | None = None

    def __new__(cls) -> "ModelManager":  # pragma: no cover - simple singleton
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.models = {}
            cls._instance.data = None
            cls._instance.feature_names: List[str] | None = None
            cls._instance.scaler = None
            cls._instance.status = "idle"
            cls._instance.last_predictions = []
        return cls._instance

    # ------------------------------------------------------------------
    def train(self, file_path: str, time_format: str | None = None) -> None:
        """Train models using data from ``file_path``.

        ``time_format`` is forwarded to :func:`handle_uploaded_file` to
        control how the time column is parsed.
        """
        self.status = "training"
        result = handle_uploaded_file(file_path, time_format)
        self.models = result["models"]
        self.data = result["data"]
        self.feature_names = result["feature_names"]
        self.scaler = result["scaler"]
        self.last_predictions = result["predictions"]
        self.status = "completed"

    # ------------------------------------------------------------------
    def predict(self, steps: int = 5):
        """Run multi‑step forecasting with already trained models."""
        if not self.models:
            raise RuntimeError("模型尚未训练")
        self.last_predictions = predict_next_steps(
            self.models, self.data, self.feature_names, self.scaler, steps
        )
        return self.last_predictions

