from __future__ import annotations

import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder

from .fault_level_estimator import FaultLevelEstimator
from .model_store import ensure_root, JsonRegistry


class FaultModelManager:
    """Manage saving/loading of :class:`FaultLevelEstimator` models.

    Models are stored under ``models_saved/fault_level`` with a registry file
    describing basic metadata (method, feature names, scaler, label column).
    The interface mimics a subset of ``timeseries_interface.ModelManager`` so
    that the same ``ModelManagerDialog`` can be reused."""

    def __init__(self) -> None:
        self.models_dir = ensure_root("fault_level")
        self.registry = JsonRegistry(self.models_dir)

    # ------------------------------------------------------------------
    def _write_registry(self) -> None:
        self.registry.write()

    # ------------------------------------------------------------------
    def refresh_models(self) -> List[Dict[str, Any]]:
        """Return metadata of all existing models."""
        return self.registry.refresh()

    # ------------------------------------------------------------------
    def save_model(
        self,
        estimator: FaultLevelEstimator,
        name: str,
        *,
        label_col: str | None = None,
        df: pd.DataFrame | None = None,
    ) -> str:
        """Persist estimator along with its training data.

        Parameters
        ----------
        estimator: FaultLevelEstimator
            Fitted estimator to persist.
        name: str
            Friendly name for registry.
        label_col: str | None
            Column name of labels in ``df``.
        df: pd.DataFrame | None
            Training dataframe to store for later reloading.
        """

        model_id = uuid.uuid4().hex
        path = self.models_dir / f"{model_id}.joblib"
        estimator.save(str(path))

        data_path = ""
        if df is not None:
            data_path = str(self.models_dir / f"{model_id}.csv")
            try:
                df.to_csv(data_path, index=False)
            except Exception:
                data_path = ""

        meta = {
            "model_id": model_id,
            "name": name,
            "path": str(path),
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_type": estimator.method,
            # 兼容 ModelManagerDialog，用 dataset_id 显示训练数据文件名
            "dataset_id": Path(data_path).name if data_path else "",
            "data_path": data_path,
            "label_col": label_col or "",
            "metrics": {
                "features": estimator.feature_names,
                "scaler": estimator.scaler_spec,
            },
        }
        self.registry.data[model_id] = meta
        self._write_registry()
        return model_id

    # ------------------------------------------------------------------
    def load_model(self, model_id: str) -> Tuple[FaultLevelEstimator, pd.DataFrame | None, Dict[str, Any]]:
        """Load estimator and associated dataset."""

        # Refresh registry to include models saved by other processes
        self.refresh_models()

        meta = self.registry.data.get(model_id)
        if not meta:
            raise KeyError(f"model_id '{model_id}' not found")

        est = FaultLevelEstimator.load(meta["path"])

        df: pd.DataFrame | None = None
        data_path = meta.get("data_path")
        if data_path:
            p = Path(data_path)
            if p.exists():
                try:
                    df = pd.read_csv(p)
                except Exception:
                    df = None

        return est, df, meta

    # ------------------------------------------------------------------
    def rename_model(self, model_id: str, new_name: str) -> bool:
        meta = self.registry.data.get(model_id)
        if not meta:
            return False
        meta["name"] = new_name
        self._write_registry()
        return True

    # ------------------------------------------------------------------
    def predict_many(self, estimators: List[FaultLevelEstimator], X: pd.DataFrame) -> np.ndarray:
        """Aggregate predictions from multiple estimators.

        Classification models vote, regression models average.
        """
        if not estimators:
            return np.array([])
        preds: list[np.ndarray] = []
        enc: LabelEncoder | None = None
        numeric = True
        for est in estimators:
            feats = est.feature_names or []
            arr = np.stack([
                X.get(c, pd.Series([np.nan] * len(X))).to_numpy()
                for c in feats
            ], axis=1)
            y = np.asarray(est.predict(arr, decode=False))
            preds.append(y)
            if enc is None and est.label_encoder is not None:
                enc = est.label_encoder
            if numeric and not np.issubdtype(y.dtype, np.number):
                numeric = False
        stack = np.vstack(preds)
        if numeric and np.array_equal(stack, stack.astype(int)):
            agg = pd.DataFrame(stack.T).mode(axis=1)[0].to_numpy()
        elif numeric:
            agg = np.nanmean(stack.astype(float), axis=0)
        else:
            agg = pd.DataFrame(stack.T).mode(axis=1)[0].to_numpy()
        if enc is not None:
            try:
                agg = enc.inverse_transform(np.asarray(agg).astype(int))
            except Exception:
                pass
        return agg

    # ------------------------------------------------------------------
    def delete_model(self, model_id: str) -> bool:
        meta = self.registry.data.pop(model_id, None)
        if not meta:
            return False
        try:
            Path(meta.get("path", "")).unlink(missing_ok=True)
            Path(meta.get("data_path", "")).unlink(missing_ok=True)
        except Exception:
            pass
        self._write_registry()
        return True

    # ------------------------------------------------------------------
    def export_model(self, model_id: str, dest: str) -> bool:
        meta = self.registry.data.get(model_id)
        if not meta:
            return False
        try:
            shutil.copyfile(meta["path"], dest)
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    def import_model(self, src: str) -> bool:
        try:
            est = FaultLevelEstimator.load(src)
        except Exception:
            return False
        name = Path(src).stem
        mid = uuid.uuid4().hex
        dest = self.models_dir / f"{mid}.joblib"
        try:
            shutil.copyfile(src, dest)
        except Exception:
            return False
        meta = {
            "model_id": mid,
            "name": name,
            "path": str(dest),
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_type": est.method,
            "dataset_id": "",
            "data_path": "",
            "label_col": "",
            "metrics": {
                "features": est.feature_names,
                "scaler": est.scaler_spec,
            },
        }
        self.registry.data[mid] = meta
        self._write_registry()
        return True
