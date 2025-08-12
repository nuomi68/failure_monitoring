from __future__ import annotations

import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from .ml_interface import ML
from .model_store import ensure_root, JsonRegistry


class MLModelManager:
    """Manage saving/loading of models from :mod:`backend.ml_interface`.

    Models are stored under ``models_saved/ml`` with a registry file so the
    generic :class:`ModelManagerDialog` can list them just like time-series and
    fault-level models.
    """

    def __init__(self) -> None:
        self.models_dir = ensure_root("ml")
        self.registry = JsonRegistry(self.models_dir)

    # ------------------------------------------------------------------
    def _write_registry(self) -> None:
        self.registry.write()

    # ------------------------------------------------------------------
    def refresh_models(self) -> List[Dict[str, Any]]:
        """Return metadata of all existing models."""
        return self.registry.refresh()

    # ------------------------------------------------------------------
    def save_current(self, name: str) -> str:
        """Persist current ML model managed by :mod:`ml_interface`.

        Parameters
        ----------
        name: str
            Friendly name displayed in manager dialog.
        """
        meta = ML.get_meta()
        if not meta:
            raise RuntimeError("暂无可保存的模型")
        model_id = uuid.uuid4().hex
        path = self.models_dir / f"{model_id}.joblib"
        ML.save(str(path))
        reg = {
            "model_id": model_id,
            "name": name,
            "path": str(path),
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_type": meta.get("model_type", ""),
            "dataset_id": "",
            "metrics": meta.get("metrics", {}),
        }
        self.registry.data[model_id] = reg
        self._write_registry()
        return model_id

    # ------------------------------------------------------------------
    def load_models(self, model_ids: List[str]) -> Dict[str, Any]:
        """Load one or multiple models into :mod:`ml_interface`.

        Returns backend meta after loading.
        """
        self.refresh_models()
        paths: List[str] = []
        for mid in model_ids:
            meta = self.registry.data.get(mid)
            if not meta:
                raise KeyError(f"model_id '{mid}' not found")
            paths.append(meta["path"])
        if not paths:
            raise KeyError("no model ids provided")
        if len(paths) == 1:
            meta = ML.load(paths[0])
        else:
            meta = ML.load_many(paths, method="mean")
        return meta

    # ------------------------------------------------------------------
    def rename_model(self, model_id: str, new_name: str) -> bool:
        meta = self.registry.data.get(model_id)
        if not meta:
            return False
        meta["name"] = new_name
        self._write_registry()
        return True

    # ------------------------------------------------------------------
    def delete_model(self, model_id: str) -> bool:
        meta = self.registry.data.pop(model_id, None)
        if not meta:
            return False
        try:
            Path(meta.get("path", "")).unlink(missing_ok=True)
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
            shutil.copyfile(src, src)  # just to ensure readable
        except Exception:
            return False
        mid = uuid.uuid4().hex
        dest = self.models_dir / f"{mid}.joblib"
        try:
            shutil.copyfile(src, dest)
        except Exception:
            return False
        meta = {
            "model_id": mid,
            "name": Path(src).stem,
            "path": str(dest),
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_type": "",
            "dataset_id": "",
            "metrics": {},
        }
        self.registry.data[mid] = meta
        self._write_registry()
        return True
