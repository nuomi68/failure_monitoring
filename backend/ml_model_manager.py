from __future__ import annotations

import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from .ml_interface import ML
from .model_store import ensure_root, JsonRegistry, generate_model_storage_id


class MLModelManager:
    """Manage saving/loading of models from :mod:`backend.ml_interface`.

    Models are stored under ``models_saved/ml`` with a registry file so the
    generic :class:`ModelManagerDialog` can list them just like time-series and
    fault-level models.
    """

    def __init__(self) -> None:
        self.root = ensure_root("ml")
        self.models_dir = self.root / "models"
        self.registries_dir = self.root / "registries"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.registries_dir.mkdir(parents=True, exist_ok=True)
        self.registry = JsonRegistry(self.root, "registries/models_registry.json")

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
        name = (name or "").strip()
        meta = ML.get_meta()
        if not meta:
            raise RuntimeError("暂无可保存的模型")
        model_id = generate_model_storage_id(meta.get("model_type", ""), name or "model", self.models_dir)
        display_name = name or model_id
        model_dir = self.models_dir / model_id
        model_dir.mkdir(parents=True, exist_ok=True)
        path = model_dir / "model.joblib"
        ML.save(str(path))
        reg = {
            "model_id": model_id,
            "name": display_name,
            "path": str(path.relative_to(self.root)),
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_type": meta.get("model_type", ""),
            "dataset_id": "",
            "metrics": meta.get("metrics", {}),
            "features": list(meta.get("features", []) or []),
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
            paths.append(str(self._resolve_path(meta.get("path", ""))))
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
            model_path = self._resolve_path(meta.get("path", ""))
        except ValueError:
            model_path = None
        try:
            if model_path and model_path.exists():
                if model_path.is_dir():
                    shutil.rmtree(model_path, ignore_errors=True)
                else:
                    model_path.unlink(missing_ok=True)
                    parent = model_path.parent
                    if parent != self.models_dir and self.models_dir in parent.parents:
                        shutil.rmtree(parent, ignore_errors=True)
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
            src = self._resolve_path(meta.get("path", ""))
            shutil.copyfile(src, dest)
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    def import_model(self, src: str) -> bool:
        try:
            shutil.copyfile(src, src)
        except Exception:
            return False
        stem = Path(src).stem
        model_id = generate_model_storage_id("import", stem, self.models_dir)
        model_dir = self.models_dir / model_id
        model_dir.mkdir(parents=True, exist_ok=True)
        dest = model_dir / "model.joblib"
        try:
            shutil.copyfile(src, dest)
        except Exception:
            return False
        meta = {
            "model_id": model_id,
            "name": stem,
            "path": str(dest.relative_to(self.root)),
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_type": "",
            "dataset_id": "",
            "metrics": {},
        }
        self.registry.data[model_id] = meta
        self._write_registry()
        return True

    # ------------------------------------------------------------------
    def _resolve_path(self, path_str: str) -> Path:
        if not path_str:
            raise ValueError("empty path")
        path = Path(path_str)
        if not path.is_absolute():
            path = self.root / path
        return path
