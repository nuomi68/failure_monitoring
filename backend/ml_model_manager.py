from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from .ml_interface import ML


class MLModelManager:
    """Manage saving/loading of models from :mod:`backend.ml_interface`.

    Models are stored under ``models_saved/ml`` with a registry file so the
    generic :class:`ModelManagerDialog` can list them just like time-series and
    fault-level models.
    """

    def __init__(self) -> None:
        root = Path(__file__).resolve().parents[1] / "models_saved" / "ml"
        self.models_dir = root
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.models_dir / "registry.json"
        if self.registry_file.exists():
            try:
                self.registry: Dict[str, Dict[str, Any]] = json.loads(
                    self.registry_file.read_text(encoding="utf-8")
                )
            except Exception:
                self.registry = {}
        else:
            self.registry = {}

    # ------------------------------------------------------------------
    def _write_registry(self) -> None:
        self.registry_file.write_text(
            json.dumps(self.registry, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------
    def refresh_models(self) -> List[Dict[str, Any]]:
        """Return metadata of all existing models."""
        # Reload registry from disk to pick up models saved by other manager
        # instances. Without this, pages that keep a long-lived manager would
        # never see freshly saved models.
        if self.registry_file.exists():
            try:
                self.registry = json.loads(
                    self.registry_file.read_text(encoding="utf-8")
                )
            except Exception:
                self.registry = {}
        else:
            self.registry = {}

        changed = False
        items: List[Dict[str, Any]] = []
        for mid, meta in list(self.registry.items()):
            path = Path(meta.get("path", ""))
            if path.exists():
                items.append(meta)
            else:
                changed = True
                self.registry.pop(mid, None)
        if changed:
            self._write_registry()
        return items

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
        self.registry[model_id] = reg
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
            meta = self.registry.get(mid)
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
        meta = self.registry.get(model_id)
        if not meta:
            return False
        meta["name"] = new_name
        self._write_registry()
        return True

    # ------------------------------------------------------------------
    def delete_model(self, model_id: str) -> bool:
        meta = self.registry.pop(model_id, None)
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
        meta = self.registry.get(model_id)
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
        self.registry[mid] = meta
        self._write_registry()
        return True
