from __future__ import annotations

"""Utility helpers for storing models and their JSON registries.

This module centralizes the logic for creating model directories under the
project's ``models_saved`` folder and for reading/writing simple JSON-based
registries.  Several model manager implementations previously carried their own
boilerplate for these tasks; consolidating them here keeps the managers
consistent and easier to maintain."""

from pathlib import Path
import json
from typing import Any, Dict, List


def ensure_root(kind: str) -> Path:
    """Return the directory for a given model *kind* and ensure it exists."""
    root = Path(__file__).resolve().parents[1] / "models_saved" / kind
    root.mkdir(parents=True, exist_ok=True)
    return root


class JsonRegistry:
    """Small helper around a ``registry.json`` file.

    The class reloads from disk before each refresh so multiple manager
    instances can cooperate safely.
    """

    def __init__(self, root: Path, filename: str = "registry.json") -> None:
        self.file = root / filename
        self.data: Dict[str, Dict[str, Any]] = {}
        self.reload()

    # ------------------------------------------------------------------
    def reload(self) -> None:
        if self.file.exists():
            try:
                self.data = json.loads(self.file.read_text(encoding="utf-8"))
            except Exception:
                self.data = {}
        else:
            self.data = {}

    # ------------------------------------------------------------------
    def write(self) -> None:
        self.file.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------
    def refresh(self) -> List[Dict[str, Any]]:
        """Return existing items and drop records with missing files."""
        self.reload()
        changed = False
        items: List[Dict[str, Any]] = []
        for mid, meta in list(self.data.items()):
            if Path(meta.get("path", "")).exists():
                items.append(meta)
            else:
                changed = True
                self.data.pop(mid, None)
        if changed:
            self.write()
        return items

    # ------------------------------------------------------------------
    def get(self, key: str, default: Any = None) -> Any:
        """Return a registry record by *key*.

        The method reloads the underlying JSON file before every access so
        that concurrent manager instances see up-to-date information.
        """
        self.reload()
        return self.data.get(key, default)

    # ------------------------------------------------------------------
    def __contains__(self, key: str) -> bool:  # pragma: no cover - trivial
        self.reload()
        return key in self.data

    # ------------------------------------------------------------------
    def __getitem__(self, key: str) -> Dict[str, Any]:  # pragma: no cover
        self.reload()
        return self.data[key]

