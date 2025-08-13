from __future__ import annotations

"""用于存储模型及其 JSON 注册表的实用工具。

该模块集中管理在项目的 ``models_saved`` 文件夹下创建模型目录以及读写
简单 JSON 注册表的逻辑。此前多个模型管理器各自携带这些样板代码；将它们
整合在此可以保持管理器一致并更易维护。"""

from pathlib import Path
import json
from typing import Any, Dict, List


def ensure_root(kind: str) -> Path:
    """返回指定模型 *kind* 的目录并确保其存在。"""
    root = Path(__file__).resolve().parents[1] / "models_saved" / kind
    root.mkdir(parents=True, exist_ok=True)
    return root


class JsonRegistry:
    """围绕 ``registry.json`` 文件的轻量辅助类。

    该类在每次刷新前都会从磁盘重新加载，以便多个管理器实例能够安全协作。
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
        """返回已存在的项目，并删除缺失文件的记录。"""
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
        """根据 *key* 返回注册表记录。

        该方法在每次访问前都会重新加载底层 JSON 文件，以便并发的管理器实例
        能看到最新信息。
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

