from pathlib import Path
import json
import time
import uuid

ROOT = Path.home() / ".ml_app" / "models"
REG = ROOT / "models.json"
ROOT.mkdir(parents=True, exist_ok=True)
if not REG.exists():
    REG.write_text("[]", encoding="utf8")


def _load():
    return json.loads(REG.read_text(encoding="utf8"))


def _dump(js):
    REG.write_text(json.dumps(js, ensure_ascii=False, indent=2), encoding="utf8")


def register(path: Path, meta: dict):
    data = _load()
    data.append({
        "id": uuid.uuid4().hex,
        "name": path.stem,
        "path": str(path),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "meta": meta,
    })
    _dump(data)


def list_all() -> list[dict]:
    return _load()
