from pathlib import Path
import json
import time
import uuid

ROOT =   Path("models_saved")
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
    data = _load()
    updated = []
    changed = False

    for item in data:
        model_path = Path(item["path"])
        if model_path.exists():
            updated.append(item)
        else:
            changed = True  # 文件不存在，需要更新 json

    if changed:
        _dump(updated)

    return updated
