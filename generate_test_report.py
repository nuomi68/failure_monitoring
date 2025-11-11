from __future__ import annotations

import sys
from pathlib import Path

import pytest


def main() -> int:
    try:
        import pytest_html  # noqa: F401
    except ImportError:  # pragma: no cover
        print("pytest-html 插件未安装。请先执行 `pip install pytest-html` 后重试。")
        return 1

    report_path = Path("reports") / "unit_test_report.html"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    args = [
        "tests",
        "--html",
        str(report_path),
        "--self-contained-html",
    ]
    return pytest.main(args)


if __name__ == "__main__":
    sys.exit(main())
