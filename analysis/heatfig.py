from __future__ import annotations

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import font_manager, rcParams


def pick_cn_font():
    candidates = [
        "Microsoft YaHei",
        "Microsoft YaHei UI",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "PingFang SC",
        "WenQuanYi Zen Hei",
        "Arial Unicode MS",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = next((n for n in candidates if n in available), None)
    if chosen:
        rcParams["font.family"] = "sans-serif"
        rcParams["font.sans-serif"] = [chosen]
    rcParams["axes.unicode_minus"] = False
    print("[INFO] chosen font =", chosen)
    return font_manager.FontProperties(family=chosen) if chosen else None


def load_matrix(base_path: Path) -> pd.DataFrame:
    """
    base_path: without suffix, e.g. ".../clf_heatmap监督学习破损状态判别_matrix"
    Prefer JSON (split), fallback to CSV.
    """
    json_path = base_path.with_suffix(".json")
    csv_path = base_path.with_suffix(".csv")

    if json_path.exists():
        # orient="split"
        return pd.read_json(json_path, orient="split")
    if csv_path.exists():
        return pd.read_csv(csv_path, index_col=0)
    raise FileNotFoundError(f"matrix file not found: {json_path} / {csv_path}")


def plot_heatmap(df: pd.DataFrame, title: str, out_png: Path, fmt: str):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fp = pick_cn_font()

    plt.figure(figsize=(6.8, 3.6))
    ax = sns.heatmap(
        df,
        annot=True,
        fmt=fmt,
        cmap="viridis",
        cbar_kws={"shrink": 0.8},
        annot_kws={"fontproperties": fp} if fp else None,  # 强制 annot 字体
    )

    if fp:
        ax.set_title(title, fontproperties=fp)
        ax.set_xlabel("指标", fontproperties=fp)
        ax.set_ylabel("模型", fontproperties=fp)
        for t in ax.get_xticklabels():
            t.set_fontproperties(fp)
        for t in ax.get_yticklabels():
            t.set_fontproperties(fp)
    else:
        ax.set_title(title)
        ax.set_xlabel("指标")
        ax.set_ylabel("模型")

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print("[OK] saved:", out_png)


def main():
    # 你的目录：E:\failure_monitoring\reports\figures1
    fig_dir = Path(r"E:\failure_monitoring\reports\figures1")

    # 你的文件名（按你打印出来的完整前缀写）
    sup_base = fig_dir / "clf_heatmap监督学习破损状态判别_matrix"
    unsup_base = fig_dir / "unsup_heatmap无监督学习破损检测_matrix"

    sup_df = load_matrix(sup_base)
    unsup_df = load_matrix(unsup_base)

    plot_heatmap(sup_df, "监督学习破损状态判别", fig_dir / "clf_heatmap_redraw.png", fmt=".4f")
    plot_heatmap(unsup_df, "无监督学习破损检测", fig_dir / "unsup_heatmap_redraw.png", fmt=".3f")


if __name__ == "__main__":
    main()
