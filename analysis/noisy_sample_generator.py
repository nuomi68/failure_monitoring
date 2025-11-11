from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# -------------------
# Paths
# -------------------
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"
SOURCE_FILE = "20230510-20240924.xlsx"
NON_FEATURE_COLUMNS = {"TIME", "TIMESTAMP", "source_file", "label"}


# -------------------
# Noise configuration
# -------------------
@dataclass
class NoiseConfig:
    fs: int = 1
    window_sec: int = 30
    n_windows: int = 12
    seed: int = 42
    gaussian_sigma: float = 0.8
    drift_max_std: float = 2.0


# -------------------
# Data loading helpers
# -------------------
def parse_time(series: pd.Series) -> pd.Series:
    raw = series.astype(str).str.replace(r"\s+", "", regex=True)
    digits = raw.str.replace(r"\D", "", regex=True)
    dt = pd.to_datetime(digits, format="%Y%m%d%H%M", errors="coerce")
    dt = dt.fillna(pd.to_datetime(digits.str[:8], format="%Y%m%d", errors="coerce"))
    return dt


def load_single() -> pd.DataFrame:
    path = DATA_DIR / SOURCE_FILE
    if not path.exists():
        raise FileNotFoundError(path)
    xl = pd.ExcelFile(path)
    df = xl.parse(xl.sheet_names[0])
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]

    if "TIMESTAMP" not in df.columns:
        if "TIME" in df.columns:
            df["TIMESTAMP"] = parse_time(df["TIME"])
        else:
            df["TIMESTAMP"] = pd.RangeIndex(len(df), name="t")
    df["source_file"] = SOURCE_FILE

    num_cols = [c for c in df.columns if c not in {"TIME", "TIMESTAMP", "source_file"}]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values("TIMESTAMP").reset_index(drop=True)
    df[num_cols] = df[num_cols].interpolate(limit_direction="both")
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    return df


# -------------------
# Noise injection
# -------------------
def _random_non_overlapping_windows(
    n: int, win: int, N: int, rng: np.random.Generator
) -> List[Tuple[int, int]]:
    if N < win:
        return [(0, N - 1)]
    anchors = np.linspace(0, N - win, num=n, dtype=int)
    used = np.zeros(N, dtype=bool)
    spans: List[Tuple[int, int]] = []
    for a in anchors:
        s = int(min(max(0, a + rng.integers(-win // 4, win // 4 + 1)), N - win))
        e = s + win - 1
        if not used[s : e + 1].any():
            spans.append((s, e))
            used[s : e + 1] = True
    tries = 0
    while len(spans) < n and tries < 5000:
        s = int(rng.integers(0, N - win + 1))
        e = s + win - 1
        if not used[s : e + 1].any():
            spans.append((s, e))
            used[s : e + 1] = True
        tries += 1
    return spans


def _inject_gaussian(
    seg: pd.DataFrame, cols: List[str], sigma: float, rng: np.random.Generator
):
    out = seg.copy()
    std = seg[cols].std(ddof=0).replace(0, 1.0)
    noise = rng.normal(0, 1, size=out[cols].shape) * std.values * sigma
    out[cols] = out[cols] + noise
    labels = np.ones(len(seg), dtype=bool)
    return out, labels


def _inject_drift(
    seg: pd.DataFrame, cols: List[str], max_std: float, rng: np.random.Generator
):
    out = seg.copy()
    L = len(seg)
    t = np.linspace(0, 1, L)
    std = seg[cols].std(ddof=0).replace(0, 1.0).values
    drift_end = (rng.uniform(-max_std, max_std, size=len(cols))) * std
    drift = np.outer(t, drift_end)
    out[cols] = out[cols] + drift
    labels = np.ones(L, dtype=bool)
    return out, labels


def make_noisy_dataset(
    df: pd.DataFrame, cfg: Optional[NoiseConfig] = None
) -> Dict[str, Dict]:
    cfg = cfg or NoiseConfig()
    rng = np.random.default_rng(cfg.seed)

    feat_cols = [
        c
        for c in df.columns
        if c not in NON_FEATURE_COLUMNS and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not feat_cols:
        raise RuntimeError("没有可用的数值列。")

    N = len(df)
    win = max(5, cfg.window_sec * cfg.fs)
    spans = _random_non_overlapping_windows(cfg.n_windows, win, N, rng)

    results: Dict[str, Dict] = {}
    for noise_type in ["gaussian", "drift"]:
        noisy = df.copy()
        noisy[feat_cols] = noisy[feat_cols].astype(float)
        labels = np.zeros(N, dtype=int)

        for (s, e) in spans:
            seg = noisy.iloc[s : e + 1][feat_cols]
            if noise_type == "gaussian":
                seg2, mask = _inject_gaussian(seg, feat_cols, cfg.gaussian_sigma, rng)
            elif noise_type == "drift":
                seg2, mask = _inject_drift(seg, feat_cols, cfg.drift_max_std, rng)
            else:
                raise ValueError(noise_type)

            noisy.loc[noisy.index[s : e + 1], feat_cols] = seg2.values
            labels[s : e + 1] = np.maximum(labels[s : e + 1], mask.astype(int))

        results[noise_type] = {
            "data": noisy,
            "labels": pd.Series(labels, index=df.index, name="label", dtype=int),
            "spans": spans,
        }
    return results


# -------------------
# Evaluation helper
# -------------------
def train_eval_temporal(
    df: pd.DataFrame, labels: pd.Series, feature_cols: List[str]
) -> Dict[str, float]:
    X = df[feature_cols].to_numpy(dtype=float)
    y = labels.to_numpy(dtype=int)

    split = int(len(X) * 0.75)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=400, class_weight="balanced", random_state=42
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    pos = report.get("1", {})
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
    except Exception:
        auc = 0.0

    return {
        "accuracy": float(report.get("accuracy", 0.0)),
        "precision_1": float(pos.get("precision", 0.0)),
        "recall_1": float(pos.get("recall", 0.0)),
        "f1_macro": float(report.get("macro avg", {}).get("f1-score", 0.0)),
        "roc_auc": float(auc),
    }


# -------------------
# Export helpers
# -------------------
def export_noise_reports(
    datasets: Dict[str, Dict], metrics: Dict[str, Dict[str, float]], cfg: NoiseConfig
) -> None:
    out_dir = REPORTS_DIR / "noisy_samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "source_file": SOURCE_FILE,
        "noise_config": asdict(cfg),
        "metrics": metrics,
        "datasets": {},
    }
    for name, payload in datasets.items():
        df_out = payload["data"].copy()
        df_out["label"] = payload["labels"].astype(int)
        csv_path = out_dir / f"{name}.csv"
        df_out.to_csv(csv_path, index=False, encoding="utf-8-sig")
        summary["datasets"][name] = {
            "rows": int(len(df_out)),
            "label_positive_fraction": float(payload["labels"].mean()),
            "num_windows": int(len(payload["spans"])),
            "csv_path": str(csv_path.relative_to(REPORTS_DIR)),
            "spans": [[int(s), int(e)] for s, e in payload["spans"]],
        }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


# -------------------
# Entrypoint
# -------------------
def main() -> None:
    df = load_single()
    cfg = NoiseConfig()

    datasets = make_noisy_dataset(df, cfg)
    feat_cols = [
        c
        for c in df.columns
        if c not in NON_FEATURE_COLUMNS and pd.api.types.is_numeric_dtype(df[c])
    ]

    metrics: Dict[str, Dict[str, float]] = {}
    for name, payload in datasets.items():
        metrics[name] = train_eval_temporal(
            payload["data"], payload["labels"], feat_cols
        )

    export_noise_reports(datasets, metrics, cfg)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "Noisy Sample Evaluation (gaussian & drift only)",
        "=" * 44,
        f"Source file: {SOURCE_FILE}",
        "",
    ]
    for k, v in metrics.items():
        m = ", ".join(f"{kk}={vv:.4f}" for kk, vv in v.items())
        lines.append(f"{k}: {m}")
    (REPORTS_DIR / "noisy_sample_metrics.txt").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print("Done. Reports under:", REPORTS_DIR)


if __name__ == "__main__":
    main()
