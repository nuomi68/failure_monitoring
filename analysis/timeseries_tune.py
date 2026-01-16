from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

import timeseries_forecast as bench


def _parse_models(raw: str) -> Tuple[str, ...]:
    items = [m.strip() for m in raw.split(",") if m.strip()]
    if not items:
        return ("gru", "tcn", "tsmixer", "timesnet")
    return tuple(items)


def _train_test_split_windows(
    data_scaled: np.ndarray, look_back: int, *, test_ratio: float, seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_all, y_all = bench.build_windows(data_scaled, look_back)
    if len(X_all) < 6:
        raise RuntimeError("Not enough windows to train/test.")

    idx = np.arange(len(X_all))
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)

    test_n = int(round(len(idx) * float(test_ratio)))
    test_n = max(1, min(test_n, len(idx) - 1))

    test_idx = np.sort(idx[:test_n])
    train_idx = idx[test_n:]

    return X_all[train_idx], X_all[test_idx], y_all[train_idx], y_all[test_idx]


def _evaluate_train_test(
    model: str,
    adapters: Dict[str, Dict[str, Any]],
    data_scaled: np.ndarray,
    scaler,
    feature_cols: List[str],
    params: Dict[str, Any],
    *,
    test_ratio: float,
    split_seed: int,
    train_seed: int,
    generate_plots: bool,
) -> Dict[str, Any]:
    """Train on train set, evaluate only on test set (no separate validation set)."""
    if model not in adapters:
        raise KeyError(f"Model '{model}' is not available (dependency missing).")

    params = params.copy()
    look_back = int(params.pop("look_back", bench.DEFAULT_LOOKBACK.get(model, 14)))
    X_train, X_test, y_train, y_test = _train_test_split_windows(
        data_scaled, look_back, test_ratio=test_ratio, seed=split_seed
    )

    bench._set_global_seeds(train_seed)

    start = time.time()
    train_fn = adapters[model]["train"]
    result = train_fn(
        X_train,
        y_train,
        X_train,
        y_train,
        log_callback=None,
        **params,
    )
    duration = time.time() - start

    model_obj = result[0] if isinstance(result, tuple) else result
    predict_fn = adapters[model]["predict"]
    n_features = data_scaled.shape[1]

    preds = [predict_fn(model_obj, seq) for seq in X_test]
    y_pred = np.asarray(preds, dtype=np.float32).reshape(len(preds), n_features)

    test_metrics = bench._regression_metrics(y_test, y_pred, scaler)
    rel_err = bench._relative_mae(y_test, y_pred, scaler, feature_cols)
    rel_vals = np.asarray(list(rel_err.values()), dtype=float) if rel_err else np.asarray([])
    rel_mean = float(np.nanmean(rel_vals)) if rel_vals.size and np.isfinite(rel_vals).any() else float("nan")

    plots: Dict[str, str] = {}
    if generate_plots:
        plots = bench._plot_holdout_error(
            model,
            "test",
            y_test,
            y_pred,
            scaler,
            feature_cols,
            bench.FIGURES_DIR,
            targets=bench.PLOT_FEATURES,
            log_fn=None,
        )

    return {
        "status": "ok",
        "look_back": look_back,
        "params": {"look_back": look_back, **params},
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "metrics": {"test": test_metrics},
        "relative_error_test": rel_err,
        "relative_error_test_mean": rel_mean,
        "test_error_plots": plots,
        "train_time_sec": float(duration),
    }


def _score_result(result: Dict[str, Any], *, objective: str) -> float:
    objective = (objective or "rel_mean").strip().lower()
    if objective == "mae":
        return float(result.get("metrics", {}).get("test", {}).get("mae", float("inf")))
    if objective == "rel_mean":
        return float(result.get("relative_error_test_mean", float("inf")))
    raise ValueError(f"Unknown objective: {objective!r} (use 'rel_mean' or 'mae').")


def _suggest_params(trial, model: str) -> Dict[str, Any]:
    # 公共超参
    params: Dict[str, Any] = {
        "look_back": trial.suggest_categorical("look_back", [12, 14, 16, 18, 24, 28, 32, 40]),
        "lr": trial.suggest_categorical("lr", [1e-3, 7e-4, 5e-4, 3e-4]),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        "epochs": trial.suggest_categorical("epochs", [20, 30, 40, 50]),
    }

    if model == "tsmixer":
        params.update(
            {
                "num_blocks": trial.suggest_categorical("num_blocks", [2, 3, 4, 5]),
                "ff_dim": trial.suggest_categorical("ff_dim", [64, 96, 128, 192, 256]),
                "dropout": trial.suggest_categorical("dropout", [0.0, 0.05, 0.1, 0.2]),
            }
        )
    elif model == "tcn":
        params.update(
            {
                "hid": trial.suggest_categorical("hid", [32, 64, 96, 128]),
                "levels": trial.suggest_categorical("levels", [2, 3, 4]),
                "k": trial.suggest_categorical("k", [2, 3, 5]),
                "drop": trial.suggest_categorical("drop", [0.0, 0.1, 0.2, 0.3]),
            }
        )
    elif model == "gru":
        num_layers = trial.suggest_categorical("num_layers", [1, 2, 3])
        dropout = 0.0 if num_layers == 1 else trial.suggest_categorical("dropout", [0.0, 0.1, 0.2, 0.3])
        params.update(
            {
                "hidden_size": trial.suggest_categorical("hidden_size", [32, 64, 96, 128]),
                "num_layers": num_layers,
                "dropout": dropout,
            }
        )
    elif model == "timesnet":
        params.update(
            {
                "d_model": trial.suggest_categorical("d_model", [16, 24, 32, 48, 64]),
                "num_blocks": trial.suggest_categorical("num_blocks", [2, 3, 4, 5]),
            }
        )
    else:
        raise ValueError(f"Unknown model: {model}")

    return params


def main() -> None:
    parser = argparse.ArgumentParser(description="Bayesian (TPE) per-model tuner (train/test only).")
    parser.add_argument("--models", default="gru,tcn,tsmixer,timesnet", help="Comma-separated model names.")
    parser.add_argument("--trials", type=int, default=30, help="Optuna trials per model (excluding baseline run).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--split-seed", type=int, default=bench.SPLIT_SEED, help="Random seed for train/test split.")
    parser.add_argument("--test-ratio", type=float, default=bench.TEST_RATIO, help="Test split ratio for windows.")
    parser.add_argument("--objective", choices=("rel_mean", "mae"), default="rel_mean", help="Optimization target.")
    parser.add_argument("--out", default=str(bench.REPORTS_DIR / "timeseries_tuning_bayes.json"), help="Output JSON path.")
    args = parser.parse_args()

    try:
        import optuna
    except ImportError:
        raise SystemExit("Missing dependency: optuna. Install with: pip install optuna")

    models = _parse_models(args.models)

    data_scaled, feature_cols, scaler = bench._load_clean_features()
    adapters, failures = bench._load_model_adapters()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {
        "dataset": {
            "source_file": bench.SOURCE_FILE,
            "num_rows": int(data_scaled.shape[0]),
            "num_features": len(feature_cols),
            "features": feature_cols,
        },
        "split": {"mode": "random_windows", "test_ratio": float(args.test_ratio), "seed": int(args.split_seed)},
        "objective": args.objective,
        "mode": "per_model",
        "models": {},
        "failures": failures,
    }

    for model in models:
        if model in failures or model not in adapters:
            summary["models"][model] = {"status": "unavailable", "reason": failures.get(model, "adapter missing")}
            continue

        base_overrides = bench.MODEL_PARAM_OVERRIDES.get(model, {}).copy()

        # 先跑一次 baseline（只用 base_overrides / 默认参数），方便对比
        baseline_info: Dict[str, Any] = {}
        try:
            t0 = time.time()
            baseline_result = _evaluate_train_test(
                model,
                adapters,
                data_scaled,
                scaler,
                feature_cols,
                base_overrides,
                test_ratio=float(args.test_ratio),
                split_seed=int(args.split_seed),
                train_seed=int(args.seed),
                generate_plots=False,
            )
            baseline_elapsed = time.time() - t0
            baseline_score = _score_result(baseline_result, objective=args.objective)
            baseline_info = {
                "score": float(baseline_score),
                "test": baseline_result.get("metrics", {}).get("test", {}),
                "relative_error_test_mean": float(baseline_result.get("relative_error_test_mean", float("nan"))),
                "params": baseline_result.get("params", base_overrides),
                "train_samples": int(baseline_result.get("train_samples", 0)),
                "test_samples": int(baseline_result.get("test_samples", 0)),
                "train_time_sec": float(baseline_elapsed),
            }
        except Exception as exc:
            baseline_info = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}", "params": base_overrides}

        trials_log: List[Dict[str, Any]] = []

        def objective(trial):
            suggested = _suggest_params(trial, model)
            merged = base_overrides.copy()
            merged.update(suggested)

            t0 = time.time()
            try:
                result = _evaluate_train_test(
                    model,
                    adapters,
                    data_scaled,
                    scaler,
                    feature_cols,
                    merged,
                    test_ratio=float(args.test_ratio),
                    split_seed=int(args.split_seed),
                    train_seed=int(args.seed),
                    generate_plots=False,
                )
                elapsed = time.time() - t0
                score = _score_result(result, objective=args.objective)

                trials_log.append(
                    {
                        "trial_number": int(trial.number),
                        "score": float(score),
                        "test": result.get("metrics", {}).get("test", {}) or {},
                        "relative_error_test_mean": float(result.get("relative_error_test_mean", float("nan"))),
                        "params": result.get("params", merged),
                        "train_samples": int(result.get("train_samples", 0)),
                        "test_samples": int(result.get("test_samples", 0)),
                        "train_time_sec": float(elapsed),
                    }
                )
                return float(score)
            except Exception as exc:
                elapsed = time.time() - t0
                # 失败给个大值，让优化器避开，并记录错误
                trials_log.append(
                    {
                        "trial_number": int(trial.number),
                        "score": float("inf"),
                        "error": f"{type(exc).__name__}: {exc}",
                        "params": merged,
                        "train_time_sec": float(elapsed),
                    }
                )
                return float("inf")

        sampler = optuna.samplers.TPESampler(
            seed=args.seed,
            n_startup_trials=max(5, min(10, args.trials)),  # 前面先随机探索几次
            multivariate=True,
        )
        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(objective, n_trials=max(1, args.trials))

        best_trial = study.best_trial
        best_params = dict(best_trial.params)

        # 用最优参数再跑一次，生成图
        merged_best = base_overrides.copy()
        merged_best.update(best_params)

        try:
            best_run = _evaluate_train_test(
                model,
                adapters,
                data_scaled,
                scaler,
                feature_cols,
                merged_best,
                test_ratio=float(args.test_ratio),
                split_seed=int(args.split_seed),
                train_seed=int(args.seed),
                generate_plots=True,
            )
            best_status = "ok" if best_run and best_run.get("status") == "ok" else "failed"
        except Exception as exc:
            best_run = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}", "params": merged_best}
            best_status = "failed"

        ok_trials = [t for t in trials_log if np.isfinite(float(t.get("score", float("inf"))))]
        ok_trials.sort(key=lambda x: float(x.get("score", float("inf"))))

        summary["models"][model] = {
            "status": best_status,
            "baseline": baseline_info,
            "best_score": float(best_trial.value),
            "best_params": best_params,
            "best": best_run,
            "top_trials": ok_trials[:5],
            "num_trials": len(trials_log),
        }

        print(
            f"{model}: baseline={baseline_info.get('score', float('nan')):.6f} "
            f"best={best_trial.value:.6f} (trial #{best_trial.number})"
        )

    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved tuning summary to: {out_path}")


if __name__ == "__main__":
    main()
