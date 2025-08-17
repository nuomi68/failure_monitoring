"""
统一接口层（仅维护一个模型状态）+ 可插拔归一化：
- 指令式：train / predict / transform / save / load / clear / get_meta
- 新训练覆盖旧模型
- 支持可选归一化器（standard/minmax/robust/maxabs/power/quantile/normalizer/none），并将拟合后的 scaler 与模型一起保存
- ★ 新增：支持“计算器公式（calc_recipes）”的保存与加载；预测时自动补齐派生特征
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Protocol, runtime_checkable, Literal
from pathlib import Path
import re

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score,
)
from sklearn.preprocessing import LabelEncoder
import joblib

# Machine learning interface with single-model state and helpers.
# 引入适配器
from .models.supervised_core import ADAPTERS as SUPERVISED_ADAPTERS
from .models.unsupervised_core import ADAPTERS as UNSUPERVISED_ADAPTERS
from .tools import make_scaler, SCALERS_DISPLAY

# ---------------------------- 数据结构 ----------------------------
@dataclass
class ModelArtifact:
    model: Any
    scaler: Optional[Any]
    meta: Dict[str, Any]
    label_encoder: Optional[LabelEncoder] = None
    x_encoders: Optional[Dict[str, LabelEncoder]] = None


@dataclass
class MultiOutputArtifact:
    """多目标分组：同一目标下可集成，不同目标分别输出"""
    groups: dict[str, list[ModelArtifact]]
    method: str = "mean"  # 'mean' | 'vote'


@dataclass
class EnsembleArtifact:
    members: list[ModelArtifact]
    method: str = "mean"  # "mean" | "vote"


@dataclass
class TrainReport:
    y_true: Optional[np.ndarray] = None
    y_pred: Optional[np.ndarray] = None
    scores: Optional[np.ndarray] = None
    metrics_text: str = ""


# ---------------------------- 协议与注册表 ----------------------------
@runtime_checkable
class AlgoAdapter(Protocol):
    code: str
    kind: Literal["supervised_clf", "supervised_reg", "unsupervised"]

    def build(self, **params) -> Any: ...

    def fit(self, model: Any, X: np.ndarray, y: Optional[np.ndarray] = None) -> Any: ...

    def predict(self, model: Any, X: np.ndarray) -> np.ndarray: ...

    def scores(self, model: Any, X: np.ndarray, *, classes_: Optional[np.ndarray] = None) -> Optional[np.ndarray]: ...

    def default_tau(self, scores: Optional[np.ndarray], *, classes_: Optional[np.ndarray] = None) -> Optional[float]: ...

    def meta_model_type(self) -> str: ...


_REGISTRY: dict[str, AlgoAdapter] = {}


def register(adapter: AlgoAdapter):
    _REGISTRY[adapter.code] = adapter


def get_adapter(code: str) -> AlgoAdapter:
    if code not in _REGISTRY:
        raise KeyError(f"Unknown algorithm code: {code}. Registered: {list(_REGISTRY)}")
    return _REGISTRY[code]


for _ad in [*SUPERVISED_ADAPTERS, *UNSUPERVISED_ADAPTERS]:
    register(_ad)


# ---------------------------- 单例模型状态 ----------------------------
class _State:
    current: Optional[ModelArtifact | EnsembleArtifact | MultiOutputArtifact] = None


STATE = _State()

# ★：暂存“计算器公式”，用于在训练后写入 meta，或在加载后同步
_PENDING_CALC_RECIPES: list[dict] | None = None


# ---------------------------- 工具：公式求值 ----------------------------
def _normalize_expr(expr: str) -> str:
    return (str(expr).replace("^", "**")
                    .replace("√", "sqrt")
                    .replace("ln", "log")
                    .replace("log10", "log10"))


def _apply_calc_recipes_to_table(
    X_table: dict[str, np.ndarray],
    recipes: list[dict],
    encoders: Dict[str, LabelEncoder] | None = None,
) -> dict[str, np.ndarray]:
    """给定原始列字典 + 公式，计算派生列并返回 *新的列字典*（不修改输入）。
    - 顺序执行，允许链式引用
    - 若提供 `encoders`，则对已知的字符串列先行转码，避免表达式中直接出现字符导致计算失败
    - 失败的条目以 NaN 兜底
    """
    if not recipes:
        return dict(X_table)
    # 转为 DataFrame 统一计算
    n = 0
    for v in X_table.values():
        n = len(v); break
    df = pd.DataFrame(index=range(n))
    for k, v in X_table.items():
        arr = np.asarray(v).ravel()
        le = (encoders or {}).get(k)
        if le is not None:
            try:
                arr = le.transform(arr.astype(str))
            except Exception:
                pass
        df[k] = arr
    for item in recipes:
        try:
            name = str(item.get("name"))
            expr = _normalize_expr(str(item.get("expr")))
            res = df.eval(
                expr,
                engine="python",
                local_dict={
                    "np": np,
                    "sqrt": np.sqrt,
                    "log": np.log,
                    "log10": np.log10,
                    "abs": np.abs,
                },
            )
            res = pd.Series(res).replace([np.inf, -np.inf], np.nan).fillna(0)
            df[name] = res
            le = (encoders or {}).get(name)
            if le is not None:
                try:
                    df[name] = le.transform(df[name].astype(str))
                except Exception:
                    pass
        except Exception:
            df[name] = np.nan
    return {c: df[c].to_numpy() for c in df.columns}


def _fit_encode_X(X: np.ndarray, feature_names: list[str]) -> tuple[np.ndarray, Dict[str, LabelEncoder]]:
    """检测并编码 X 中的非数值列，返回编码后的 X 及每列的编码器。"""
    X_work = np.asarray(X, dtype=object).copy()
    encoders: Dict[str, LabelEncoder] = {}
    for i in range(X_work.shape[1]):
        col = X_work[:, i]
        try:
            X_work[:, i] = col.astype(float)
        except Exception:
            le = LabelEncoder()
            X_work[:, i] = le.fit_transform(col)
            feat = feature_names[i] if i < len(feature_names) else f"f{i}"
            encoders[feat] = le
    return X_work.astype(float), encoders


def _encode_X(X: np.ndarray, feature_names: list[str], encoders: Dict[str, LabelEncoder]) -> np.ndarray:
    if not encoders:
        return np.asarray(X, dtype=float)
    X_work = np.asarray(X, dtype=object).copy()
    for i, feat in enumerate(feature_names):
        le = encoders.get(feat)
        if le is None:
            try:
                X_work[:, i] = X_work[:, i].astype(float)
            except Exception:
                pass
            continue
        mapping = {cls: idx for idx, cls in enumerate(le.classes_)}
        col = X_work[:, i]
        X_work[:, i] = [mapping.get(v, -1) for v in col]
    return X_work.astype(float)


# ---------------------------- 工具：分数归一化 ----------------------------
def _build_score_normalizer(scores: np.ndarray, bins: int = 256) -> dict:
    """用训练集 scores 建立 ECDF 标定器，保存若干分位点，便于正反查表。"""
    s = np.asarray(scores, dtype=float)
    s = s[np.isfinite(s)]
    q = np.linspace(0.0, 1.0, bins)
    v = np.quantile(s, q)
    return {"method": "ecdf", "q": q.tolist(), "v": v.tolist()}


def _score_to_q(x: np.ndarray | float, norm: dict | None) -> np.ndarray | float:
    """原始分数 -> [0,1]。"""
    if not norm:
        return x
    q = np.asarray(norm["q"], dtype=float)
    v = np.asarray(norm["v"], dtype=float)
    return np.interp(x, v, q)


def _q_to_score(qv: np.ndarray | float, norm: dict | None) -> np.ndarray | float:
    """[0,1] -> 原始分数阈值。"""
    if not norm:
        return qv
    q = np.asarray(norm["q"], dtype=float)
    v = np.asarray(norm["v"], dtype=float)
    return np.interp(qv, q, v)


# ---------------------------- 工具：公式依赖 ----------------------------

_EXPR_FUNC_TOKENS = {"np", "sqrt", "log", "log10", "abs"}


def _extract_vars_from_expr(expr: str) -> set[str]:
    """从规范化表达式中提取变量名（包含反引号列名）。"""
    expr = _normalize_expr(str(expr))
    vars_: set[str] = set(re.findall(r"`([^`]+)`", expr))
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expr)
    for tok in tokens:
        if tok not in _EXPR_FUNC_TOKENS:
            vars_.add(tok)
    return vars_


def infer_input_features(features: list[str], recipes: list[dict]) -> list[str]:
    """给定模型特征和计算配方，推断预测时需提供的原始特征列。"""
    deps = {str(r.get("name")): _extract_vars_from_expr(r.get("expr", "")) for r in (recipes or [])}
    required = set(features)
    changed = True
    while changed:
        changed = False
        for name, vars_ in deps.items():
            if name in required:
                for v in vars_:
                    if v not in required:
                        required.add(v)
                        changed = True
    derived = set(deps.keys())
    base = sorted(f for f in required if f not in derived)
    return base


# ---------------------------- 数据预处理（缩放） ----------------------------
def _fit_transform_supervised(X: np.ndarray, y: np.ndarray, scaler_spec: Any, *, test_size: float, random_state: int,
                              stratify: Optional[np.ndarray]):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)
    scaler = make_scaler(scaler_spec)
    if scaler is not None:
        X_trs = scaler.fit_transform(X_tr)
        X_tes = scaler.transform(X_te)
    else:
        X_trs, X_tes = X_tr, X_te
    return (X_trs, X_tes, y_tr, y_te, scaler)


def _fit_transform_unsupervised(X: np.ndarray, scaler_spec: Any):
    scaler = make_scaler(scaler_spec)
    if scaler is not None:
        Xs = scaler.fit_transform(X)
    else:
        Xs = X
    return Xs, scaler


# ---------------------------- 训练与预测实现 ----------------------------
def _train_impl(
        alg: str,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        *,
        params: Optional[Dict[str, Any]] = None,
        scaler: Any = None,
        test_size: float = 0.2,
        random_state: int = 0,
        stratify: Optional[np.ndarray] = None,
        feature_names: Optional[list[str]] = None,
        calc_recipes: Optional[list[dict]] = None,
) -> Tuple[ModelArtifact, TrainReport]:
    # 统一：保存预测目标名称（监督：真实列名；无监督：默认“是否破损”）
    params = dict(params or {})
    target_name = None
    if "target_name" in params and params.get("target_name"):
        target_name = str(params.pop("target_name"))

    adapter = get_adapter(alg)

    if adapter.kind.startswith("supervised"):
        if y is None:
            raise ValueError("监督学习需要提供 y")
        feature_names = list(feature_names or [f"X{i}" for i in range(X.shape[1])])
        X_enc, x_encoders = _fit_encode_X(X, feature_names)
        label_encoder: Optional[LabelEncoder] = None
        y_work = y
        if adapter.kind == "supervised_clf" and not np.issubdtype(np.asarray(y).dtype, np.number):
            label_encoder = LabelEncoder()
            y_work = label_encoder.fit_transform(y)
        strat = y_work if (adapter.kind == "supervised_clf" and stratify is None and len(np.unique(y_work)) > 1) else stratify
        if strat is not None:
            try:
                _, cnts = np.unique(strat, return_counts=True)
                if cnts.min() < 2:
                    strat = None
            except Exception:
                strat = None
        X_tr, X_te, y_tr, y_te, scaler_obj = _fit_transform_supervised(
            X_enc, y_work, scaler_spec=scaler, test_size=test_size, random_state=random_state, stratify=strat
        )
        n_classes = int(len(np.unique(y_work))) if adapter.kind == "supervised_clf" else None
        is_binary = (adapter.kind == "supervised_clf" and n_classes == 2)
        model = adapter.build(**params)
        model = adapter.fit(model, X_tr, y_tr)

        y_pred = adapter.predict(model, X_te)
        try:
            scores = adapter.scores(model, X_te, classes_=getattr(model, "classes_", None))
        except Exception:
            scores = None
        if adapter.kind == "supervised_clf" and not is_binary:
            scores = None

        metrics_text = ""
        if adapter.kind == "supervised_clf":
            parts = []
            if is_binary and scores is not None:
                try:
                    auc = roc_auc_score(y_te, scores)
                    parts.append(f"AUC={auc:.3f}")
                except Exception:
                    pass
            try:
                acc = accuracy_score(y_te, y_pred)
                parts.append(f"Accuracy={acc:.3f}")
                avg = "binary" if is_binary else "macro"
                prec = precision_score(y_te, y_pred, average=avg, zero_division=0)
                parts.append(f"Precision={prec:.3f}")
                rec = recall_score(y_te, y_pred, average=avg, zero_division=0)
                parts.append(f"Recall={rec:.3f}")
                f1 = f1_score(y_te, y_pred, average=avg, zero_division=0)
                parts.append(f"F1={f1:.3f}")
            except Exception:
                pass
            metrics_text = " | ".join(parts)
        else:
            try:
                mae = mean_absolute_error(y_te, y_pred)
                mse = mean_squared_error(y_te, y_pred)
                r2 = r2_score(y_te, y_pred)
                metrics_text = f"MAE={mae:.3f} | MSE={mse:.3f} | R2={r2:.3f}"
            except Exception:
                pass

        tau = adapter.default_tau(None) if is_binary else None

        # ★ 合并公式：优先用调用方传入的 calc_recipes；否则回退到 _PENDING
        recipes_final = list(calc_recipes or _PENDING_CALC_RECIPES or [])

        classes_meta = (
            list(label_encoder.classes_) if label_encoder is not None
            else getattr(model, "classes_", None)
        )

        meta = {
            "model_type": adapter.meta_model_type(),
            "tau": tau,
            "advanced": params,
            "classes_": classes_meta,
            "scaler": type(scaler_obj).__name__ if scaler_obj is not None else "None",
            "features": list(feature_names) if feature_names else [f"X{i}" for i in range(X.shape[1])],
            "task": adapter.kind,                      # "supervised_clf" | "supervised_reg"
            "n_classes": n_classes if adapter.kind == "supervised_clf" else 1,
            "is_binary": bool(is_binary) if adapter.kind == "supervised_clf" else False,
            "target": (target_name if target_name else "目标"),
            "calc_recipes": recipes_final,            # ★ 保存“计算器公式”
        }
        y_te_out = label_encoder.inverse_transform(y_te) if label_encoder is not None else y_te
        y_pred_out = label_encoder.inverse_transform(y_pred) if label_encoder is not None else y_pred
        art = ModelArtifact(
            model=model,
            scaler=scaler_obj,
            meta=meta,
            label_encoder=label_encoder,
            x_encoders=x_encoders or None,
        )
        rep = TrainReport(y_true=y_te_out, y_pred=y_pred_out, scores=scores, metrics_text=metrics_text)
        return art, rep

    # 无监督
    feature_names = list(feature_names or [f"X{i}" for i in range(X.shape[1])])
    X_enc, x_encoders = _fit_encode_X(X, feature_names)
    Xs, scaler_obj = _fit_transform_unsupervised(X_enc, scaler_spec=scaler)
    model = adapter.build(**params)
    model = adapter.fit(model, Xs, None)

    scores_raw = adapter.scores(model, Xs)                   # 原始分数
    norm = _build_score_normalizer(scores_raw)              # 保存分布标定器
    scores = _score_to_q(scores_raw, norm)                  # 统一到 [0,1]

    tau_raw = adapter.default_tau(scores_raw)               # 模型给出的原始阈值
    tau = float(_score_to_q(float(tau_raw), norm))          # 统一到 [0,1]

    recipes_final = list(calc_recipes or _PENDING_CALC_RECIPES or [])
    le = LabelEncoder()
    le.fit(["否", "是"])

    meta = {
        "model_type": adapter.meta_model_type(),
        "tau": tau,
        "tau_is_normalized": True,
        "score_norm": norm,
        "advanced": params,
        "scaler": type(scaler_obj).__name__ if scaler_obj is not None else "None",
        "features": list(feature_names) if feature_names else [f"X{i}" for i in range(X.shape[1])],
        "task": adapter.kind,                         # "unsupervised"
        "n_classes": 2,
        "is_binary": True,
        "target": (target_name if target_name else "是否破损"),
        "calc_recipes": recipes_final,
    }
    art = ModelArtifact(
        model=model,
        scaler=scaler_obj,
        meta=meta,
        label_encoder=le,
        x_encoders=x_encoders or None,
    )
    rep = TrainReport(scores=scores)
    return art, rep


def _predict_impl(artifact: ModelArtifact, X: np.ndarray):
    feats = artifact.meta.get("features", [])
    enc = artifact.x_encoders or {}
    X_enc = _encode_X(X, feats, enc)
    Xs = artifact.scaler.transform(X_enc) if artifact.scaler is not None else X_enc
    mtype = artifact.meta.get("model_type")
    mapping = {"knn_clf": "knn_clf", "rf": "rf_clf", "knn_reg": "knn_reg", "rf_reg": "rf_reg", "knn": "knn",
               "iforest": "iforest", "autoencoder": "autoencoder"}
    alg = mapping.get(mtype, mtype)
    adapter = get_adapter(alg)

    if adapter.kind == "unsupervised":
        scores_raw = adapter.scores(artifact.model, Xs)      # 原始分数
        norm = artifact.meta.get("score_norm")
        scores = _score_to_q(scores_raw, norm)               # 返回给前端的 0~1 分数

        tau_meta = artifact.meta.get("tau")
        if scores_raw is None:
            return np.array([]), None
        try:
            t_meta = float(tau_meta) if tau_meta is not None else None
        except Exception:
            t_meta = None

        if t_meta is None:
            labels = np.zeros_like(scores_raw, dtype=int)
        else:
            if artifact.meta.get("tau_is_normalized", False) and norm:
                t_raw = float(_q_to_score(t_meta, norm))     # 0~1 -> 原始阈值
            else:
                t_raw = t_meta
            labels = (scores_raw >= t_raw).astype(int)       # 一律用原始分数比较
        return labels, scores

    y_pred = adapter.predict(artifact.model, Xs)
    try:
        scores = adapter.scores(artifact.model, Xs, classes_=artifact.meta.get("classes_"))
    except Exception:
        scores = None
    if artifact.meta.get("task") == "supervised_clf" and not artifact.meta.get("is_binary", False):
        scores = None
    return y_pred, scores


def _predict_ensemble(bundle: EnsembleArtifact, X_table: Dict[str, np.ndarray]):
    preds = []
    scores_list = []
    for art in bundle.members:
        f_order = art.meta.get("features", [])
        length = len(next(iter(X_table.values())))
        X_sub = np.stack([X_table.get(f, np.full(length, np.nan)) for f in f_order], axis=1)
        y, sc = _predict_impl(art, X_sub)
        preds.append(y)
        scores_list.append(sc)

    if bundle.method == "mean":
        y_mean = np.nanmean(np.stack(preds, axis=0), axis=0)
        if bundle.members and bundle.members[0].label_encoder is not None:
            try:
                y_mean = bundle.members[0].label_encoder.inverse_transform(y_mean.astype(int))
            except Exception:
                pass
        sc_mean = (None if any(s is None for s in scores_list)
                   else np.nanmean(np.stack(scores_list, axis=0), axis=0))
        return y_mean, sc_mean
    votes = np.stack(preds, axis=1)
    maj = np.apply_along_axis(lambda r: np.bincount(r.astype(int)).argmax(), 1, votes)
    if bundle.members and bundle.members[0].label_encoder is not None:
        try:
            maj = bundle.members[0].label_encoder.inverse_transform(maj.astype(int))
        except Exception:
            pass
    sc_mean = (None if any(s is None for s in scores_list)
               else np.nanmean(np.stack(scores_list, axis=0), axis=0))
    return maj, sc_mean


def _predict_grouped(moa: MultiOutputArtifact, X_table: Dict[str, np.ndarray]):
    out: dict[str, tuple[np.ndarray, Optional[np.ndarray]]] = {}
    for target, arts in moa.groups.items():
        preds, scores_list = [], []
        for art in arts:
            f_order = art.meta.get("features", [])
            length = len(next(iter(X_table.values())))
            X_sub = np.stack([X_table.get(f, np.full(length, np.nan)) for f in f_order], axis=1)
            y, sc = _predict_impl(art, X_sub)
            preds.append(np.asarray(y))
            scores_list.append(None if sc is None else np.asarray(sc))
        if len(preds) == 1:
            y_single = preds[0]
            le = arts[0].label_encoder
            if le is not None:
                try:
                    y_single = le.inverse_transform(y_single.astype(int))
                except Exception:
                    pass
            out[target] = (y_single, scores_list[0])
        else:
            task = arts[0].meta.get("task")
            le = arts[0].label_encoder
            if task == "supervised_clf":
                votes = np.stack(preds, axis=1)
                maj = np.apply_along_axis(lambda r: np.bincount(r.astype(int)).argmax(), 1, votes)
                if le is not None:
                    try:
                        maj = le.inverse_transform(maj.astype(int))
                    except Exception:
                        pass
                sc_mean = (
                    None
                    if any(s is None for s in scores_list)
                    else np.nanmean(np.stack(scores_list, axis=0), axis=0)
                )
                out[target] = (maj, sc_mean)
            else:
                y_mean = np.nanmean(np.stack(preds, axis=0), axis=0)
                if le is not None:
                    try:
                        y_mean = le.inverse_transform(y_mean.astype(int))
                    except Exception:
                        pass
                sc_mean = (
                    None
                    if any(s is None for s in scores_list)
                    else np.nanmean(np.stack(scores_list, axis=0), axis=0)
                )
                out[target] = (y_mean, sc_mean)
    return out


def _dict_to_array_for_model(artifact: ModelArtifact, X_table: dict[str, np.ndarray]) -> np.ndarray:
    """单模型也支持列字典输入：按 meta['features'] 顺序取列，缺失补 NaN。"""
    feats = artifact.meta.get("features", [])
    if not feats:
        feats = sorted(X_table.keys())
    # 找到长度
    n = 0
    for f in feats:
        if f in X_table:
            n = len(X_table[f]); break
    if n == 0 and X_table:
        n = len(next(iter(X_table.values())))
    if n == 0:
        return np.zeros((0, len(feats)))
    cols = []
    for f in feats:
        if f in X_table:
            col = np.asarray(X_table[f]).ravel()
            if len(col) != n:
                col = np.resize(col, (n,))
        else:
            col = np.full((n,), np.nan)
        cols.append(col)
    return np.stack(cols, axis=1)


# ---------------------------- 对外 API（单例） ----------------------------
class ML:
    @classmethod
    def train(
            cls,
            alg: str,
            X: np.ndarray,
            y: Optional[np.ndarray] = None,
            *,
            feature_names: Optional[list[str]] = None,
            calc_recipes: Optional[list[dict]] = None,
            **kwargs,
    ) -> TrainReport:
        art, rep = _train_impl(
            alg=alg,
            X=X,
            y=y,
            feature_names=feature_names,
            calc_recipes=calc_recipes,
            **kwargs,
        )
        STATE.current = art  # 覆盖旧模型
        # 训练结束后清理 pending（以模型内的为准）
        global _PENDING_CALC_RECIPES
        _PENDING_CALC_RECIPES = list(art.meta.get("calc_recipes", []) or [])
        return rep

    @classmethod
    def predict(cls, X: np.ndarray | Dict[str, np.ndarray]):
        if STATE.current is None:
            raise RuntimeError("当前没有已训练/加载的模型。请先训练/加载。")
        cur = STATE.current

        # 多目标：返回 {target: {"labels": y, "scores": s}}
        if isinstance(cur, MultiOutputArtifact):
            assert isinstance(X, dict), "MultiOutput 预测需要传入列字典：{feature: ndarray}"
            # ★ 对列字典先补齐计算列，并应用各模型的编码器
            recipes: list[dict] = []
            enc_all: Dict[str, LabelEncoder] = {}
            for arts in cur.groups.values():
                for a in arts:
                    recipes.extend(a.meta.get("calc_recipes", []) or [])
                    if a.x_encoders:
                        enc_all.update(a.x_encoders)
            X = _apply_calc_recipes_to_table(X, recipes, encoders=enc_all)
            grouped = _predict_grouped(cur, X)
            return {t: {"labels": ys, "scores": sc} for t, (ys, sc) in grouped.items()}

        # 旧式集合（同目标）
        if isinstance(cur, EnsembleArtifact):
            assert isinstance(X, dict)
            recipes: list[dict] = []
            enc_all: Dict[str, LabelEncoder] = {}
            for a in cur.members:
                recipes.extend(a.meta.get("calc_recipes", []) or [])
                if a.x_encoders:
                    enc_all.update(a.x_encoders)
            X = _apply_calc_recipes_to_table(X, recipes, encoders=enc_all)
            y, sc = _predict_ensemble(cur, X)
            tgt = None
            try:
                tgt = cur.members[0].meta.get("target")
            except Exception:
                tgt = None
            return {"target": (tgt or "目标"), "labels": y, "scores": sc}

        # 单模型：支持 ndarray 或 列字典
        if isinstance(X, dict):
            recipes = STATE.current.meta.get("calc_recipes", []) or []
            enc = cur.x_encoders or {}
            X_table = _apply_calc_recipes_to_table(X, recipes, encoders=enc)
            X_arr = _dict_to_array_for_model(cur, X_table)
        else:
            X_arr = X
        y, sc = _predict_impl(cur, X_arr)
        if cur.label_encoder is not None:
            try:
                y = cur.label_encoder.inverse_transform(y.astype(int))
            except Exception:
                pass
        return {"target": cur.meta.get("target", "目标"), "labels": y, "scores": sc}

    @classmethod
    def transform(cls, X: np.ndarray):
        if STATE.current is None:
            raise RuntimeError("当前没有已训练/加载的模型。请先训练/加载。")
        cur = STATE.current
        sc = None
        if isinstance(cur, (EnsembleArtifact, MultiOutputArtifact)):
            # 简化：集合/多输出不提供 transform
            return X
        if isinstance(cur, ModelArtifact):
            X_enc = _encode_X(X, cur.meta.get("features", []), cur.x_encoders or {})
            sc = cur.scaler
            return sc.transform(X_enc) if sc is not None else X_enc
        return X

    @classmethod
    def get_meta(cls) -> Dict[str, Any]:
        if STATE.current is None:
            return {}
        cur = STATE.current
        if isinstance(cur, EnsembleArtifact):
            return {
                "ensemble": True,
                "method": cur.method,
                "members": [m.meta for m in cur.members],
            }
        if isinstance(cur, MultiOutputArtifact):
            return {
                "multi_output": True,
                "method": cur.method,
                "groups": {t: [m.meta for m in arts] for t, arts in cur.groups.items()},
            }
        return dict(cur.meta)

    @classmethod
    def set_tau(cls, tau: float, *, normalized: bool = True) -> None:
        if STATE.current is None:
            raise RuntimeError("没有可更新的模型。")
        STATE.current.meta["tau"] = float(tau)
        STATE.current.meta["tau_is_normalized"] = bool(normalized)

    # ★ 新增：在训练前后、或任意时机注入/覆盖“计算器公式”
    @classmethod
    def set_calc_recipes(cls, recipes: list[dict] | None):
        global _PENDING_CALC_RECIPES
        _PENDING_CALC_RECIPES = list(recipes or [])
        if isinstance(STATE.current, ModelArtifact):
            STATE.current.meta["calc_recipes"] = list(_PENDING_CALC_RECIPES)

    @classmethod
    def get_calc_recipes(cls) -> list[dict]:
        if isinstance(STATE.current, ModelArtifact):
            return list(STATE.current.meta.get("calc_recipes", []) or [])
        return list(_PENDING_CALC_RECIPES or [])

    @classmethod
    def save(cls, path: str) -> None:
        if STATE.current is None:
            raise RuntimeError("没有可保存的模型。")
        save_artifact(path, STATE.current)

    @classmethod
    def save_auto(cls, name: str | None = None) -> Dict[str, Any]:
        if STATE.current is None:
            raise RuntimeError("没有可保存的模型。")
        # 利用统一的模型管理器保存并写入注册表
        from .ml_model_manager import MLModelManager  # 避免循环依赖

        mgr = MLModelManager()
        title = name or STATE.current.meta.get("model_type", "model")
        model_id = mgr.save_current(title)
        meta = mgr.registry.data.get(model_id, {})
        return {"ok": True, "path": meta.get("path", "")}

    @classmethod
    def load(cls, path: str) -> Dict[str, Any]:
        art = load_artifact(path)
        STATE.current = art
        # 同步 pending，便于前端需要时读取
        global _PENDING_CALC_RECIPES
        _PENDING_CALC_RECIPES = list(art.meta.get("calc_recipes", []) or [])
        return dict(art.meta)

    @classmethod
    def load_many(cls, paths: list[str], *, method: str = "mean") -> Dict[str, Any]:
        arts = [load_artifact(p) for p in paths]
        groups: dict[str, list[ModelArtifact]] = {}
        for art in arts:
            t = art.meta.get("target")
            if not t:
                t = art.meta.get("model_type", "目标")
            groups.setdefault(str(t), []).append(art)
        STATE.current = MultiOutputArtifact(groups=groups, method=method)
        features_union = sorted({
            f
            for art in arts
            for f in infer_input_features(art.meta.get("features", []), art.meta.get("calc_recipes", []))
        })
        # 多模型情况下，pending 公式暂不合并（按 predict() 时动态收集）
        return {"ok": True, "method": method, "groups": {k: len(v) for k, v in groups.items()},
                "features_union": features_union}

    @classmethod
    def clear(cls) -> None:
        global _PENDING_CALC_RECIPES
        _PENDING_CALC_RECIPES = None
        STATE.current = None

    @classmethod
    def run(cls, action: str, **kwargs):
        action = action.lower()
        if action == "train":
            return cls.train(**kwargs)
        if action == "predict":
            return cls.predict(**kwargs)
        if action == "transform":
            return cls.transform(**kwargs)
        if action == "save":
            cls.save(**kwargs)
            return {"ok": True}
        if action == "save_auto":
            return cls.save_auto(**kwargs)
        if action == "load":
            meta = cls.load(**kwargs)
            return {"ok": True, "meta": meta}
        if action == "clear":
            cls.clear()
            return {"ok": True}
        if action == "get_meta":
            return cls.get_meta()
        raise ValueError(f"未知指令: {action}")

    @staticmethod
    def available_scalers():
        return SCALERS_DISPLAY.copy()


# ---------------------------- 持久化 ----------------------------
def save_artifact(path: str | Path, artifact: ModelArtifact) -> None:
    # 保存逻辑保持不变：模型、scaler、meta 打包
    obj: Dict[str, Any] = {
        "scaler": artifact.scaler,
        "meta": artifact.meta,
        "label_encoder": artifact.label_encoder,
        "x_encoders": artifact.x_encoders,
        "_interface": "ml_interface.singleton.scaler.v5",
    }
    model = artifact.model
    mtype = artifact.meta.get("model_type")
    adapter = get_adapter(mtype)

    if hasattr(adapter, "persist"):
        try:
            payload = adapter.persist(model)  # type: ignore[attr-defined]
            obj["model"] = {"_adapter": mtype, "payload": payload}
            joblib.dump(obj, str(path)); return
        except Exception:
            obj["model"] = model
    else:
        obj["model"] = model
    joblib.dump(obj, str(path))


def load_artifact(path: str | Path) -> ModelArtifact:
    obj = joblib.load(str(path))
    meta = obj.get("meta", {})
    scaler = obj.get("scaler")
    label_encoder = obj.get("label_encoder")
    x_encoders = obj.get("x_encoders")
    model_field = obj.get("model")

    if isinstance(model_field, dict) and model_field.get("_adapter"):
        mtype = model_field["_adapter"]
        adapter = get_adapter(mtype)
        if not hasattr(adapter, "restore"):
            raise RuntimeError(f"适配器 {mtype} 不支持 restore()。")
        arch = meta.get("ae_arch", {})
        model = adapter.restore(model_field["payload"], arch)  # type: ignore[attr-defined]
    else:
        model = model_field

    if "features" not in meta:
        try:
            n_feat = model.n_features_in_
            meta["features"] = [f"X{i}" for i in range(n_feat)]
        except Exception:
            meta["features"] = []
    # 兼容：若旧模型没有 calc_recipes 字段，补空
    meta.setdefault("calc_recipes", [])
    return ModelArtifact(model=model, scaler=scaler, meta=meta, label_encoder=label_encoder, x_encoders=x_encoders)
