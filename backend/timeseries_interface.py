"""
时间序列模型的后端管理模块。

此模块提供 `ModelManager` 类用于处理数据集的注册与加载、模型的训练与保存等工作。
数据集经过弹窗清洗后会持久化保存为 parquet 文件，并记录时间列和格式等元信息；
模型训练完毕后可以保存到磁盘，并在统一的 JSON 注册表中登记其类型、参数、绑定的数据集等信息。

由于当前环境缺少真实的模型训练依赖，本实现中的 `train` 方法仅提供示例性的占位训练，
生成随机指标而不执行真正的深度学习训练。
需要对接实际模型训练代码时，可在此模块中引入模型库并替换 `train` 方法。
"""

from __future__ import annotations

import json
import uuid
import hashlib
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import torch
import numpy as np
import pandas as pd
import joblib
import os

# 新增：运行时单例，保存当前训练得到的模型、缩放器等
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
from sklearn.model_selection import train_test_split


from backend.data_utils import build_windows
from backend.models import (
    gru_model,
    tcn_model,
    tsmixer_model,
    random_forest_model,
    xgboost_model,
    timesnet_model
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# 把所有模型的 train 和 predict 函数映射到同一个结构
MODEL_REGISTRY = {
    "gru": {
        "train": lambda X_tr, y_tr, X_val, y_val, **kw: gru_model.train_gru(
            X_tr, y_tr, X_val, y_val, device=DEVICE, **kw
        ),
        "predict": lambda model, seq: gru_model.predict(model, seq, device=DEVICE),
    },
    "tcn": {
        "train": lambda X_tr, y_tr, X_val, y_val, **kw: tcn_model.train_tcn(
            X_tr, y_tr, X_val, y_val, device=DEVICE, **kw
        ),
        "predict": lambda model, seq: tcn_model.predict(model, seq, device=DEVICE),
    },
    "tsmixer": {
        "train": lambda X_tr, y_tr, X_val, y_val, **kw: tsmixer_model.train_tsmixer(
            X_tr, y_tr, X_val, y_val, device=DEVICE, **kw
        ),
        "predict": lambda model, seq: tsmixer_model.predict(model, seq, device=DEVICE),
    },
    "rf": {
        "train": lambda X_tr, y_tr, X_val, y_val, **kw: random_forest_model.train_rf(
            X_tr, y_tr, X_val, y_val, **kw
        ),
        "predict": lambda model, seq: random_forest_model.predict(model, seq),
    },
    "xgb": {
        "train": lambda X_tr, y_tr, X_val, y_val, **kw: xgboost_model.train_xgb(
            X_tr, y_tr, X_val, y_val, **kw
        ),
        "predict": lambda model, seq: xgboost_model.predict(model, seq),
    },
    "timesnet": {
        "train": lambda X_tr, y_tr, X_val, y_val, **kw: timesnet_model.train_timesnet(
            X_tr, y_tr, X_val, y_val, device=DEVICE, **kw
        ),
        "predict": lambda model, seq: timesnet_model.predict(model, seq, device=DEVICE),
    },
}
DATA_CFG = {
    "gru": 32,
    "tcn": 14,
    "tsmixer": 32,
    "rf": 5,
    "xgb": 14,
    "timesnet": 32,       # 新增
}

@dataclass
class RuntimeStore:
    """Singleton store for a single trained model and its context."""

    trained_model: Any = None  # currently loaded or trained model object
    model_type: Optional[str] = None  # name of the current model
    look_back: Optional[int] = None  # window size used for training
    scaler: Any = None
    feature_names: Optional[List[str]] = None
    data_scaled: Optional[np.ndarray] = None
    dataset_id: Optional[str] = None

class _RuntimeSingleton:
    _inst: Optional[RuntimeStore] = None

    @classmethod
    def get(cls) -> RuntimeStore:
        if cls._inst is None:
            cls._inst = RuntimeStore()
        return cls._inst

    @classmethod
    def reset(cls) -> None:
        cls._inst = RuntimeStore()

class ModelManager:
    """负责管理数据集、模型注册与伪训练逻辑的管理类。

    该类的主要职责包括：

    1. 在磁盘上持久化保存清洗后的数据集，并生成数据集清单（manifest）。
    2. 提供加载数据集、列出数据集信息的接口。
    3. 管理模型的注册表，保存模型元数据及其训练指标、参数等信息。
    4. 提供训练、保存、加载模型的接口。由于缺乏真正的后端依赖，这里的训练过程是一个示例性的占位实现。

    使用时，可通过 ``register_dataset`` 将清洗后的 ``pandas.DataFrame`` 保存并注册为数据集，
    再通过 ``train`` 函数传入数据集 ID、模型类型以及参数进行训练，训练后通过 ``save_model`` 存档。

    注意：所有文件都会存放在项目根目录的 ``artifacts`` 目录下，包括 ``datasets``、``models`` 和 ``registries`` 子目录。
    """

    def __init__(self) -> None:
        # 确定基础目录：使路径与其它模型管理器一致
        # 统一使用 ``models_saved/timeseries`` 作为时间序列相关文件的根目录，
        # 避免因工作目录不同导致在旧的 ``artifacts`` 路径下找不到文件。
        project_root = Path(__file__).resolve().parents[1] / "models_saved" / "timeseries"
        # 为兼容既有变量命名，仍使用 artifacts_dir 表示根目录
        self.artifacts_dir = project_root
        self.datasets_dir = self.artifacts_dir / "datasets"
        self.models_dir = self.artifacts_dir / "models"
        self.registries_dir = self.artifacts_dir / "registries"
        # 注册表文件路径
        self.datasets_registry_file = self.registries_dir / "datasets_registry.json"
        self.models_registry_file = self.registries_dir / "models_registry.json"

        # 创建必要的目录
        for d in [self.artifacts_dir, self.datasets_dir, self.models_dir, self.registries_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # 加载或初始化注册表
        self.datasets_registry: Dict[str, Dict[str, Any]] = self._load_json(self.datasets_registry_file) or {}
        self.models_registry: Dict[str, Dict[str, Any]] = self._load_json(self.models_registry_file) or {}
        self.datasets_cache: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # 基础工具函数
    # ------------------------------------------------------------------
    def _load_json(self, path: Path) -> Optional[Dict[str, Any]]:
        """从 JSON 文件中读取数据，如果文件不存在则返回 ``None``。"""
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    def _write_json(self, data: Dict[str, Any], path: Path) -> None:
        """将字典写入 JSON 文件，使用 UTF-8 编码并带缩进。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _calc_sha256(self, df: pd.DataFrame) -> str:
        """计算 ``DataFrame`` 的 SHA256 摘要。

        这里通过将数据表转换为 CSV 字节流来计算哈希值；
        对于大表而言，这可能会有一定开销，但在持久化注册时可以辅助唯一性检查。
        """
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        return hashlib.sha256(csv_bytes).hexdigest()

    # ------------------------------------------------------------------
    # 数据集管理
    # ------------------------------------------------------------------
    def register_dataset(self, df: pd.DataFrame, time_col: str, time_format: str) -> Dict[str, Any]:
        """注册数据集（仅缓存，不立即持久化）。"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        uid = uuid.uuid4().hex[:8]
        dataset_id = f"{timestamp}-DATA-{uid}"

        self.datasets_cache[dataset_id] = {
            "df": df.copy(),
            "time_col": time_col,
            "time_format": time_format,
        }

        manifest: Dict[str, Any] = {
            "dataset_id": dataset_id,
            "time_col": time_col,
            "time_format": time_format,
            "n_rows": len(df),
            "n_cols": len(df.columns),
        }
        return manifest

    def load_dataset(self, dataset_id: str) -> pd.DataFrame:
        """根据数据集 ID 载入数据表。

        :param dataset_id: 已注册的数据集 ID。
        :returns: 载入的数据 ``DataFrame``。
        :raises KeyError: 当数据集 ID 不存在时抛出异常。
        """
        if dataset_id in self.datasets_cache:
            return self.datasets_cache[dataset_id]["df"].copy()
        if dataset_id not in self.datasets_registry:
            raise KeyError(f"未找到数据集 {dataset_id}")
        manifest = self.datasets_registry[dataset_id]
        data_path = self.artifacts_dir / manifest["path"]
        return pd.read_parquet(data_path)

    def list_datasets(self) -> List[Dict[str, Any]]:
        """列出所有已注册的数据集的清单。"""
        return list(self.datasets_registry.values())

    # ------------------------------------------------------------------
    # 模型管理
    # ------------------------------------------------------------------
    def list_models(self) -> List[Dict[str, Any]]:
        """列出所有已注册模型的元数据。"""
        return list(self.models_registry.values())

    def get_model_meta(self, model_id: str) -> Optional[Dict[str, Any]]:
        """获取某一模型的元数据，如果不存在则返回 ``None``。"""
        return self.models_registry.get(model_id)

    def current_feature_names(self):
        """返回运行时中模型训练用的特征列（若尚未训练则为 None）"""
        try:
            return _RuntimeSingleton.get().feature_names
        except Exception:
            return None

    def refresh_models(self) -> List[Dict[str, Any]]:
        """刷新模型与数据集注册表，清理已失效的记录。

        - 若数据集文件缺失，将其从数据集注册表移除；
        - 若模型文件缺失，或其关联的数据集不存在，同样移除模型记录。

        :returns: 更新后的模型元数据列表。
        """
        # 每次调用都从磁盘重新读取注册表，以获取外部新增的模型
        self.datasets_registry = self._load_json(self.datasets_registry_file) or {}
        self.models_registry = self._load_json(self.models_registry_file) or {}

        # 先刷新数据集，移除磁盘上不存在的文件
        removed_datasets: List[str] = []
        for did, meta in list(self.datasets_registry.items()):
            data_path = self.artifacts_dir / meta["path"]
            if not data_path.exists():
                removed_datasets.append(did)
                self.datasets_registry.pop(did, None)
        if removed_datasets:
            self._write_json(self.datasets_registry, self.datasets_registry_file)

        # 再刷新模型，如果模型文件不存在或其数据集已经移除，则删除模型记录
        removed_models: List[str] = []
        for mid, meta in list(self.models_registry.items()):
            model_path = self.artifacts_dir / meta["artifacts"]["model_path"]
            dataset_id = meta.get("dataset_id")
            dataset_missing = dataset_id not in self.datasets_registry
            if (not model_path.exists()) or dataset_missing:
                removed_models.append(mid)
                self.models_registry.pop(mid, None)
        if removed_models:
            self._write_json(self.models_registry, self.models_registry_file)

        return list(self.models_registry.values())

    # ---------- 模型维护 API ----------
    def rename_model(self, model_id: str, new_name: str) -> bool:
        """仅修改注册表中的 name 字段。"""
        try:
            if model_id not in self.models_registry:
                return False
            self.models_registry[model_id]["name"] = new_name
            self._write_json(self.models_registry, self.models_registry_file)
            return True
        except Exception:
            return False

    def delete_model(self, model_id: str) -> bool:
        """删除注册表条目并清理模型目录/文件。"""
        import shutil
        try:
            meta = self.models_registry.pop(model_id, None)
            if meta:
                model_path = self.artifacts_dir / meta["artifacts"]["model_path"]
                shutil.rmtree(model_path.parent, ignore_errors=True)
            self._write_json(self.models_registry, self.models_registry_file)
            return True
        except Exception:
            return False

    def export_model(self, model_id: str, out_zip_path: str) -> bool:
        """打包模型目录及 meta.json 为 zip。"""
        import os, json, zipfile
        try:
            if model_id not in self.models_registry:
                return False
            meta = self.models_registry[model_id]
            model_dir = (self.artifacts_dir / meta["artifacts"]["model_path"]).parent
            with zipfile.ZipFile(out_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for root, _, files in os.walk(model_dir):
                    for f in files:
                        p = os.path.join(root, f)
                        zf.write(p, arcname=os.path.relpath(p, model_dir))
                zf.writestr("meta.json", json.dumps(meta, ensure_ascii=False, indent=2))
            return True
        except Exception:
            return False

    def import_model(self, zip_path: str) -> bool:
        """从 zip 包导入模型并登记注册表。"""
        import zipfile, json, uuid, os
        try:
            if not os.path.exists(zip_path):
                return False
            with zipfile.ZipFile(zip_path, "r") as zf:
                try:
                    meta = json.loads(zf.read("meta.json").decode("utf-8"))
                except Exception:
                    meta = {}
                new_id = meta.get("model_id")
                while (not new_id) or (new_id in self.models_registry):
                    new_id = uuid.uuid4().hex[:8]
                meta["model_id"] = new_id
                target_dir = self.models_dir / new_id
                target_dir.mkdir(parents=True, exist_ok=True)
                for name in zf.namelist():
                    if name.endswith("/") or name == "meta.json":
                        continue
                    out = target_dir / name
                    out.parent.mkdir(parents=True, exist_ok=True)
                    with open(out, "wb") as f:
                        f.write(zf.read(name))
                meta["artifacts"] = {
                    "model_path": str((target_dir / "model.pkl").relative_to(self.artifacts_dir)),
                    "config_path": str((target_dir / "config.json").relative_to(self.artifacts_dir)),
                    "metrics_path": str((target_dir / "metrics.json").relative_to(self.artifacts_dir)),
                }
                self.models_registry[new_id] = meta
                self._write_json(self.models_registry, self.models_registry_file)
            return True
        except Exception:
            return False

    def train(
        self,
        dataset_id: str,
        model_type: str,
        params: Dict[str, Any],
        log_callback: Optional[Callable[[str], None]] = None,
        use_color: bool = True,
    ) -> Dict[str, Any]:
        """
        使用 controller 中真实模型对指定数据集进行训练，并把模型放入单例 RuntimeStore。
        """
        # 校验并载入数据
        if dataset_id in self.datasets_cache:
            df = self.datasets_cache[dataset_id]["df"]
            time_col = self.datasets_cache[dataset_id]["time_col"]
        elif dataset_id in self.datasets_registry:
            df = self.load_dataset(dataset_id)
            time_col = self.datasets_registry[dataset_id]["time_col"]
        else:
            raise KeyError(f"未找到数据集 {dataset_id}")

        # 只用数值特征（去掉时间列），并按需筛选特征列
        feat_df = df.drop(columns=[time_col], errors="ignore").select_dtypes(include=[np.number])
        feature_cols = params.pop("feature_cols", None)
        if feature_cols:
            try:
                feat_df = feat_df[feature_cols]
            except KeyError as exc:
                raise ValueError(f"训练特征列缺失: {exc}") from exc
        if feat_df.empty:
            raise ValueError("数据集中不包含数值列，无法训练模型。")

        feature_names = feat_df.columns.tolist()

        # 标准化
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(feat_df.values.astype(float))

        # 构造窗口
        default_lb = DATA_CFG.get(model_type, 14)  # controller 里给的默认窗口长度
        train_params = params.copy()
        look_back = int(train_params.pop("look_back", default_lb))
        holdout = int(train_params.pop("holdout", 5))  # 留出最后 N 条不参与窗口构造，默认 5（与 controller 用法一致）
        batch_size = int(train_params.pop("batch_size", 16))
        epochs = int(train_params.pop("epochs", 50))
        X, y = build_windows(data_scaled[:-holdout], look_back)  # 与 controller 的写法对齐
        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

        # 训练
        if model_type not in MODEL_REGISTRY:
            raise KeyError(f"未知的模型类型: {model_type}")
        train_fn = MODEL_REGISTRY[model_type]["train"]
        if model_type in {"rf", "xgb"}:
            result = train_fn(
                X_tr, y_tr, X_val, y_val, log_callback=log_callback, **train_params
            )
        else:
            result = train_fn(
                X_tr,
                y_tr,
                X_val,
                y_val,
                batch_size=batch_size,
                epochs=epochs,
                log_callback=log_callback,
                **train_params,
            )

        # 统一评估训练/验证误差
        if isinstance(result, tuple):
            model_obj = result[0]
        else:
            model_obj = result

        predict_fn = MODEL_REGISTRY[model_type]["predict"]
        train_preds = np.array([predict_fn(model_obj, seq) for seq in X_tr])
        test_preds = np.array([predict_fn(model_obj, seq) for seq in X_val])
        train_mae = mean_absolute_error(y_tr, train_preds)
        test_mae = mean_absolute_error(y_val, test_preds)
        if log_callback:
            msg = (
                f"[{model_type.upper()}] 训练集误差={train_mae:.4f} "
                f"测试集误差={test_mae:.4f}"
            )
            if use_color:
                msg = f"\033[92m{msg}\033[0m"
            log_callback(msg)
        metrics = {"train_mae": float(train_mae), "test_mae": float(test_mae)}

        # 6) 写入运行时单例（保持单模型状态）
        _RuntimeSingleton.reset()
        rt = _RuntimeSingleton.get()
        rt.trained_model = model_obj
        rt.model_type = model_type
        rt.look_back = look_back
        rt.scaler = scaler
        rt.feature_names = feature_names
        rt.data_scaled = data_scaled
        rt.dataset_id = dataset_id

        # 7) 返回训练摘要（无需持久化时，可不调用 save_model；如需保存，可按你的序列化策略扩展）
        return {
            "model": f"in-memory:{model_type}",
            "metrics": metrics,
            "extra": {"look_back": look_back, "n_features": len(feature_names)}
        }

    def predict(self, steps: int = 1, use_ensemble: bool = True) -> List[Dict[str, Any]]:
        """
        用单例中的已训练模型做多步逐步预测。
        返回一个列表，每个元素包含第 i 步的各模型预测与（可选）eval 集成。
        """
        rt = _RuntimeSingleton.get()
        if rt.trained_model is None:
            raise RuntimeError("没有已训练的模型，请先调用 train。")
        if rt.data_scaled is None or rt.scaler is None or rt.feature_names is None:
            raise RuntimeError("运行时状态不完整，缺少 scaler/feature_names/data_scaled。")

        results = []
        model_name = rt.model_type or "model"
        look_back = rt.look_back or 14
        for i in range(1, int(steps) + 1):
            seq = rt.data_scaled[-look_back - i: -i]
            pred = MODEL_REGISTRY[model_name]["predict"](rt.trained_model, seq)
            pred_inv = rt.scaler.inverse_transform(pred.reshape(1, -1)).squeeze()
            series = pd.Series(pred_inv, index=rt.feature_names)
            df = pd.DataFrame({model_name: series}, index=rt.feature_names)
            results.append({"step": i, "table": df})
        return results

    # ===== 追加新观测到运行时 ===== #
    def append_observations(self, df_new: pd.DataFrame) -> None:
        """
        把 **原始量纲** 的观测行追加到运行时 cache，供后续预测滚动窗口使用。
        `df_new` 只需包含训练阶段用到的特征列（不用时间列）。
        """
        if df_new is None or df_new.empty:
            return

        rt = _RuntimeSingleton.get()
        if rt.scaler is None or rt.data_scaled is None or rt.feature_names is None:
            raise RuntimeError("请先完成训练，再追加观测。")

        # 1) 按训练时的列顺序取数值
        try:
            df_use = df_new[rt.feature_names].astype(float)
        except KeyError as exc:
            raise ValueError(f"追加观测缺少列: {exc}") from exc

        # 2) 同一 scaler 做 transform，拼接到 data_scaled
        new_scaled = rt.scaler.transform(df_use.values)
        rt.data_scaled = np.vstack([rt.data_scaled, new_scaled])

    def _ensure_dataset_persisted(self, dataset_id: str) -> None:
        """若数据集尚未持久化，则在保存模型前写入磁盘并登记。"""
        if dataset_id in self.datasets_registry or dataset_id not in self.datasets_cache:
            return

        info = self.datasets_cache.pop(dataset_id)
        df = info["df"]
        time_col = info["time_col"]
        time_format = info["time_format"]

        dataset_dir = self.datasets_dir / dataset_id
        dataset_dir.mkdir(parents=True, exist_ok=True)
        data_path = dataset_dir / "data.parquet"
        df.to_parquet(data_path, index=False)

        manifest: Dict[str, Any] = {
            "dataset_id": dataset_id,
            "path": str(data_path.relative_to(self.artifacts_dir)),
            "time_col": time_col,
            "time_format": time_format,
            "n_rows": len(df),
            "n_cols": len(df.columns),
            "columns": df.columns.tolist(),
            "sha256": self._calc_sha256(df),
            "created_at": datetime.now().isoformat(),
        }

        manifest_path = dataset_dir / "dataset_manifest.json"
        self._write_json(manifest, manifest_path)

        self.datasets_registry[dataset_id] = manifest
        self._write_json(self.datasets_registry, self.datasets_registry_file)

    def save_model(
        self,
        model_id: Optional[str],
        name: str,
        model_type: str,
        dataset_id: str,
        params: Dict[str, Any],
        model_obj: Any,
        metrics: Dict[str, Any],
    ) -> str:
        """保存模型及其元数据，并返回模型 ID。

        :param model_id: 如果为 ``None`` 则生成新模型 ID；否则覆盖对应模型。
        :param name: 给模型取一个友好的名称。
        :param model_type: 模型类型字符串。
        :param dataset_id: 训练该模型所用的数据集 ID。
        :param params: 训练参数字典。
        :param model_obj: 训练得到的模型对象，本例中为字典；可根据真实需求替换为序列化对象。
        :param metrics: 训练评估指标字典。
        :returns: 最终保存的模型 ID。
        """
        # 保存模型前确保数据集已落盘
        self._ensure_dataset_persisted(dataset_id)

        # 如果没有提供 ID，则创建一个新的
        is_new = model_id is None
        if is_new:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            uid = uuid.uuid4().hex[:6]
            model_id = f"{timestamp}-{model_type.upper()}-{uid}"

        model_dir = self.models_dir / model_id
        model_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(model_obj, str) and model_obj.startswith("in-memory:"):
            in_mem_type = model_obj.split(":", 1)[1]
            rt = _RuntimeSingleton.get()
            real = rt.trained_model if rt.model_type == in_mem_type else None
            if real is None:
                raise ValueError("内存中找不到对应已训练模型，请先训练后再保存。")
            model_obj = real
        # 保存模型对象到二进制文件，避免字符串化导致加载后类型错误
        model_path = model_dir / "model.pkl"
        joblib.dump(model_obj, model_path)

        # 保存参数和指标
        config_path = model_dir / "config.json"
        metrics_path = model_dir / "metrics.json"
        self._write_json(params, config_path)
        self._write_json(metrics, metrics_path)

        # 更新模型注册表，附带训练所用特征列
        rt = _RuntimeSingleton.get()
        dataset_meta = self.datasets_registry.get(dataset_id, {})
        meta = {
            "model_id": model_id,
            "name": name,
            "model_type": model_type,
            "dataset_id": dataset_id,
            "created_at": datetime.now().isoformat(),
            "artifacts": {
                "model_path": str(model_path.relative_to(self.artifacts_dir)),
                "config_path": str(config_path.relative_to(self.artifacts_dir)),
                "metrics_path": str(metrics_path.relative_to(self.artifacts_dir)),
            },
            "params": params,
            "metrics": metrics,
            "feature_names": rt.feature_names or [],
            "time_col": dataset_meta.get("time_col"),
            "time_format": dataset_meta.get("time_format"),
        }
        self.models_registry[model_id] = meta
        self._write_json(self.models_registry, self.models_registry_file)

        return model_id

    def load_model(self, model_id: str) -> Dict[str, Any]:
        """加载模型对象及其元信息。

        :param model_id: 模型 ID。
        :returns: 包含 ``model`` 和 ``meta`` 的字典；其中 ``model`` 是保存的模型对象，``meta`` 是注册表中的元数据。
        :raises KeyError: 如果模型 ID 不存在。
        """
        # 首先刷新注册表，清除可能已被删除的模型文件
        self.refresh_models()
        if model_id not in self.models_registry:
            raise KeyError(f"未找到模型 {model_id}")

        meta = self.models_registry[model_id]
        if "time_col" not in meta or meta.get("time_col") is None:
            ds_meta = self.datasets_registry.get(meta.get("dataset_id", ""), {})
            meta["time_col"] = ds_meta.get("time_col")
            meta["time_format"] = ds_meta.get("time_format")
        model_path = self.artifacts_dir / meta["artifacts"]["model_path"]
        # 使用 joblib 反序列化模型；若旧模型仍为 JSON，保留回退处理
        if model_path.suffix == ".json":
            with open(model_path, "r", encoding="utf-8") as f:
                model_obj = json.load(f)
        else:
            model_obj = joblib.load(model_path)

        # 载入数据集并恢复运行时状态，便于后续预测
        dataset_id = meta.get("dataset_id")
        if dataset_id not in self.datasets_registry:
            # 数据集已丢失，同步移除模型记录
            self.models_registry.pop(model_id, None)
            self._write_json(self.models_registry, self.models_registry_file)
            raise KeyError(f"模型关联的数据集 {dataset_id} 不存在")
        df = self.load_dataset(dataset_id)

        feature_names = meta.get("feature_names")
        if feature_names:
            try:
                feat_df = df[feature_names].astype(float)
            except KeyError as exc:
                raise ValueError(f"数据集中缺少列: {exc}") from exc
        else:
            time_col = self.datasets_registry[dataset_id]["time_col"]
            feat_df = df.drop(columns=[time_col], errors="ignore").select_dtypes(include=[np.number])
            if feat_df.empty:
                raise ValueError("数据集中不包含数值列，无法加载模型。")
            feature_names = feat_df.columns.tolist()

        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(feat_df.values.astype(float))

        # ---- 在加载阶段也需构造与训练完成后一致的运行时状态 ----
        # 避免旧状态残留，先重置再写入
        _RuntimeSingleton.reset()
        rt = _RuntimeSingleton.get()

        model_type = meta.get("model_type")
        rt.trained_model = model_obj
        rt.model_type = model_type
        look_back = int(meta.get("params", {}).get("look_back", DATA_CFG.get(model_type, 14)))
        rt.look_back = look_back
        rt.scaler = scaler
        rt.feature_names = feature_names
        rt.data_scaled = data_scaled
        rt.dataset_id = dataset_id

        return {"model": model_obj, "meta": meta}

    def get_advanced_params(self, model_type: str) -> Dict[str, Any]:
        """返回不同模型可调的高级参数默认值"""
        presets = {
            "gru": {"lr": 1e-3, "hidden_size": 32, "num_layers": 2, "dropout": 0.3},
            "tcn": {"lr": 1e-3, "hid": 32, "levels": 2, "k": 2, "drop": 0.2},
            "tsmixer": {"lr": 1e-3, "num_blocks": 4, "ff_dim": 128, "dropout": 0.1},
            "rf": {"n_estimators": 400, "random_state": 42, "n_jobs": -1},
            "xgb": {"n_estimators": 600, "learning_rate": 0.05, "max_depth": 6},
            "timesnet": {"lr": 1e-3, "d_model": 32, "num_blocks": 3},
        }
        return presets.get(model_type, {})


    def get_all_model_name(self):
        MODEL_NAME_MAP = {
            "gru": "GRU",
            "tcn": "TCN",
            "tsmixer": "TSMixer",
            "rf": "随机森林",
            "xgb": "XGBoost",
            "timesnet": "TimesNet"
        }
        # 直接构造筛过的字典
        return {k: v for k, v in MODEL_NAME_MAP.items() if k in MODEL_REGISTRY}