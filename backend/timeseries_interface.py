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
import  torch
import numpy as np
import pandas as pd

# 新增：运行时单例，保存当前训练得到的模型、缩放器等
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
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
    trained_models: Dict[str, Any] = field(default_factory=dict)  # {model_name: model_obj}
    look_back_map: Dict[str, int] = field(default_factory=dict)  # {model_name: look_back}
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
        # 确定基础目录：当前文件位于 backend 子目录下，因此父目录的父目录为项目根
        project_root = Path(__file__).resolve().parents[1]
        self.artifacts_dir = project_root / "artifacts"
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
        """持久化保存清洗后的数据集并返回数据集清单。

        :param df: 清洗好的 ``DataFrame``；时间列应当已经解析为 ``datetime`` 或正确的字符串格式。
        :param time_col: 时间列的列名。
        :param time_format: 时间列的字符串格式，用于后续解析。
        :returns: 数据集清单字典，包含数据集 ID、保存路径、时间列信息、行列数等元数据。
        """
        # 生成唯一的数据集 ID，使用时间戳和短 UUID 组合
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        uid = uuid.uuid4().hex[:8]
        dataset_id = f"{timestamp}-DATA-{uid}"

        dataset_dir = self.datasets_dir / dataset_id
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # 保存数据为 parquet 格式（适合存储表格数据）
        data_path = dataset_dir / "data.parquet"
        df.to_parquet(data_path, index=False)

        # 计算一些元信息
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

        # 写入清单文件
        manifest_path = dataset_dir / "dataset_manifest.json"
        self._write_json(manifest, manifest_path)

        # 更新全局注册表并持久化
        self.datasets_registry[dataset_id] = manifest
        self._write_json(self.datasets_registry, self.datasets_registry_file)

        return manifest

    def load_dataset(self, dataset_id: str) -> pd.DataFrame:
        """根据数据集 ID 载入数据表。

        :param dataset_id: 已注册的数据集 ID。
        :returns: 载入的数据 ``DataFrame``。
        :raises KeyError: 当数据集 ID 不存在时抛出异常。
        """
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

    def train(self, dataset_id: str, model_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        使用 controller 中真实模型对指定数据集进行训练，并把模型放入单例 RuntimeStore。
        """
        # 1) 校验并载入数据
        if dataset_id not in self.datasets_registry:
            raise KeyError(f"未找到数据集 {dataset_id}")
        df = self.load_dataset(dataset_id)
        time_col = self.datasets_registry[dataset_id]["time_col"]

        # 2) 只用数值特征（去掉时间列）
        feat_df = df.drop(columns=[time_col], errors="ignore").select_dtypes(include=[np.number])
        if feat_df.empty:
            raise ValueError("数据集中不包含数值列，无法训练模型。")

        feature_names = feat_df.columns.tolist()

        # 3) 标准化
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(feat_df.values.astype(float))

        # 4) 构造窗口
        default_lb = DATA_CFG.get(model_type, 14)  # controller 里给的默认窗口长度
        train_params = params.copy()
        look_back = int(train_params.pop("look_back", default_lb))
        holdout = int(train_params.pop("holdout", 5))  # 留出最后 N 条不参与窗口构造，默认 5（与 controller 用法一致）
        batch_size = int(train_params.pop("batch_size", 16))
        epochs = int(train_params.pop("epochs", 50))
        X, y = build_windows(data_scaled[:-holdout], look_back)  # 与 controller 的写法对齐
        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

        # 5) 训练
        if model_type not in MODEL_REGISTRY:
            raise KeyError(f"未知的模型类型: {model_type}")
        train_fn = MODEL_REGISTRY[model_type]["train"]
        if model_type in {"rf", "xgb"}:
            result = train_fn(X_tr, y_tr, X_val, y_val, **train_params)
        else:
            result = train_fn(
                X_tr,
                y_tr,
                X_val,
                y_val,
                batch_size=batch_size,
                epochs=epochs,
                **train_params,
            )

        # controller 的返回有两种：直接模型 或 (模型, metric) 元组
        if isinstance(result, tuple):
            model_obj, metric = result
            metrics = {"val_metric": float(metric)}
        else:
            model_obj = result
            metrics = {}

        # 6) 写入运行时单例
        rt = _RuntimeSingleton.get()
        rt.trained_models[model_type] = model_obj
        rt.look_back_map[model_type] = look_back
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
        if not rt.trained_models:
            raise RuntimeError("没有已训练的模型，请先调用 train。")
        if rt.data_scaled is None or rt.scaler is None or rt.feature_names is None:
            raise RuntimeError("运行时状态不完整，缺少 scaler/feature_names/data_scaled。")

        results = []
        for i in range(1, int(steps) + 1):
            step_out = {}
            # 针对每个已训练模型做一步预测
            for name, model_obj in rt.trained_models.items():
                look_back = rt.look_back_map.get(name, 14)
                seq = rt.data_scaled[-look_back - i: -i]  # 与 controller 的窗口取法一致
                pred = MODEL_REGISTRY[name]["predict"](model_obj, seq)
                pred_inv = rt.scaler.inverse_transform(pred.reshape(1, -1)).squeeze()
                step_out[name] = pd.Series(pred_inv, index=rt.feature_names)

            # 可选：与 controller 同样的 eval 集成（tsmixer & timesnet 均存在时）
            if use_ensemble and ("tsmixer" in step_out) and ("timesnet" in step_out):
                step_out["eval"] = 0.5 * (step_out["tsmixer"].values + step_out["timesnet"].values)
                step_out["eval"] = pd.Series(step_out["eval"], index=rt.feature_names)

            # 组织结果为 DataFrame（列是模型名，行为各特征）
            df = pd.DataFrame({k: v for k, v in step_out.items()}, index=rt.feature_names)
            results.append({"step": i, "table": df})
        return results

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
        # 如果没有提供 ID，则创建一个新的
        is_new = model_id is None
        if is_new:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            uid = uuid.uuid4().hex[:6]
            model_id = f"{timestamp}-{model_type.upper()}-{uid}"

        model_dir = self.models_dir / model_id
        model_dir.mkdir(parents=True, exist_ok=True)

        # 保存模型对象（示例将字典保存为 JSON）
        model_path = model_dir / "model.json"
        with open(model_path, "w", encoding="utf-8") as f:
            json.dump(model_obj, f, ensure_ascii=False, indent=2)

        # 保存参数和指标
        config_path = model_dir / "config.json"
        metrics_path = model_dir / "metrics.json"
        self._write_json(params, config_path)
        self._write_json(metrics, metrics_path)

        # 更新模型注册表
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
        if model_id not in self.models_registry:
            raise KeyError(f"未找到模型 {model_id}")
        meta = self.models_registry[model_id]
        model_path = self.artifacts_dir / meta["artifacts"]["model_path"]
        with open(model_path, "r", encoding="utf-8") as f:
            model_obj = json.load(f)
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
