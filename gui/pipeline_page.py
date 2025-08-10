from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional, Dict, List

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QGroupBox,
    QCheckBox,
    QPushButton,
    QMessageBox,
)

from backend.ml_interface import ML, infer_input_features
from backend.ml_model_manager import MLModelManager
from backend.fault_level_estimator import FaultLevelEstimator
from backend.fault_model_manager import FaultModelManager
from backend.timeseries_interface import ModelManager

from gui.smart_table import SmartTable, SmartTableConfig
from gui.nuclide_prediction.model_manager_dialog import ModelManagerDialog


class PipelinePage(QWidget):
    """模型流水线页面：

    - 支持按块启用时间序列、通用 ML 与故障等级模型
    - 仅做模型加载与推理，不涉及训练
    - 若启用了时间序列模块，预测结果会自动追加到底部共用表
    - 下游表可再喂给 ML 或故障等级模块进行最终推理
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("模型流水线")

        # 后端管理器 / 模型状态
        self.ts_manager = ModelManager()
        self.ts_model_ids: List[str] = []
        self.ts_names: Dict[str, str] = {}
        self.ts_features: Dict[str, List[str]] = {}

        self.fault_manager = FaultModelManager()
        self.fault_models: Dict[str, FaultLevelEstimator] = {}
        self.fault_names: Dict[str, str] = {}
        self.fault_features: Dict[str, List[str]] = {}

        self.ml_manager = MLModelManager()
        self.ml_model_ids: List[str] = []
        self.ml_names: Dict[str, str] = {}
        self.ml_features: Dict[str, List[str]] = {}

        # ----- 四宫格布局 -----
        self.grp_ts = self._build_ts_block()
        self.grp_ml = self._build_ml_block()
        self.grp_fault = self._build_fault_block()

        # 时间序列输入表（初始隐藏）
        self.ts_input_wrap = QWidget()
        right_top = QVBoxLayout(self.ts_input_wrap)
        right_top.addWidget(QLabel("时间序列输入"))
        self.tbl_ts = SmartTable(SmartTableConfig(default_headers=["feat1", "feat2"]))
        right_top.addWidget(self.tbl_ts)
        right_top.addWidget(self._make_ts_actions())
        self.ts_input_wrap.hide()

        # 下游共用表（常驻）
        self.common_wrap = QWidget()
        right_bottom = QVBoxLayout(self.common_wrap)
        right_bottom.addWidget(QLabel("下游共用输入"))
        self.tbl_common = SmartTable(SmartTableConfig())
        right_bottom.addWidget(self.tbl_common)
        right_bottom.addWidget(self._make_final_actions())

        # 左下角：ML 与故障块
        self.bottom_left = QWidget()
        left_bottom = QVBoxLayout(self.bottom_left)
        left_bottom.addWidget(self.grp_ml)
        left_bottom.addWidget(self.grp_fault)
        left_bottom.addStretch(1)

        # 根布局 2x2
        root = QGridLayout(self)
        root.addWidget(self.grp_ts, 0, 0)
        root.addWidget(self.ts_input_wrap, 0, 1)
        root.addWidget(self.bottom_left, 1, 0)
        root.addWidget(self.common_wrap, 1, 1)
        root.setColumnStretch(1, 1)
        root.setRowStretch(1, 1)

        # 关联时间序列启用开关
        self.chk_ts.toggled.connect(self._on_ts_toggle)

    # ------------------------------------------------------------------
    # 工具函数
    # ------------------------------------------------------------------
    def _rebuild_model_chips(self, layout: QHBoxLayout, ids: List[str], names: Dict[str, str], remove_cb) -> None:
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        for mid in ids:
            w = QWidget()
            l = QHBoxLayout(w)
            l.setContentsMargins(2, 2, 2, 2)
            l.addWidget(QLabel(names.get(mid, mid)))
            btn = QPushButton("x")
            btn.setFixedSize(16, 16)
            btn.clicked.connect(lambda _, m=mid: remove_cb(m))
            l.addWidget(btn)
            layout.addWidget(w)
        layout.addStretch(1)

    def _set_table_headers(self, tbl: SmartTable, headers: List[str]) -> None:
        df = tbl.dataframe()
        df_new = pd.DataFrame(columns=headers)
        for c in df.columns:
            if c in df_new.columns:
                df_new[c] = df[c]
        with tbl.no_record():
            tbl.set_dataframe(df_new, record_state=False)

    def _required_ts_features(self) -> List[str]:
        feats: set[str] = set()
        for fs in self.ts_features.values():
            feats.update(fs)
        return sorted(feats)

    def _required_common_features(self) -> List[str]:
        feats: set[str] = set()
        for fs in self.ml_features.values():
            feats.update(fs)
        for fs in self.fault_features.values():
            feats.update(fs)
        return sorted(feats)

    def _refresh_ts_models(self) -> None:
        self._rebuild_model_chips(self.ts_model_layout, self.ts_model_ids, self.ts_names, self._remove_ts_model)
        headers = self._required_ts_features()
        if headers:
            self._set_table_headers(self.tbl_ts, headers)
        else:
            self._set_table_headers(self.tbl_ts, [])

    def _refresh_ml_models(self) -> None:
        self._rebuild_model_chips(self.ml_model_layout, self.ml_model_ids, self.ml_names, self._remove_ml_model)
        self._update_common_headers()

    def _refresh_fault_models(self) -> None:
        ids = list(self.fault_models.keys())
        self._rebuild_model_chips(self.fault_model_layout, ids, self.fault_names, self._remove_fault_model)
        self._update_common_headers()

    def _update_common_headers(self) -> None:
        headers = self._required_common_features()
        self._set_table_headers(self.tbl_common, headers)

    # ------------------------------------------------------------------
    # 左侧块构建
    # ------------------------------------------------------------------
    def _build_ts_block(self) -> QGroupBox:
        g = QGroupBox("时间序列模型")
        lay = QVBoxLayout(g)
        self.chk_ts = QCheckBox("启用")
        lay.addWidget(self.chk_ts)

        self.ts_controls = QWidget()
        row = QHBoxLayout(self.ts_controls)
        self.btn_ts_load = QPushButton("加载模型…")
        self.btn_ts_clear = QPushButton("清空")
        row.addWidget(self.btn_ts_load)
        row.addWidget(self.btn_ts_clear)
        lay.addWidget(self.ts_controls)

        self.ts_model_wrap = QWidget()
        self.ts_model_layout = QHBoxLayout(self.ts_model_wrap)
        self.ts_model_layout.addStretch(1)
        lay.addWidget(self.ts_model_wrap)

        self.ts_controls.hide()
        self.ts_model_wrap.hide()

        self.btn_ts_load.clicked.connect(self._open_ts_dialog)
        self.btn_ts_clear.clicked.connect(self._on_clear_ts)
        return g

    def _build_ml_block(self) -> QGroupBox:
        g = QGroupBox("通用 ML 模型")
        lay = QVBoxLayout(g)
        self.chk_ml = QCheckBox("启用")
        self.btn_ml_load = QPushButton("加载模型…")
        self.btn_ml_clear = QPushButton("清空")
        row = QHBoxLayout(); row.addWidget(self.chk_ml); row.addStretch(1)
        lay.addLayout(row)
        row2 = QHBoxLayout(); row2.addWidget(self.btn_ml_load); row2.addWidget(self.btn_ml_clear)
        lay.addLayout(row2)
        self.ml_model_wrap = QWidget()
        self.ml_model_layout = QHBoxLayout(self.ml_model_wrap)
        self.ml_model_layout.addStretch(1)
        lay.addWidget(self.ml_model_wrap)

        self.btn_ml_load.clicked.connect(self._open_ml_dialog)
        self.btn_ml_clear.clicked.connect(self._on_clear_ml)
        return g

    def _build_fault_block(self) -> QGroupBox:
        g = QGroupBox("故障等级模型")
        lay = QVBoxLayout(g)
        self.chk_fault = QCheckBox("启用")
        self.btn_fault_load = QPushButton("加载模型…")
        self.btn_fault_clear = QPushButton("清空")
        row = QHBoxLayout(); row.addWidget(self.chk_fault); row.addStretch(1)
        lay.addLayout(row)
        row2 = QHBoxLayout(); row2.addWidget(self.btn_fault_load); row2.addWidget(self.btn_fault_clear)
        lay.addLayout(row2)
        self.fault_model_wrap = QWidget()
        self.fault_model_layout = QHBoxLayout(self.fault_model_wrap)
        self.fault_model_layout.addStretch(1)
        lay.addWidget(self.fault_model_wrap)

        self.btn_fault_load.clicked.connect(self._open_fault_dialog)
        self.btn_fault_clear.clicked.connect(self._on_clear_fault)
        return g

    # ------------------------------------------------------------------
    # 右侧动作
    # ------------------------------------------------------------------
    def _make_ts_actions(self) -> QWidget:
        box = QWidget(); lay = QHBoxLayout(box)
        self.btn_ts_to_down = QPushButton("生成预测 ➜ 下游")
        self.btn_ts_to_down.clicked.connect(self._on_ts_generate)
        lay.addStretch(1); lay.addWidget(self.btn_ts_to_down)
        return box

    def _make_final_actions(self) -> QWidget:
        box = QWidget(); lay = QHBoxLayout(box)
        self.btn_clear_results = QPushButton("清空结果列")
        self.btn_run = QPushButton("推理")
        self.btn_clear_results.clicked.connect(self._on_clear_results)
        self.btn_run.clicked.connect(self._on_run)
        lay.addStretch(1); lay.addWidget(self.btn_clear_results); lay.addWidget(self.btn_run)
        return box

    # ------------------------------------------------------------------
    # 时间序列启用开关
    # ------------------------------------------------------------------
    def _on_ts_toggle(self, checked: bool) -> None:
        self.ts_controls.setVisible(checked)
        self.ts_model_wrap.setVisible(checked)
        self.ts_input_wrap.setVisible(checked)

    # ------------------------------------------------------------------
    # 时间序列加载/清空
    # ------------------------------------------------------------------
    def _open_ts_dialog(self) -> None:
        dlg = ModelManagerDialog(self.ts_manager, self)
        dlg.model_loaded.connect(self._add_ts_model)
        dlg.exec()

    def _add_ts_model(self, model_id: str) -> None:
        try:
            self.ts_manager.load_model(model_id)
            feats = self.ts_manager.current_feature_names() or []
            name = self.ts_manager.models_registry.get(model_id, {}).get("name", model_id)
            if model_id not in self.ts_model_ids:
                self.ts_model_ids.append(model_id)
            self.ts_features[model_id] = feats
            self.ts_names[model_id] = name
            self.chk_ts.setChecked(True)
            self._refresh_ts_models()
        except Exception as e:
            QMessageBox.critical(self, "加载失败", str(e))

    def _remove_ts_model(self, model_id: str) -> None:
        if model_id in self.ts_model_ids:
            self.ts_model_ids.remove(model_id)
        self.ts_features.pop(model_id, None)
        self.ts_names.pop(model_id, None)
        self._refresh_ts_models()

    def _on_clear_ts(self) -> None:
        from backend.timeseries_interface import _RuntimeSingleton
        _RuntimeSingleton.reset()
        self.ts_manager = ModelManager()
        self.ts_model_ids.clear()
        self.ts_features.clear()
        self.ts_names.clear()
        self._refresh_ts_models()
        self.chk_ts.setChecked(False)

    # ------------------------------------------------------------------
    # ML 加载/清空
    # ------------------------------------------------------------------
    def _open_ml_dialog(self) -> None:
        dlg = ModelManagerDialog(self.ml_manager, self, multi_select=True)
        dlg.models_loaded.connect(self._add_ml_models)
        dlg.exec()

    def _add_ml_models(self, model_ids: list[str]) -> None:
        for mid in model_ids:
            try:
                meta = self.ml_manager.load_models([mid])
                name = self.ml_manager.registry.get(mid, {}).get("name", mid)
                feats = infer_input_features(meta.get("features", []), meta.get("calc_recipes", []))
                if mid not in self.ml_model_ids:
                    self.ml_model_ids.append(mid)
                self.ml_names[mid] = name
                self.ml_features[mid] = feats
            except Exception as e:
                QMessageBox.warning(self, "加载失败", str(e))
        if self.ml_model_ids:
            self.chk_ml.setChecked(True)
        self._refresh_ml_models()

    def _remove_ml_model(self, model_id: str) -> None:
        if model_id in self.ml_model_ids:
            self.ml_model_ids.remove(model_id)
        self.ml_names.pop(model_id, None)
        self.ml_features.pop(model_id, None)
        self._refresh_ml_models()

    def _on_clear_ml(self) -> None:
        try:
            ML.clear()
        except Exception:
            pass
        self.ml_model_ids.clear()
        self.ml_names.clear()
        self.ml_features.clear()
        self._refresh_ml_models()
        self.chk_ml.setChecked(False)

    # ------------------------------------------------------------------
    # 故障等级加载/清空
    # ------------------------------------------------------------------
    def _open_fault_dialog(self) -> None:
        dlg = ModelManagerDialog(self.fault_manager, self)
        dlg.model_loaded.connect(self._add_fault_model)
        dlg.exec()

    def _add_fault_model(self, model_id: str) -> None:
        try:
            est, _, meta = self.fault_manager.load_model(model_id)
            name = meta.get("name", model_id)
            feats = list(est.feature_names)
            self.fault_models[model_id] = est
            self.fault_names[model_id] = name
            self.fault_features[model_id] = feats
            self.chk_fault.setChecked(True)
            self._refresh_fault_models()
        except Exception as e:
            QMessageBox.critical(self, "加载失败", str(e))

    def _remove_fault_model(self, model_id: str) -> None:
        self.fault_models.pop(model_id, None)
        self.fault_names.pop(model_id, None)
        self.fault_features.pop(model_id, None)
        self._refresh_fault_models()

    def _on_clear_fault(self) -> None:
        self.fault_models.clear()
        self.fault_names.clear()
        self.fault_features.clear()
        self._refresh_fault_models()
        self.chk_fault.setChecked(False)

    # ------------------------------------------------------------------
    # 时间序列预测追加到底表
    # ------------------------------------------------------------------
    def _on_ts_generate(self) -> None:
        if not (self.chk_ts.isChecked() and self.ts_model_ids):
            QMessageBox.information(self, "提示", "请先加载并启用时间序列模型。")
            return
        df_ts = self.tbl_ts.dataframe()
        if df_ts.empty:
            QMessageBox.information(self, "提示", "时间序列表为空。")
            return
        df_ts = df_ts.apply(pd.to_numeric, errors="coerce")
        df_ts = df_ts.dropna()
        if df_ts.empty:
            QMessageBox.information(self, "提示", "时间序列表无有效数值行。")
            return

        pred_total = pd.DataFrame()
        for mid in self.ts_model_ids:
            feats = self.ts_features.get(mid, [])
            miss = [f for f in feats if f not in df_ts.columns]
            if miss:
                QMessageBox.warning(self, "特征缺失", f"模型 {self.ts_names.get(mid, mid)} 缺少列: {miss}")
            df_use = df_ts.reindex(columns=feats).dropna()
            if df_use.empty:
                continue
            try:
                self.ts_manager.load_model(mid)
                self.ts_manager.append_observations(df_use)
                pred = self.ts_manager.predict(steps=1)[0]["table"].T
                for c in pred.columns:
                    pred_total[c] = np.asarray(pred[c]).ravel()
            except Exception as e:
                QMessageBox.warning(self, "预测失败", str(e))

        if pred_total.empty:
            QMessageBox.information(self, "提示", "没有生成任何预测。")
            return

        required = set(self._required_common_features())
        miss_req = [f for f in required if f not in pred_total.columns]
        if miss_req:
            QMessageBox.warning(self, "特征缺失", f"未生成特征：{miss_req}")

        headers = list(self.tbl_common.table.horizontalHeaderLabels()) if hasattr(self.tbl_common, "table") else []
        for c in pred_total.columns:
            if c not in headers:
                headers.append(c)
        self._set_table_headers(self.tbl_common, headers)

        df_common = self.tbl_common.dataframe()
        if df_common.empty:
            df_common = pd.DataFrame(index=range(len(pred_total)))
        df_out = df_common.copy()
        for c in pred_total.columns:
            df_out[c] = np.asarray(pred_total[c]).ravel()[: len(df_out)]
        with self.tbl_common.no_record():
            self.tbl_common.set_dataframe(df_out, record_state=False)
        QMessageBox.information(self, "完成", f"已写入列：{list(pred_total.columns)}")

    # ------------------------------------------------------------------
    # 最终推理
    # ------------------------------------------------------------------
    def _on_run(self) -> None:
        df = self.tbl_common.dataframe()
        if df.empty:
            QMessageBox.information(self, "提示", "下游输入表为空。")
            return
        df_num = df.apply(pd.to_numeric, errors="coerce")
        valid = ~df_num.isna().any(axis=1)
        idx = np.where(valid.to_numpy())[0]
        if idx.size == 0:
            QMessageBox.information(self, "提示", "无完整数值行。")
            return
        X_valid = df_num.iloc[idx].reset_index(drop=True)
        result_cols: Dict[str, np.ndarray] = {}

        if self.chk_ml.isChecked() and self.ml_model_ids:
            for mid in self.ml_model_ids:
                feats = self.ml_features.get(mid, [])
                try:
                    self.ml_manager.load_models([mid])
                    X_table = {
                        f: X_valid.get(f, pd.Series([np.nan] * len(X_valid))).to_numpy()
                        for f in feats
                    }
                    ret = ML.predict(X_table)
                    y = np.asarray(ret.get("labels") if isinstance(ret, dict) else ret).ravel()
                    col = f"ml_pred_{self.ml_names.get(mid, mid)}".replace(" ", "_")
                    result_cols[col] = y
                except Exception as e:
                    QMessageBox.warning(self, f"ML 模型 {self.ml_names.get(mid, mid)} 预测失败", str(e))

        if self.chk_fault.isChecked() and self.fault_models:
            for mid, est in self.fault_models.items():
                feats = self.fault_features.get(mid, [])
                miss = [f for f in feats if f not in X_valid.columns]
                if miss:
                    QMessageBox.warning(self, "特征缺失", f"故障模型{self.fault_names.get(mid, mid)}缺失列：{miss}")
                try:
                    arr = np.stack([
                        X_valid.get(c, pd.Series([np.nan] * len(X_valid))).to_numpy()
                        for c in feats
                    ], axis=1)
                    preds = est.predict(arr)
                    col = f"fault_pred_{self.fault_names.get(mid, mid)}".replace(" ", "_")
                    result_cols[col] = np.asarray(preds).ravel()
                except Exception as e:
                    QMessageBox.warning(self, f"故障模型{self.fault_names.get(mid, mid)}预测失败", str(e))

        if not result_cols:
            QMessageBox.information(self, "提示", "没有生成任何结果。")
            return
        df_out = df.copy()
        for name, arr in result_cols.items():
            col = np.full(len(df_out), np.nan)
            col[idx[: len(arr)]] = arr[: len(idx)]
            df_out[name] = col
        with self.tbl_common.no_record():
            self.tbl_common.set_dataframe(df_out, record_state=False)
        QMessageBox.information(self, "完成", f"已写入结果列：{list(result_cols.keys())}")

    def _on_clear_results(self) -> None:
        df = self.tbl_common.dataframe()
        drop = [c for c in df.columns if c.startswith("ml_pred_") or c.startswith("fault_pred_")]
        if not drop:
            return
        df2 = df.drop(columns=drop)
        with self.tbl_common.no_record():
            self.tbl_common.set_dataframe(df2, record_state=False)
