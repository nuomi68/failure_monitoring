from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional, Dict

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QGroupBox, QCheckBox,
    QPushButton, QMessageBox
)

from backend.ml_interface import ML
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
        self.ts_loaded = False

        self.fault_manager = FaultModelManager()
        self.fault_model: Optional[FaultLevelEstimator] = None

        self.ml_manager = MLModelManager()
        self.ml_loaded = False

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
    # 左侧块构建
    # ------------------------------------------------------------------
    def _build_ts_block(self) -> QGroupBox:
        g = QGroupBox("时间序列模型")
        lay = QVBoxLayout(g)
        self.chk_ts = QCheckBox("启用")
        lay.addWidget(self.chk_ts)

        self.ts_controls = QWidget()
        row = QHBoxLayout(self.ts_controls)
        self.btn_ts_load = QPushButton("选择模型…")
        self.btn_ts_clear = QPushButton("清空")
        row.addWidget(self.btn_ts_load)
        row.addWidget(self.btn_ts_clear)
        lay.addWidget(self.ts_controls)

        self.lab_ts = QLabel("未加载")
        lay.addWidget(self.lab_ts)

        self.ts_controls.hide()
        self.lab_ts.hide()

        self.btn_ts_load.clicked.connect(self._open_ts_dialog)
        self.btn_ts_clear.clicked.connect(self._on_clear_ts)
        return g

    def _build_ml_block(self) -> QGroupBox:
        g = QGroupBox("通用 ML 模型")
        lay = QVBoxLayout(g)
        self.chk_ml = QCheckBox("启用")
        self.btn_ml_load = QPushButton("选择模型…")
        self.btn_ml_clear = QPushButton("清空")
        self.lab_ml = QLabel("未加载")
        row = QHBoxLayout(); row.addWidget(self.chk_ml); row.addStretch(1)
        lay.addLayout(row)
        row2 = QHBoxLayout(); row2.addWidget(self.btn_ml_load); row2.addWidget(self.btn_ml_clear)
        lay.addLayout(row2)
        lay.addWidget(self.lab_ml)

        self.btn_ml_load.clicked.connect(self._open_ml_dialog)
        self.btn_ml_clear.clicked.connect(self._on_clear_ml)
        return g

    def _build_fault_block(self) -> QGroupBox:
        g = QGroupBox("故障等级模型")
        lay = QVBoxLayout(g)
        self.chk_fault = QCheckBox("启用")
        self.btn_fault_load = QPushButton("选择模型…")
        self.btn_fault_clear = QPushButton("清空")
        self.lab_fault = QLabel("未加载")
        row = QHBoxLayout(); row.addWidget(self.chk_fault); row.addStretch(1)
        lay.addLayout(row)
        row2 = QHBoxLayout(); row2.addWidget(self.btn_fault_load); row2.addWidget(self.btn_fault_clear)
        lay.addLayout(row2)
        lay.addWidget(self.lab_fault)

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
        self.lab_ts.setVisible(checked)
        self.ts_input_wrap.setVisible(checked)

    # ------------------------------------------------------------------
    # 时间序列加载/清空
    # ------------------------------------------------------------------
    def _open_ts_dialog(self) -> None:
        dlg = ModelManagerDialog(self.ts_manager, self)
        dlg.model_loaded.connect(self._load_ts_by_id)
        dlg.exec()

    def _load_ts_by_id(self, model_id: str) -> None:
        try:
            self.ts_manager.load_model(model_id)
            self.lab_ts.setText(model_id)
            self.chk_ts.setChecked(True)
            self.ts_loaded = True
        except Exception as e:
            QMessageBox.critical(self, "加载失败", str(e))

    def _on_clear_ts(self) -> None:
        from backend.timeseries_interface import _RuntimeSingleton
        _RuntimeSingleton.reset()
        self.ts_manager = ModelManager()
        self.lab_ts.setText("未加载")
        self.chk_ts.setChecked(False)
        self.ts_loaded = False

    # ------------------------------------------------------------------
    # ML 加载/清空
    # ------------------------------------------------------------------
    def _open_ml_dialog(self) -> None:
        dlg = ModelManagerDialog(self.ml_manager, self, multi_select=True)
        dlg.models_loaded.connect(self._load_ml_by_ids)
        dlg.exec()

    def _load_ml_by_ids(self, model_ids: list[str]) -> None:
        try:
            self.ml_manager.load_models(model_ids)
            self.lab_ml.setText(" | ".join(model_ids))
            self.chk_ml.setChecked(True)
            self.ml_loaded = True
        except Exception as e:
            QMessageBox.critical(self, "加载失败", str(e))

    def _on_clear_ml(self) -> None:
        try:
            ML.clear()
        except Exception:
            pass
        self.lab_ml.setText("未加载")
        self.chk_ml.setChecked(False)
        self.ml_loaded = False

    # ------------------------------------------------------------------
    # 故障等级加载/清空
    # ------------------------------------------------------------------
    def _open_fault_dialog(self) -> None:
        dlg = ModelManagerDialog(self.fault_manager, self)
        dlg.model_loaded.connect(self._load_fault_by_id)
        dlg.exec()

    def _load_fault_by_id(self, model_id: str) -> None:
        try:
            est, _, _ = self.fault_manager.load_model(model_id)
            self.fault_model = est
            self.lab_fault.setText(model_id)
            self.chk_fault.setChecked(True)
        except Exception as e:
            QMessageBox.critical(self, "加载失败", str(e))

    def _on_clear_fault(self) -> None:
        self.fault_model = None
        self.lab_fault.setText("未加载")
        self.chk_fault.setChecked(False)

    # ------------------------------------------------------------------
    # 时间序列预测追加到底表
    # ------------------------------------------------------------------
    def _on_ts_generate(self) -> None:
        if not (self.chk_ts.isChecked() and self.ts_loaded):
            QMessageBox.information(self, "提示", "请先加载并启用时间序列模型。")
            return
        df_ts = self.tbl_ts.dataframe()
        if df_ts.empty:
            QMessageBox.information(self, "提示", "时间序列表为空。")
            return
        df_ts = df_ts.apply(pd.to_numeric, errors="coerce").dropna()
        if df_ts.empty:
            QMessageBox.information(self, "提示", "时间序列表无有效数值行。")
            return
        feats = self.ts_manager.current_feature_names() or []
        miss = [f for f in feats if f not in df_ts.columns]
        if miss:
            QMessageBox.warning(self, "特征缺失", f"缺少列: {miss}")
            return
        try:
            self.ts_manager.append_observations(df_ts[feats])
            pred = self.ts_manager.predict(steps=1)[0]["table"].T
            rename = {c: (c if c.startswith("ts_pred") else f"ts_pred_{c}") for c in pred.columns}
            pred_df = pred.rename(columns=rename)
        except Exception as e:
            QMessageBox.critical(self, "预测失败", str(e))
            return

        df_common = self.tbl_common.dataframe()
        df_out = df_common.copy()
        for c in pred_df.columns:
            df_out[c] = np.asarray(pred_df[c]).ravel()[: len(df_out)]
        with self.tbl_common.no_record():
            self.tbl_common.set_dataframe(df_out, record_state=False)
        QMessageBox.information(self, "完成", f"已追加列：{list(pred_df.columns)}")

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

        if self.chk_ml.isChecked() and self.ml_loaded:
            try:
                X_table = {c: X_valid[c].to_numpy() for c in X_valid.columns}
                ret = ML.predict(X_table)
                if isinstance(ret, dict) and "labels" in ret:
                    result_cols["ml_pred"] = np.asarray(ret["labels"]).ravel()
                elif isinstance(ret, dict):
                    for t, sub in ret.items():
                        if isinstance(sub, dict) and "labels" in sub:
                            result_cols[f"ml_pred_{t}"] = np.asarray(sub["labels"]).ravel()
            except Exception as e:
                QMessageBox.warning(self, "ML 预测失败", str(e))

        if self.chk_fault.isChecked() and self.fault_model is not None:
            try:
                feats = self.fault_model.feature_names
                miss = [f for f in feats if f not in X_valid.columns]
                if miss:
                    QMessageBox.warning(self, "特征缺失", f"故障模型所需列缺失：{miss}")
                arr = np.stack([
                    X_valid.get(c, pd.Series([np.nan] * len(X_valid))).to_numpy() for c in feats
                ], axis=1)
                preds = self.fault_model.predict(arr)
                result_cols["fault_pred"] = np.asarray(preds).ravel()
            except Exception as e:
                QMessageBox.warning(self, "故障预测失败", str(e))

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
        drop = [c for c in df.columns if c.startswith("ts_pred_") or c.startswith("ml_pred") or c == "fault_pred"]
        if not drop:
            return
        df2 = df.drop(columns=drop)
        with self.tbl_common.no_record():
            self.tbl_common.set_dataframe(df2, record_state=False)
