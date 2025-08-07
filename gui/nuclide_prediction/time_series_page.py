
# -*- coding: utf-8 -*-
"""
合并版前端（最终）：
- 弹窗数据清洗（沿用你的外部 data_load_dialog.DataLoadDialog，不改接口）。
- 训练/保存交互采用 time_series_page2 的单模型流程（只针对单一模型，不再一次性训练全部）。
- 后端统一通过 backend.timeseries_interface.ModelManager：
    * register_dataset(df, time_col, time_format) -> manifest{dataset_id,...}
    * list_models() / get_model_meta(model_id) / load_dataset(dataset_id)
    * train(dataset_id, model_type, params) -> {model, metrics, extra?}
    * save_model(model_id|None, name, model_type, dataset_id, params, model_obj, metrics) -> model_id

注意：
- 本前端不再依赖 controller.run_all_models；controller 仅供你对照后端实现参考。
- 主界面左侧表格开启“按内容自适应 + 横向滚动”，便于查看宽表。
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTableView,
    QHeaderView, QComboBox, QDialog, QMessageBox, QInputDialog
)

# —— 后端 ——
from backend.timeseries_interface import ModelManager
from gui.tools import logger

from .data_load_dialog import DataLoadDialog, DataFrameModel
from .param_panel import ParamPanel

class TimeSeriesPage(QWidget):
    """主页面：左侧数据表 + 右侧模型控制（单模型工作流）。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.manager = ModelManager()

        # 清洗后的 DataFrame 与时间设定
        self._df: Optional[pd.DataFrame] = None
        self._time_col: Optional[str] = None
        self._time_fmt: str = "%Y年%m月%d日%H%M"

        # 注册后的数据集 ID（训练/保存均依赖它）
        self._dataset_id: Optional[str] = None

        # 最近一次训练产物（保存时使用）
        self._trained_model: Optional[Any] = None
        self._trained_metrics: Optional[Dict[str, Any]] = None
        self._trained_params: Optional[Dict[str, Any]] = None
        self._trained_model_type: Optional[str] = None

        # 当前选中（或刚保存）的模型 ID（用于“加载已有模型”时）
        self._model_id: Optional[str] = None

        # ================= 左侧：数据表格 =================
        self.table = QTableView()
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        hh.setStretchLastSection(False)
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        left = QVBoxLayout()
        left.addWidget(QLabel("数据预览"))
        left.addWidget(self.table, stretch=1)

        # ================= 右侧：控制面板 =================
        right = QVBoxLayout()

        # 数据集状态与“加载数据”按钮
        self.dataset_label = QLabel("未加载数据集")
        right.addWidget(self.dataset_label)
        btn_load = QPushButton("加载数据…")
        btn_load.clicked.connect(self._open_load_dialog)
        right.addWidget(btn_load)

        # 已保存模型下拉（从注册表读取）
        right.addWidget(QLabel("选择已保存模型"))
        self.model_list_combo = QComboBox()
        self.model_list_combo.currentIndexChanged.connect(self._on_model_selected)
        right.addWidget(self.model_list_combo)
        self._refresh_model_list()

        # 模型类型
        right.addWidget(QLabel("模型类型"))
        self.model_type_combo = QComboBox()
        # 后端可选模型
        self.model_type_combo.addItems(["gru", "tcn", "tsmixer", "rf", "xgb", "timesnet"])
        right.addWidget(self.model_type_combo)

        # 训练参数
        right.addWidget(QLabel("训练参数"))
        self.param_panel = ParamPanel(self.manager, self)
        right.addWidget(self.param_panel)

        # 训练/保存
        self.btn_train = QPushButton("训练模型")
        self.btn_train.clicked.connect(self._on_train)
        right.addWidget(self.btn_train)

        self.btn_save = QPushButton("保存模型")
        self.btn_save.clicked.connect(self._on_save_model)
        right.addWidget(self.btn_save)

        # 状态与结果
        self.status_label = QLabel("")
        right.addWidget(self.status_label)

        # 总体布局
        root = QHBoxLayout(self)
        root.addLayout(left, stretch=3)
        root.addLayout(right, stretch=2)

    # ============================================================
    # 数据加载（外部弹窗 → 本地注册数据集）
    # ============================================================
    def _open_load_dialog(self):
        """弹出清洗弹窗；确认后注册数据集并在左侧显示。"""
        dlg = DataLoadDialog(self, default_time_fmt=self._time_fmt)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        df = dlg.loaded_dataframe()
        if df is None:
            QMessageBox.warning(self, "提示", "未获取到清洗后的数据。")
            return

        self._time_col = dlg.time_column()
        self._time_fmt = dlg.time_format()

        # 左侧展示
        self._df = df.copy()
        model =DataFrameModel(self._df)
        self.table.setModel(model)
        if self._time_col is not None:
            for col in range(model.columnCount()):
                hh = self.table.horizontalHeader()
                header_text = model.headerData(col, Qt.Orientation.Horizontal)
                if header_text == self._time_col:
                    hh.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
                    self.table.resizeColumnsToContents()
                    hh.setSectionResizeMode(col, QHeaderView.ResizeMode.Interactive)
                    break
        # 通过 ModelManager 注册数据集，拿到 dataset_id
        try:
            manifest = self.manager.register_dataset(df, self._time_col or "", self._time_fmt or "")
        except Exception as exc:
            QMessageBox.critical(self, "错误", f"注册数据集失败：{exc}")
            return
        self._dataset_id = manifest.get("dataset_id")
        self.dataset_label.setText(f"数据集：{self._dataset_id}（行 {manifest.get('n_rows')}, 列 {manifest.get('n_cols')}）")

        # 提前把时间信息传给后端（若有对应方法）
        if hasattr(self.manager, "set_time_column") and self._time_col:
            try:
                self.manager.set_time_column(self._time_col)
            except Exception:
                pass
        if hasattr(self.manager, "set_time_format") and self._time_fmt:
            try:
                self.manager.set_time_format(self._time_fmt)
            except Exception:
                pass

    # ============================================================
    # 已保存模型列表 & 选择
    # ============================================================
    def _refresh_model_list(self):
        """从注册表刷新模型列表。"""
        self.model_list_combo.blockSignals(True)
        self.model_list_combo.clear()
        self.model_list_combo.addItem("(无)")
        try:
            for meta in self.manager.list_models():
                text = f"{meta.get('name', meta.get('model_id'))} ({meta.get('model_id')})"
                self.model_list_combo.addItem(text, userData=meta.get("model_id"))
        except Exception:
            pass
        self.model_list_combo.blockSignals(False)

    def _on_model_selected(self):
        """选择已保存模型 → 自动加载对应数据集与参数。"""
        idx = self.model_list_combo.currentIndex()
        if idx <= 0:
            return
        model_id = self.model_list_combo.currentData()
        if not model_id:
            return

        try:
            meta = self.manager.get_model_meta(model_id) or {}
        except Exception as exc:
            QMessageBox.warning(self, "提示", f"读取模型元数据失败：{exc}")
            return

        # 载入数据集
        dataset_id = meta.get("dataset_id")
        if dataset_id:
            try:
                df = self.manager.load_dataset(dataset_id)
                self._df = df
                self._dataset_id = dataset_id
                self.table.setModel(DataFrameModel(df))
                self.dataset_label.setText(f"数据集：{dataset_id}（来自模型）")
            except Exception as exc:
                QMessageBox.warning(self, "提示", f"加载模型关联数据集失败：{exc}")

        # 回填模型类型 & 参数
        model_type = meta.get("model_type", "")
        if model_type and self.model_type_combo.findText(model_type) == -1:
            self.model_type_combo.addItem(model_type)
        if model_type:
            self.model_type_combo.setCurrentText(model_type)

        params = meta.get("params", {})
        self.param_panel.set_params(params)

        metrics = meta.get("metrics", {})
        for line in [f"{k}: {v}" for k, v in metrics.items()]:
            logger.info(line)
        self.status_label.setText(f"已加载模型 {model_id}")
        self._model_id = model_id

        # 清空未保存的训练缓存
        self._trained_model = None
        self._trained_metrics = None
        self._trained_params = None
        self._trained_model_type = None

    # ============================================================
    # 训练
    # ============================================================
    def _on_train(self):
        """使用当前数据集与参数训练单一模型。"""
        if not self._dataset_id:
            QMessageBox.warning(self, "提示", "请先通过弹窗加载并清洗数据集。")
            return

        model_type = self.model_type_combo.currentText().strip()
        params = self.param_panel.params()

        self.status_label.setText("训练中…")

        try:
            result = self.manager.train(
                self._dataset_id, model_type, params, log_callback=lambda m: logger.info(m)
            )
        except Exception as exc:
            QMessageBox.critical(self, "错误", f"训练失败：{exc}")
            self.status_label.setText("训练失败")
            return

        # 缓存训练结果用于保存
        self._trained_model = result.get("model")
        self._trained_metrics = result.get("metrics") or {}
        self._trained_params = params
        self._trained_model_type = model_type

        # 展示指标与额外信息
        lines = [f"{k}: {v}" for k, v in self._trained_metrics.items()]
        extra = result.get("extra", {})
        if extra:
            lines.append("预测值: " + str(extra.get("prediction", {})))
            lines.append("真实值: " + str(extra.get("last_true", {})))
        for line in lines:
            logger.info(line)
        self.status_label.setText("训练完成（未保存）")

        # 新训练的模型，清空已选模型 ID
        self._model_id = None

    # ============================================================
    # 保存
    # ============================================================
    def _on_save_model(self):
        """保存当前训练结果为新模型，并刷新模型列表。"""
        if self._trained_model is None or self._trained_metrics is None:
            QMessageBox.warning(self, "提示", "请先训练模型，再保存。")
            return
        if not self._dataset_id:
            QMessageBox.warning(self, "提示", "缺少数据集 ID，无法保存模型。")
            return

        name, ok = QInputDialog.getText(self, "模型名称", "请输入模型名称：")
        if not ok or not name.strip():
            return

        try:
            model_id = self.manager.save_model(
                model_id=None,
                name=name.strip(),
                model_type=self._trained_model_type or self.model_type_combo.currentText(),
                dataset_id=self._dataset_id,
                params=self._trained_params or {},
                model_obj=self._trained_model,
                metrics=self._trained_metrics or {},
            )
        except Exception as exc:
            QMessageBox.critical(self, "错误", f"保存模型失败：{exc}")
            return

        self.status_label.setText(f"模型已保存：{model_id}")
        QMessageBox.information(self, "成功", f"模型已保存为 {model_id}")
        self._refresh_model_list()
