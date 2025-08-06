
# -*- coding: utf-8 -*-
"""
主要功能：
   - 左侧显示最终清洗后的数据表（开启横向滚动条、列宽随内容自适应）。
   - 右侧为模型控制区：选择模型、参数(JSON)、训练/加载/保存模型。
   - 与后端 ModelManager 兼容：仍然支持 manager.train(path, fmt) 的旧接口；
     若后端实现了 list_models/set_model/set_params/set_time_column/set_time_format/load_model/save_model 等，
     会自动调用，无则忽略。

注意：
- 弹窗中强制“缺失清零”后才能返回主界面，因此主界面的表格不再需要缺失提示。
"""

from __future__ import annotations

import json
import os
from typing import Optional, Any, Dict, Set

import pandas as pd

from PyQt6.QtCore import Qt, QAbstractTableModel, QModelIndex, QVariant
from PyQt6.QtGui import QBrush
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGridLayout,
    QPushButton, QLabel, QFileDialog, QTextEdit, QLineEdit, QTableView,
    QHeaderView, QComboBox, QDialog, QDialogButtonBox, QMessageBox, QGroupBox,
    QCheckBox
)


# ====== 引入后端 ======
from backend.timeseries_interface import ModelManager
from data_load_dialog import DataLoadDialog,DataFrameModel


class TimeSeriesPage(QWidget):
    """主页面：左侧数据表 + 右侧模型控制。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.manager = ModelManager()

        self._df: Optional[pd.DataFrame] = None           # 清洗后的数据
        self._data_path: Optional[str] = None              # 源文件路径（可供后端使用）
        self._time_col: Optional[str] = None               # 选择的时间列名
        self._time_fmt: str = "%Y年%m月%d日%H%M"            # 时间格式

        # ---------- 左侧：数据表（主界面也加横向滚动） ----------
        self.table = QTableView()
        hh = self.table.horizontalHeader()
        # 主界面改为“按内容自适应 + 横向滚动”，避免列过多时被挤压
        hh.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        hh.setStretchLastSection(False)
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.table.setAlternatingRowColors(True)

        # ---------- 顶部工具栏 ----------
        btn_load = QPushButton("加载数据...")
        btn_load.clicked.connect(self._open_load_dialog)

        btn_template = QPushButton("导出模板")
        btn_template.clicked.connect(self._save_template)

        self.status_label = QLabel("未训练")
        self.status_label.setMinimumWidth(120)

        # ---------- 右侧：模型控制区 ----------
        model_box = QGroupBox("模型与参数")
        model_layout = QFormLayout()

        # 模型选择：若后端提供 list_models() 则填充；否则允许手动输入
        self.model_combo = QComboBox()
        model_names = []
        if hasattr(self.manager, "list_models"):
            try:
                model_names = list(getattr(self.manager, "list_models")())
            except Exception:
                model_names = []
        if not model_names:
            self.model_combo.setEditable(True)
            self.model_combo.setPlaceholderText("模型名称（可留空使用默认）")
        else:
            self.model_combo.addItems(model_names)
        model_layout.addRow(QLabel("选择模型"), self.model_combo)

        # 参数 JSON
        self.param_edit = QTextEdit()
        self.param_edit.setPlaceholderText('可选：输入参数 JSON，例如 {"learning_rate": 0.01, "max_depth": 6}')
        self.param_edit.setFixedHeight(90)
        model_layout.addRow(QLabel("参数(JSON)"), self.param_edit)

        # 时间设置显示（只读，来源于弹窗选择）
        self.time_info = QLabel("时间列：-   格式：-")
        model_layout.addRow(QLabel("时间设置"), self.time_info)

        # 按钮：训练/加载/保存模型
        self.btn_train = QPushButton("训练模型")
        self.btn_train.clicked.connect(self._train)

        self.btn_load_model = QPushButton("加载已训练模型...")
        self.btn_load_model.clicked.connect(self._load_model)
        if not hasattr(self.manager, "load_model"):
            self.btn_load_model.setEnabled(False)

        self.btn_save_model = QPushButton("保存当前模型...")
        self.btn_save_model.clicked.connect(self._save_model)
        if not hasattr(self.manager, "save_model"):
            self.btn_save_model.setEnabled(False)

        model_layout.addRow(self.btn_train)
        model_layout.addRow(self.btn_load_model)
        model_layout.addRow(self.btn_save_model)
        model_box.setLayout(model_layout)

        # 训练日志/结果
        self.result_view = QTextEdit()
        self.result_view.setReadOnly(True)

        # ---------- 总体布局 ----------
        top_bar = QHBoxLayout()
        top_bar.addWidget(btn_load)
        top_bar.addWidget(btn_template)
        top_bar.addStretch(1)
        top_bar.addWidget(self.status_label)

        left = QVBoxLayout()
        left.addLayout(top_bar)
        left.addWidget(QLabel("数据预览"))
        left.addWidget(self.table, stretch=1)
        left.addWidget(QLabel("训练日志 / 结果"))
        left.addWidget(self.result_view, stretch=1)

        right = QVBoxLayout()
        right.addWidget(model_box)
        right.addStretch(1)

        container = QHBoxLayout(self)
        container.addLayout(left, stretch=3)
        container.addLayout(right, stretch=2)

    # ---------- 业务逻辑 ----------

    def _open_load_dialog(self):
        """打开“加载数据”弹窗；只有清洗完成的数据才会放入主界面表格。"""
        dlg = DataLoadDialog(self, default_time_fmt=self._time_fmt)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            df = dlg.loaded_dataframe()
            if df is None:
                return

            # 弹窗保证“缺失清零”，直接载入主界面
            self._df = df
            self._data_path = dlg.file_path()
            self._time_col = dlg.time_column()
            self._time_fmt = dlg.time_format()

            self.time_info.setText(f"时间列：{self._time_col or '-'}   格式：{self._time_fmt or '-'}")

            # 放入主界面的表格
            model = DataFrameModel(self._df)
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
            # 告知后端（若实现了对应方法）
            if self._time_col and hasattr(self.manager, "set_time_column"):
                try:
                    getattr(self.manager, "set_time_column")(self._time_col)
                except Exception:
                    pass

            if hasattr(self.manager, "set_time_format"):
                try:
                    getattr(self.manager, "set_time_format")(self._time_fmt)
                except Exception:
                    pass

    def _parse_params(self) -> Optional[Dict[str, Any]]:
        """解析参数 JSON（为空则返回 {}）。"""
        text = self.param_edit.toPlainText().strip()
        if not text:
            return {}
        try:
            params = json.loads(text)
            if not isinstance(params, dict):
                raise ValueError("参数 JSON 须为对象（形如 {\"key\": value}）。")
            return params
        except Exception as e:
            QMessageBox.warning(self, "参数错误", f"解析参数失败：{e}")
            return None

    def _apply_model_and_params(self) -> bool:
        """将模型名与参数（若提供）下发给后端（若后端实现了相关方法）。"""
        model_name = self.model_combo.currentText().strip()
        if model_name and hasattr(self.manager, "set_model"):
            try:
                getattr(self.manager, "set_model")(model_name)
            except Exception as e:
                QMessageBox.warning(self, "提示", f"设置模型失败：{e}")
                return False

        params = self._parse_params()
        if params is None:
            return False
        if params and hasattr(self.manager, "set_params"):
            try:
                getattr(self.manager, "set_params")(params)
            except Exception as e:
                QMessageBox.warning(self, "提示", f"设置参数失败：{e}")
                return False
        return True

    def _train(self):
        """训练入口：要求先完成数据加载与清洗。"""
        if self._df is None:
            QMessageBox.warning(self, "提示", "请先加载并清洗数据。")
            return
        if not self._apply_model_and_params():
            return

        self.status_label.setText("训练中...")
        self.result_view.clear()

        fmt = self._time_fmt or None
        try:
            # 兼容旧接口（你当前后端就是这样使用的）
            self.manager.train(self._data_path, fmt)
        except Exception as e:
            self.status_label.setText("训练失败")
            QMessageBox.critical(self, "训练失败", f"{e}")
            return

        # 结果/状态展示（尽量容错）
        try:
            self.status_label.setText(getattr(self.manager, "status", "训练完成"))
            lines = []
            last_preds = getattr(self.manager, "last_predictions", None)
            if last_preds is not None:
                for res in last_preds:
                    step = res.get("step", "?")
                    msg = f"第 {step} 步"
                    if "max_err" in res:
                        msg += f": max_err={float(pd.Series(res['max_err']).max()):.4f}"
                    if "mean_err" in res:
                        msg += f", mean_err={float(pd.Series(res['mean_err']).mean()):.4f}"
                    lines.append(msg)
            self.result_view.setPlainText("\n".join(lines) if lines else "训练完成。")
        except Exception:
            self.result_view.setPlainText("训练完成。")

    def _load_model(self):
        """加载已训练模型（若后端实现）。"""
        if not hasattr(self.manager, "load_model"):
            return
        path, _ = QFileDialog.getOpenFileName(self, "选择模型文件", "", "All Files (*)")
        if not path:
            return
        try:
            getattr(self.manager, "load_model")(path)
            QMessageBox.information(self, "成功", "模型加载完成。")
        except Exception as e:
            QMessageBox.critical(self, "失败", f"加载模型失败：{e}")

    def _save_model(self):
        """保存当前模型（若后端实现）。"""
        if not hasattr(self.manager, "save_model"):
            return
        path, _ = QFileDialog.getSaveFileName(self, "保存模型为", "model.bin", "All Files (*)")
        if not path:
            return
        try:
            getattr(self.manager, "save_model")(path)
            QMessageBox.information(self, "成功", "模型保存完成。")
        except Exception as e:
            QMessageBox.critical(self, "失败", f"保存模型失败：{e}")

    def _save_template(self):
        """导出一个最小 Excel 模板供参考。"""
        path, _ = QFileDialog.getSaveFileName(
            self, "保存模板", "time_series_template.xlsx", "Excel Files (*.xlsx)"
        )
        if not path:
            return
        df = pd.DataFrame(
            {
                "TIME": ["2024年01月01日0000", "2024年01月01日0100"],
                "value1": [1.0, 1.5],
                "value2": [2.0, 2.5],
            }
        )
        try:
            df.to_excel(path, index=False)
        except Exception as e:
            QMessageBox.critical(self, "失败", f"保存模板失败：{e}")
            return
        QMessageBox.information(self, "成功", f"已保存模板：{path}")
