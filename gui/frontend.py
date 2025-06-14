# coding: utf-8
"""PyQt6 GUI for machine learning tasks.

This implements a simple interface with three modules:
1. 在线监测 (placeholder)
2. 异常检测 (main implementation)
3. 损伤计算 (placeholder)

The anomaly detection module allows loading a CSV/XLSX file, selecting
columns for training, and running a basic algorithm.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QFileDialog,
    QTableWidget,
    QTableWidgetItem,
    QListWidget,
    QListWidgetItem,
    QStackedWidget,
    QTabWidget,
    QComboBox,
    QSpinBox,
    QTextEdit,
    QMessageBox,
)

# local helpers
from data_loader import load_dataframe
from tools import scale_features, logger
from model import train_knn, train_iforest


class AnomalyDetectionWidget(QWidget):
    """Widget implementing the anomaly detection workflow."""

    def __init__(self) -> None:
        super().__init__()
        self.df: pd.DataFrame | None = None
        self.features: List[str] = []
        self.labels: List[str] = []

        layout = QVBoxLayout(self)
        self.stack = QStackedWidget()
        layout.addWidget(self.stack)

        # -------- Step 1: Load data ---------
        page_load = QWidget()
        l1 = QVBoxLayout(page_load)
        self.btn_load = QPushButton("选择数据文件")
        self.btn_load.clicked.connect(self._load_file)
        l1.addWidget(self.btn_load)
        self.table = QTableWidget()
        l1.addWidget(self.table)
        self.btn_to_select = QPushButton("下一步")
        self.btn_to_select.setEnabled(False)
        self.btn_to_select.clicked.connect(lambda: self.stack.setCurrentIndex(1))
        l1.addWidget(self.btn_to_select, alignment=Qt.AlignmentFlag.AlignRight)
        self.stack.addWidget(page_load)

        # -------- Step 2: Select columns ---------
        page_select = QWidget()
        l2 = QVBoxLayout(page_select)
        l2.addWidget(QLabel("选择特征列"))
        self.list_features = QListWidget()
        l2.addWidget(self.list_features)
        l2.addWidget(QLabel("选择标签列 (可选)"))
        self.list_labels = QListWidget()
        l2.addWidget(self.list_labels)
        btn_row = QHBoxLayout()
        self.btn_back_load = QPushButton("返回")
        self.btn_back_load.clicked.connect(lambda: self.stack.setCurrentIndex(0))
        btn_row.addWidget(self.btn_back_load)
        btn_row.addStretch()
        self.btn_to_train = QPushButton("进入训练")
        self.btn_to_train.clicked.connect(self._to_train)
        btn_row.addWidget(self.btn_to_train)
        l2.addLayout(btn_row)
        self.stack.addWidget(page_select)

        # -------- Step 3: Train ---------
        page_train = QWidget()
        l3 = QVBoxLayout(page_train)
        form_row = QHBoxLayout()
        form_row.addWidget(QLabel("算法:"))
        self.alg_combo = QComboBox()
        self.alg_combo.addItems(["kNN", "IsolationForest"])
        form_row.addWidget(self.alg_combo)
        form_row.addWidget(QLabel("参数:"))
        self.spin_param = QSpinBox()
        self.spin_param.setValue(5)
        form_row.addWidget(self.spin_param)
        l3.addLayout(form_row)
        self.train_btn = QPushButton("开始训练")
        self.train_btn.clicked.connect(self._train)
        l3.addWidget(self.train_btn)
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        l3.addWidget(self.log_edit)
        btn_row2 = QHBoxLayout()
        self.btn_back_select = QPushButton("返回")
        self.btn_back_select.clicked.connect(lambda: self.stack.setCurrentIndex(1))
        btn_row2.addWidget(self.btn_back_select)
        btn_row2.addStretch()
        l3.addLayout(btn_row2)
        self.stack.addWidget(page_train)

    # ------------------ Actions ------------------
    def _load_file(self) -> None:
        file_name, _ = QFileDialog.getOpenFileName(self, "选择数据文件", "", "Data Files (*.csv *.xlsx *.xls)")
        if not file_name:
            return
        try:
            self.df = load_dataframe(Path(file_name))
        except Exception as exc:  # pragma: no cover - GUI feedback
            QMessageBox.critical(self, "加载失败", str(exc))
            logger.exception("failed to load file")
            return
        self._show_preview()
        self.btn_to_select.setEnabled(True)

    def _show_preview(self) -> None:
        assert self.df is not None
        preview = self.df.head(10)
        self.table.setRowCount(len(preview))
        self.table.setColumnCount(len(preview.columns))
        self.table.setHorizontalHeaderLabels(list(preview.columns))
        for r_idx, (_, row) in enumerate(preview.iterrows()):
            for c_idx, value in enumerate(row):
                self.table.setItem(r_idx, c_idx, QTableWidgetItem(str(value)))

    def _to_train(self) -> None:
        if self.df is None:
            return
        self.features = self._checked_items(self.list_features)
        self.labels = self._checked_items(self.list_labels)
        if not self.features:
            QMessageBox.warning(self, "提示", "请选择至少一个特征列")
            return
        self.stack.setCurrentIndex(2)

    def _checked_items(self, widget: QListWidget) -> List[str]:
        res: List[str] = []
        for i in range(widget.count()):
            item = widget.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                res.append(item.text())
        return res

    def _train(self) -> None:
        if self.df is None:
            return
        X = self.df[self.features].values.astype(np.float32)
        X_scaled, _ = scale_features(X)
        alg = self.alg_combo.currentText()
        param = self.spin_param.value()
        self.log_edit.append("开始训练...")
        if alg == "kNN":
            _, tau = train_knn(X_scaled, k=param)
        else:
            _, tau = train_iforest(X_scaled, n_estimators=param)
        self.log_edit.append(f"训练完成，阈值 τ={tau:.4f}")
        logger.info("training finished tau=%.4f", tau)

    def update_columns(self) -> None:
        """Update column check lists after data is loaded."""
        if self.df is None:
            return
        self.list_features.clear()
        self.list_labels.clear()
        for col in self.df.columns:
            item_f = QListWidgetItem(col)
            item_f.setCheckState(Qt.CheckState.Unchecked)
            self.list_features.addItem(item_f)
            item_l = QListWidgetItem(col)
            item_l.setCheckState(Qt.CheckState.Unchecked)
            self.list_labels.addItem(item_l)

    def showEvent(self, event):  # noqa: D401
        super().showEvent(event)
        if self.df is not None and self.list_features.count() == 0:
            self.update_columns()


class MainWindow(QMainWindow):
    """Main application window with three tabs."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ML 前端")
        self.resize(800, 600)
        tabs = QTabWidget()
        self.setCentralWidget(tabs)

        # 在线监测 - placeholder
        placeholder1 = QWidget()
        p1_layout = QVBoxLayout(placeholder1)
        p1_layout.addWidget(QLabel("在线监测模块待实现"))
        tabs.addTab(placeholder1, "在线监测")

        # 异常检测
        self.anom_widget = AnomalyDetectionWidget()
        tabs.addTab(self.anom_widget, "异常检测")

        # 损伤计算 - placeholder
        placeholder2 = QWidget()
        p2_layout = QVBoxLayout(placeholder2)
        p2_layout.addWidget(QLabel("损伤计算模块待实现"))
        tabs.addTab(placeholder2, "损伤计算")


def main() -> None:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":  # pragma: no cover
    main()
