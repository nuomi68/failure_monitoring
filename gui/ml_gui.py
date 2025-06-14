import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QPushButton,
    QFileDialog,
    QListWidget,
    QListWidgetItem,
    QHBoxLayout,
    QMessageBox,
    QLabel,
    QComboBox,
    QSpinBox,
)
from PyQt6.QtCore import Qt

from data_loader import load_dataframe
from tools import scale_features, logger
from model import train_knn, train_iforest


class MLWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ML Frontend")
        self.resize(600, 400)

        self.file_path: Path | None = None
        self.df: pd.DataFrame | None = None

        layout = QVBoxLayout()
        self.setLayout(layout)

        btn_layout = QHBoxLayout()
        self.load_btn = QPushButton("Load File")
        self.load_btn.clicked.connect(self.load_file)
        btn_layout.addWidget(self.load_btn)

        self.alg_combo = QComboBox()
        self.alg_combo.addItems(["knn", "iforest"])
        btn_layout.addWidget(QLabel("Algorithm:"))
        btn_layout.addWidget(self.alg_combo)

        self.k_spin = QSpinBox()
        self.k_spin.setValue(5)
        btn_layout.addWidget(QLabel("k / estimators:"))
        btn_layout.addWidget(self.k_spin)

        self.train_btn = QPushButton("Train")
        self.train_btn.clicked.connect(self.train_model)
        btn_layout.addWidget(self.train_btn)

        layout.addLayout(btn_layout)

        self.column_list = QListWidget()
        layout.addWidget(self.column_list)

    def load_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Data", "", "Data Files (*.csv *.xlsx *.xls)")
        if not file_name:
            return
        try:
            self.df = load_dataframe(Path(file_name))
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to load file: {exc}")
            logger.exception("load failed")
            return
        self.file_path = Path(file_name)
        self.column_list.clear()
        for col in self.df.columns:
            item = QListWidgetItem(col)
            item.setCheckState(Qt.CheckState.Unchecked)
            self.column_list.addItem(item)

    def selected_columns(self) -> List[str]:
        cols: List[str] = []
        for i in range(self.column_list.count()):
            item = self.column_list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                cols.append(item.text())
        return cols

    def train_model(self):
        if self.df is None:
            QMessageBox.warning(self, "Warning", "Please load data first")
            return
        cols = self.selected_columns()
        if not cols:
            QMessageBox.warning(self, "Warning", "Please select columns")
            return

        X = self.df[cols].values.astype(np.float32)
        X_scaled, _ = scale_features(X)

        alg = self.alg_combo.currentText()
        if alg == "knn":
            _, tau = train_knn(X_scaled, k=self.k_spin.value())
        else:
            _, tau = train_iforest(X_scaled, n_estimators=self.k_spin.value())

        QMessageBox.information(self, "Result", f"Training finished. Tau = {tau:.4f}")
        logger.info("Model trained with tau=%.4f", tau)


def main():
    app = QApplication(sys.argv)
    win = MLWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()