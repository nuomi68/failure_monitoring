import sys
from pathlib import Path
from typing import List, Any

import numpy as np
import pandas as pd
from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QPushButton,
    QListWidget,
    QListWidgetItem,
    QSplitter,
    QHBoxLayout,
    QMessageBox,
    QLabel,
    QComboBox,
    QSpinBox,
    QDoubleSpinBox,
    QTextEdit,
    QFileDialog,
)
from PyQt6.QtCore import Qt

from tools import scale_features, save_model, plot_scores
from model import train_knn, train_iforest


class MLWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ML Frontend")
        self.resize(600, 400)

        self.df: pd.DataFrame | None = None
        self.model = None
        self.scaler = None
        self.meta: dict[str, Any] = {}
        self.scores: np.ndarray | None = None

        layout = QVBoxLayout()
        self.setLayout(layout)

        btn_layout = QHBoxLayout()

        self.alg_combo = QComboBox()
        self.alg_combo.addItems(["knn", "iforest"])
        btn_layout.addWidget(QLabel("Algorithm:"))
        btn_layout.addWidget(self.alg_combo)

        self.k_spin = QSpinBox()
        self.k_spin.setValue(5)
        btn_layout.addWidget(QLabel("k / estimators:"))
        btn_layout.addWidget(self.k_spin)

        self.contam_spin = QDoubleSpinBox()
        self.contam_spin.setDecimals(3)
        self.contam_spin.setSingleStep(0.001)
        self.contam_spin.setRange(0.0, 1.0)
        self.contam_spin.setValue(0.01)
        btn_layout.addWidget(QLabel("contam:"))
        btn_layout.addWidget(self.contam_spin)

        self.train_btn = QPushButton("Train")
        self.train_btn.clicked.connect(self.train_model)
        btn_layout.addWidget(self.train_btn)

        self.save_btn = QPushButton("Save Model")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self.save_model)
        btn_layout.addWidget(self.save_btn)

        layout.addLayout(btn_layout)

        body_split = QSplitter(Qt.Orientation.Horizontal)
        self.column_list = QListWidget()
        body_split.addWidget(self.column_list)

        body_split.setStretchFactor(0, 1)
        body_split.setStretchFactor(1, 1)
        layout.addWidget(body_split)

    def set_data(self, df: pd.DataFrame, columns: List[str]) -> None:
        """Inject dataframe and populate column list."""
        self.df = df
        for col in df.columns:
            item = QListWidgetItem(col)
            state = Qt.CheckState.Checked if col in columns else Qt.CheckState.Unchecked
            item.setCheckState(state)
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
            QMessageBox.warning(self, "Warning", "No data provided")
            return
        cols = self.selected_columns()
        if not cols:
            QMessageBox.warning(self, "Warning", "Please select columns")
            return

        X = self.df[cols].values.astype(np.float32)
        X_scaled, self.scaler = scale_features(X)

        alg = self.alg_combo.currentText()
        if alg == "knn":
            self.model, tau = train_knn(X_scaled, k=self.k_spin.value())
            self.meta = {"model_type": "knn", "tau": tau}
            self.scores = self.model.kneighbors(X_scaled)[0][:, -1]
        else:
            self.model, tau = train_iforest(
                X_scaled,
                n_estimators=self.k_spin.value(),
                contamination=self.contam_spin.value(),
            )
            self.meta = {"model_type": "iforest", "tau": tau}
            self.scores = -self.model.decision_function(X_scaled)

        self.save_btn.setEnabled(True)

        plot_scores(
            range(len(self.scores)),
            self.scores,
            threshold=self.meta.get("tau"),
            title="Training Scores",
        )

    def save_model(self) -> None:
        if self.model is None:
            return
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Model", "", "Joblib Files (*.joblib)")
        if not file_name:
            return
        save_model(Path(file_name), self.model, self.scaler, self.meta)


def main():
    app = QApplication(sys.argv)
    win = MLWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()