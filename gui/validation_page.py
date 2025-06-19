from typing import List, Any, Dict
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QLineEdit, QLabel,
    QPushButton, QMessageBox, QHBoxLayout
)
from PyQt6.QtCore import Qt

class ValidationPage(QWidget):
    """Simple page to validate a single sample using a trained model."""

    def __init__(self) -> None:
        super().__init__()
        self.model: Any | None = None
        self.scaler: Any | None = None
        self.meta: Dict[str, Any] = {}
        self.inputs: List[QLineEdit] = []

        layout = QVBoxLayout(self)
        self.form = QFormLayout()
        layout.addLayout(self.form)

        btn_row = QHBoxLayout()
        self.predict_btn = QPushButton("计算")
        self.predict_btn.clicked.connect(self.on_predict)
        btn_row.addWidget(self.predict_btn)
        self.result_lbl = QLabel("结果: ")
        btn_row.addWidget(self.result_lbl)
        btn_row.addStretch()
        layout.addLayout(btn_row)

    def configure(self, features: List[str], model: Any, scaler: Any, meta: Dict[str, Any]) -> None:
        """Configure input fields and model."""
        # rebuild form
        while self.form.rowCount():
            self.form.removeRow(0)
        self.inputs = []
        for name in features:
            le = QLineEdit()
            le.setPlaceholderText(name)
            self.form.addRow(QLabel(name+":"), le)
            self.inputs.append(le)
        self.model = model
        self.scaler = scaler
        self.meta = meta or {}

    def on_predict(self) -> None:
        if self.model is None or self.scaler is None:
            QMessageBox.warning(self, "提示", "请先在上一页训练模型")
            return
        try:
            values = [float(le.text()) for le in self.inputs]
        except ValueError:
            QMessageBox.warning(self, "提示", "请输入有效数值")
            return
        X = np.array(values, dtype=np.float32).reshape(1, -1)
        Xs = self.scaler.transform(X)
        mtype = self.meta.get("model_type")
        tau = self.meta.get("tau", 0)
        if mtype == "knn":
            dists, _ = self.model.kneighbors(Xs)
            score = dists[:, -1][0]
        elif mtype == "iforest":
            score = -self.model.decision_function(Xs)[0]
        else:
            QMessageBox.warning(self, "错误", "未知模型类型")
            return
        is_abnormal = score > tau
        self.result_lbl.setText(f"结果: {score:.4f} - {'异常' if is_abnormal else '正常'}")
