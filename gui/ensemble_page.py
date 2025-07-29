from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QComboBox, QPushButton
)
from PyQt6.QtCore import Qt

from backend import ml_interface as ML
from backend.model_registry import list_all as list_models
from gui.validation_page import ValidationPage


class EnsemblePage(QWidget):
    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)

        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["✔", "名字", "时间", "类型", "备注"])
        layout.addWidget(self.table)

        ctrl = QHBoxLayout()
        self.method_combo = QComboBox()
        self.method_combo.addItems(["平均", "投票"])
        ctrl.addWidget(self.method_combo)
        self.load_btn = QPushButton("加载")
        self.load_btn.clicked.connect(self.on_load)
        ctrl.addWidget(self.load_btn)
        self.open_btn = QPushButton("打开验证页")
        self.open_btn.clicked.connect(self.open_validation)
        ctrl.addWidget(self.open_btn)
        ctrl.addStretch()
        layout.addLayout(ctrl)

        self.validation_page = ValidationPage()
        self.records = []
        self._populate()

    def _populate(self) -> None:
        self.records = list_models()
        self.table.setRowCount(len(self.records))
        for r, rec in enumerate(self.records):
            ck = QTableWidgetItem()
            ck.setCheckState(Qt.CheckState.Unchecked)
            self.table.setItem(r, 0, ck)
            self.table.setItem(r, 1, QTableWidgetItem(rec["name"]))
            self.table.setItem(r, 2, QTableWidgetItem(rec["created_at"]))
            self.table.setItem(r, 3, QTableWidgetItem(rec["meta"].get("model_type", "")))
            self.table.setItem(r, 4, QTableWidgetItem(str(rec["meta"].get("advanced", {}))))

    def on_load(self) -> None:
        paths = [
            self.records[r]["path"]
            for r in range(self.table.rowCount())
            if self.table.item(r, 0).checkState() == Qt.CheckState.Checked
        ]
        method = "mean" if self.method_combo.currentText() == "平均" else "vote"
        info = ML.ML.load_many(paths, method=method)
        self.validation_page.configure(info["features_union"])

    def open_validation(self) -> None:
        self.validation_page.show()
