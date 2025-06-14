from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QStackedLayout
)
from PyQt6.QtCore import Qt
from data_handle import DataHandlePage
from ml_gui import MLWindow

class OutlierDetectionPage(QWidget):
    """Top-level page managing the data handle and ML views."""

    def __init__(self) -> None:
        super().__init__()
        self._step = 0

        layout = QVBoxLayout(self)

        # --- step labels and navigation ---
        top = QHBoxLayout()
        self.labels = []
        for text in ["数据处理", "拟合模型", "验证预测"]:
            lbl = QLabel(text)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.labels.append(lbl)
            top.addWidget(lbl)
        top.addStretch()
        self.prev_btn = QPushButton("上一步")
        self.prev_btn.clicked.connect(self.prev_step)
        self.next_btn = QPushButton("下一步")
        self.next_btn.clicked.connect(self.next_step)
        top.addWidget(self.prev_btn)
        top.addWidget(self.next_btn)
        layout.addLayout(top)

        # --- stacked pages ---
        self.stack = QStackedLayout()
        self.data_page = DataHandlePage()
        self.ml_page = MLWindow()
        self.valid_page = QWidget()
        self.stack.addWidget(self.data_page)
        self.stack.addWidget(self.ml_page)
        self.stack.addWidget(self.valid_page)
        layout.addLayout(self.stack)

        self.update_steps()

    def update_steps(self) -> None:
        for i, lbl in enumerate(self.labels):
            lbl.setProperty("step", "current" if i == self._step else "")
            lbl.style().unpolish(lbl)
            lbl.style().polish(lbl)
        self.prev_btn.setEnabled(self._step > 0)
        self.next_btn.setEnabled(self._step < self.stack.count() - 1)
        self.stack.setCurrentIndex(self._step)
        if self._step == 1:
            self.ml_page.set_data(
                self.data_page.df, self.data_page.selected_columns()
            )

    def next_step(self) -> None:
        if self._step < self.stack.count() - 1:
            self._step += 1
            self.update_steps()

    def prev_step(self) -> None:
        if self._step > 0:
            self._step -= 1
            self.update_steps()