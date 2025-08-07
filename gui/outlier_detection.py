from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QStackedLayout
)
from PyQt6.QtCore import Qt
from gui.data_handle import DataHandlePage
from gui.unsupervised_page import UnsupervisedPage
from gui.supervised_page import SupervisedPage
from gui.validation_page import ValidationPage

class OutlierDetectionPage(QWidget):
    """Top-level page managing the data handle and ML views."""

    def __init__(self) -> None:
        super().__init__()
        self._step = 0

        layout = QVBoxLayout(self)

        # --- step labels and navigation ---
        top = QHBoxLayout()
        self.labels = []
        for text in ["数据处理", "异常检测", "监督学习", "验证预测"]:
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
        self.unsup_page = UnsupervisedPage()
        self.sup_page = SupervisedPage()
        self.valid_page = ValidationPage()
        self.stack.addWidget(self.data_page)
        self.stack.addWidget(self.unsup_page)
        self.stack.addWidget(self.sup_page)
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
            self.unsup_page.set_data(
                self.data_page.df,
                self.data_page.selected_columns(),
            )
        elif self._step == 2:
            self.sup_page.set_data(
                self.data_page.df,
                self.data_page.selected_columns(),
                target=self.data_page.target_column(),
            )
        elif self._step == 3:
            src = self.sup_page if self.data_page.has_target() else self.unsup_page
            self.valid_page.configure(
                src.selected_columns(),
            )

    def next_step(self) -> None:
        if self._step == 0:
            self._step = 2 if self.data_page.has_target() else 1
        elif self._step in (1, 2):
            self._step = 3
        else:
            return
        self.update_steps()

    def prev_step(self) -> None:
        if self._step == 3:
            self._step = 2 if self.data_page.has_target() else 1
        elif self._step in (1, 2):
            self._step = 0
        else:
            return
        self.update_steps()
