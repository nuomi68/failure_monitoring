from __future__ import annotations

from typing import List

from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QListWidget,
    QAbstractItemView,
    QPushButton,
    QLabel,
)


class FeatureSelectorWidget(QWidget):
    """
    轻量级特征选择器（左右双列表 + 箭头按钮）

    - ``set_columns``: 设置可选列（默认全选到右侧）
    - ``set_selected``: 回填已选列
    - ``selected``: 当前选择结果

    ``selectionChanged`` 信号在选择变化时发出，参数为 ``list[str]``。
    """

    selectionChanged = pyqtSignal(list)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._suppress_signal = False

        root = QVBoxLayout(self)

        row = QHBoxLayout()
        root.addLayout(row)

        # 左：全部特征
        left_col = QVBoxLayout()
        left_col.addWidget(QLabel("全部特征"))
        self.list_all = QListWidget()
        self.list_all.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        left_col.addWidget(self.list_all)
        row.addLayout(left_col, 1)

        # 中间：箭头按钮
        mid = QVBoxLayout()
        mid.addWidget(QLabel(" "))
        mid.addStretch(1)
        for text, slot in [
            ("→", self._move_one_right),
            ("←", self._move_one_left),
            ("≫", self._move_all_right),
            ("≪", self._move_all_left),
        ]:
            btn = QPushButton(text)
            btn.setMaximumWidth(40)
            btn.clicked.connect(slot)
            mid.addWidget(btn)
        mid.addStretch(1)
        row.addLayout(mid)

        # 右：已选特征
        right_col = QVBoxLayout()
        right_col.addWidget(QLabel("已选特征"))
        self.list_sel = QListWidget()
        self.list_sel.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        right_col.addWidget(self.list_sel)
        row.addLayout(right_col, 1)

    # ---------------- 公共 API ---------------- #
    def set_columns(self, cols: List[str]) -> None:
        """设置可选列并默认全选到右侧。"""
        self._suppress_signal = True
        try:
            self.list_all.clear()
            self.list_sel.clear()
            for c in cols:
                self.list_sel.addItem(c)
        finally:
            self._suppress_signal = False
            self._emit()

    def set_selected(self, cols: List[str]) -> None:
        """回填选择"""
        all_cols = set(self.columns())
        cols = [c for c in cols if c in all_cols]
        left = [c for c in self.columns() if c not in cols]

        self._suppress_signal = True
        try:
            self.list_all.clear()
            self.list_sel.clear()
            for c in left:
                self.list_all.addItem(c)
            for c in cols:
                self.list_sel.addItem(c)
        finally:
            self._suppress_signal = False
            self._emit()

    def columns(self) -> List[str]:
        """全部候选列（= 左列 + 右列的并集，按当前顺序）"""
        res: List[str] = []
        for w in (self.list_all, self.list_sel):
            for i in range(w.count()):
                res.append(w.item(i).text())
        # 去重保持顺序
        seen: set[str] = set()
        uniq: List[str] = []
        for c in res:
            if c not in seen:
                seen.add(c)
                uniq.append(c)
        return uniq

    def selected(self) -> List[str]:
        """当前已选列"""
        return [self.list_sel.item(i).text() for i in range(self.list_sel.count())]

    # -------------- 内部操作 -------------- #
    def _move_one_right(self) -> None:
        for it in self.list_all.selectedItems():
            self.list_sel.addItem(it.text())
            self.list_all.takeItem(self.list_all.row(it))
        self._emit()

    def _move_one_left(self) -> None:
        for it in self.list_sel.selectedItems():
            self.list_all.addItem(it.text())
            self.list_sel.takeItem(self.list_sel.row(it))
        self._emit()

    def _move_all_right(self) -> None:
        while self.list_all.count():
            it = self.list_all.takeItem(0)
            self.list_sel.addItem(it.text())
        self._emit()

    def _move_all_left(self) -> None:
        while self.list_sel.count():
            it = self.list_sel.takeItem(0)
            self.list_all.addItem(it.text())
        self._emit()

    def _emit(self) -> None:
        if not self._suppress_signal:
            self.selectionChanged.emit(self.selected())

