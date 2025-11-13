from __future__ import annotations

from typing import Iterable, List

from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
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
        self._col_order: list[str] = []

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
        """Set available columns and select all by default."""
        cols = self._dedup_preserve_order(cols)
        self._col_order = cols
        self._suppress_signal = True
        try:
            self._update_lists(set(cols))
        finally:
            self._suppress_signal = False
            self._emit()

    def set_selected(self, cols: List[str]) -> None:
        """Restore selected columns."""
        self._ensure_col_order()
        valid = {c for c in cols if c in set(self._col_order)}

        self._suppress_signal = True
        try:
            self._update_lists(valid)
        finally:
            self._suppress_signal = False
            self._emit()

    def columns(self) -> List[str]:
        """All available columns (union of both lists) in stable order."""
        if not self._col_order:
            self._col_order = self._collect_from_lists()
        return list(self._col_order)

    def selected(self) -> List[str]:
        """当前已选列"""
        return [self.list_sel.item(i).text() for i in range(self.list_sel.count())]

    # -------------- 内部操作 -------------- #
    def _move_one_right(self) -> None:
        left_sel, right_sel = self._capture_selection()
        if not left_sel:
            return
        new_selected = set(self.selected()).union(left_sel)
        self._update_lists(
            new_selected,
            left_selected=set(),
            right_selected=right_sel,
        )
        self._emit()

    def _move_one_left(self) -> None:
        left_sel, right_sel = self._capture_selection()
        if not right_sel:
            return
        new_selected = set(self.selected()).difference(right_sel)
        self._update_lists(
            new_selected,
            left_selected=left_sel,
            right_selected=set(),
        )
        self._emit()

    def _move_all_right(self) -> None:
        left_sel, right_sel = self._capture_selection()
        self._ensure_col_order()
        self._update_lists(
            set(self._col_order),
            left_selected=set(),
            right_selected=right_sel,
        )
        self._emit()

    def _move_all_left(self) -> None:
        left_sel, right_sel = self._capture_selection()
        self._update_lists(
            set(),
            left_selected=left_sel,
            right_selected=set(),
        )
        self._emit()

    def _emit(self) -> None:
        if not self._suppress_signal:
            self.selectionChanged.emit(self.selected())

    def _capture_selection(self) -> tuple[set[str], set[str]]:
        left: set[str] = set()
        for i in range(self.list_all.count()):
            item = self.list_all.item(i)
            if item.isSelected():
                left.add(item.text())

        right: set[str] = set()
        for i in range(self.list_sel.count()):
            item = self.list_sel.item(i)
            if item.isSelected():
                right.add(item.text())

        return left, right

    def _update_lists(
        self,
        selected_set: Iterable[str],
        *,
        left_selected: set[str] | None = None,
        right_selected: set[str] | None = None,
    ) -> None:
        self._ensure_col_order()
        col_set = set(self._col_order)
        selected = {c for c in selected_set if c in col_set}
        left_keep = (left_selected or set()) & col_set
        right_keep = (right_selected or set()) & col_set

        self.list_all.clear()
        self.list_sel.clear()
        for col in self._col_order:
            if col in selected:
                item = QListWidgetItem(col)
                if col in right_keep:
                    item.setSelected(True)
                self.list_sel.addItem(item)
            else:
                item = QListWidgetItem(col)
                if col in left_keep:
                    item.setSelected(True)
                self.list_all.addItem(item)

    def _ensure_col_order(self) -> None:
        if not self._col_order:
            self._col_order = self._collect_from_lists()

    def _collect_from_lists(self) -> list[str]:
        res: list[str] = []
        for widget in (self.list_all, self.list_sel):
            for i in range(widget.count()):
                res.append(widget.item(i).text())
        return self._dedup_preserve_order(res)

    @staticmethod
    def _dedup_preserve_order(cols: Iterable[str]) -> list[str]:
        seen: set[str] = set()
        uniq: list[str] = []
        for col in cols:
            if col not in seen:
                seen.add(col)
                uniq.append(col)
        return uniq
