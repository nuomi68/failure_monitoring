from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, List, Optional

import pandas as pd

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QTableWidgetItem

from .smart_table import SmartTable, SmartTableConfig

# ========================== 行配色常量 ========================== #
GREEN = QColor(Qt.GlobalColor.green).lighter(160)
BLUE = QColor(Qt.GlobalColor.blue).lighter(160)
PEND = QColor(Qt.GlobalColor.lightGray).lighter(170)


class TimeSeriesTable(SmartTable):
    """继承 SmartTable 的时间序列表格，额外支持按行背景染色。"""

    def __init__(
        self,
        headers: Optional[List[str]] = None,
        look_back: int = 14,
        parent=None,
    ) -> None:
        cfg = SmartTableConfig(
            show_label_selector=False,
            show_toolbar=False,
            min_rows=0,
            default_headers=headers or [],
        )
        # 需要在父类初始化前准备内部状态，避免 set_headers 调用时属性不存在
        self._look_back = look_back
        self._input_rows: List[int] | None = None
        self._pend_row: int | None = None
        self._pred_row: int | None = None
        self._row_colors: Dict[int, QColor] = {}
        self._updating = False
        super().__init__(cfg)
        if parent is not None:
            self.setParent(parent)

        self.table.itemChanged.connect(self._on_item_changed_local)

        self.ensure_blank_row()
        self.apply_row_colors()

    # ------------------ 公共接口 ------------------ #
    def dataframe(self) -> pd.DataFrame:  # type: ignore[override]
        return super().dataframe()

    def set_headers(self, headers: List[str]) -> None:  # type: ignore[override]
        super().set_headers(headers)
        with self.no_record():
            self.table.setRowCount(0)
        self.ensure_blank_row(force=True)
        self.apply_row_colors()

    def set_dataframe(self, df: pd.DataFrame) -> None:  # type: ignore[override]
        with self.no_record():
            super().set_dataframe(df)
        self.ensure_blank_row()
        self.apply_row_colors()

    def set_look_back(self, n: int) -> None:
        self._look_back = int(n)
        self.init_input_window()

    # 预测前调用，移动窗口
    def advance_window_before_predict(self) -> None:
        if self._pred_row is not None and self._input_rows is not None:
            self._input_rows.append(self._pred_row)
            while len(self._input_rows) > self._look_back:
                self._input_rows.pop(0)
            self._pred_row = None
            self.apply_row_colors()

    # 将预测结果写入最后一行
    def fill_last_row(self, row: Dict[str, float]) -> None:
        if self.table.columnCount() == 0:
            return
        r = self.table.rowCount() - 1
        with self._silent_update():
            for c, col in enumerate(self.dataframe().columns):
                val = row.get(col, pd.NA)
                item = self.table.item(r, c)
                if item is None:
                    item = QTableWidgetItem()
                    self.table.setItem(r, c, item)
                item.setText("" if pd.isna(val) else str(val))

    # 注册新的预测行并准备下一空行
    def register_new_prediction(self) -> None:
        self._pred_row = self.table.rowCount() - 1
        self.ensure_blank_row(force=True)
        self._pend_row = self.table.rowCount() - 1
        self.apply_row_colors()

    # 初始化窗口，通常在训练或加载模型后调用
    def init_input_window(self) -> None:
        row_cnt = self.table.rowCount()
        if row_cnt <= 1:
            self._input_rows = []
            self._pend_row = 0
            self._pred_row = None
        else:
            win_end = max(0, row_cnt - 2)
            win_start = max(0, win_end - self._look_back + 1)
            self._input_rows = list(range(win_start, win_end + 1))
            self._pend_row = row_cnt - 1
            self._pred_row = None
        self.apply_row_colors()

    # ------------------ 内部辅助 ------------------ #
    def _row_filled(self, r: int) -> bool:
        for c in range(self.table.columnCount()):
            item = self.table.item(r, c)
            if item is None or item.text().strip() == "":
                return False
        return True

    def _on_item_changed_local(self, item: QTableWidgetItem) -> None:
        if self._updating:
            return
        if item.row() == self.table.rowCount() - 1 and self._row_filled(item.row()):
            self._on_row_filled(item.row())

    def _on_row_filled(self, row: int) -> None:
        if row != self._pend_row or self._input_rows is None:
            return
        self._input_rows.append(row)
        while len(self._input_rows) > self._look_back:
            self._input_rows.pop(0)
        self.ensure_blank_row()
        self._pend_row = self.table.rowCount() - 1
        self.apply_row_colors()

    def ensure_blank_row(self, force: bool = False) -> None:
        cols = self.table.columnCount()
        if cols == 0:
            return
        r = self.table.rowCount()
        need_new = force or r == 0 or self._row_filled(r - 1)
        if need_new:
            with self._silent_update():
                self.table.insertRow(r)

    def apply_row_colors(self) -> None:
        for r in list(self._row_colors):
            self._set_row_color(r, QColor())
        self._row_colors.clear()
        if self._input_rows:
            for r in self._input_rows:
                self._set_row_color(r, GREEN)
        if self._pend_row is not None:
            self._set_row_color(self._pend_row, PEND)
        if self._pred_row is not None:
            self._set_row_color(self._pred_row, BLUE)

    def _set_row_color(self, row: int, color: QColor) -> None:
        if row < 0 or row >= self.table.rowCount():
            return
        with self._silent_update():
            for c in range(self.table.columnCount()):
                item = self.table.item(row, c)
                if item is None:
                    item = QTableWidgetItem("")
                    self.table.setItem(row, c, item)
                item.setBackground(color)
        if color.isValid():
            self._row_colors[row] = color
        elif row in self._row_colors:
            del self._row_colors[row]

    @contextmanager
    def _silent_update(self):
        with self.no_record():
            self._updating = True
            try:
                yield
            finally:
                self._updating = False

    # 方便外部调用滚动到底部
    def scrollToBottom(self) -> None:  # pragma: no cover - 简单转调
        self.table.scrollToBottom()

