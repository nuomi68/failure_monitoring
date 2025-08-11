from __future__ import annotations

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
    """带有背景色滑动窗口效果的时间序列表格."""

    def __init__(self, headers: Optional[List[str]] = None, look_back: int = 14, parent=None) -> None:
        cfg = SmartTableConfig(default_headers=headers or [])
        super().__init__(cfg)
        self._look_back = look_back
        self._input_rows: List[int] | None = None
        self._pend_row: int | None = None
        self._pred_row: int | None = None

        self.table.itemChanged.connect(self._check_row_complete)
        self.ensure_blank_row()
        self.apply_row_colors()

    # ------------------ 公共接口 ------------------ #
    def dataframe(self) -> pd.DataFrame:
        return super().dataframe()

    def set_headers(self, headers: List[str]) -> None:
        super().set_headers(headers)
        self.init_input_window()

    def set_dataframe(self, df: pd.DataFrame) -> None:
        super().set_dataframe(df)
        self.init_input_window()

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
        if self.table.rowCount() == 0:
            return
        r = self.table.rowCount() - 1
        headers = [self.table.horizontalHeaderItem(c).text() for c in range(self.table.columnCount())]
        for c, col in enumerate(headers):
            val = row.get(col, pd.NA)
            it = self.table.item(r, c)
            if it is None:
                it = QTableWidgetItem()
                self._apply_editable_flag(it, c, None)
                self.table.setItem(r, c, it)
            it.setText("" if pd.isna(val) else str(val))

    # 注册新的预测行并准备下一空行
    def register_new_prediction(self) -> None:
        self._pred_row = self.table.rowCount() - 1
        self.ensure_blank_row(force=True)
        self._pend_row = self.table.rowCount() - 1
        self.apply_row_colors()

    # 初始化窗口，通常在训练或加载模型后调用
    def init_input_window(self) -> None:
        df = self.dataframe()
        if df.empty:
            self._input_rows = []
            self._pend_row = 0
            self._pred_row = None
        else:
            win_end = max(0, len(df) - 2)
            win_start = max(0, win_end - self._look_back + 1)
            self._input_rows = list(range(win_start, win_end + 1))
            self._pend_row = len(df) - 1
            self._pred_row = None
        self.apply_row_colors()

    # ------------------ 内部辅助 ------------------ #
    def _check_row_complete(self, item: QTableWidgetItem) -> None:
        r = item.row()
        if r != self.table.rowCount() - 1:
            return
        cols = self.table.columnCount()
        if all(self._cell_text(r, c).strip() != "" for c in range(cols)):
            self._on_row_filled(r)

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
        rc = self.table.rowCount()
        need_new = force or rc == 0 or all(
            self._cell_text(rc - 1, c).strip() != "" for c in range(self.table.columnCount())
        )
        if need_new:
            self.table.insertRow(rc)
            for c in range(self.table.columnCount()):
                it = QTableWidgetItem("")
                self._apply_editable_flag(it, c, None)
                self.table.setItem(rc, c, it)

    def apply_row_colors(self) -> None:
        for r in range(self.table.rowCount()):
            for c in range(self.table.columnCount()):
                it = self.table.item(r, c)
                if it is not None:
                    it.setBackground(QColor())
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
        for c in range(self.table.columnCount()):
            it = self.table.item(row, c)
            if it is None:
                it = QTableWidgetItem("")
                self._apply_editable_flag(it, c, None)
                self.table.setItem(row, c, it)
            it.setBackground(color)

    # 方便外部调用滚动到底部
    def scrollToBottom(self) -> None:  # pragma: no cover - 简单转调
        self.table.scrollToBottom()
