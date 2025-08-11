from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QTableView,
    QHeaderView,
)

from .data_load_dialog import DataFrameModel
from .smart_table import _CellEditorDelegate

# ========================== 行配色常量 ========================== #
GREEN = QColor(Qt.GlobalColor.green).lighter(160)
BLUE = QColor(Qt.GlobalColor.blue).lighter(160)
PEND = QColor(Qt.GlobalColor.lightGray).lighter(170)


class _SmartColorDelegate(_CellEditorDelegate):
    """在填充背景色的同时继承 SmartTable 的编辑器样式."""

    def paint(self, painter, option, index):  # type: ignore[override]
        bg = index.data(Qt.ItemDataRole.BackgroundRole)
        if bg:
            painter.fillRect(option.rect, bg)
        super().paint(painter, option, index)


class TimeSeriesTable(QWidget):
    """带有背景色滑动窗口效果的时间序列表格."""

    def __init__(self, headers: Optional[List[str]] = None, look_back: int = 14, parent=None) -> None:
        super().__init__(parent)
        self._look_back = look_back
        self._input_rows: List[int] | None = None
        self._pend_row: int | None = None
        self._pred_row: int | None = None

        self.table = QTableView(self)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        hh.setStretchLastSection(False)
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.table)

        self._df = pd.DataFrame(columns=headers or [])
        self._model = DataFrameModel(self._df)
        self._model.row_filled_sig.connect(self._on_row_filled)
        self.table.setModel(self._model)
        # 使用 SmartTable 的编辑样式，同时保持完整背景色渲染
        self.table.setItemDelegate(_SmartColorDelegate(self.table))
        self.ensure_blank_row()
        self.apply_row_colors()

    # ------------------ 公共接口 ------------------ #
    def dataframe(self) -> pd.DataFrame:
        return self._model.dataframe()

    def set_headers(self, headers: List[str]) -> None:
        self.set_dataframe(pd.DataFrame(columns=headers))

    def set_dataframe(self, df: pd.DataFrame) -> None:
        self._df = df.copy()
        self._model.setDataFrame(self._df)
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
        if self._df.empty:
            return
        for col in self._df.columns:
            val = row.get(col, pd.NA)
            self._df.iloc[-1, self._df.columns.get_loc(col)] = val
        self._model.setDataFrame(self._df)

    # 注册新的预测行并准备下一空行
    def register_new_prediction(self) -> None:
        self._pred_row = len(self._df) - 1
        self.ensure_blank_row(force=True)
        self._pend_row = len(self._df) - 1
        self.apply_row_colors()

    # 初始化窗口，通常在训练或加载模型后调用
    def init_input_window(self) -> None:
        df = self._df
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
    def _on_row_filled(self, row: int) -> None:
        if row != self._pend_row or self._input_rows is None:
            return
        self._input_rows.append(row)
        while len(self._input_rows) > self._look_back:
            self._input_rows.pop(0)
        self.ensure_blank_row()
        self._pend_row = len(self._df) - 1
        self.apply_row_colors()

    def ensure_blank_row(self, force: bool = False) -> None:
        if self._df is None:
            return
        need_new = force or self._df.empty or self._df.iloc[-1].notna().all()
        if need_new:
            self._df = self._df.reset_index(drop=True)
            new_len = len(self._df) + 1
            self._df = self._df.reindex(range(new_len))
            self._model.setDataFrame(self._df)

    def apply_row_colors(self) -> None:
        self._model.clear_row_colors()
        if self._input_rows:
            for r in self._input_rows:
                self._model.set_row_color(r, GREEN)
        if self._pend_row is not None:
            self._model.set_row_color(self._pend_row, PEND)
        if self._pred_row is not None:
            self._model.set_row_color(self._pred_row, BLUE)

    # 方便外部调用滚动到底部
    def scrollToBottom(self) -> None:  # pragma: no cover - 简单转调
        self.table.scrollToBottom()
