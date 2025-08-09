from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import pandas as pd
import numpy as np

from PyQt6.QtCore import Qt, pyqtSignal, QPoint, QTimer
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem, QLabel,
    QPushButton, QFileDialog, QComboBox, QMenu
)
from PyQt6.QtGui import QGuiApplication, QKeySequence, QShortcut, QCursor


@dataclass
class SmartTableConfig:
    show_label_selector: bool = False
    show_toolbar: bool = True
    min_rows: int = 5
    max_undo: int = 50
    editable: bool = True
    default_headers: Optional[List[str]] = None


class SmartTable(QWidget):
    dataframeChanged = pyqtSignal(pd.DataFrame)
    schemaChanged = pyqtSignal(list)
    labelColumnChanged = pyqtSignal(str)

    def __init__(self, cfg: SmartTableConfig):
        super().__init__()
        self.cfg = cfg
        self._restoring = False
        self._undo_stack: list[list[list[str]]] = []
        self._redo_stack: list[list[list[str]]] = []
        self._last_edit_cell: Tuple[int, int] | None = None
        self._label_col: Optional[str] = None
        self._features_sink: Optional["SmartTable"] = None

        root = QVBoxLayout(self)

        if self.cfg.show_toolbar:
            tools = QHBoxLayout()
            self.btn_import = QPushButton("导入")
            self.btn_import.clicked.connect(self._import)
            self.btn_export = QPushButton("导出CSV")
            self.btn_export.clicked.connect(self._export_csv)
            self.btn_add = QPushButton("新增行")
            self.btn_add.clicked.connect(self._add_row)
            self.btn_del = QPushButton("删除选中行")
            self.btn_del.clicked.connect(self._del_selected_rows)
            self.btn_clear = QPushButton("清空")
            self.btn_clear.clicked.connect(self._clear_all)
            tools.addWidget(self.btn_import)
            tools.addWidget(self.btn_export)
            tools.addStretch(1)
            tools.addWidget(self.btn_add)
            tools.addWidget(self.btn_del)
            tools.addWidget(self.btn_clear)
            root.addLayout(tools)

        if self.cfg.show_label_selector:
            lab = QHBoxLayout()
            lab.addWidget(QLabel("等级列："))
            self.cb_label = QComboBox()
            self.cb_label.currentIndexChanged.connect(self._on_label_changed)
            lab.addWidget(self.cb_label)
            lab.addStretch(1)
            root.addLayout(lab)

        self.table = QTableWidget()
        root.addWidget(self.table)
        self._init_shortcuts_and_menu()

        headers = self.cfg.default_headers or []
        self.set_headers(headers)
        self._ensure_min_rows()
        self._push_state()

    def set_headers(self, headers: Iterable[str]) -> None:
        headers = [str(h) for h in headers]
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        if self.cfg.show_label_selector:
            self.cb_label.blockSignals(True)
            self.cb_label.clear()
            self.cb_label.addItems(headers)
            self.cb_label.blockSignals(False)
        self.schemaChanged.emit(headers)

    def set_dataframe(self, df: pd.DataFrame, editable: Optional[bool] = None) -> None:
        self._restoring = True
        try:
            headers = [str(c) for c in df.columns]
            self.set_headers(headers)
            self.table.setRowCount(max(len(df), self.cfg.min_rows))
            for r in range(len(df)):
                for c, col in enumerate(df.columns):
                    val = "" if pd.isna(df.iloc[r, c]) else str(df.iloc[r, c])
                    it = QTableWidgetItem(val)
                    self._apply_editable_flag(it, c, editable)
                    self.table.setItem(r, c, it)
        finally:
            self._restoring = False
        self._push_state()
        self.table.resizeColumnsToContents()
        self.dataframeChanged.emit(self.dataframe())
        if self._features_sink is not None:
            self._sync_features_sink_headers()

    def dataframe(self) -> pd.DataFrame:
        headers = [self.table.horizontalHeaderItem(c).text() for c in range(self.table.columnCount())]
        rows: list[dict[str, str]] = []
        for r in range(self.table.rowCount()):
            row = {}
            empty = True
            for c, h in enumerate(headers):
                item = self.table.item(r, c)
                txt = "" if item is None else str(item.text())
                if txt != "":
                    empty = False
                row[h] = txt
            if not empty:
                rows.append(row)
        return pd.DataFrame(rows, columns=headers)

    def dataframe_numeric(self, *, drop_na_rows: bool = True, use_features_only: bool = False) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        df = self.dataframe()
        if use_features_only and self._label_col and self._label_col in df.columns:
            df = df[[c for c in df.columns if c != self._label_col]]
        df_num = df.apply(pd.to_numeric, errors="coerce")
        if not drop_na_rows:
            return df_num, None
        keep = ~df_num.isna().any(axis=1)
        df_num = df_num[keep].reset_index(drop=True)
        return df_num, keep

    def set_label_column(self, name: str) -> None:
        if not self.cfg.show_label_selector:
            return
        idx = self.cb_label.findText(name)
        self.cb_label.setCurrentIndex(idx if idx >= 0 else 0)

    def label_column(self) -> Optional[str]:
        return self._label_col

    def bind_features_sink(self, sink: "SmartTable") -> None:
        self._features_sink = sink
        self._sync_features_sink_headers()

    def _import(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择表格文件", "",
            "表格文件 (*.csv *.xlsx *.xls);;CSV 文件 (*.csv);;Excel 文件 (*.xlsx *.xls);;所有文件 (*)"
        )
        if not path:
            return
        try:
            if path.endswith(".csv"):
                df = pd.read_csv(path)
            elif path.endswith(".xlsx") or path.endswith(".xls"):
                df = pd.read_excel(path)
            else:
                try:
                    df = pd.read_csv(path)
                except Exception:
                    df = pd.read_excel(path)
        except Exception as e:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "读取失败", f"无法读取：\n{path}\n\n{e}")
            return
        self.set_dataframe(df)

    def _export_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "导出到 CSV", "", "CSV Files (*.csv)")
        if not path:
            return
        self.dataframe().to_csv(path, index=False)

    def _add_row(self):
        self._push_state()
        self.table.insertRow(self.table.rowCount())

    def _del_selected_rows(self):
        idxs = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        if not idxs:
            return
        self._push_state()
        for r in idxs:
            self.table.removeRow(r)
        self.dataframeChanged.emit(self.dataframe())

    def _clear_all(self):
        self._push_state()
        self.set_headers([self.table.horizontalHeaderItem(c).text() for c in range(self.table.columnCount())])
        self._ensure_min_rows()
        self._redo_stack.clear()
        self.dataframeChanged.emit(self.dataframe())

    def _init_shortcuts_and_menu(self):
        QShortcut(QKeySequence("Ctrl+V"), self, activated=self._handle_paste)
        QShortcut(QKeySequence("Ctrl+Z"), self, activated=self._undo)
        QShortcut(QKeySequence("Ctrl+Shift+Z"), self, activated=self._redo)
        QShortcut(QKeySequence("Ctrl+Y"), self, activated=self._redo)
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._open_menu)
        self.table.itemChanged.connect(self._on_item_changed)

    def _open_menu(self, pos: QPoint):
        r = self.table.rowAt(pos.y())
        c = self.table.columnAt(pos.x())
        if r < 0:
            return
        m = QMenu(self.table)
        m.addAction("在上方插入行", lambda: self._record_then(self.table.insertRow, r))
        m.addAction("在下方插入行", lambda: self._record_then(self.table.insertRow, r + 1))
        m.addSeparator()
        m.addAction("删除当前行",    lambda: self._record_then(self.table.removeRow, r))
        m.addAction("清空当前行",    lambda: self._record_then(self._clear_row, r))
        if 0 <= c < self.table.columnCount():
            m.addAction("清空当前列", lambda: self._record_then(self._clear_col, c))
        m.exec(QCursor.pos())

    def _record_then(self, fn, *args):
        self._push_state()
        fn(*args)
        self.dataframeChanged.emit(self.dataframe())

    def _clear_row(self, r: int):
        for c in range(self.table.columnCount()):
            self.table.setItem(r, c, QTableWidgetItem(""))

    def _clear_col(self, c: int):
        for r in range(self.table.rowCount()):
            self.table.setItem(r, c, QTableWidgetItem(""))

    def _handle_paste(self):
        text = QGuiApplication.clipboard().text()
        if not text.strip():
            return
        self._push_state()
        rows = [row for row in text.splitlines() if row.strip()]
        start = max(self.table.currentRow(), 0)
        need = start + len(rows)
        if need > self.table.rowCount():
            self.table.setRowCount(need)
        for r_idx, line in enumerate(rows):
            cells = line.split("\t")
            for c_idx, val in enumerate(cells):
                if c_idx >= self.table.columnCount():
                    break
                self.table.setItem(start + r_idx, c_idx, QTableWidgetItem(val.strip()))
        self.dataframeChanged.emit(self.dataframe())

    def _undo(self):
        if len(self._undo_stack) <= 1:
            return
        self._redo_stack.append(self._undo_stack.pop())
        self._restore_state(self._undo_stack[-1])

    def _redo(self):
        if not self._redo_stack:
            return
        state = self._redo_stack.pop()
        self._undo_stack.append(state)
        self._restore_state(state)

    # ---- public undo/redo helpers ----
    def undo(self) -> None:
        """Undo the last operation if possible."""
        self._undo()

    def redo(self) -> None:
        """Redo the last undone operation if possible."""
        self._redo()

    def can_undo(self) -> bool:
        """Return True if there is an undo history."""
        return len(self._undo_stack) > 1

    def can_redo(self) -> bool:
        """Return True if there is a redo history."""
        return bool(self._redo_stack)

    def clear_history(self) -> None:
        """Clear undo/redo stacks and record current snapshot as base."""
        self._undo_stack = [self._snapshot()]
        self._redo_stack.clear()
        self._last_edit_cell = None

    def _push_state(self):
        if self._restoring:
            return
        self._undo_stack.append(self._snapshot())
        if len(self._undo_stack) > self.cfg.max_undo:
            self._undo_stack.pop(0)
        self._redo_stack.clear()
        self._last_edit_cell = None

    def _snapshot(self) -> list[list[str]]:
        return [[self._cell_text(r, c) for c in range(self.table.columnCount())]
                for r in range(self.table.rowCount())]

    def _restore_state(self, state: list[list[str]]):
        self._restoring = True
        try:
            rows = max(len(state), self.cfg.min_rows)
            cols = len(state[0]) if state else self.table.columnCount()
            self.table.setRowCount(rows)
            self.table.setColumnCount(cols)
            for r in range(len(state)):
                for c in range(len(state[r])):
                    it = QTableWidgetItem(state[r][c])
                    self._apply_editable_flag(it, c, None)
                    self.table.setItem(r, c, it)
        finally:
            self._restoring = False
        self.table.resizeColumnsToContents()
        self.dataframeChanged.emit(self.dataframe())

    def _on_item_changed(self, item: QTableWidgetItem):
        if self._restoring or not self.cfg.editable:
            return
        cell = (item.row(), item.column())
        if self._last_edit_cell != cell:
            QTimer.singleShot(0, self._push_state)
            self._last_edit_cell = cell
        self.dataframeChanged.emit(self.dataframe())

    def _cell_text(self, r: int, c: int) -> str:
        it = self.table.item(r, c)
        return "" if it is None else it.text()

    def _apply_editable_flag(self, item: QTableWidgetItem, col: int, editable: Optional[bool]):
        can_edit = self.cfg.editable if editable is None else editable
        if not can_edit:
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)

    def _ensure_min_rows(self):
        if self.table.rowCount() < self.cfg.min_rows:
            self.table.setRowCount(self.cfg.min_rows)

    def _on_label_changed(self, _idx: int):
        self._label_col = self.cb_label.currentText() if self.cfg.show_label_selector else None
        self.labelColumnChanged.emit(self._label_col or "")
        if self._features_sink is not None:
            self._sync_features_sink_headers()

    def _feature_headers(self) -> List[str]:
        headers = [self.table.horizontalHeaderItem(c).text() for c in range(self.table.columnCount())]
        if self._label_col and self._label_col in headers:
            return [h for h in headers if h != self._label_col]
        return headers

    def _sync_features_sink_headers(self):
        if self._features_sink is None:
            return
        self._features_sink.set_headers(self._feature_headers())
