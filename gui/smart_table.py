from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import pandas as pd

from contextlib import contextmanager
from PyQt6.QtCore import Qt, pyqtSignal, QPoint, QEvent
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTableWidget,
    QTableWidgetItem,
    QLabel,
    QPushButton,
    QFileDialog,
    QComboBox,
    QMenu,
    QAbstractItemView,
    QStyledItemDelegate,
    QLineEdit,
)
from PyQt6.QtGui import (
    QGuiApplication,
    QKeySequence,
    QShortcut,
    QCursor,
    QPalette,
    QColor,
)

from gui.data_load_dialog import DataLoadDialog
@dataclass
class SmartTableConfig:
    show_label_selector: bool = False
    show_toolbar: bool = True
    min_rows: int = 5
    max_undo: int = 50
    editable: bool = True
    default_headers: Optional[List[str]] = None
    # 是否使用 DataLoadDialog 作为导入界面
    use_data_load_dialog: bool = True
    # 若使用 DataLoadDialog，是否强制选择时间列
    require_time_column: bool = False
    # DataLoadDialog 默认时间格式
    data_load_default_time_fmt: str = "%Y年%m月%d日%H%M"


class _CellEditorDelegate(QStyledItemDelegate):
    """Delegate customizing editor appearance."""

    def destroyEditor(self, editor, index):
        # Restore the underlying item's foreground when editor is destroyed
        tbl = None
        w = editor.parent()
        while w is not None and getattr(w, 'metaObject', None) is not None:
            if w.metaObject().className() in ('QTableWidget', 'QTableView'):
                tbl = w
                break
            w = w.parent()
        try:
            r, c = editor.property('st_row'), editor.property('st_col')
            prev_brush = editor.property('st_prev_brush')
            if (
                tbl is not None
                and r is not None
                and c is not None
                and prev_brush is not None
            ):
                try:
                    if hasattr(tbl, 'item'):
                        it = tbl.item(int(r), int(c))
                        if it is not None:
                            it.setForeground(prev_brush)
                except Exception:
                    pass
        finally:
            super().destroyEditor(editor, index)

    def createEditor(self, parent, option, index):
        editor = super().createEditor(parent, option, index)
        if isinstance(editor, QLineEdit):
            editor.setFrame(False)
            editor.setStyleSheet(
                "border: none; border-radius: 0; padding: 0; background: palette(base); color: black;"
            )
            pal = editor.palette()
            pal.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
            pal.setColor(QPalette.ColorRole.Highlight, QColor("#c1c1c1"))
            pal.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.white)
            editor.setPalette(pal)
            editor.setAutoFillBackground(True)
            editor.setProperty('st_row', index.row())
            editor.setProperty('st_col', index.column())
            try:
                it = parent.item(index.row(), index.column())
                if it is not None:
                    editor.setProperty('st_prev_brush', it.foreground())
            except Exception:
                editor.setProperty('st_prev_brush', None)
        return editor

    def setEditorData(self, editor, index):
        super().setEditorData(editor, index)
        if isinstance(editor, QLineEdit):
            editor.deselect()
            editor.setCursorPosition(len(editor.text()))


class SmartTable(QWidget):
    dataframeChanged = pyqtSignal(pd.DataFrame)
    schemaChanged = pyqtSignal(list)
    labelColumnChanged = pyqtSignal(str)

    def __init__(self, cfg: SmartTableConfig):
        super().__init__()
        self.cfg = cfg
        self._restoring = False
        self._record_enabled = True
        self._edit_dirty = False
        self._undo_stack: list[list[list[str]]] = []
        self._redo_stack: list[list[list[str]]] = []
        self._label_col: Optional[str] = None
        self._features_sink: Optional["SmartTable"] = None

        root = QVBoxLayout(self)

        if self.cfg.show_toolbar:
            tools_bar = QWidget()
            tools_bar.setObjectName("SmartTableTools")
            tools = QHBoxLayout(tools_bar)
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
            self.btn_undo = QPushButton("撤销")
            self.btn_undo.clicked.connect(self._undo)
            self.btn_redo = QPushButton("重做")
            self.btn_redo.clicked.connect(self._redo)
            tools.addWidget(self.btn_import)
            tools.addWidget(self.btn_export)
            tools.addStretch(1)
            tools.addWidget(self.btn_add)
            tools.addWidget(self.btn_del)
            tools.addWidget(self.btn_clear)
            tools.addWidget(self.btn_undo)
            tools.addWidget(self.btn_redo)
            root.addWidget(tools_bar)

        if self.cfg.show_label_selector:
            lab = QHBoxLayout()
            lab.addWidget(QLabel("等级列："))
            self.cb_label = QComboBox()
            self.cb_label.currentIndexChanged.connect(self._on_label_changed)
            lab.addWidget(self.cb_label)
            lab.addStretch(1)
            root.addLayout(lab)

        self.table = QTableWidget()
        # 允许拖拽框选 / Shift 扩选 / Ctrl 多选
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._delegate = _CellEditorDelegate(self.table)
        self.table.setItemDelegate(self._delegate)
        root.addWidget(self.table)
        self._init_shortcuts_and_menu()
        self.table.itemDelegate().closeEditor.connect(self._on_edit_closed)
        self.table.itemDelegate().commitData.connect(self._on_commit_data)
        self.table.itemChanged.connect(self._on_item_changed)
        self.table.installEventFilter(self)

        headers = self.cfg.default_headers or []
        self.set_headers(headers)
        self._ensure_min_rows()
        self._push_state()
        self._update_undo_redo_buttons()

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

    def set_dataframe(
        self,
        df: pd.DataFrame,
        editable: Optional[bool] = None,
        *,
        record_state: bool = True,
    ) -> None:
        """Fill table with DataFrame contents.

        Parameters
        ----------
        df : pd.DataFrame
            Data to populate the table.
        editable : Optional[bool], default None
            Override the table's editable flag for this operation.
        record_state : bool, default True
            Whether to push an undo snapshot after filling.
        """

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
        if record_state:
            self._push_state()
        self.table.resizeColumnsToContents()
        self.dataframeChanged.emit(self.dataframe())
        self._update_undo_redo_buttons()
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
        if self.cfg.use_data_load_dialog:


            dlg = DataLoadDialog.from_file_dialog(
                self,
                default_time_fmt=self.cfg.data_load_default_time_fmt,
                require_time_column=self.cfg.require_time_column,
            )
            if dlg is None:
                return
            df = dlg.loaded_dataframe()
            if df is None:
                return
        else:
            path, _ = QFileDialog.getOpenFileName(
                self,
                "选择表格文件",
                "",
                "表格文件 (*.csv *.xlsx *.xls);;CSV 文件 (*.csv);;Excel 文件 (*.xlsx *.xls);;所有文件 (*)",
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
        headers = [self.table.horizontalHeaderItem(c).text() for c in range(self.table.columnCount())]
        default_headers = self.cfg.default_headers or []
        if any(h.strip() for h in headers):
            df.columns = [str(c) for c in df.columns]
            if headers != default_headers:
                df = df.reindex(columns=headers)
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
        # 只清内容，不影响表头
        self.table.blockSignals(True)
        try:
            self.table.clearContents()
            self.table.setRowCount(self.cfg.min_rows)  # 强制恢复到最小行数
        finally:
            self.table.blockSignals(False)
        self._redo_stack.clear()
        self.dataframeChanged.emit(self.dataframe())
        self._update_undo_redo_buttons()

    def _init_shortcuts_and_menu(self):
        ctx = Qt.ShortcutContext.WidgetWithChildrenShortcut
        sc = QShortcut(QKeySequence("Ctrl+V"), self.table)
        sc.setContext(ctx)
        sc.activated.connect(self._handle_paste)
        # 复制快捷键
        sc = QShortcut(QKeySequence("Ctrl+C"), self.table)
        sc.setContext(ctx)
        sc.activated.connect(self._handle_copy)
        sc = QShortcut(QKeySequence("Ctrl+Z"), self.table)
        sc.setContext(ctx)
        sc.activated.connect(self._undo)
        sc = QShortcut(QKeySequence("Ctrl+Shift+Z"), self.table)
        sc.setContext(ctx)
        sc.activated.connect(self._redo)
        sc = QShortcut(QKeySequence("Ctrl+Y"), self.table)
        sc.setContext(ctx)
        sc.activated.connect(self._redo)
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._open_menu)

    def _open_menu(self, pos: QPoint):
        r = self.table.rowAt(pos.y())
        c = self.table.columnAt(pos.x())
        if r < 0:
            return
        m = QMenu(self.table)
        m.addAction("在上方插入行", lambda: self._record_then(self.table.insertRow, r))
        m.addAction("在下方插入行", lambda: self._record_then(self.table.insertRow, r + 1))
        m.addSeparator()
        m.addAction("复制", self._handle_copy)
        m.addAction("粘贴", self._handle_paste)
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
        self._update_undo_redo_buttons()

    def _clear_row(self, r: int):
        for c in range(self.table.columnCount()):
            self.table.setItem(r, c, QTableWidgetItem(""))

    def _clear_col(self, c: int):
        for r in range(self.table.rowCount()):
            self.table.setItem(r, c, QTableWidgetItem(""))

    def _handle_copy(self):
        ranges = self.table.selectedRanges()
        if not ranges:
            return
        blocks = []
        for rg in ranges:
            lines = []
            for r in range(rg.topRow(), rg.bottomRow() + 1):
                cells = []
                for c in range(rg.leftColumn(), rg.rightColumn() + 1):
                    it = self.table.item(r, c)
                    cells.append("" if it is None else it.text())
                lines.append("\t".join(cells))
            blocks.append("\n".join(lines))
        QGuiApplication.clipboard().setText("\n".join(blocks))

    def _handle_paste(self):
        text = QGuiApplication.clipboard().text()
        if not text.strip():
            return
        self._push_state()
        rows = [row for row in text.splitlines() if row.strip()]
        # 从当前单元格开始粘贴
        start_row = max(self.table.currentRow(), 0)
        start_col = max(self.table.currentColumn(), 0)
        need = start_row + len(rows)
        if need > self.table.rowCount():
            self.table.setRowCount(need)
        for r_idx, line in enumerate(rows):
            cells = line.split("\t")
            for c_idx, val in enumerate(cells):
                cc = start_col + c_idx
                if cc >= self.table.columnCount():
                    break
                it = self.table.item(start_row + r_idx, cc)
                if it is None:
                    it = QTableWidgetItem("")
                    self._apply_editable_flag(it, cc, None)
                    self.table.setItem(start_row + r_idx, cc, it)
                it.setText(val.strip())
        self.dataframeChanged.emit(self.dataframe())

    def _undo(self):
        if len(self._undo_stack) <= 1:
            return
        self._redo_stack.append(self._undo_stack.pop())
        self._restore_state(self._undo_stack[-1])
        self._update_undo_redo_buttons()

    def _redo(self):
        if not self._redo_stack:
            return
        state = self._redo_stack.pop()
        self._undo_stack.append(state)
        self._restore_state(state)
        self._update_undo_redo_buttons()

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

    def _update_undo_redo_buttons(self):
        if hasattr(self, "btn_undo"):
            self.btn_undo.setEnabled(self.can_undo())
        if hasattr(self, "btn_redo"):
            self.btn_redo.setEnabled(self.can_redo())

    def clear_history(self) -> None:
        """Clear undo/redo stacks and record current snapshot as base."""
        self._undo_stack = [self._snapshot()]
        self._redo_stack.clear()
        self._update_undo_redo_buttons()

    @contextmanager
    def no_record(self):
        """Context manager to temporarily disable undo stack recording."""
        prev = self._record_enabled
        self._record_enabled = False
        try:
            yield
        finally:
            self._record_enabled = prev

    def _push_state(self):
        if self._restoring or not self._record_enabled:
            return
        self._undo_stack.append(self._snapshot())
        if len(self._undo_stack) > self.cfg.max_undo:
            self._undo_stack.pop(0)
        self._redo_stack.clear()
        self._update_undo_redo_buttons()

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
        self._update_undo_redo_buttons()

    def _on_item_changed(self, item: QTableWidgetItem):
        if self._restoring or not self.cfg.editable:
            return
        if self.table.state() == QAbstractItemView.State.EditingState:
            self._edit_dirty = True
        self.dataframeChanged.emit(self.dataframe())

    def _on_commit_data(self, _editor):
        self._edit_dirty = True

    def _on_edit_closed(self, _editor, _hint):
        if self._edit_dirty:
            self._push_state()
            self._edit_dirty = False

    def eventFilter(self, obj, event):
        if obj is self.table and event.type() == QEvent.Type.FocusOut:
            if self._edit_dirty:
                self._push_state()
                self._edit_dirty = False
        return super().eventFilter(obj, event)

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
