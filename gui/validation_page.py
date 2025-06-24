from __future__ import annotations

from typing import List, Any, Dict, Tuple
import numpy as np
from functools import partial

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem,
    QLabel, QPushButton, QHBoxLayout, QMessageBox, QMenu
)
from PyQt6.QtGui import (
    QGuiApplication, QKeySequence, QShortcut, QCursor
)
from PyQt6.QtCore import Qt, QPoint, QTimer


class ValidationPage(QWidget):
    """
    支持批量验证样本，并提供完备的 **撤销 / 重做**：

    * `Ctrl+Z` 撤销，`Ctrl+Y`/`Ctrl+Shift+Z` 重做
    * 对 **任意单元格增删改**（包括键入、粘贴、行列操作）记录快照
      - 单元格连续输入期间只记录一次快照，避免一键撤回到修改前状态
    * 快照包含整张表（行列结构 + 内容），最大深度由 ``MAX_UNDO`` 控制
    """

    MIN_ROWS = 5  # 初始空行数
    MAX_UNDO = 50   # 撤销栈深度

    def __init__(self) -> None:  # noqa: D401
        super().__init__()

        # -------------------- 数据属性 --------------------
        self.model: Any | None = None
        self.scaler: Any | None = None
        self.meta: Dict[str, Any] = {}
        self.features: List[str] = []

        # 撤销 / 重做
        self._undo_stack: list[list[list[str]]] = []
        self._redo_stack: list[list[list[str]]] = []
        self._restoring = False  # 正在还原栈，避免递归 push
        self._last_edit_cell: Tuple[int, int] | None = None  # (row, col) 用于去重

        # -------------------- UI --------------------
        layout = QVBoxLayout(self)
        self.table = QTableWidget(); layout.addWidget(self.table)

        # 右键菜单
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._open_context_menu)

        # 单元格编辑完成
        self.table.itemChanged.connect(self._on_item_changed)

        # 快捷键
        QShortcut(QKeySequence("Ctrl+V"), self, activated=self._handle_paste)
        QShortcut(QKeySequence("Ctrl+Z"), self, activated=self._undo)
        QShortcut(QKeySequence("Ctrl+Shift+Z"), self, activated=self._redo)
        QShortcut(QKeySequence("Ctrl+Y"), self, activated=self._redo)

        # 底部按钮
        btn_row = QHBoxLayout()
        self.undo_btn = QPushButton("撤销"); self.undo_btn.clicked.connect(self._undo); btn_row.addWidget(self.undo_btn)
        self.redo_btn = QPushButton("重做"); self.redo_btn.clicked.connect(self._redo); btn_row.addWidget(self.redo_btn)
        self.predict_btn = QPushButton("计算"); self.predict_btn.clicked.connect(self.on_predict); btn_row.addWidget(self.predict_btn)
        self.clear_btn = QPushButton("清空全部"); self.clear_btn.clicked.connect(self._clear_all); btn_row.addWidget(self.clear_btn)
        self.result_lbl = QLabel("结果: "); btn_row.addWidget(self.result_lbl); btn_row.addStretch(1)
        layout.addLayout(btn_row)

        self._update_undo_redo_state()

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------
    def configure(self, features: List[str], model: Any, scaler: Any, meta: Dict[str, Any] | None = None) -> None:
        self.features = list(features)
        self.model = model; self.scaler = scaler; self.meta = meta or {}
        self._setup_table()
        self._clear_history(); self._push_state()  # 初始快照

    # ------------------------------------------------------------------
    # 撤销 / 重做 栈
    # ------------------------------------------------------------------
    def _push_state(self) -> None:
        """保存当前快照到撤销栈；自动裁剪深度并重置重做栈。"""
        if self._restoring:
            return
        self._undo_stack.append(self._snapshot())
        if len(self._undo_stack) > self.MAX_UNDO:
            self._undo_stack.pop(0)
        self._redo_stack.clear()
        self._update_undo_redo_state()
        self._last_edit_cell = None  # 重置

    def _undo(self):
        if len(self._undo_stack) <= 1:
            return
        self._redo_stack.append(self._undo_stack.pop())
        self._restore_state(self._undo_stack[-1])
        self._update_undo_redo_state()

    def _redo(self):
        if not self._redo_stack:
            return
        state = self._redo_stack.pop()
        self._undo_stack.append(state)
        self._restore_state(state)
        self._update_undo_redo_state()

    def _clear_history(self):
        self._undo_stack.clear(); self._redo_stack.clear(); self._update_undo_redo_state()

    def _update_undo_redo_state(self):
        self.undo_btn.setEnabled(len(self._undo_stack) > 1)
        self.redo_btn.setEnabled(bool(self._redo_stack))

    # ------------------------------------------------------------------
    # 快照工具
    # ------------------------------------------------------------------
    def _snapshot(self) -> list[list[str]]:
        return [[self._cell_text(r, c) for c in range(self.table.columnCount())] for r in range(self.table.rowCount())]

    def _restore_state(self, state: list[list[str]]):
        self._restoring = True
        try:
            self.table.setRowCount(len(state))
            self.table.setColumnCount(len(state[0]) if state else len(self.features) + 2)
            for r, row in enumerate(state):
                for c, val in enumerate(row):
                    item = QTableWidgetItem(val)
                    if c >= len(self.features):  # “得分” 和 “标签”列设为不可编辑
                        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    self.table.setItem(r, c, item)
        finally:
            self._restoring = False

    # ------------------------------------------------------------------
    # 单元格编辑跟踪
    # ------------------------------------------------------------------
    def _on_item_changed(self, item: QTableWidgetItem):
        if self._restoring:
            return
        # 忽略结果列的修改
        if item.column() >= len(self.features):
            return
        cell = (item.row(), item.column())
        # 若连续编辑同一单元格，不重复 push；编辑焦点移开再记录
        if self._last_edit_cell != cell:
            # 延迟 0ms 以确保撤销时包含最新值（Qt 会在文本提交后再触发 itemChanged）
            QTimer.singleShot(0, self._push_state)
            self._last_edit_cell = cell

    # ------------------------------------------------------------------
    # 右键菜单 & 行列操作
    # ------------------------------------------------------------------
    def _open_context_menu(self, pos: QPoint):
        r = self.table.rowAt(pos.y()); c = self.table.columnAt(pos.x())
        if r < 0:
            return
        menu = QMenu(self.table)
        menu.addAction("在上方插入行", partial(self._record_then, self._insert_row, r))
        menu.addAction("在下方插入行", partial(self._record_then, self._insert_row, r + 1))
        menu.addSeparator()
        menu.addAction("删除当前行",    partial(self._record_then, self._delete_row, r))
        menu.addAction("清空当前行",    partial(self._record_then, self._clear_row, r))
        if 0 <= c < len(self.features):
            menu.addAction("清空当前列", partial(self._record_then, self._clear_column, c))
        menu.exec(QCursor.pos())

    def _record_then(self, fn, *args):
        self._push_state(); fn(*args)

    # 行列基础操作
    def _insert_row(self, idx):
        self.table.insertRow(idx)
    def _delete_row(self, idx):
        self.table.removeRow(idx)
    def _clear_row(self, idx):
        for col in range(self.table.columnCount()):
            self.table.setItem(idx, col, QTableWidgetItem(""))
    def _clear_column(self, col):
        for row in range(self.table.rowCount()):
            self.table.setItem(row, col, QTableWidgetItem(""))

    def _clear_all(self):
        self._push_state()
        self._setup_table()
        self.result_lbl.setText("结果: ")

    # ------------------------------------------------------------------
    # 粘贴操作
    # ------------------------------------------------------------------
    def _handle_paste(self):
        text = QGuiApplication.clipboard().text()
        if not text.strip():
            return
        self._push_state()
        rows = [r for r in text.splitlines() if r.strip()]
        start = max(self.table.currentRow(), 0)
        need = start + len(rows)
        if need > self.table.rowCount():
            self.table.setRowCount(need)
        for r_idx, line in enumerate(rows):
            cells = line.split("\t")
            for c_idx, val in enumerate(cells):
                if c_idx >= len(self.features):
                    break
                self.table.setItem(start + r_idx, c_idx, QTableWidgetItem(val.strip()))
            for c_idx in range(len(self.features),len(self.features)+2):
                item = QTableWidgetItem()
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.table.setItem(start + r_idx, c_idx, item)
    # ------------------------------------------------------------------
    # 预测逻辑（略微整理）
    # ------------------------------------------------------------------
    def on_predict(self):
        if self.model is None or self.scaler is None:
            QMessageBox.warning(self, "提示", "请先在上一页训练模型"); return
        data_rows, row_map = [], []
        for r in range(self.table.rowCount()):
            try:
                vals = [float(self._cell_text(r, c)) for c in range(len(self.features))]
            except ValueError:
                continue
            data_rows.append(vals); row_map.append(r)
        if not data_rows:
            QMessageBox.warning(self, "提示", "请在表格中输入 / 粘贴有效数值"); return
        Xs = self.scaler.transform(np.asarray(data_rows, dtype=np.float32))
        mtype = self.meta.get("model_type"); tau = float(self.meta.get("tau", 0))
        if mtype == "knn":
            scores = self.model.kneighbors(Xs)[0][:, -1]
        elif mtype == "iforest":
            scores = -self.model.decision_function(Xs)
        elif mtype in ("rf", "knn_clf"):
            try:
                scores = self.model.predict_proba(Xs)[:, 1]
            except Exception:
                scores = self.model.predict(Xs).astype(float)
        else:
            QMessageBox.warning(self, "错误", "未知模型类型"); return
        abn = 0
        for idx, row in enumerate(row_map):
            s = scores[idx]; flag = s > tau; abn += flag
            self.table.setItem(row, len(self.features), QTableWidgetItem(f"{s:.4f}"))
            self.table.setItem(row, len(self.features) + 1, QTableWidgetItem("异常" if flag else "正常"))
        self.result_lbl.setText(f"结果: 共 {len(row_map)} 行, 异常 {abn} 行")

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------
    def _setup_table(self):
        self.table.setColumnCount(len(self.features) + 2)
        self.table.setHorizontalHeaderLabels(self.features + ["得分", "标签"])
        self.table.setRowCount(self.MIN_ROWS)
        self.table.clearContents()
        for row in range(self.MIN_ROWS):
            for col in range(len(self.features) + 2):
                item = QTableWidgetItem()
                if col >= len(self.features):  # 最后两列是“得分”和“标签”
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.table.setItem(row, col, item)
    def _cell_text(self, row: int, col: int) -> str:
        item = self.table.item(row, col)
        return item.text() if item else ""