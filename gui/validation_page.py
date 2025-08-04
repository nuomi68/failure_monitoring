from __future__ import annotations

from typing import List, Any, Dict, Tuple
import pandas as pd
import numpy as np
from backend.ml_interface import ML
from functools import partial

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem,
    QLabel, QPushButton, QHBoxLayout, QMessageBox, QMenu,
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
        self.meta: Dict[str, Any] = {}
        self.features: List[str] = []  # 若为空，将在 _setup_table/enable_external 中回落为 ["X0"]
        self._external_mode: bool = False
        self._external_cb = None  # type: ignore
        # 撤销 / 重做
        self._undo_stack: list[list[list[str]]] = []
        self._redo_stack: list[list[list[str]]] = []
        self._restoring = False  # 正在还原栈，避免递归 push
        self._last_edit_cell: Tuple[int, int] | None = None  # (row, col) 用于去重

        # -------------------- UI --------------------
        layout = QVBoxLayout(self)
        self.table = QTableWidget()
        layout.addWidget(self.table)

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
        self.undo_btn = QPushButton("撤销")
        self.undo_btn.clicked.connect(self._undo)
        btn_row.addWidget(self.undo_btn)
        self.redo_btn = QPushButton("重做")
        self.redo_btn.clicked.connect(self._redo)
        btn_row.addWidget(self.redo_btn)
        self.predict_btn = QPushButton("计算")
        self.predict_btn.clicked.connect(self.on_predict)
        btn_row.addWidget(self.predict_btn)
        self.clear_btn = QPushButton("清空全部")
        self.clear_btn.clicked.connect(self._clear_all)
        btn_row.addWidget(self.clear_btn)

        self.result_lbl = QLabel("结果: ")
        btn_row.addWidget(self.result_lbl)
        btn_row.addStretch(1)
        self.save_btn = QPushButton("保存模型")
        self.save_btn.clicked.connect(self.save_model)
        btn_row.addWidget(self.save_btn)
        layout.addLayout(btn_row)

        self._update_undo_redo_state()

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------
    def configure(self, features: List[str]) -> None:
        self.features = list(features)
        self.meta = {}
        self._setup_table()
        self._clear_history()
        self._push_state()  # 初始快照
        self.save_btn.setEnabled(bool(ML.get_meta()))

    def enable_external(self, features: List[str], predict_cb) -> None:
        """
        外部预测模式：
        - features: 特征列名
        - predict_cb: 回调(df: pandas.DataFrame) -> Dict[str, np.ndarray]，
          返回 {目标名: 预测数组}，表格会据此动态生成结果列
        """
        self._external_mode = True
        self._external_cb = predict_cb
        self.configure(features)  # 用现有初始化流程
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
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._update_undo_redo_state()

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
            # ✅ 没有历史时，也给最少行数，列数按模式取
            rows = max(len(state), self.MIN_ROWS)
            # 新接口：初始仅含特征列，预测后再追加结果列
            cols = (len(state[0]) if state else len(self.features))
            self.table.setRowCount(rows)
            self.table.setColumnCount(cols)
            for r in range(len(state)):
                for c in range(len(state[r])):
                    item = QTableWidgetItem(state[r][c])
                    if c >= len(self.features):
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
        r = self.table.rowAt(pos.y())
        c = self.table.columnAt(pos.x())
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
        self._push_state()
        fn(*args)

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

    def on_predict(self):
        # ---------- 外部模式：由回调提供结果，动态生成目标列 ----------
        if self._external_mode and self._external_cb is not None:
            # 读表 -> DataFrame（空值用 NaN）
            rows = []
            for r in range(self.table.rowCount()):
                row = []
                for c in range(len(self.features)):
                    txt = (self._cell_text(r, c) or "").strip()
                    if txt == "":
                        row.append(np.nan)
                    else:
                        try:
                            row.append(float(txt))
                        except Exception:
                            row.append(np.nan)
                rows.append(row)
            df = pd.DataFrame(rows, columns=self.features)

            # 调回调：期望 {目标名: 一维数组}
            res = self._external_cb(df) or {}
            if not isinstance(res, dict) or not res:
                self.result_lbl.setText("结果: 空")
                return

            targets = list(res.keys())
            # 重建表头：特征列 + 各目标列
            self.table.setColumnCount(len(self.features) + len(targets))
            self.table.setHorizontalHeaderLabels(self.features + targets)

            # 写入结果并将结果列设为只读
            for i, t in enumerate(targets):
                arr = np.asarray(res[t]).ravel()
                for r in range(self.table.rowCount()):
                    val = "" if r >= len(arr) or (arr[r] != arr[r]) else str(arr[r])  # NaN 检测
                    item = self.table.item(r, len(self.features) + i)
                    if item is None:
                        item = QTableWidgetItem("")
                        self.table.setItem(r, len(self.features) + i, item)
                    item.setText(val)
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)

            self.result_lbl.setText(f"结果: 共 {self.table.rowCount()} 行, 目标列 {len(targets)} 个")
            return
        # ---------- 非外部模式：按最新 ML 接口调用 ----------
        meta = ML.get_meta() or {}
        # 收集 DataFrame
        rows = []
        for r in range(self.table.rowCount()):
            row = []
            for c in range(len(self.features)):
                txt = (self._cell_text(r, c) or "").strip()
                if txt == "":
                    row.append(np.nan)
                else:
                    try:
                        row.append(float(txt))
                    except Exception:
                        row.append(np.nan)
            rows.append(row)
        df = pd.DataFrame(rows, columns=self.features)
        # 选择输入形态
        is_multi = bool(meta.get("multi_output", False))
        is_ens   = bool(meta.get("ensemble", False))
        try:
            if is_multi or is_ens:
                X = {c: df[c].to_numpy() for c in df.columns}
                ret = ML.predict(X)
            else:
                ret = ML.predict(df.to_numpy())
        except Exception:
            # 兜底双形态
            try:
                ret = ML.predict({c: df[c].to_numpy() for c in df.columns})
            except Exception:
                ret = ML.predict(df.to_numpy())
        # 标准化：得到 {target: 1d-array}
        res = self._normalize_backend_result(ret)
        if not isinstance(res, dict) or not res:
            self.result_lbl.setText("结果: 空")
            return
        targets = list(res.keys())
        # 重建表头：特征列 + 各目标列
        self.table.setColumnCount(len(self.features) + len(targets))
        self.table.setHorizontalHeaderLabels(self.features + targets)
        # 写入各目标列并设为只读
        for i, t in enumerate(targets):
            arr = np.asarray(res[t]).ravel()
            for r in range(self.table.rowCount()):
                val = "" if r >= len(arr) or (arr[r] != arr[r]) else str(arr[r])
                item = self.table.item(r, len(self.features) + i)
                if item is None:
                    item = QTableWidgetItem("")
                    self.table.setItem(r, len(self.features) + i, item)
                item.setText(val)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.result_lbl.setText(f"结果: 共 {self.table.rowCount()} 行, 目标列 {len(targets)} 个")

    def save_model(self):
        if not ML.get_meta():
            QMessageBox.warning(self, "提示", "暂无可保存的模型")
            return
        try:
            ret = ML.save_auto()
            QMessageBox.information(self, "已保存", f"模型已保存到:\n{ret['path']}")
        except Exception as e:
            QMessageBox.warning(self, "错误", str(e))

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------
    def _setup_table(self):
        # 新接口：初始化仅包含特征列；预测时动态追加目标列
        self.table.setColumnCount(len(self.features))
        self.table.setHorizontalHeaderLabels(self.features)
        # ✅ 初始化空行
        self.table.setRowCount(self.MIN_ROWS)

    def _cell_text(self, row: int, col: int) -> str:
        item = self.table.item(row, col)
        return item.text() if item else ""

    # --------- 与 EnsemblePage 一致：标准化后端返回 ---------
    def _normalize_backend_result(self, ret) -> Dict[str, Any]:
        # MultiOutput: {t: {"labels": y, "scores": s}}
        if isinstance(ret, dict) and all(isinstance(v, dict) and "labels" in v for v in ret.values()):
            return {t: v.get("labels") for t, v in ret.items()}
        # 单模型/旧式集合：{"target": name, "labels": y, "scores": s}
        if isinstance(ret, dict) and "labels" in ret:
            return {str(ret.get("target", "输出")): ret.get("labels")}
        # 兼容极少数旧返回：(y, scores) 或 直接 y
        if isinstance(ret, tuple) and len(ret) >= 1:
            return {"输出": ret[0]}
        return {"输出": ret}
