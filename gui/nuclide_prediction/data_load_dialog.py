
# -*- coding: utf-8 -*-
"""
主要功能：
“加载数据”弹窗：
   - 读取 CSV/Excel，预览表格（按内容自适应列宽 + 横向滚动条）。
   - 仅显示“含缺失行及其前后各 1 行”的动态子集，便于集中处理问题数据；
     一旦修复该行（或时间列解析正确），该行会自动从视图中消失。
   - 选择“时间列”与“时间格式”，使用 pandas.to_datetime(..., errors="coerce") 做解析，
     解析失败自动记为缺失参与高亮与筛选。
   - 在弹窗内完成缺失值处理（前向/后向/均值/中位数/常数填充，以及删除含缺失行）。
   - 只有当数据不再包含缺失时，OK 按钮才允许点击；OK 返回的是“清洗后的 DataFrame”。

注意：
- 弹窗中强制“缺失清零”后才能返回主界面，因此主界面的表格不再需要缺失提示。
"""

from __future__ import annotations

import json
import os
from typing import Optional, Any, Dict, Set

import pandas as pd

from PyQt6.QtCore import Qt, QAbstractTableModel, QModelIndex, QVariant, pyqtSignal
from PyQt6.QtGui import QBrush, QColor
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QFileDialog,  QLineEdit, QTableView,
    QHeaderView, QComboBox, QDialog, QDialogButtonBox, QMessageBox,
    QCheckBox
)
from PyQt6.QtCore import QSortFilterProxyModel

# ========================= 工具函数与模型 =========================

def is_nan_like(val: Any) -> bool:
    """判断一个值是否“缺失”——包含 None/NaN/NaT/空字符串等。"""
    if val is None:
        return True
    try:
        # pandas 对 NaN/NaT/None 均返回 True
        return pd.isna(val)
    except Exception:
        pass
    if isinstance(val, str) and val.strip() == "":
        return True
    return False


def is_bad_str(val: Any) -> bool:
    """判断是否为包含非数字字符的“坏值”字符串。"""
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return False
        try:
            float(s)
            return False
        except ValueError:
            return True
    return False


class DataFrameModel(QAbstractTableModel):
    """将 pandas.DataFrame 映射到 QTableView 使用的模型，并对缺失值(蓝色)及无法解析的字符串(红色)做背景高亮。"""

    row_filled_sig = pyqtSignal(int)

    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self._df = df
        self._row_colors: Dict[int, QColor] = {}

    # 行数/列数
    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._df)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._df.columns)

    FLOAT_FMT = "%.6g"
    # 单元格数据与显示角色
    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return QVariant()

        r, c = index.row(), index.column()
        val = self._df.iat[r, c]

        if role == Qt.ItemDataRole.BackgroundRole:
            color = self._row_colors.get(r)
            if color is not None:
                return color
            if is_bad_str(val):
                return QBrush(Qt.GlobalColor.red).color().lighter(170)
            if is_nan_like(val):
                return QBrush(Qt.GlobalColor.blue).color().lighter(170)

        if role == Qt.ItemDataRole.DisplayRole:
            val = self._df.iat[index.row(), index.column()]

            if pd.isna(val):
                return ""

            # ✨ 浮点统一格式化，减少 “6....” 省略号
            if isinstance(val, float):
                return self.FLOAT_FMT % val
            return str(val)

        return None


    # 表头
    def headerData(self, section: int, orientation: Qt.Orientation,
                   role: int = Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole:
            return QVariant()
        if orientation == Qt.Orientation.Horizontal:
            try:
                return str(self._df.columns[section])
            except Exception:
                return QVariant()
        else:
            return str(section)

    # 替换整个 DataFrame
    def setDataFrame(self, df: pd.DataFrame):
        self.beginResetModel()
        self._df = df
        self.endResetModel()

    def dataframe(self) -> pd.DataFrame:
        return self._df

    # 允许编辑最后一行
    def flags(self, index: QModelIndex):
        base = super().flags(index)
        return base | Qt.ItemFlag.ItemIsEditable

    def setData(self, index: QModelIndex, value, role: int = Qt.ItemDataRole.EditRole):
        if role != Qt.ItemDataRole.EditRole or not index.isValid():
            return False
        r, c = index.row(), index.column()
        col_name = self._df.columns[c]
        try:
            if pd.api.types.is_numeric_dtype(self._df[col_name]):
                if value == "":
                    self._df.iat[r, c] = pd.NA
                else:
                    self._df.iat[r, c] = float(value)
            else:
                self._df.iat[r, c] = value
        except Exception:
            self._df.iat[r, c] = pd.NA
        self.dataChanged.emit(index, index, [role])
        if r == self.rowCount() - 1 and self._df.iloc[r].notna().all():
            self.row_filled_sig.emit(r)
        return True

    # 行背景色控制
    def set_row_color(self, row: int, color: QColor):
        self._row_colors[row] = color
        if 0 <= row < self.rowCount() and self.columnCount() > 0:
            tl = self.index(row, 0)
            br = self.index(row, self.columnCount() - 1)
            self.dataChanged.emit(tl, br, [Qt.ItemDataRole.BackgroundRole])

    def clear_row_colors(self):
        if not self._row_colors:
            return
        rows = list(self._row_colors.keys())
        self._row_colors.clear()
        if rows and self.columnCount() > 0:
            tl = self.index(min(rows), 0)
            br = self.index(max(rows), self.columnCount() - 1)
            self.dataChanged.emit(tl, br, [Qt.ItemDataRole.BackgroundRole])


class MissingRowsProxyModel(QSortFilterProxyModel):
    """
    仅展示“含缺失行及其前后各 k 行”的代理模型。
    - 通过 set_allowed_rows(集合) 指定允许显示的“源模型行索引”；
    - 通过 set_only_missing_context(True/False) 控制是否启用该过滤。
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._only_missing_context = True
        self._allowed_rows: Set[int] = set()

    def set_only_missing_context(self, enabled: bool):
        self._only_missing_context = enabled
        self.invalidateFilter()

    def set_allowed_rows(self, rows: Set[int]):
        self._allowed_rows = set(rows)
        self.invalidateFilter()

    def filterAcceptsRow(self, source_row: int, source_parent: QModelIndex) -> bool:
        if not self._only_missing_context:
            return True
        return source_row in self._allowed_rows


# ========================= “加载数据”弹窗 =========================

class DataLoadDialog(QDialog):
    """
    读取与清洗数据的弹窗：
    - 读取 CSV/Excel；
    - 选择“时间列 + 时间格式”，即时解析并统计成功/失败；
    - 仅显示“缺失行±1”的动态子集；
    - 在弹窗内完成缺失值处理；
    - 缺失清零后才能点击 OK 返回主界面。

    Parameters
    ----------
    require_time_column : bool, default False
        是否强制要求用户选择时间列。若为 True，则在未选择时间列时即便
        数据已无缺失，OK 按钮也不会启用。
    """

    def __init__(
        self,
        parent: QWidget | None = None,
        default_time_fmt: str = "%Y年%m月%d日%H%M",
        require_time_column: bool = False,
    ):
        super().__init__(parent)
        self.setWindowTitle("加载数据")
        self.resize(1100, 700)

        self._require_time_column = require_time_column

        # 原始数据（来自文件）；工作数据（在弹窗内进行解析/填充/删除等）
        self._raw_df: Optional[pd.DataFrame] = None
        self._work_df: Optional[pd.DataFrame] = None
        self._path: Optional[str] = None

        # 当前被解析为时间的列及其原始值副本
        self._current_time_col: Optional[str] = None
        self._time_col_raw: Optional[pd.Series] = None

        # 记录含有非数字字符的字符串单元格，以避免每次刷新都全表扫描
        self._bad_mask: Optional[pd.DataFrame] = None
        # 用户已处理过的行，保持可见避免“消失”
        self._sticky_rows: Set[int] = set()

        # ---------- 顶部：选择文件 / 时间列与格式 ----------
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("选择 CSV 或 Excel 文件...")
        self.path_edit.setReadOnly(True)
        btn_load = QPushButton("选择并加载")
        btn_load.clicked.connect(self._choose_and_load)

        self.ignore_first_col_chk = QCheckBox("忽略首列")
        self.ignore_first_col_chk.setEnabled(False)
        self.ignore_first_col_chk.stateChanged.connect(self._apply_ignore_first_column)

        self.time_col_chk = QCheckBox("时间列")
        self.time_col_chk.stateChanged.connect(self._time_col_chk_changed)
        if self._require_time_column:
            self.time_col_chk.setChecked(True)
            self.time_col_chk.setEnabled(False)

        self.time_col_combo = QComboBox()
        self.time_col_combo.setEnabled(False)

        self.time_fmt_label = QLabel("时间格式")
        self.time_fmt_edit = QLineEdit(default_time_fmt)
        self.time_fmt_edit.setPlaceholderText('例如：%Y年%m月%d日%H%M 或 %Y-%m-%d %H:%M:%S')
        self.time_fmt_label.setEnabled(False)
        self.time_fmt_edit.setEnabled(False)

        # ---------- 表格预览（按内容自适应 + 横向滚动） ----------
        self.preview = QTableView()
        hh = self.preview.horizontalHeader()
        # 按内容自适应（注意：超大表可能有性能开销）
        hh.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        hh.setStretchLastSection(False)
        self.preview.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.preview.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.preview.setAlternatingRowColors(True)

        # 底层模型 + 只显示“缺失行±1”的代理模型
        self._base_model: Optional[DataFrameModel] = None
        self._proxy = MissingRowsProxyModel(self.preview)
        self.preview.setModel(self._proxy)

        # 统计信息区
        self.stats_label = QLabel("尚未读取数据。")
        self.parsed_label = QLabel("")
        self.remaining_label = QLabel("")

        # 只看“缺失行±1”的开关
        self.only_missing_chk = QCheckBox("仅显示缺失行±1（动态）")
        self.only_missing_chk.setChecked(True)
        self.only_missing_chk.stateChanged.connect(self._refresh_missing_display)

        # 缺失处理工具条：列选择 + 方法 + 常数值 + 应用
        self.col_combo = QComboBox()
        self.col_combo.setEnabled(False)

        self.method_combo = QComboBox()
        self.method_combo.addItems([
            "前向填充 (ffill)",
            "后向填充 (bfill)",
            "均值填充",
            "中位数填充",
            "常数填充",
            "删除含缺失行"
        ])
        self.const_edit = QLineEdit()
        self.const_edit.setPlaceholderText("常数值（仅常数填充时填写）")
        self.apply_btn = QPushButton("应用填充/删除")
        self.apply_btn.setEnabled(False)
        self.apply_btn.clicked.connect(self._apply_fix)

        # 绑定时间解析：选择列或结束时间格式编辑时再解析
        self.time_col_combo.currentIndexChanged.connect(self._time_col_changed)
        self.time_fmt_edit.editingFinished.connect(self._time_fmt_edit_finished)

        # 顶部布局
        top = QGridLayout()
        top.addWidget(QLabel("文件"), 0, 0)
        top.addWidget(self.path_edit, 0, 1, 1, 2)
        top.addWidget(btn_load, 0, 3)
        top.addWidget(self.ignore_first_col_chk, 0, 4)

        top.addWidget(self.time_col_chk, 1, 0)
        top.addWidget(self.time_col_combo, 1, 1)
        top.addWidget(self.time_fmt_label, 1, 2)
        top.addWidget(self.time_fmt_edit, 1, 3, 1, 2)

        # 工具条布局
        tool_row = QHBoxLayout()
        tool_row.addWidget(QLabel("列"))
        tool_row.addWidget(self.col_combo, 2)
        tool_row.addWidget(QLabel("方法"))
        tool_row.addWidget(self.method_combo, 2)
        tool_row.addWidget(self.const_edit, 2)
        tool_row.addWidget(self.apply_btn)
        tool_row.addStretch(1)

        # 确认/取消按钮
        self.btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.btns.button(QDialogButtonBox.StandardButton.Ok).setEnabled(False)  # 缺失清零后才可 OK
        self.btns.accepted.connect(self._accept_if_clean)
        self.btns.rejected.connect(self.reject)

        # 主体布局
        main = QVBoxLayout(self)
        main.addLayout(top)
        main.addWidget(self.preview, stretch=1)
        main.addWidget(self.only_missing_chk)
        main.addLayout(tool_row)
        main.addWidget(self.stats_label)
        main.addWidget(self.parsed_label)
        main.addWidget(self.remaining_label)
        main.addWidget(self.btns)

    # ---------- 槽函数 ----------

    def _choose_and_load(self):
        """选择文件并直接加载预览。"""
        path, _ = QFileDialog.getOpenFileName(
            self, "选择数据文件", "", "All Supported (*.csv *.xlsx);;CSV Files (*.csv);;Excel Files (*.xlsx)"
        )
        if path:
            self.path_edit.setText(path)
            self._read_preview(path)

    def _read_preview(self, path: Optional[str] = None):
        """读取文件，初始化工作 DataFrame 和各个控件。"""
        if path is None:
            path = self.path_edit.text().strip()
        if not path:
            QMessageBox.warning(self, "提示", "请选择文件。")
            return
        if not os.path.isfile(path):
            QMessageBox.warning(self, "提示", "文件不存在。")
            return

        try:
            if path.lower().endswith(".csv"):
                df = pd.read_csv(path)
            else:
                df = pd.read_excel(path)
        except Exception as e:
            QMessageBox.critical(self, "读取失败", f"无法读取文件：{e}")
            return

        if df.empty:
            QMessageBox.warning(self, "提示", "文件为空。")
            return

        self._raw_df = df.copy()
        self._path = path
        self.ignore_first_col_chk.setEnabled(True)
        self._apply_ignore_first_column(initial=True)

    def _apply_ignore_first_column(self, _state=None, *, initial: bool = False):
        """根据复选框状态决定是否忽略首列，并刷新控件。"""
        if self._raw_df is None:
            return
        self._work_df = self._raw_df.copy()
        if self.ignore_first_col_chk.isChecked() and self._work_df.shape[1] > 0:
            self._work_df = self._work_df.iloc[:, 1:].copy()
        self._reset_controls_after_df_change(initial=initial)

    def _reset_controls_after_df_change(self, initial: bool = False):
        if self._work_df is None:
            return

        # 切换数据源后重置时间列状态
        self._current_time_col = None
        self._time_col_raw = None

        # 时间列/普通列选择器
        self.time_col_combo.clear()
        self.time_col_combo.addItems(list(self._work_df.columns.astype(str)))

        self.col_combo.clear()
        self.col_combo.addItem("（全部列）")
        self.col_combo.addItems(list(self._work_df.columns.astype(str)))
        self.col_combo.setEnabled(True)
        self.apply_btn.setEnabled(True)

        # 初始化坏值掩码和已处理行集合
        # DataFrame.applymap 在 pandas 2.1 后已弃用，改用 DataFrame.map
        self._bad_mask = self._work_df.map(is_bad_str).fillna(False)
        self._sticky_rows = set()

        # 模型设置
        self._base_model = DataFrameModel(self._work_df)
        self._base_model.dataChanged.connect(self._on_data_changed)
        self._proxy.setSourceModel(self._base_model)

        # 根据复选框状态启用时间相关控件
        self._toggle_time_controls(self.time_col_chk.isChecked())
        if self.time_col_chk.isChecked():
            self.time_col_combo.setCurrentIndex(0 if self.time_col_combo.count() > 0 else -1)
            if not self._auto_detect_time_column(initial=initial):
                self._reparse_time_and_refresh(initial=initial)
        else:
            self.time_col_combo.setCurrentIndex(-1)
            self.parsed_label.setText("未选择时间列。")
            self._update_preview(initial=initial)
            self._refresh_missing_display()

    def _toggle_time_controls(self, enabled: bool):
        self.time_col_combo.setEnabled(enabled)
        self.time_fmt_label.setEnabled(enabled)
        self.time_fmt_edit.setEnabled(enabled)

    def _time_col_chk_changed(self, _state):
        """启用/禁用时间列相关控件。"""
        if self._work_df is None:
            return
        enabled = self.time_col_chk.isChecked()
        self._toggle_time_controls(enabled)
        if not enabled:
            if self._current_time_col and self._time_col_raw is not None and self._current_time_col in self._work_df.columns:
                self._work_df[self._current_time_col] = self._time_col_raw
                if self._bad_mask is not None:
                    self._bad_mask[self._current_time_col] = (
                        self._work_df[self._current_time_col].map(is_bad_str).fillna(False)
                    )
            self._current_time_col = None
            self._time_col_raw = None
            self.time_col_combo.setCurrentIndex(-1)
            self.parsed_label.setText("未选择时间列。")
            self._update_preview()
            self._refresh_missing_display()
        else:
            if self.time_col_combo.count() > 0 and self.time_col_combo.currentIndex() < 0:
                self.time_col_combo.setCurrentIndex(0)
            if not self._auto_detect_time_column():
                self._reparse_time_and_refresh(show_fail_msg=False)

    def _auto_detect_time_column(self, initial: bool = False) -> bool:
        if self._work_df is None:
            return False
        text_cols = [c for c in self._work_df.columns if pd.api.types.is_string_dtype(self._work_df[c])]
        if not text_cols:
            return False
        cand = str(text_cols[0])
        idx = self.time_col_combo.findText(cand)
        if idx < 0:
            return False
        self.time_col_combo.blockSignals(True)
        self.time_col_combo.setCurrentIndex(idx)
        self.time_col_combo.blockSignals(False)
        self._current_time_col = cand
        self._time_col_raw = self._work_df[cand].copy()
        ser = pd.to_datetime(self._time_col_raw, errors="coerce")
        self._work_df[cand] = ser
        ok, bad = ser.notna().sum(), ser.isna().sum()
        self.parsed_label.setText(f"时间解析（通用）：成功 {ok:,} 条，失败 {bad:,} 条。")
        self._update_preview(initial=initial)
        self._refresh_missing_display()
        return True

    def _time_col_changed(self):
        if not self.time_col_chk.isChecked():
            return
        self._reparse_time_and_refresh(show_fail_msg=True)

    def _time_fmt_edit_finished(self):
        if not self.time_col_chk.isChecked():
            return
        self._reparse_time_and_refresh(show_fail_msg=True)

    def _on_data_changed(self, topLeft: QModelIndex, bottomRight: QModelIndex, roles: list[int]):
        """同步用户编辑到原始时间列，并刷新坏值掩码。"""
        if self._work_df is None or self._bad_mask is None:
            return
        changed_rows: Set[int] = set()
        for c in range(topLeft.column(), bottomRight.column() + 1):
            col_name = self._work_df.columns[c]
            for r in range(topLeft.row(), bottomRight.row() + 1):
                val = self._work_df.iloc[r, c]
                self._bad_mask.iat[r, c] = is_bad_str(val)
                changed_rows.add(self._work_df.index[r])
        if self.time_col_chk.isChecked() and self._current_time_col and self._time_col_raw is not None:
            col_idx = self._work_df.columns.get_loc(self._current_time_col)
            if topLeft.column() <= col_idx <= bottomRight.column():
                for row in range(topLeft.row(), bottomRight.row() + 1):
                    if row < len(self._time_col_raw):
                        self._time_col_raw.iloc[row] = self._work_df.iloc[row, col_idx]
        self._sticky_rows |= changed_rows
        self._update_preview()

    def _reparse_time_and_refresh(self, *, initial: bool = False, show_fail_msg: bool = False, refresh_missing: bool = True):
        """根据“时间列 + 格式”解析时间列；刷新统计，并按需更新缺失行显示。"""
        if self._work_df is None:
            return

        if not self.time_col_chk.isChecked():
            self.parsed_label.setText("未选择时间列。")
            self._update_preview(initial=initial)
            if refresh_missing:
                self._refresh_missing_display()
            return

        col = self.time_col_combo.currentText()
        fmt = self.time_fmt_edit.text().strip()

        # 切换时间列时恢复旧列的原始值，并缓存新列的原始值
        if col != self._current_time_col:
            if self._current_time_col and self._time_col_raw is not None and self._current_time_col in self._work_df.columns:
                self._work_df[self._current_time_col] = self._time_col_raw
            self._current_time_col = col or None
            self._time_col_raw = self._work_df[col].copy() if col else None
            if self._bad_mask is not None and col:
                self._bad_mask[col] = (
                    self._work_df[col].map(is_bad_str).fillna(False)
                )
        else:
            if col and self._time_col_raw is not None:
                self._work_df[col] = self._time_col_raw.copy()
                if self._bad_mask is not None:
                    self._bad_mask[col] = self._work_df[col].map(is_bad_str).fillna(False)

        # 解析：先通用解析，再对未成功部分尝试用户指定格式
        if col:
            try:
                ser = pd.to_datetime(self._work_df[col])
                mask = ser.isna() & self._work_df[col].notna()
                if mask.any() and fmt:
                    ser_fmt = pd.to_datetime(self._work_df.loc[mask, col], format=fmt)
                    ser.loc[mask] = ser_fmt
                self._work_df[col] = ser
                if self._bad_mask is not None:
                    self._bad_mask[col] = self._work_df[col].map(is_bad_str).fillna(False)
                ok, bad = ser.notna().sum(), ser.isna().sum()
                if fmt:
                    self.parsed_label.setText(
                        f"时间解析：成功 {ok:,} 条，失败 {bad:,} 条。失败将以缺失值高亮显示。"
                    )
                    if bad and show_fail_msg:
                        QMessageBox.warning(
                            self,
                            "时间解析失败",
                            f"格式 {fmt} 未能解析 {bad} 条记录。",
                        )
                else:
                    self.parsed_label.setText(
                        f"时间解析（通用）：成功 {ok:,} 条，失败 {bad:,} 条。"
                    )
                    if bad and show_fail_msg:
                        QMessageBox.warning(
                            self,
                            "时间解析失败",
                            f"未能解析 {bad} 条记录，请检查时间格式。",
                        )
            except Exception as e:
                QMessageBox.warning(
                    self,
                    "时间解析失败",
                    f"{e}",
                )
        else:
            self.parsed_label.setText("未选择时间列。")

        self._update_preview(initial=initial)
        if refresh_missing:
            self._refresh_missing_display()

    def _current_missing_indices(self) -> Set[int]:
        """返回当前工作表中“任一列缺失”的行索引集合。"""
        if self._work_df is None:
            return set()
        mask_bad = (
            self._bad_mask if self._bad_mask is not None else self._work_df.map(is_bad_str)
        ).fillna(False)
        mask = self._work_df.isna() | mask_bad
        rows = mask.any(axis=1)
        return set(self._work_df.index[rows].tolist())

    def _indices_with_context(self, base: Set[int], k: int = 1) -> Set[int]:
        """在缺失行的基础上，加入前后各 k 行的“上下文行索引”。"""
        if self._work_df is None or not base:
            return base
        all_idx = list(self._work_df.index)
        pos_map = {i: p for p, i in enumerate(all_idx)}
        out: Set[int] = set()
        for i in base:
            p = pos_map.get(i, None)
            if p is None:
                continue
            start = max(0, p - k)
            end = min(len(all_idx) - 1, p + k)
            for q in range(start, end + 1):
                out.add(all_idx[q])
        return out

    def _refresh_missing_display(self):
        """根据当前缺失行刷新代理模型的可见行集合。"""
        if self._work_df is None:
            return
        missing_rows = self._current_missing_indices()
        allowed = self._indices_with_context(missing_rows, k=1)
        allowed |= self._sticky_rows
        self._proxy.set_only_missing_context(self.only_missing_chk.isChecked())
        self._proxy.set_allowed_rows(allowed)

    def _update_preview(self, initial: bool = False):
        """刷新统计并控制 OK 按钮状态。"""
        if self._work_df is None:
            return

        # 统计缺失 + 坏值
        mask_na = self._work_df.isna()
        mask_bad = (
            self._bad_mask if self._bad_mask is not None else self._work_df.map(is_bad_str)
        ).fillna(False)
        total_cells = self._work_df.shape[0] * self._work_df.shape[1]
        total_na = int((mask_na | mask_bad).sum().sum())
        self.stats_label.setText(
            f"行数: {len(self._work_df):,}，列数: {self._work_df.shape[1]}，缺失单元格: {total_na:,} "
            f"（{total_na / max(1, total_cells):.2%}）"
        )

        missing_rows = self._current_missing_indices()

        # 保证底层模型数据最新
        if self._base_model is None:
            self._base_model = DataFrameModel(self._work_df)
            self._base_model.dataChanged.connect(self._on_data_changed)
            self._proxy.setSourceModel(self._base_model)
        else:
            self._base_model.setDataFrame(self._work_df)

        # 控制 OK 状态 / 显示剩余缺失行数
        remain = len(missing_rows)
        time_text = self.time_col_combo.currentText()
        time_selected = self.time_col_chk.isChecked() and bool(time_text.strip())
        ok_enabled = (remain == 0) and (not self._require_time_column or time_selected)

        if remain == 0:
            if self._require_time_column and not time_selected:
                self.remaining_label.setText(
                    "⚠️ 缺失值已处理完成，但未选择时间列。请先选择时间列再继续。"
                )
            else:
                self.remaining_label.setText("✅ 所有缺失值已处理完成。可以点击 OK。")
        else:
            self.remaining_label.setText(
                f"⚠️ 仍有 {remain} 行包含缺失值。请处理后再继续。仅显示缺失行及其前后各 1 行。"
            )

        self.btns.button(QDialogButtonBox.StandardButton.Ok).setEnabled(ok_enabled)

        # 首次读取时给一点提示
        if initial:
            self.only_missing_chk.setToolTip("勾选：只显示含缺失行及其前后各 1 行；取消：查看全部数据。")

    def _apply_fix(self):
        """根据用户选择的“列 + 方法 + 常数”执行缺失处理，并刷新视图。"""
        if self._work_df is None:
            return

        # 目标列
        selected = self.col_combo.currentText()
        if selected and selected != "（全部列）":
            cols = [selected] if selected in self._work_df.columns else []
        else:
            cols = list(self._work_df.columns)

        if not cols:
            QMessageBox.warning(self, "提示", "未选择有效列。")
            return

        # 记录本次受影响的行，填充后保持可见
        mask_before = self._work_df[cols].isna()
        if self._bad_mask is not None:
            mask_before |= self._bad_mask[cols]
        affected_rows = set(self._work_df.index[mask_before.any(axis=1)].tolist())

        method = self.method_combo.currentText()
        refresh_missing = False
        try:
            if "前向填充" in method:
                self._work_df[cols] = self._work_df[cols].ffill()
            elif "后向填充" in method:
                self._work_df[cols] = self._work_df[cols].bfill()
            elif "均值填充" in method:
                num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(self._work_df[c])]
                for c in num_cols:
                    self._work_df[c] = self._work_df[c].fillna(self._work_df[c].mean())
            elif "中位数填充" in method:
                num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(self._work_df[c])]
                for c in num_cols:
                    self._work_df[c] = self._work_df[c].fillna(self._work_df[c].median())
            elif "常数填充" in method:
                val_text = self.const_edit.text()
                if val_text == "":
                    QMessageBox.warning(self, "提示", "请输入常数值。")
                    return
                try:
                    val = float(val_text)
                except ValueError:
                    val = val_text
                self._work_df[cols] = self._work_df[cols].fillna(val)
            elif "删除含缺失行" in method:
                self._work_df.dropna(axis=0, how='any', inplace=True)
                if self._bad_mask is not None:
                    self._bad_mask = self._bad_mask.loc[self._work_df.index]
                refresh_missing = True
            else:
                QMessageBox.warning(self, "提示", "未知方法。")
                return
        except Exception as e:
            QMessageBox.critical(self, "失败", f"处理失败：{e}")
            return

        # 更新坏值掩码并记录受影响行
        if self._bad_mask is not None:
            # DataFrame.applymap 在新版 pandas 中已弃用，使用 map 逐元素判断
            self._bad_mask[cols] = (
                self._work_df[cols].map(is_bad_str).fillna(False)
            )
        self._sticky_rows |= affected_rows
        self._sticky_rows &= set(self._work_df.index)

        # 若当前时间列被修改，需同步原始副本
        if self._current_time_col and self._current_time_col in self._work_df.columns:
            self._time_col_raw = self._work_df[self._current_time_col].copy()

        # 处理后需要重新解析时间列（防止该列也有缺失）并刷新
        self._reparse_time_and_refresh(refresh_missing=refresh_missing)

    def _accept_if_clean(self):
        """只有在工作表不存在任何缺失时才允许关闭对话框。"""
        if self._work_df is None:
            QMessageBox.warning(self, "提示", "请先读取数据。")
            return
        bad = self._bad_mask.any().any() if self._bad_mask is not None else self._work_df.map(is_bad_str).any().any()
        if self._work_df.isna().any().any() or bad:
            QMessageBox.warning(self, "提示", "仍存在缺失或非法值，请先处理干净再继续。")
            return
        if self._require_time_column and not self.time_column():
            QMessageBox.warning(self, "提示", "请先选择时间列。")
            return
        self.accept()

    # ---------- 结果导出接口 ----------

    def loaded_dataframe(self) -> Optional[pd.DataFrame]:
        """返回清洗完成的 DataFrame（副本）。"""
        return None if self._work_df is None else self._work_df.copy()

    def file_path(self) -> Optional[str]:
        return self._path

    def time_column(self) -> Optional[str]:
        if not self.time_col_chk.isChecked():
            return None
        col = self.time_col_combo.currentText()
        return col or None

    def time_format(self) -> str:
        return self.time_fmt_edit.text().strip()

    # ---------- 工厂方法：先选文件再弹窗 ----------

    @classmethod
    def from_file_dialog(
        cls,
        parent: QWidget | None = None,
        default_time_fmt: str = "%Y年%m月%d日%H%M",
        require_time_column: bool = False,
    ) -> Optional["DataLoadDialog"]:
        """先弹出文件选择框，随后打开本对话框。如果用户取消则返回 None。"""
        path, _ = QFileDialog.getOpenFileName(
            parent,
            "选择数据文件",
            "",
            "All Supported (*.csv *.xlsx);;CSV Files (*.csv);;Excel Files (*.xlsx)",
        )
        if not path:
            return None
        dlg = cls(parent, default_time_fmt=default_time_fmt, require_time_column=require_time_column)
        dlg.path_edit.setText(path)
        dlg._read_preview(path)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            return dlg
        return None
