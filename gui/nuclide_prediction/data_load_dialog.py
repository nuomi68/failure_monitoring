
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


class DataFrameModel(QAbstractTableModel):
    """将 pandas.DataFrame 映射到 QTableView 使用的模型，并对缺失值做背景高亮。"""

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

    # 单元格数据与显示角色
    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return QVariant()

        r, c = index.row(), index.column()
        val = self._df.iat[r, c]

        if role == Qt.ItemDataRole.DisplayRole:
            # 显示时对缺失值用空字符串，美观一些
            if is_nan_like(val):
                return ""
            return str(val)

        if role == Qt.ItemDataRole.BackgroundRole:
            color = self._row_colors.get(r)
            if color is not None:
                return color
            # 缺失值浅红背景
            if is_nan_like(val):
                return QBrush(Qt.GlobalColor.red).color().lighter(170)

        return QVariant()

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
        if index.row() == self.rowCount() - 1:
            return base | Qt.ItemFlag.ItemIsEditable
        return base

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
        if self._df.iloc[r].notna().all():
            self.row_filled_sig.emit(r)
        return True

    # 行背景色控制
    def set_row_color(self, row: int, color: QColor | None):
        if color is None:
            self._row_colors.pop(row, None)
        else:
            self._row_colors[row] = color
        if 0 <= row < self.rowCount() and self.columnCount() > 0:
            tl = self.index(row, 0)
            br = self.index(row, self.columnCount() - 1)
            self.dataChanged.emit(tl, br, [Qt.ItemDataRole.BackgroundRole])

    def clear_row_colors(self):
        if not self._row_colors:
            return
        self._row_colors.clear()
        if self.rowCount() > 0 and self.columnCount() > 0:
            tl = self.index(0, 0)
            br = self.index(self.rowCount() - 1, self.columnCount() - 1)
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
    """

    def __init__(self, parent: QWidget | None = None, default_time_fmt: str = "%Y年%m月%d日%H%M"):
        super().__init__(parent)
        self.setWindowTitle("加载数据")
        self.resize(1100, 700)

        # 原始数据（来自文件）；工作数据（在弹窗内进行解析/填充/删除等）
        self._raw_df: Optional[pd.DataFrame] = None
        self._work_df: Optional[pd.DataFrame] = None
        self._path: Optional[str] = None

        # ---------- 顶部：选择文件 / 时间列与格式 ----------
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("选择 CSV 或 Excel 文件...")
        btn_browse = QPushButton("浏览...")
        btn_browse.clicked.connect(self._browse)

        btn_read = QPushButton("读取/预览")
        btn_read.clicked.connect(self._read_preview)

        self.time_col_combo = QComboBox()
        self.time_col_combo.setEnabled(False)

        self.time_fmt_edit = QLineEdit(default_time_fmt)
        self.time_fmt_edit.setPlaceholderText('例如：%Y年%m月%d日%H%M 或 %Y-%m-%d %H:%M:%S')

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
        self.only_missing_chk.stateChanged.connect(self._update_preview)

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

        # 绑定时间解析的即时刷新
        self.time_col_combo.currentIndexChanged.connect(self._reparse_time_and_refresh)
        self.time_fmt_edit.textChanged.connect(self._reparse_time_and_refresh)

        # 顶部布局
        top = QGridLayout()
        top.addWidget(QLabel("文件"), 0, 0)
        top.addWidget(self.path_edit, 0, 1, 1, 2)
        top.addWidget(btn_browse, 0, 3)
        top.addWidget(btn_read, 0, 4)

        top.addWidget(QLabel("时间列"), 1, 0)
        top.addWidget(self.time_col_combo, 1, 1)
        top.addWidget(QLabel("时间格式"), 1, 2)
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

    def _browse(self):
        """选择文件路径。"""
        path, _ = QFileDialog.getOpenFileName(
            self, "选择数据文件", "", "All Supported (*.csv *.xlsx);;CSV Files (*.csv);;Excel Files (*.xlsx)"
        )
        if path:
            self.path_edit.setText(path)

    def _read_preview(self):
        """读取文件，初始化工作 DataFrame 和各个控件。"""
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
        self._work_df = self._raw_df.copy()
        self._path = path

        # 时间列/普通列选择器
        self.time_col_combo.clear()
        self.time_col_combo.addItems(list(self._work_df.columns.astype(str)))
        self.time_col_combo.setEnabled(True)

        self.col_combo.clear()
        self.col_combo.addItem("（全部列）")
        self.col_combo.addItems(list(self._work_df.columns.astype(str)))
        self.col_combo.setEnabled(True)
        self.apply_btn.setEnabled(True)

        # 模型设置
        self._base_model = DataFrameModel(self._work_df)
        self._proxy.setSourceModel(self._base_model)

        # 解析时间 + 刷新视图
        self._reparse_time_and_refresh(initial=True)

    def _reparse_time_and_refresh(self, initial: bool = False):
        """根据“时间列 + 格式”解析时间列；刷新统计与视图。"""
        if self._work_df is None:
            return

        col = self.time_col_combo.currentText().strip()
        fmt = self.time_fmt_edit.text().strip()

        # 解析：指定格式优先；否则尝试通用解析
        if col and fmt:
            ser = pd.to_datetime(self._work_df[col], format=fmt, errors="coerce")
            self._work_df[col] = ser
            ok, bad = ser.notna().sum(), ser.isna().sum()
            self.parsed_label.setText(f"时间解析：成功 {ok:,} 条，失败 {bad:,} 条。失败将以缺失值高亮显示。")
        elif col:
            ser = pd.to_datetime(self._work_df[col], errors="coerce")
            self._work_df[col] = ser
            ok, bad = ser.notna().sum(), ser.isna().sum()
            self.parsed_label.setText(f"时间解析（通用）：成功 {ok:,} 条，失败 {bad:,} 条。")
        else:
            self.parsed_label.setText("请选择时间列并输入格式。")

        self._update_preview(initial=initial)

    def _current_missing_indices(self) -> Set[int]:
        """返回当前工作表中“任一列缺失”的行索引集合。"""
        if self._work_df is None:
            return set()
        mask = self._work_df.isna().any(axis=1)
        return set(self._work_df.index[mask].tolist())

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

    def _update_preview(self, initial: bool = False):
        """刷新统计、计算“缺失行±1”、更新代理过滤，并控制 OK 按钮状态。"""
        if self._work_df is None:
            return

        # 统计缺失
        na_counts = self._work_df.isna().sum()
        total_cells = self._work_df.shape[0] * self._work_df.shape[1]
        total_na = int(na_counts.sum())
        self.stats_label.setText(
            f"行数: {len(self._work_df):,}，列数: {self._work_df.shape[1]}，缺失单元格: {total_na:,} "
            f"（{total_na / max(1, total_cells):.2%}）"
        )

        # 计算“缺失行±1”
        missing_rows = self._current_missing_indices()
        allowed = self._indices_with_context(missing_rows, k=1)

        # 保证底层模型数据最新
        if self._base_model is None:
            self._base_model = DataFrameModel(self._work_df)
            self._proxy.setSourceModel(self._base_model)
        else:
            self._base_model.setDataFrame(self._work_df)

        # 应用过滤
        self._proxy.set_only_missing_context(self.only_missing_chk.isChecked())
        self._proxy.set_allowed_rows(allowed)

        # 控制 OK 状态 / 显示剩余缺失行数
        remain = len(missing_rows)
        if remain == 0:
            self.remaining_label.setText("✅ 所有缺失值已处理完成。可以点击 OK。")
            self.btns.button(QDialogButtonBox.StandardButton.Ok).setEnabled(True)
        else:
            self.remaining_label.setText(f"⚠️ 仍有 {remain} 行包含缺失值。请处理后再继续。仅显示缺失行及其前后各 1 行。")
            self.btns.button(QDialogButtonBox.StandardButton.Ok).setEnabled(False)

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

        method = self.method_combo.currentText()
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
                # 尝试转为数值，失败则按字符串处理
                try:
                    val = float(val_text)
                except ValueError:
                    val = val_text
                self._work_df[cols] = self._work_df[cols].fillna(val)
            elif "删除含缺失行" in method:
                self._work_df.dropna(axis=0, how='any', inplace=True)
            else:
                QMessageBox.warning(self, "提示", "未知方法。")
                return
        except Exception as e:
            QMessageBox.critical(self, "失败", f"处理失败：{e}")
            return

        # 处理后需要重新解析时间列（防止该列也有缺失）并刷新
        self._reparse_time_and_refresh()

    def _accept_if_clean(self):
        """只有在工作表不存在任何缺失时才允许关闭对话框。"""
        if self._work_df is None:
            QMessageBox.warning(self, "提示", "请先读取数据。")
            return
        if self._work_df.isna().any().any():
            QMessageBox.warning(self, "提示", "仍存在缺失值，请先处理干净再继续。")
            return
        self.accept()

    # ---------- 结果导出接口 ----------

    def loaded_dataframe(self) -> Optional[pd.DataFrame]:
        """返回清洗完成的 DataFrame（副本）。"""
        return None if self._work_df is None else self._work_df.copy()

    def file_path(self) -> Optional[str]:
        return self._path

    def time_column(self) -> Optional[str]:
        return self.time_col_combo.currentText().strip() if self.time_col_combo.isEnabled() else None

    def time_format(self) -> str:
        return self.time_fmt_edit.text().strip()