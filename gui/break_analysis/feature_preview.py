# ================= feature_preview.py =================
from typing import Any

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.colors import to_hex

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QSizePolicy,
    QPushButton, QListWidget, QListWidgetItem, QFrame, QCheckBox,
    QStyledItemDelegate, QListView, QDialog, QMessageBox
)
from PyQt6.QtGui import QStandardItem, QStandardItemModel
from PyQt6.QtCore import Qt, pyqtSignal, QPoint, QSize

class HeatmapCanvas(FigureCanvas):
    """相关系数热力图——嵌入式 Matplotlib 画布"""

    def __init__(self, parent=None):
        self.fig, self.ax = plt.subplots()
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Expanding)
        self.colorbar = None

        # 状态：用于滚轮缩放与字体自适应
        self._has_data = False
        self._current_labels: list[str] = []
        self._x_full_limits = (-0.5, 0.5)
        self._y_full_limits = (0.5, -0.5)
        self._min_span = 0.8  # 允许的最小视域，约等于 1 个单元格

        # 交互事件
        self.mpl_connect("scroll_event", self._on_scroll)
        self.mpl_connect("button_press_event", self._on_mouse_press)
        self.mpl_connect("button_release_event", self._on_mouse_release)
        self.mpl_connect("motion_notify_event", self._on_mouse_move)

        self._is_panning = False
        self._last_mouse_pos: tuple[float, float] | None = None

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_tick_fontsize()

    def plot_corr(self, df: pd.DataFrame):
        """给定 DF，画皮尔逊相关矩阵；空 DF 时清屏。"""
        if self.colorbar:
            self.colorbar.remove()
            self.colorbar = None

        self.ax.clear()
        self._current_labels = []
        self._has_data = False

        if df.empty or df.shape[1] < 2:
            self.draw()
            return

        corr = df.corr(numeric_only=True)
        im = self.ax.imshow(corr.values, vmin=-1, vmax=1, cmap="coolwarm")
        # 默认 imshow 会把纵横比锁定为 1:1，一旦缩放/拖拽只改变某一方向，
        # Matplotlib 会在坐标轴内部留出大片留白，给人“坐标轴也被拖走”的错觉。
        # 设为 auto 让图像始终铺满坐标轴，可避免 UI 中出现这种错位。
        self.ax.set_aspect("auto")

        labels = corr.columns.astype(str).tolist()
        positions = np.arange(len(labels))
        self.ax.set_xticks(positions)
        self.ax.set_xticklabels(labels, rotation=45, ha="right")
        self.ax.set_yticks(positions)
        self.ax.set_yticklabels(labels)

        self.colorbar = self.fig.colorbar(im, ax=self.ax, shrink=0.7)

        extent = len(labels) - 0.5
        self._x_full_limits = (-0.5, extent)
        self._y_full_limits = (extent, -0.5)
        self._has_data = True
        self._current_labels = labels
        self._reset_limits()
        self._update_tick_fontsize()

        self.fig.tight_layout()
        self.draw()

    # ───────────────────────── 交互逻辑 ──────────────────────────
    def _on_scroll(self, event):
        if not self._has_data or event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        base_scale = 1.2
        if event.button == "up":
            scale_factor = 1 / base_scale  # 缩小视域 => 放大
        elif event.button == "down":
            scale_factor = base_scale
        else:
            return

        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()
        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
        relx = (cur_xlim[1] - event.xdata) / (cur_xlim[1] - cur_xlim[0])
        rely = (cur_ylim[1] - event.ydata) / (cur_ylim[1] - cur_ylim[0])

        new_xlim = (
            event.xdata - new_width * (1 - relx),
            event.xdata + new_width * relx,
        )
        new_ylim = (
            event.ydata - new_height * (1 - rely),
            event.ydata + new_height * rely,
        )

        if abs(new_xlim[1] - new_xlim[0]) < self._min_span or \
           abs(new_ylim[1] - new_ylim[0]) < self._min_span:
            return

        new_xlim = self._clamp_limits(new_xlim, self._x_full_limits)
        new_ylim = self._clamp_limits(new_ylim, self._y_full_limits)

        self.ax.set_xlim(new_xlim)
        self.ax.set_ylim(new_ylim)
        self.draw_idle()

    def _on_mouse_press(self, event):
        if not self._has_data:
            return
        if event.dblclick:
            self._reset_limits()
            self.draw_idle()
            return

        if event.button == 1 and event.inaxes == self.ax and \
                event.xdata is not None and event.ydata is not None:
            self._is_panning = True
            self._last_mouse_pos = (event.xdata, event.ydata)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def _clamp_limits(self, limits, full_limits):
        """确保缩放边界不会超出完整矩阵范围，并保留原有方向"""
        lower = min(full_limits)
        upper = max(full_limits)
        start = min(max(limits[0], lower), upper)
        end = min(max(limits[1], lower), upper)
        return start, end

    def _reset_limits(self):
        self.ax.set_xlim(self._x_full_limits)
        self.ax.set_ylim(self._y_full_limits)

    def _on_mouse_release(self, event):
        if event.button == 1 and self._is_panning:
            self._is_panning = False
            self._last_mouse_pos = None
            self.unsetCursor()

    def _on_mouse_move(self, event):
        if not self._is_panning or event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None or self._last_mouse_pos is None:
            return

        dx = event.xdata - self._last_mouse_pos[0]
        dy = event.ydata - self._last_mouse_pos[1]
        if dx == 0 and dy == 0:
            return

        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()
        new_xlim = (cur_xlim[0] - dx, cur_xlim[1] - dx)
        new_ylim = (cur_ylim[0] - dy, cur_ylim[1] - dy)

        new_xlim = self._clamp_limits(new_xlim, self._x_full_limits)
        new_ylim = self._clamp_limits(new_ylim, self._y_full_limits)

        self.ax.set_xlim(new_xlim)
        self.ax.set_ylim(new_ylim)
        self._last_mouse_pos = (event.xdata, event.ydata)
        self.draw_idle()

    def _update_tick_fontsize(self):
        if not self._current_labels:
            return

        num_labels = len(self._current_labels)
        max_len = max(len(lbl) for lbl in self._current_labels)
        width = max(self.width(), 1)

        pixels_per_label = width / max(num_labels, 1)
        penalty = max(1.0, max_len / 10)  # 标签越长字体越小
        font_size = pixels_per_label / (4 * penalty)
        font_size = max(6, min(14, font_size))

        self.ax.tick_params(axis="both", labelsize=font_size)
        if self.colorbar:
            self.colorbar.ax.tick_params(labelsize=max(font_size - 2, 6))


class _ColorCheckDelegate(QStyledItemDelegate):
    def sizeHint(self, option, index):
        size = super().sizeHint(option, index)
        height = max(size.height(), 26)
        return QSize(size.width(), height)


class _CategoryPopup(QFrame):
    """Popup checklist container that closes only when focus leaves."""

    closed = pyqtSignal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent, Qt.WindowType.Popup)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFrameShadow(QFrame.Shadow.Raised)

    def hideEvent(self, event):
        super().hideEvent(event)
        self.closed.emit()


class CategoryFilterWidget(QWidget):
    """
    Custom multi-select control used by the scatter plot panel.
    Always shows “选择类别” on the button, keeps the popup open while toggling.
    """

    selectionChanged = pyqtSignal(list)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.button = QPushButton("选择类别")
        self.button.setCheckable(True)
        self.button.setMinimumWidth(130)
        self.button.clicked.connect(self._toggle_popup)
        layout.addWidget(self.button)

        self.popup = _CategoryPopup(self)
        popup_layout = QVBoxLayout(self.popup)
        popup_layout.setContentsMargins(8, 8, 8, 8)
        popup_layout.setSpacing(6)
        self.list = QListWidget(self.popup)
        self.list.setSpacing(4)
        self.list.setAlternatingRowColors(True)
        self.list.setSelectionMode(QListWidget.SelectionMode.NoSelection)
        popup_layout.addWidget(self.list)
        self.popup.closed.connect(self._popup_closed)

        self._checkbox_map: dict[str, QCheckBox] = {}

    def setVisible(self, visible: bool):
        if not visible:
            self.popup.hide()
            self.button.setChecked(False)
        super().setVisible(visible)

    def _toggle_popup(self, checked: bool):
        if checked and self.has_entries():
            height = 60 + 34 * max(self.list.count(), 1)
            self.popup.resize(max(self.width(), 220), min(320, height))
            pos = self.button.mapToGlobal(QPoint(0, self.button.height()))
            self.popup.move(pos)
            self.popup.show()
        else:
            self.popup.hide()

    def _popup_closed(self):
        self.button.setChecked(False)

    def set_categories(self, entries: list[tuple[str, str]], _color_map: dict[str, str] | None = None):
        """Populate checklist with label/value pairs; color map unused but kept for API parity."""
        self.list.clear()
        self._checkbox_map.clear()

        for label, key in entries:
            item = QListWidgetItem(self.list)
            checkbox = QCheckBox(label)
            checkbox.setChecked(True)
            checkbox.stateChanged.connect(self._on_checkbox_changed)
            checkbox.setProperty("_category_key", key)
            checkbox.setCursor(Qt.CursorShape.PointingHandCursor)
            checkbox.setStyleSheet("QCheckBox { padding: 4px 12px; }")
            self.list.setItemWidget(item, checkbox)
            item.setSizeHint(checkbox.sizeHint() + QSize(20, 8))
            self._checkbox_map[key] = checkbox

        has_entries = bool(self._checkbox_map)
        self.setVisible(has_entries)
        self.button.setEnabled(has_entries)
        if not has_entries:
            self.popup.hide()
            self.selectionChanged.emit([])
        else:
            self.selectionChanged.emit(self.selected_values())

    def clear_categories(self):
        self.list.clear()
        self._checkbox_map.clear()
        self.setVisible(False)
        self.selectionChanged.emit([])

    def has_entries(self) -> bool:
        return bool(self._checkbox_map)

    def selected_values(self) -> list[str]:
        return [
            key for key, checkbox in self._checkbox_map.items()
            if checkbox.isChecked()
        ]

    def _on_checkbox_changed(self, _state: int):
        self.selectionChanged.emit(self.selected_values())


class PreviewEnlargeDialog(QDialog):
    """放大 FeaturePreview 图表的独立窗口，带导航工具栏。"""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("图表放大预览")
        self.resize(900, 720)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        self.canvas = _PreviewCanvas(self)
        toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(toolbar)
        layout.addWidget(self.canvas)

class _PreviewCanvas(FigureCanvas):
    """承载 Matplotlib 的底层画布"""
    def __init__(self, parent=None):
        self.fig, self.ax = plt.subplots()
        super().__init__(self.fig)
        self.setParent(parent)
        # 让画布在 QSplitter 里可以正常伸缩
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Expanding)


class FeaturePreviewWidget(QWidget):
    """
    多图形预览：
        - 折线图：多列随 index/TIME 变化
        - 散点图：两列 X vs Y
        - 直方图：单列分布
        - 箱线图：多列箱须
    外部需要调用：
        set_dataframe(df)              # 上传或重新载入数据集后
        set_selected_columns(list[str])# 左侧“已选择特征”变化后
    """
    CHART_TYPES = ["折线图", "散点图", "直方图", "箱线图"]

    def __init__(self, parent=None):
        super().__init__(parent)

        # --- 布局骨架 ------------------------------------------------------
        root = QVBoxLayout(self)

        #  顶部：控制条 ---------------------------------------------------
        ctrl = QHBoxLayout()
        ctrl.setSpacing(12)
        # 样本名称 / 行索引选择 ----------------------
        ctrl.addWidget(QLabel("样本名称:"))
        self.cmb_idx = QComboBox()           
        self.cmb_idx.currentIndexChanged.connect(self._update_plot)
        self.cmb_idx.currentIndexChanged.connect(self._update_control_visibility)
        ctrl.addWidget(self.cmb_idx)

        ctrl.addWidget(QLabel("图形类型:"))
        self.cmb_type = QComboBox()
        self.cmb_type.addItems(self.CHART_TYPES)
        self.cmb_type.currentIndexChanged.connect(self._update_plot)
        self.cmb_type.currentIndexChanged.connect(self._update_control_visibility)
        ctrl.addWidget(self.cmb_type)

        self.category_filter = CategoryFilterWidget()
        self.category_filter.setVisible(False)
        self.category_filter.selectionChanged.connect(self._on_category_selection_changed)
        ctrl.addWidget(self.category_filter)

        # X / Y 轴列选择（仅散点/直方用） -------------------------
        self.cmb_x = QComboBox()
        self.cmb_y = QComboBox()
        self.lbl_x = QLabel("X 轴:")
        self.lbl_y = QLabel("Y 轴:")
        self.cmb_x.currentIndexChanged.connect(self._update_plot)
        self.cmb_y.currentIndexChanged.connect(self._update_plot)
        ctrl.addWidget(self.lbl_x)
        ctrl.addWidget(self.cmb_x)
        ctrl.addWidget(self.lbl_y)
        ctrl.addWidget(self.cmb_y)
        self._axis_widgets = [self.lbl_x, self.cmb_x, self.lbl_y, self.cmb_y]

        ctrl.addStretch()

        self.btn_enlarge = QPushButton("放大查看")
        self.btn_enlarge.setFixedHeight(28)
        self.btn_enlarge.clicked.connect(self._open_enlarge_dialog)
        ctrl.addWidget(self.btn_enlarge)
        root.addLayout(ctrl)

        # 中部：Matplotlib 画布 ------------------------------
        self.canvas = _PreviewCanvas()
        root.addWidget(self.canvas)
        self.ax = self.canvas.ax

        # --- 状态 ----------------------------------------------------------
        self._df: pd.DataFrame = pd.DataFrame()
        self._sel_cols: list[str] = []
        self._target_column: str | None = None
        self._category_colors: dict[str, str] = {}
        self._update_control_visibility()

    # ======= 公共 API ======================================================
    def set_dataframe(self, df: pd.DataFrame):
        self._df = df

        # ── 刷新“样本名称”下拉框 ──
        # 第一项固定为 “递增数列”，随后是所有列名
        self.cmb_idx.blockSignals(True)
        try:
            self.cmb_idx.clear()
            self.cmb_idx.addItem("递增数列")   
            self.cmb_idx.addItems(df.columns.astype(str).tolist())
            self.cmb_idx.setCurrentIndex(0)         
            self.cmb_idx.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
            self.cmb_idx.setMinimumContentsLength(6)
        finally:
            # 解锁信号，确保切换“样本名称”时能即时刷新图形
            self.cmb_idx.blockSignals(False)

        # 刷新 X/Y 轴候选 + 绘图
        self._refresh_axis_comboboxes(df.columns.tolist(), trigger_plot=False)
        self._refresh_category_filter()
        self._update_plot()

    def set_selected_columns(self, cols: list[str]):
        self._sel_cols = cols
        # 只把“已选特征”放到 X/Y 轴候选里，方便用户定位
        self._refresh_axis_comboboxes(cols)

    def index_column(self) -> str | None:
        """Return the currently selected sample name column, if any."""
        idx_choice = self.cmb_idx.currentText()
        if idx_choice != "递增数列" and idx_choice in self._df.columns:
            return idx_choice
        return None

    def set_target_column(self, column: str | None):
        """
        开启监督学习后，传入样本标签列名；传 ``None`` 则关闭类别筛选。
        """
        if column and column in self._df.columns:
            self._target_column = column
        else:
            self._target_column = None
        self._refresh_category_filter()
        self._update_plot()

    # ======= 内部工具 ======================================================
    def _refresh_axis_comboboxes(self, cols: list[str], *, trigger_plot: bool = True):
        """
        用新的列名列表刷新 X/Y 轴下拉框。若当前选中列已不在列表中，
        自动 fallback 到列表第 1 个；并保证至少有选项被选中。
        """
        for cmb in (self.cmb_x, self.cmb_y):
            current = cmb.currentText()
            cmb.blockSignals(True)  # 避免中途触发 _update_plot
            cmb.clear()
            cmb.addItems(cols)

            # ① 尝试保持原来的列
            if current in cols:
                cmb.setCurrentText(current)
            # ② 否则默认选第一列（如果有的话）
            elif cols:
                cmb.setCurrentIndex(0)

            cmb.blockSignals(False)

        if trigger_plot:
            self._update_plot()

    def _update_control_visibility(self):
        show_axes = self.cmb_type.currentText() == "散点图"
        for widget in getattr(self, "_axis_widgets", []):
            widget.setVisible(show_axes)

    def _refresh_category_filter(self):
        """根据当前标签列刷新类别筛选下拉框。"""
        if not self._target_column or self._target_column not in self._df.columns:
            self._category_colors = {}
            if self.category_filter.has_entries():
                self.category_filter.clear_categories()
            else:
                self.category_filter.setVisible(False)
            return

        series = self._df[self._target_column].dropna()
        if series.empty:
            self._category_colors = {}
            if self.category_filter.has_entries():
                self.category_filter.clear_categories()
            else:
                self.category_filter.setVisible(False)
            return

        labels = pd.Index(series.astype(str)).drop_duplicates().tolist()
        if not labels:
            self._category_colors = {}
            if self.category_filter.has_entries():
                self.category_filter.clear_categories()
            else:
                self.category_filter.setVisible(False)
            return

        cmap = plt.get_cmap("tab20", max(len(labels), 1))
        color_map: dict[str, str] = {}
        entries: list[tuple[str, str]] = []
        for idx, label in enumerate(labels):
            color_hex = to_hex(cmap(idx))
            color_map[label] = color_hex
            entries.append((label, label))

        self._category_colors = color_map
        self.category_filter.set_categories(entries, color_map)

    def _on_category_selection_changed(self, _values: list[str]):
        self._update_plot()

    # ======= 绘图核心 ======================================================
    def _plot_on_axes(self, ax) -> bool:
        ax.clear()
        if self._df.empty:
            return False

        idx_choice = self.cmb_idx.currentText()
        if idx_choice != "递增数列" and idx_choice in self._df.columns:
            df_use = self._df.set_index(idx_choice, drop=False)
        else:
            df_use = self._df

        chart = self.cmb_type.currentText()
        has_target = bool(self._target_column) and self._target_column in df_use.columns
        show_category_filter = (
            chart == "散点图"
            and has_target
            and self.category_filter.has_entries()
        )
        self.category_filter.setVisible(show_category_filter)

        if chart == "折线图":
            if not self._sel_cols:
                return False
            for col in self._sel_cols:
                ax.plot(df_use.index, df_use[col], label=col)
            ax.legend()
            ax.set_xlabel("样本")
            num_ticks = min(5, len(df_use))
            tick_indices = np.linspace(0, len(df_use) - 1, num=num_ticks, dtype=int)
            tick_values = df_use.index[tick_indices]
            ax.set_xticks(tick_values)
            if np.issubdtype(df_use.index.dtype, np.str_) or df_use.index.dtype == object:
                ax.set_xticklabels(tick_values, rotation=45)

        elif chart == "散点图":
            x, y = self.cmb_x.currentText(), self.cmb_y.currentText()
            if not x or not y or x == y:
                return False
            applied_coloring = False
            if show_category_filter:
                selected = self.category_filter.selected_values()
                if selected:
                    label_series = df_use[self._target_column].astype(str)
                    mask = label_series.isin(selected)
                    filtered = df_use[mask]
                    label_series = label_series[mask]
                    if filtered.empty:
                        return False
                    for label in pd.Index(label_series).drop_duplicates():
                        subset = filtered[label_series == label]
                        kwargs: dict[str, Any] = {"alpha": 0.75}
                        color_hex = self._category_colors.get(label)
                        if color_hex:
                            kwargs["color"] = color_hex
                        ax.scatter(subset[x], subset[y], label=label, **kwargs)
                    ax.legend(title=self._target_column, loc="best")
                    applied_coloring = True
            if not applied_coloring:
                ax.scatter(df_use[x], df_use[y], alpha=0.6)
            ax.set_xlabel(x)
            ax.set_ylabel(y)

        elif chart == "直方图":
            col = self.cmb_x.currentText()
            if not col:
                return False
            ax.hist(df_use[col].dropna(), bins=30, alpha=0.7)
            ax.set_xlabel(col)

        elif chart == "箱线图":
            if not self._sel_cols:
                return False
            data = [df_use[c].dropna() for c in self._sel_cols]
            ax.boxplot(data, labels=self._sel_cols, vert=False)

        else:
            return False

        return True

    def _update_plot(self):
        self._update_control_visibility()
        rendered = self._plot_on_axes(self.ax)
        if rendered:
            self.canvas.figure.tight_layout()
        self.canvas.draw()

    def _open_enlarge_dialog(self):
        dlg = PreviewEnlargeDialog(self)
        if not self._plot_on_axes(dlg.canvas.ax):
            dlg.close()
            QMessageBox.information(self, "提示", "暂无可放大的图表。")
            return
        dlg.canvas.figure.tight_layout()
        dlg.canvas.draw()
        dlg.show()
