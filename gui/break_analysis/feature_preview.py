# ================= feature_preview.py =================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QSizePolicy
)
from PyQt6.QtCore import Qt

class HeatmapCanvas(FigureCanvas):
    """相关系数热力图——嵌入式 Matplotlib 画布"""

    def __init__(self, parent=None):
        self.fig, self.ax = plt.subplots()
        super().__init__(self.fig)
        self.setParent(parent)
        self.colorbar = None

    def plot_corr(self, df: pd.DataFrame):
        """给定 DF，画皮尔逊相关矩阵；空 DF 时清屏。"""
        # 清除旧的 colorbar（如果有）
        if self.colorbar:
            self.colorbar.remove()
            self.colorbar = None

        self.ax.clear()

        if df.empty or df.shape[1] < 2:
            self.draw()
            return

        corr = df.corr(numeric_only=True)
        im = self.ax.imshow(corr.values, vmin=-1, vmax=1, cmap="coolwarm")
        # 坐标轴标签
        self.ax.set_xticks(np.arange(len(corr.columns)), corr.columns, rotation=90)
        self.ax.set_yticks(np.arange(len(corr.columns)), corr.columns)
        # 颜色条
        self.colorbar = self.fig.colorbar(im, ax=self.ax, shrink=0.7)
        self.fig.tight_layout()
        self.draw()


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
        # 样本名称 / 行索引选择 ----------------------
        ctrl.addWidget(QLabel("样本名称:"))
        self.cmb_idx = QComboBox()              # ← 新增
        self.cmb_idx.currentIndexChanged.connect(self._update_plot)
        ctrl.addWidget(self.cmb_idx)

        ctrl.addWidget(QLabel("图形类型:"))
        self.cmb_type = QComboBox()
        self.cmb_type.addItems(self.CHART_TYPES)
        self.cmb_type.currentIndexChanged.connect(self._update_plot)
        ctrl.addWidget(self.cmb_type)

        # X / Y 轴列选择（仅散点/直方用） -------------------------
        self.cmb_x = QComboBox()
        self.cmb_y = QComboBox()
        for cmb, name in [(self.cmb_x, "X 轴"), (self.cmb_y, "Y 轴")]:
            ctrl.addWidget(QLabel(name + ":"))
            cmb.currentIndexChanged.connect(self._update_plot)
            ctrl.addWidget(cmb)

        ctrl.addStretch()
        root.addLayout(ctrl)

        # 中部：Matplotlib 画布 ------------------------------
        self.canvas = _PreviewCanvas()
        root.addWidget(self.canvas)
        self.ax = self.canvas.ax

        # --- 状态 ----------------------------------------------------------
        self._df: pd.DataFrame = pd.DataFrame()
        self._sel_cols: list[str] = []

    # ======= 公共 API ======================================================
    def set_dataframe(self, df: pd.DataFrame):
        self._df = df

        # ── 刷新“样本名称”下拉框 ──
        # 第一项固定为 “递增数列(默认)”，随后是所有列名
        self.cmb_idx.blockSignals(True)
        self.cmb_idx.clear()
        self.cmb_idx.addItem("递增数列(默认)")       # index == 0 → 不改索引
        self.cmb_idx.addItems(df.columns.astype(str).tolist())
        self.cmb_idx.setCurrentIndex(0)            # 保持默认
        self.cmb_idx.blockSignals(False)

        # 刷新 X/Y 轴候选 + 绘图
        self._refresh_axis_comboboxes(df.columns.tolist())
        self._update_plot()

    def set_selected_columns(self, cols: list[str]):
        self._sel_cols = cols
        # 只把“已选特征”放到 X/Y 轴候选里，方便用户定位
        self._refresh_axis_comboboxes(cols)
        self._update_plot()

    # ======= 内部工具 ======================================================
    def _refresh_axis_comboboxes(self, cols: list[str]):
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

        # 手动再调一次 _update_plot，确保画布重绘
        self._update_plot()

    # ======= 绘图核心 ======================================================
    def _update_plot(self):
        self.ax.clear()
        if self._df.empty:
            self.canvas.draw();
            return

        # —— 取用户选择的行索引（样本名称） ————————————————
        idx_choice = self.cmb_idx.currentText()
        if idx_choice != "递增数列(默认)" and idx_choice in self._df.columns:
            df_use = self._df.set_index(idx_choice, drop=False)
        else:
            df_use = self._df

        chart = self.cmb_type.currentText()

        # -------- 折线图 ---------------------------------------------------
        if chart == "折线图":
            if not self._sel_cols:
                self.canvas.draw()
                return
            for col in self._sel_cols:
                self.ax.plot(df_use.index, df_use[col], label=col)
            self.ax.legend()
            self.ax.set_xlabel("样本")

            # ✅ 等距选择 4~5 个横坐标标签
            num_ticks = min(5, len(df_use))
            tick_indices = np.linspace(0, len(df_use) - 1, num=num_ticks, dtype=int)
            tick_values = df_use.index[tick_indices]
            self.ax.set_xticks(tick_values)

            # 如果 index 是文本（如时间标签），设置文字标签
            if np.issubdtype(df_use.index.dtype, np.str_) or df_use.index.dtype == object:
                self.ax.set_xticklabels(tick_values, rotation=45)

        # -------- 散点图 ---------------------------------------------------
        elif chart == "散点图":
            x, y = self.cmb_x.currentText(), self.cmb_y.currentText()
            if not x or not y or x == y:
                self.canvas.draw();
                return
            self.ax.scatter(df_use[x], df_use[y], alpha=0.6)
            self.ax.set_xlabel(x);
            self.ax.set_ylabel(y)

        # -------- 直方图 ---------------------------------------------------
        elif chart == "直方图":
            col = self.cmb_x.currentText()
            if not col:
                self.canvas.draw();
                return
            self.ax.hist(df_use[col].dropna(), bins=30, alpha=0.7)
            self.ax.set_xlabel(col)

        # -------- 箱线图 ---------------------------------------------------
        elif chart == "箱线图":
            if not self._sel_cols:
                self.canvas.draw();
                return
            data = [df_use[c].dropna() for c in self._sel_cols]
            self.ax.boxplot(data, labels=self._sel_cols, vert=False)

        self.canvas.figure.tight_layout()
        self.canvas.draw()
