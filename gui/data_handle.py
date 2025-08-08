import pandas as pd

from PyQt6.QtWidgets import (
    QWidget, QPushButton, QFileDialog, QSplitter, QStackedLayout,
    QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem, QHeaderView,
    QAbstractItemView, QCheckBox, QComboBox,
    QLabel, QSizePolicy, QSpacerItem
)
from PyQt6.QtCore import Qt

from gui.calculator_widget import CalculatorWidget
from gui.feature_preview import FeaturePreviewWidget,HeatmapCanvas
from .feature_selector_widget import FeatureSelectorWidget
from gui.tools import logger

class DataHandlePage(QWidget):
    """Data preprocessing interface embedding the ML window."""

    def __init__(self):
        super().__init__()

        self.df = pd.DataFrame()

        layout = QVBoxLayout(self)


        self.data_page = QWidget()
        main = QVBoxLayout(self.data_page)          # 顶层垂直
        title = QLabel("选择数据")
        title.setStyleSheet("font-weight:600; font-size:26px;")
        title.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        main.addWidget(title, alignment=Qt.AlignmentFlag.AlignCenter)
        # ── 1. 上半部分：用栈布局切换（示例页 / 数据表） ──
        upper_container = QWidget()
        upper_v = QVBoxLayout(upper_container)
        self.top_stack = QStackedLayout()
        upper_v.addLayout(self.top_stack)
        main.addWidget(upper_container, stretch=1)  # 给上半部分多一点伸展空间

        # ---------- Page 0：示例表 + 上传按钮 ----------
        upper_placeholder = QWidget()
        # 用布局替代 QSplitter
        ph_h = QHBoxLayout(upper_placeholder)
        ph_h.setSpacing(32)  # 两者间距
        ph_h.setContentsMargins(0, 0, 0, 0)

        # === ① 标题 + 示例表 =========
        demo_wrap = QWidget()
        demo_v = QVBoxLayout(demo_wrap)
        title = QLabel("表格内容示例")
        title.setStyleSheet("font-weight:600; font-size:20px;")  # 粗体、稍大
        title.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        demo_v.addWidget(title, alignment=Qt.AlignmentFlag.AlignCenter)

        # 行×列数量
        rows, cols = 10,10  # ← 根据需要增减

        demo = QTableWidget(rows, cols)
        demo.setHorizontalHeaderLabels([f"测量值{i + 1}" for i in range(cols)])
        demo.setVerticalHeaderLabels([f"样本{i + 1}" for i in range(rows - 1)] + [""])

        # 自适应列宽 & 行高
        demo.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        demo.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        demo.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Expanding)  # ★ 关键：可拉伸
        demo_v.addWidget(demo)

        ph_h.addWidget(demo_wrap,1)

        # === ② 上传按钮（大号虚线框） ===
        up_wrap = QWidget()
        up_v = QVBoxLayout(up_wrap)
        up_v.addStretch()

        self.btn_open = QPushButton("选择表格文件")
        self.btn_open.clicked.connect(self.open_file)
        # ---- 样式 & 尺寸 ----
        self.btn_open.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.btn_open.setMinimumHeight(120)  # 固定高度，自适应宽度
        self.btn_open.setStyleSheet("""
            QPushButton {
                background-color: #fafafa;
                border: 2px dashed #9ca3af;        /* 灰色虚线框 */
                border-radius: 30px;
                color: #4b5563;                    /* 深灰文字 */
                font-size:30px;
            }
            QPushButton:hover {
                background-color: #f3f4f6;
                border-color: #6b7280;             /* hover 变深一点 */
            }
            QPushButton:pressed {
                background-color: #e5e7eb;
                border-style: solid;               /* 按压改成实线加强反馈 */
            }
        """)

        up_v.addWidget(self.btn_open)
        up_v.addStretch()
        ph_h.addWidget(up_wrap,1)

        self.top_stack.addWidget(upper_placeholder)  # index 0

        # ---------- Page 1：真正的数据表（表格左 2/3 + 热力图右 1/3） ----------
        page1 = QWidget()
        page1_layout = QVBoxLayout(page1)

        # 水平分割
        split = QSplitter(Qt.Orientation.Horizontal)

        # —— ① 表格 —— #
        self.table = QTableWidget()
        self.table.setSizePolicy(QSizePolicy.Policy.Expanding,
                                 QSizePolicy.Policy.Expanding)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        split.addWidget(self.table)

        # —— ② 热力图 —— #
        self.heatmap_canvas = HeatmapCanvas()  # ← 前面定义好的 Matplotlib 画布
        split.addWidget(self.heatmap_canvas)

        # 设置左右 2:1 比例
        split.setStretchFactor(0, 2)  # 表格
        split.setStretchFactor(1, 1)  # 热力图
        split.setHandleWidth(1)
        self.heatmap_canvas.setMaximumWidth(500)
        page1_layout.addWidget(split)
        self.top_stack.addWidget(page1)  # index 1

        # ---------- 底部：左右分割 ----------
        bottom_split = QSplitter(Qt.Orientation.Horizontal)
        main.addWidget(bottom_split, stretch=1)   # stretch=1 让它吃剩余空间

        # ===== 字段选择 =====
        left_panel = QWidget()
        left_v = QVBoxLayout(left_panel)
        self.feature_selector = FeatureSelectorWidget()
        left_v.addWidget(self.feature_selector)
        bottom_split.addWidget(left_panel)

        # === 中：图形预览================
        self.preview = FeaturePreviewWidget()
        bottom_split.addWidget(self.preview)
        self.feature_selector.selectionChanged.connect(self.preview.set_selected_columns)

        # ===== 计算器 ======
        right_panel = QWidget()
        right_v = QVBoxLayout(right_panel)

        # ──右下角的清空按钮 ──
        bottom_bar = QHBoxLayout()
        bottom_bar.addStretch()
        self.btn_reset = QPushButton("清空表格")
        self.btn_reset.clicked.connect(self.reset_ui)
        bottom_bar.addWidget(self.btn_reset)
        right_v.addLayout(bottom_bar)
        #=====计算器 == == =
        self.calc = CalculatorWidget()
        right_v.addWidget(self.calc)
        # -- 监督学习区 --
        sup = QHBoxLayout()
        self.chk = QCheckBox("监督学习")
        self.chk.stateChanged.connect(self.toggle_target)
        self.cmb = QComboBox()
        self.cmb.setEnabled(False)
        sup.addWidget(self.chk)
        sup.addWidget(QLabel("样本标签:"))
        sup.addWidget(self.cmb)
        sup.addItem(QSpacerItem(20, 20,
                                QSizePolicy.Policy.Expanding,
                                QSizePolicy.Policy.Minimum))
        right_v.addLayout(sup)
        bottom_split.addWidget(right_panel)

        right_v.addStretch() #空白位置

        # ---------- 设置左右初始比例 ----------
        bottom_split.setStretchFactor(0, 1)   # 左 1
        bottom_split.setStretchFactor(1, 1)   # 右 1

        layout.addWidget(self.data_page)
        self.load_dev_file()

    def load_dev_file(self):
        """开发阶段：写死加载一个本地 Excel 文件"""
        path = "./data/20230510-20240924_merged.xlsx"
        try:
            df = pd.read_excel(path)
        except Exception as e:
            logger.error("开发文件加载失败: %s", e)
            return

        removed = int(df.isna().any(axis=1).sum())
        if removed:
            df = df.dropna()
            logger.info("删除含 NaN 的行 %d 条", removed)

        self.df = df
        self.populate_table(df)
        self.populate_lists(df.columns.tolist())
        self.calc.setDataFrame(df)
        self.calc.new_column.connect(self._on_new_column)
        self.heatmap_canvas.plot_corr(self.df)
        self.top_stack.setCurrentIndex(1)
        self.preview.set_dataframe(df)
        self.preview.set_selected_columns(self.selected_columns())

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择Excel文件",
                                              "", "Excel 文件 (*.xlsx *.xls)")
        if not path:
            return
        df = pd.read_excel(path)
        removed = int(df.isna().any(axis=1).sum())
        if removed:
            df = df.dropna()
            logger.info("删除含 NaN 的行 %d 条", removed)
        self.df = df
        self.populate_table(df)
        self.populate_lists(df.columns.tolist())
        self.calc.setDataFrame(df)
        self.calc.new_column.connect(self._on_new_column)
        self.heatmap_canvas.plot_corr(self.df)
        self.top_stack.setCurrentIndex(1)      # 只切上半部分！

        self.preview.set_dataframe(df)
        self.preview.set_selected_columns(self.selected_columns())

    def reset_ui(self):
        self.table.clear()
        self.feature_selector.set_columns([])
        self.cmb.clear()
        self.calc.setDataFrame(pd.DataFrame())
        self.df = pd.DataFrame()
        self.top_stack.setCurrentIndex(0)      # 恢复到示例页

    def populate_table(self, df):
        sub = df.iloc[:100, :]
        self.table.clear()
        self.table.setColumnCount(len(sub.columns))
        self.table.setRowCount(len(sub))
        self.table.setHorizontalHeaderLabels([str(c) for c in sub.columns])
        for i, row in sub.iterrows():
            for j, val in enumerate(row):
                self.table.setItem(i, j, QTableWidgetItem(str(val)))
        self.table.resizeColumnsToContents()

    def populate_lists(self, cols):
        self.feature_selector.set_columns(cols)
        self.feature_selector.set_selected([])
        self.cmb.clear()
        for c in cols:
            self.cmb.addItem(c)

    def toggle_target(self, state):
        self.cmb.setEnabled(state == Qt.CheckState.Checked.value)

    def _on_new_column(self, name: str, col: pd.Series):
        """收到计算器的新列后，同步 UI."""
        # 1) 表格（这里只显示前 100 行）
        col_idx = self.table.columnCount()
        self.table.insertColumn(col_idx)
        self.table.setHorizontalHeaderItem(col_idx, QTableWidgetItem(name))
        for i in range(min(100, len(col))):
            self.table.setItem(i, col_idx, QTableWidgetItem(str(col.iat[i])))

        # 2) 字段选择器 & 下拉框
        prev_selected = self.feature_selector.selected()
        cols = self.feature_selector.columns() + [name]
        self.feature_selector.set_columns(cols)
        self.feature_selector.set_selected(prev_selected)
        self.cmb.addItem(name)

        self._update_heatmap()

    def selected_columns(self) -> list[str]:
        """返回已选择的特征名称列表"""
        return self.feature_selector.selected()

    def has_target(self) -> bool:
        """是否启用监督学习"""
        return self.chk.isChecked()

    def target_column(self) -> str | None:
        """返回选择的标签列名称，若未启用则为 ``None``"""
        return self.cmb.currentText() if self.has_target() and self.cmb.currentText() else None

    def _update_heatmap(self):
        """根据已选特征即时重绘热力图"""
        self.heatmap_canvas.plot_corr(self.df)

