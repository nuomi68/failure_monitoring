import pandas as pd
from PyQt6.QtWidgets import (
    QWidget, QFrame, QPushButton, QFileDialog,QSplitter,QStackedLayout,
    QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,QHeaderView,
    QListWidget, QAbstractItemView, QCheckBox, QComboBox,
    QLabel, QSizePolicy, QSpacerItem
)
from PyQt6.QtCore import Qt

from calculator_widget import CalculatorWidget
from ml_gui import MLWindow

class OutlierDetectionPage(QWidget):
    def __init__(self):
        super().__init__()

        self.df = pd.DataFrame()

        self.stack = QStackedLayout()
        self.setLayout(self.stack)

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

        # ---------- Page 1：真正的数据表 ----------
        self.table = QTableWidget()
        self.table.setSizePolicy(QSizePolicy.Policy.Expanding,
                                 QSizePolicy.Policy.Expanding)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.top_stack.addWidget(self.table)  # index 1

        # ---------- 底部：左右分割 ----------
        bottom_split = QSplitter(Qt.Orientation.Horizontal)
        main.addWidget(bottom_split, stretch=1)   # stretch=1 让它吃剩余空间

        # ===== 字段选择=====
        left_panel = QWidget()
        left_v = QVBoxLayout(left_panel)

        # -- 字段列表布局
        lists_layout = QHBoxLayout()
        # 左边：全部标签列表（带标题）
        left_column = QVBoxLayout()
        all_label_title = QLabel("全部特征")
        all_label_title.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        left_column.addWidget(all_label_title, alignment=Qt.AlignmentFlag.AlignCenter)
        self.list_all = QListWidget()
        self.list_all.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        left_column.addWidget(self.list_all)

        # 中间：箭头按钮区（含提示）
        arrows = QVBoxLayout()
        arrow_info = [
            ("→", "选择", self.move_one_right),
            ("←", "移除", self.move_one_left),
            ("≫", "全部添加", self.move_all_right),
            ("≪", "全部移除", self.move_all_left),
        ]
        for text, tooltip, slot in arrow_info:
            btn = QPushButton(text)
            btn.setMaximumWidth(40)
            btn.setToolTip(tooltip)  # 设置悬浮提示
            btn.clicked.connect(slot)
            arrows.addWidget(btn)

        # 右边：已选标签列表（带标题）
        right_column = QVBoxLayout()
        select_label_title = QLabel("选择特征")
        select_label_title.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        right_column.addWidget(select_label_title, alignment=Qt.AlignmentFlag.AlignCenter)
        self.list_selected = QListWidget()
        self.list_selected.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        right_column.addWidget(self.list_selected)

        # 加入主布局
        lists_layout.addLayout(left_column)
        lists_layout.addLayout(arrows)
        lists_layout.addLayout(right_column)

        # 加入父布局
        left_v.addLayout(lists_layout)
        bottom_split.addWidget(left_panel)

        # ===== 右半======
        right_panel = QWidget()
        right_v = QVBoxLayout(right_panel)

        # ──右下角的清空按钮 ──
        bottom_bar = QHBoxLayout()
        bottom_bar.addStretch()
        self.btn_reset = QPushButton("清空表格")
        self.btn_reset.clicked.connect(self.reset_ui)
        bottom_bar.addWidget(self.btn_reset)
        # 下一步进入算法训练界面
        self.btn_next = QPushButton("下一步")
        self.btn_next.clicked.connect(self.open_ml_window)
        bottom_bar.addWidget(self.btn_next)
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

        # ---------- 可选：设置左右初始比例 ----------
        bottom_split.setStretchFactor(0, 1)   # 左 1
        bottom_split.setStretchFactor(1, 1)   # 右 1

        self.stack.addWidget(self.data_page)
        self.ml_window = MLWindow()
        self.stack.addWidget(self.ml_window)
        self.stack.setCurrentWidget(self.data_page)

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择Excel文件",
                                              "", "Excel 文件 (*.xlsx *.xls)")
        if not path:
            return
        df = pd.read_excel(path)
        self.df = df
        self.populate_table(df)
        self.populate_lists(df.columns.tolist())
        self.calc.setDataFrame(df)
        self.calc.new_column.connect(self._on_new_column)

        self.top_stack.setCurrentIndex(1)      # 只切上半部分！

    def reset_ui(self):
        self.table.clear()
        self.list_all.clear()
        self.list_selected.clear()
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
        self.list_all.clear()
        self.list_selected.clear()
        self.cmb.clear()
        for c in cols:
            self.list_all.addItem(c)
            self.cmb.addItem(c)

    def move_one_right(self):
        for it in self.list_all.selectedItems():
            self.list_selected.addItem(it.text())
            self.list_all.takeItem(self.list_all.row(it))

    def move_one_left(self):
        for it in self.list_selected.selectedItems():
            self.list_all.addItem(it.text())
            self.list_selected.takeItem(self.list_selected.row(it))

    def move_all_right(self):
        while self.list_all.count():
            it = self.list_all.takeItem(0)
            self.list_selected.addItem(it.text())

    def move_all_left(self):
        while self.list_selected.count():
            it = self.list_selected.takeItem(0)
            self.list_all.addItem(it.text())

    def toggle_target(self, state):
        self.cmb.setEnabled(state == Qt.CheckState.Checked)

    def _on_new_column(self, name: str, col: pd.Series):
        """收到计算器的新列后，同步 UI."""
        # 1) 表格（这里只显示前 100 行）
        col_idx = self.table.columnCount()
        self.table.insertColumn(col_idx)
        self.table.setHorizontalHeaderItem(col_idx, QTableWidgetItem(name))
        for i in range(min(100, len(col))):
            self.table.setItem(i, col_idx, QTableWidgetItem(str(col.iat[i])))

        # 2) 左侧 / 右侧列表 & 下拉框
        self.list_all.addItem(name)
        self.cmb.addItem(name)
        self.cmb.addItem(name)

    def selected_columns(self) -> list[str]:
        """返回已选择的特征名称列表"""
        return [self.list_selected.item(i).text() for i in range(self.list_selected.count())]

    def open_ml_window(self):
        """打开算法训练界面"""
        if self.df.empty:
            return
        self.ml_window.set_data(self.df, self.selected_columns())
        self.stack.setCurrentWidget(self.ml_window)
