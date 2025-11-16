from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QStackedLayout, QMessageBox
)
from PyQt6.QtCore import Qt
from .data_handle import DataHandlePage
from .unsupervised_page import UnsupervisedPage
from .supervised_page import SupervisedPage
from .validation_page import ValidationPage
from backend.ml_interface import ML, infer_input_features

class OutlierDetectionPage(QWidget):
    """Top-level page managing the data handle and ML views."""

    def __init__(self) -> None:
        super().__init__()
        self._step = 0
        # flags to avoid resetting selections when navigating back
        self._unsup_inited = False
        self._sup_inited = False

        layout = QVBoxLayout(self)

        # ============== 顶部：步骤标题（居中） + 导航按钮（右侧） ==============
        header = QVBoxLayout()

        # 步骤标题行（整体居中），按钮同一行
        steps_row = QHBoxLayout()
        self.prev_btn = QPushButton("上一步")
        self.prev_btn.clicked.connect(self.prev_step)
        self.next_btn = QPushButton("下一步")
        self.next_btn.clicked.connect(self.next_step)
        steps_row.addWidget(self.prev_btn)
        steps_row.addStretch()
        self.labels = []
        for text in ["数据处理", "异常检测", "监督学习", "验证预测"]:
            lbl = QLabel(text)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            # 统一的胶囊按钮
            lbl.setMinimumHeight(32)
            lbl.setProperty("step", "")  # 用属性控制是否为当前步骤
            self.labels.append(lbl)
            steps_row.addWidget(lbl)
        steps_row.addStretch()
        steps_row.addWidget(self.next_btn)
        steps_row.setSpacing(12)  # 四个标题之间的间距
        header.addLayout(steps_row)

        layout.addLayout(header)

        # ----------------- 中间：页面堆叠 -----------------
        self.stack = QStackedLayout()
        self.data_page = DataHandlePage()
        self.unsup_page = UnsupervisedPage()
        self.sup_page = SupervisedPage()
        self.valid_page = ValidationPage()
        self.data_page.data_status_changed.connect(lambda _status: self.update_steps())
        self.stack.addWidget(self.data_page)
        self.stack.addWidget(self.unsup_page)
        self.stack.addWidget(self.sup_page)
        self.stack.addWidget(self.valid_page)
        layout.addLayout(self.stack)

        # ----------------- 统一样式 -----------------
        # 说明：
        # - 普通/未选中：轻量标签
        # - 当前步骤：带背景、圆角、加粗，模拟“不可点击按钮”
        self.setStyleSheet("""
            QLabel {
                padding: 6px 12px;
                border-radius: 8px;
                border: 1px solid transparent;
                font-size: 14px;
            }
            QLabel[step=""] {
                color: palette(window-text);
                background: transparent;
                font-weight: 500;
            }
            QLabel[step="current"] {
                /* 跟随窗口背景色（更像任务栏的浅色系） */
                background: palette(window);          
                color: palette(window-text);          
                border: 1px solid palette(midlight);  /* 边框用系统的浅中性色 */
                font-weight: 600;
                font-size: 16px;
            }
        """)
        self.update_steps()

    def update_steps(self) -> None:
        for i, lbl in enumerate(self.labels):
            lbl.setProperty("step", "current" if i == self._step else "")
            # 刷新样式以应用属性变化
            lbl.style().unpolish(lbl)
            lbl.style().polish(lbl)
        allow_next = self._step < self.stack.count() - 1
        if self._step == 0 and not self.data_page.has_loaded_data():
            allow_next = False
        self.prev_btn.setEnabled(self._step > 0)
        self.next_btn.setEnabled(allow_next)
        self.stack.setCurrentIndex(self._step)
        if self._step == 3:
            src = self.sup_page if self.data_page.has_target() else self.unsup_page
            feats = src.selected_columns()
            recipes = ML.get_calc_recipes()
            raw_feats = infer_input_features(feats, recipes)
            self.valid_page.configure(raw_feats)

    def next_step(self) -> None:
        if self._step == 0:
            if not self.data_page.has_loaded_data():
                QMessageBox.warning(self, "提示", "请先加载数据后再进行下一步。")
                return
            if self.data_page.has_target() and not self.data_page.target_column():
                QMessageBox.warning(self, "提示", "请选择监督学习的样本标签列。")
                return
        if self._step == 0:
            if self.data_page.has_target():
                if not self._sup_inited:
                    df_idx = self.data_page.dataframe_with_index()
                    self.sup_page.set_data(
                        df_idx,
                        self.data_page.selected_columns(),
                        target=self.data_page.target_column(),
                    )
                    self._sup_inited = True
                self._step = 2
            else:
                if not self._unsup_inited:
                    df_idx = self.data_page.dataframe_with_index()
                    self.unsup_page.set_data(
                        df_idx,
                        self.data_page.selected_columns(),
                    )
                    self._unsup_inited = True
                self._step = 1
        elif self._step in (1, 2):
            self._step = 3
        else:
            return
        self.update_steps()

    def prev_step(self) -> None:
        if self._step == 3:
            self._step = 2 if self.data_page.has_target() else 1
        elif self._step in (1, 2):
            self._step = 0
            # moving back to data page resets initialization
            self._unsup_inited = False
            self._sup_inited = False
        else:
            return
        self.update_steps()
