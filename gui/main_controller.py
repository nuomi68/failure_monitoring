import sys
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QPushButton,
    QStackedWidget,
    QFrame,
    QPlainTextEdit,
)

import logging

from gui.tools import logger

class QtLogHandler(logging.Handler):
    """Simple logging handler that appends logs to a QPlainTextEdit."""

    def __init__(self, widget: QPlainTextEdit):
        super().__init__()
        self.widget = widget
        self.setFormatter(
            logging.Formatter("[%(asctime)s] %(levelname)s | %(message)s", "%H:%M:%S")
        )

    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        self.widget.appendPlainText(msg)

from gui.set_style import get_sheet
from gui.outlier_detection import OutlierDetectionPage
from gui.ensemble_page import EnsemblePage
from gui.nuclide_prediction.time_series_page import TimeSeriesPage
from gui.fault_level_page import FaultLevelPage
class MainController(QMainWindow):
    """主界面控制器

    侧边栏特性
    ---------------
    1. 启动时默认 *展开*（图标 + 文本）。
    2. 点击折叠按钮后，侧边栏缩至固定窄宽度，仅保留图标。
    3. 再次点击按钮可重新展开。

    去掉了悬浮浮窗逻辑 —— 折叠时仅显示图标 + Tooltip 提示文字。
    """

    EXPANDED_WIDTH = 180  # 展开宽度
    COLLAPSED_WIDTH = 60  # 折叠宽度（容纳图标）

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("主控界面")
        self.resize(1000, 600)

        # 当前状态：False → 展开；True → 折叠
        self.sidebar_collapsed = False

        self._setup_ui()

    # ------------------------------------------------------------------
    # UI 构建
    # ------------------------------------------------------------------
    def _setup_ui(self) -> None:
        # -------------------- 主布局 --------------------
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # -------------------- 左侧侧边栏 --------------------
        self.left_frame = QFrame()
        self.left_frame.setObjectName("Sidebar")
        self.left_frame.setFixedWidth(self.COLLAPSED_WIDTH)
        left_layout = QVBoxLayout(self.left_frame)
        left_layout.setContentsMargins(4, 4, 4, 4)
        left_layout.setSpacing(4)

        # 功能按钮
        self.btn_nuclide = QPushButton()
        self.btn_anomaly = QPushButton()
        self.btn_damage = QPushButton()
        self.btn_ensemble = QPushButton()

        self.menu_items = [
            (self.btn_nuclide, "📈", "核素预测"),
            (self.btn_anomaly, "🔍", "异常检测"),
            (self.btn_damage, "🔧", "损伤评估"),
            (self.btn_ensemble, "🧩", "模型集成"),
        ]

        for btn, icon, label in self.menu_items:
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setCheckable(True)
            btn.setToolTip(label)
            btn.setText(f"{icon}")
            left_layout.addWidget(btn)

        left_layout.addStretch()

        # 折叠 / 展开切换按钮
        self.toggle_btn = QPushButton("⏴")  # 初始箭头 ← 收起
        self.toggle_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.toggle_btn.setStyleSheet("border: none; font-size: 16px;")
        self.toggle_btn.clicked.connect(self.toggle_sidebar)
        left_layout.addWidget(self.toggle_btn, alignment=Qt.AlignmentFlag.AlignLeft)

        # 右侧内容区：堆栈切换与日志
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        self.stack = QStackedWidget()
        self.anomaly_page = OutlierDetectionPage()
        self.damage_page = FaultLevelPage()
        self.ensemble_page = EnsemblePage()
        self.nuclide_page = TimeSeriesPage()
        self.stack.addWidget(self.nuclide_page)
        self.stack.addWidget(self.anomaly_page)
        self.stack.addWidget(self.damage_page)
        self.stack.addWidget(self.ensemble_page)

        # 默认选中第一页
        self.btn_nuclide.setChecked(True)
        self.stack.setCurrentWidget(self.nuclide_page)

        # 底部日志窗口
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumHeight(150)

        right_layout.addWidget(self.stack)
        right_layout.addWidget(self.log_view)

        # 按钮点击切换页面
        self.btn_nuclide.clicked.connect(lambda: self.switch_page(self.btn_nuclide, self.nuclide_page))
        self.btn_anomaly.clicked.connect(lambda: self.switch_page(self.btn_anomaly, self.anomaly_page))
        self.btn_damage.clicked.connect(lambda: self.switch_page(self.btn_damage, self.damage_page))
        self.btn_ensemble.clicked.connect(lambda: self.switch_page(self.btn_ensemble, self.ensemble_page))
        self.btn_ensemble.clicked.connect(lambda: self.switch_page(self.btn_ensemble, self.ensemble_page))
        # 布局装载
        main_layout.addWidget(self.left_frame)
        main_layout.addWidget(right_widget, 4)

        # 将日志输出到文本框
        self._qt_handler = QtLogHandler(self.log_view)
        self._qt_handler.setLevel(logging.INFO)
        logger.addHandler(self._qt_handler)

    def closeEvent(self, event):
        logger.removeHandler(self._qt_handler)
        super().closeEvent(event)


    # ------------------------------------------------------------------
    # 页切换 & 侧边栏折叠逻辑
    # ------------------------------------------------------------------
    def switch_page(self, sender_btn: QPushButton, page: QWidget) -> None:
        """切换页面并更新按钮选中状态"""
        for btn, *_ in self.menu_items:
            btn.setChecked(False)
        sender_btn.setChecked(True)
        self.stack.setCurrentWidget(page)

    def toggle_sidebar(self) -> None:
        """展开 ↔ 折叠 侧边栏"""
        self.sidebar_collapsed = not self.sidebar_collapsed

        if self.sidebar_collapsed:
            self.left_frame.setFixedWidth(self.COLLAPSED_WIDTH)
            self.toggle_btn.setText("⏵")  # → 展开
            # 仅显示图标
            for btn, icon, _ in self.menu_items:
                btn.setText(icon)
        else:
            self.left_frame.setFixedWidth(self.EXPANDED_WIDTH)
            self.toggle_btn.setText("⏴")  # ← 收起
            for btn, icon, label in self.menu_items:
                btn.setText(f"{icon} {label}")


# ------------------------------------------------------------------
# 运行入口
# ------------------------------------------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    style_sheet = get_sheet("light")
    app.setStyleSheet(style_sheet)

    win = MainController()
    win.show()

    sys.exit(app.exec())
