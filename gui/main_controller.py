import sys
from PyQt6.QtCore import Qt, QRegularExpression
from PyQt6.QtGui import QSyntaxHighlighter, QTextCharFormat, QColor
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
import re

from gui.tools import logger

class QtLogHandler(logging.Handler):
    """Simple logging handler that appends logs to a QPlainTextEdit."""

    def __init__(self, widget: QPlainTextEdit):
        super().__init__()
        self.widget = widget
        self.setFormatter(
            logging.Formatter("[%(asctime)s] %(levelname)s | %(message)s", "%H:%M:%S")
        )

    ansi_re = re.compile(r"\x1b\[[0-9;]*m")

    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        msg = self.ansi_re.sub("", msg)
        self.widget.appendPlainText(msg)


class LogHighlighter(QSyntaxHighlighter):
    def __init__(self, doc) -> None:
        super().__init__(doc)

        # 通用格式
        self.f_info = QTextCharFormat(); self.f_info.setForeground(QColor("#16a34a"))
        self.f_warn = QTextCharFormat(); self.f_warn.setForeground(QColor("#f59e0b"))
        self.f_err  = QTextCharFormat(); self.f_err.setForeground(QColor("#ef4444"))
        self.f_num  = QTextCharFormat(); self.f_num.setForeground(QColor("#16a34a"))  # 数值用绿
        self.f_key  = QTextCharFormat(); self.f_key.setForeground(QColor("#2563eb"))  # 关键字段蓝

        # 仅匹配 token 本身
        self.re_info = QRegularExpression(r"\bINFO\b")
        self.re_warn = QRegularExpression(r"\bWARN(?:ING)?\b")


        # 精确匹配 “训练集误差:xx” / “测试集误差:xx” 中的 等号后的数值
        # 捕获组1是 key，组2是数值（含小数/科学计数）
        self.re_metric = QRegularExpression(
            r"(((?:训练|测试)集)\s*(?:[A-Za-z0-9][A-Za-z0-9_\-/]*)?\s*误差)\s*[:：]\s*"
            r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
        )
        self.re_metric_simple = QRegularExpression(
            r"((?:准确率|精确率|召回率|F1|MAE|MSE|R2)(?:误差)?)\s*[:：]\s*"
            r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
        )

    def _apply_all(self, regex: QRegularExpression, text: str, fmt: QTextCharFormat, group: int = 0):
        it = regex.globalMatch(text)
        while it.hasNext():
            m = it.next()
            start = m.capturedStart(group)
            length = m.capturedLength(group)
            if start >= 0 and length > 0:
                self.setFormat(start, length, fmt)

    def highlightBlock(self, text: str) -> None:
        self._apply_all(self.re_info, text, self.f_info)
        self._apply_all(self.re_warn, text, self.f_warn)
        self._apply_all(self.re_err, text, self.f_err)

        it = self.re_metric.globalMatch(text)
        while it.hasNext():
            m = it.next()
            key_start, key_len = m.capturedStart(1), m.capturedLength(1)  # 组1：整段key（蓝色）
            val_start, val_len = m.capturedStart(3), m.capturedLength(3)  # 组3：数值（绿色）
            if key_start >= 0: self.setFormat(key_start, key_len, self.f_key)
            if val_start >= 0: self.setFormat(val_start, val_len, self.f_num)

        self._apply_all(self.re_metric_simple, text, self.f_key, group=1)
        self._apply_all(self.re_metric_simple, text, self.f_num, group=2)

from gui.set_style import get_sheet
from gui.break_analysis.outlier_detection import OutlierDetectionPage
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

    EXPANDED_WIDTH = 200  # 展开宽度
    COLLAPSED_WIDTH = 60  # 折叠宽度（容纳图标）

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("特征核素智能诊断程序v1.0")
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
        self.left_frame.setFixedWidth(self.EXPANDED_WIDTH)
        left_layout = QVBoxLayout(self.left_frame)
        left_layout.setContentsMargins(4, 4, 4, 4)
        left_layout.setSpacing(4)

        # 功能按钮
        self.btn_anomaly = QPushButton()
        self.btn_damage = QPushButton()
        self.btn_nuclide = QPushButton()
        self.btn_ensemble = QPushButton()
        self.btn_pipeline = QPushButton()

        self.menu_items = [
            (self.btn_anomaly, "🔍", "异常检测与工况评估"),
            (self.btn_damage, "🔧", "损伤程度评估"),
            (self.btn_nuclide, "📈", "核素核素活度预测"),
            (self.btn_ensemble, "🧩", "模型集成"),
        ]

        for btn, icon, label in self.menu_items:
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setCheckable(True)
            btn.setToolTip(label)
            btn.setText(f"{icon} {label}")
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
        self.stack.addWidget(self.anomaly_page)
        self.stack.addWidget(self.damage_page)
        self.stack.addWidget(self.nuclide_page)
        self.stack.addWidget(self.ensemble_page)

        # 默认选中第一页
        self.btn_anomaly.setChecked(True)
        self.stack.setCurrentWidget(self.anomaly_page)

        # 底部日志窗口
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumHeight(150)
        self.highlighter = LogHighlighter(self.log_view.document())

        right_layout.addWidget(self.stack)
        right_layout.addWidget(self.log_view)

        # 按钮点击切换页面
        self.btn_nuclide.clicked.connect(lambda: self.switch_page(self.btn_nuclide, self.nuclide_page))
        self.btn_anomaly.clicked.connect(lambda: self.switch_page(self.btn_anomaly, self.anomaly_page))
        self.btn_damage.clicked.connect(lambda: self.switch_page(self.btn_damage, self.damage_page))
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
