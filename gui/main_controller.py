import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QStackedWidget, QFrame
)
from PyQt6.QtCore import Qt

from set_style import get_sheet
from outlier_detection import OutlierDetectionPage

class MainController(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("主控界面")
        self.resize(1000, 600)
        self._setup_ui()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # 左侧按钮区
        left_frame = QFrame()
        left_frame.setObjectName("Sidebar")
        left_layout = QVBoxLayout(left_frame)
        self.btn_online = QPushButton("在线监测")
        self.btn_anomaly = QPushButton("异常检测")
        self.btn_damage = QPushButton("损伤计算")
        left_layout.addWidget(self.btn_online)
        left_layout.addWidget(self.btn_anomaly)
        left_layout.addWidget(self.btn_damage)
        left_layout.addStretch()

        # 右侧内容区：堆栈切换
        self.stack = QStackedWidget()
        # 占位页面
        self.online_page = QWidget()
        self.damage_page = QWidget()
        # 异常检测页面
        self.anomaly_page = OutlierDetectionPage()

        # 将页面加入堆栈，顺序与按钮一一对应
        self.stack.addWidget(self.online_page)
        self.stack.addWidget(self.anomaly_page)
        self.stack.addWidget(self.damage_page)

        # 按钮点击切换对应页面
        self.btn_online.clicked.connect(lambda: self.stack.setCurrentWidget(self.online_page))
        self.btn_anomaly.clicked.connect(lambda: self.stack.setCurrentWidget(self.anomaly_page))
        self.btn_damage.clicked.connect(lambda: self.stack.setCurrentWidget(self.damage_page))

        main_layout.addWidget(left_frame, 1)
        main_layout.addWidget(self.stack, 4)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    style_sheet = get_sheet("light")
    app.setStyleSheet(style_sheet)
    win = MainController()
    win.show()
    sys.exit(app.exec())
