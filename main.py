import  sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QIcon
from PyQt6.QtCore import QTimer
from gui.set_style import get_sheet
from gui.main_controller import MainController

if __name__ == "__main__":
    app = QApplication(sys.argv)
    style_sheet = get_sheet("light")
    app.setStyleSheet(style_sheet)

    win = MainController()
    win.setWindowIcon(QIcon("./gui/icons/logo.png"))
    # 居中操作
    center_point = app.primaryScreen().availableGeometry().center()
    frame_geo = win.frameGeometry()
    frame_geo.moveCenter(center_point)
    win.move(frame_geo.topLeft())

    # 显示
    win.show()

    # 再最大化（视觉上会覆盖居中效果）
    QTimer.singleShot(100, win.showMaximized)

    sys.exit(app.exec())