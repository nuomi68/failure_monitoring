import  sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QIcon

from gui.set_style import get_sheet
from gui.main_controller import MainController

if __name__ == "__main__":
    app = QApplication(sys.argv)
    style_sheet = get_sheet("light")
    app.setStyleSheet(style_sheet)

    win = MainController()
    win.setWindowIcon(QIcon("./gui/icons/logo.png"))
    win.show()

    sys.exit(app.exec())