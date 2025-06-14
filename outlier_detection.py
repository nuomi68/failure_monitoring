import sys
from PyQt6.QtWidgets import QApplication

from gui.outlier_detection import OutlierDetectionPage


def main() -> None:
    app = QApplication(sys.argv)
    win = OutlierDetectionPage()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":  # pragma: no cover
    main()
