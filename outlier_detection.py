import sys
from PyQt6.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, QTextEdit
)

from gui.ml_gui import MLWindow


class DataProcessingWindow(QWidget):
    """Simple window performing data preprocessing then jumping to ML."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Outlier Detection")
        self.resize(600, 400)
        layout = QVBoxLayout(self)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log)

        self.to_ml_btn = QPushButton("进入机器学习")
        self.to_ml_btn.clicked.connect(self.open_ml)
        layout.addWidget(self.to_ml_btn)

        self._process_data()

    def _process_data(self) -> None:
        """Placeholder for data processing steps."""
        self.log.append("数据处理完成")

    def open_ml(self) -> None:
        self.ml_win = MLWindow()
        self.ml_win.show()


def main() -> None:
    app = QApplication(sys.argv)
    win = DataProcessingWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":  # pragma: no cover
    main()
