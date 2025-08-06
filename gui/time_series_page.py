"""Simple PyQt page for training time-series forecasting models."""

import pandas as pd

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QPushButton,
    QLabel,
    QFileDialog,
    QTextEdit,
    QLineEdit,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
)

from backend.timeseries_interface import ModelManager


class TimeSeriesPage(QWidget):
    """Minimal page to upload a dataset and trigger training."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.manager = ModelManager()

        layout = QVBoxLayout(self)

        # 示例模板表格，让用户了解需要的列结构
        self.demo_table = QTableWidget(3, 3)
        self.demo_table.setHorizontalHeaderLabels(["TIME", "value1", "value2"])
        sample_times = ["2024年01月01日0000", "2024年01月01日0100", "..."]
        sample_vals = [[1.0, 2.0], [1.5, 2.5], ["", ""]]
        for r in range(3):
            self.demo_table.setItem(r, 0, QTableWidgetItem(sample_times[r]))
            for c in range(2):
                self.demo_table.setItem(r, c + 1, QTableWidgetItem(str(sample_vals[r][c])))
        self.demo_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(QLabel("表格内容模板（需包含 TIME 列）"))
        layout.addWidget(self.demo_table)
        self.btn_template = QPushButton("导出模板")
        self.btn_template.clicked.connect(self._save_template)
        layout.addWidget(self.btn_template)

        # 时间格式输入
        layout.addWidget(QLabel("时间格式，例如 %Y年%m月%d日%H%M"))
        self.time_fmt_edit = QLineEdit("%Y年%m月%d日%H%M")
        layout.addWidget(self.time_fmt_edit)

        self.status_label = QLabel("未训练")
        self.btn_select = QPushButton("选择文件并训练")
        self.btn_select.clicked.connect(self._choose_file)
        self.result_view = QTextEdit()
        self.result_view.setReadOnly(True)

        layout.addWidget(self.status_label)
        layout.addWidget(self.btn_select)
        layout.addWidget(self.result_view)

    # ------------------------------------------------------------------
    def _choose_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择数据文件", "", "Excel Files (*.xlsx);;CSV Files (*.csv)"
        )
        if not path:
            return
        self.status_label.setText("训练中...")
        fmt = self.time_fmt_edit.text().strip() or None
        self.manager.train(path, fmt)
        self.status_label.setText(self.manager.status)

        lines = []
        for res in self.manager.last_predictions:
            max_err = res["max_err"].max()
            mean_err = res["mean_err"].mean()
            lines.append(
                f"第 {res['step']} 步: max_err={max_err:.4f}, mean_err={mean_err:.4f}"
            )
        self.result_view.setPlainText("\n".join(lines))

    # ------------------------------------------------------------------
    def _save_template(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "保存模板", "time_series_template.xlsx", "Excel Files (*.xlsx)"
        )
        if not path:
            return
        df = pd.DataFrame(
            {
                "TIME": ["2024年01月01日0000", "2024年01月01日0100"],
                "value1": [1.0, 1.5],
                "value2": [2.0, 2.5],
            }
        )
        df.to_excel(path, index=False)

