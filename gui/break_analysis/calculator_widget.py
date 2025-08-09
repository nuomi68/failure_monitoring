import sys
import numpy as np
import pandas as pd
import keyword
from PyQt6.QtWidgets import (
    QApplication, QGroupBox, QWidget, QGridLayout, QLineEdit,
    QComboBox, QPushButton, QLabel, QMessageBox
)
from PyQt6.QtCore import Qt, pyqtSignal



class CalculatorWidget(QGroupBox):
    new_column = pyqtSignal(str, object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("计算新特征", parent)
        self.df: pd.DataFrame | None = None
        self._init_ui()

    # --------------------------- UI 布局 ---------------------------------
    def _init_ui(self) -> None:
        g = QGridLayout(self)

        # 网格共 5 列：0 = 标签 / 左侧按钮
        #            1-3 = 输入区或按钮区（占 3 格）
        #            4 = 右侧功能按钮
        # 结果名 -----------------------------------------------------------
        g.addWidget(QLabel("新特征名"), 0, 0)

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("请输入新特征名")
        g.addWidget(self.name_edit, 0, 1, 1, 3)   # 跨 3 列

        btn_add = QPushButton("添加")
        g.addWidget(btn_add, 0, 4)

        # 表达式 -----------------------------------------------------------
        g.addWidget(QLabel("表达式"), 1, 0)

        self.expr_edit = QLineEdit()
        self.expr_edit.setPlaceholderText("输入计算表达式")
        g.addWidget(self.expr_edit, 1, 1, 1, 3)

        btn_clear = QPushButton("清空")
        g.addWidget(btn_clear, 1, 4)

        # 按键矩阵 ---------------------------------------------------------
        key_rows = [
            ["(", ")", "+", "-", "√"],
            ["7", "8", "9", "*", "^"],
            ["4", "5", "6", "/", "abs"],
            ["1", "2", "3", "ln", "log10"],
            ["", "0", ".",  "",  ""]
        ]
        start_row = 2
        for r, row in enumerate(key_rows):
            for c, tok in enumerate(row):
                if not tok:
                    continue
                b = QPushButton(tok)
                b.clicked.connect(lambda _, t=tok: self._insert_token(t))
                g.addWidget(b, start_row + r, c)

        # ---------------------- 下拉框 --------------------------------------
        self.var_combo = QComboBox()
        self.var_combo.setEditable(False)  # 不可编辑
        self.var_combo.setPlaceholderText("选择特征")  # 占位符文字
        g.addWidget(self.var_combo, start_row + len(key_rows) - 1, 3, 2, 3)

        # 列伸缩
        for col in range(5):
            g.setColumnStretch(col, 1)

        # 信号
        btn_add.clicked.connect(self._apply_expression)
        btn_clear.clicked.connect(self.expr_edit.clear)
        self.var_combo.activated.connect(self._on_combo_activated)

    def setDataFrame(self, df: pd.DataFrame) -> None:
        self.df = df
        self.var_combo.blockSignals(True)
        self.var_combo.clear()
        self.var_combo.addItems(df.columns.tolist())
        self.var_combo.setCurrentIndex(-1)  # 关键：无选中项
        self.var_combo.blockSignals(False)

    # ---------- 帮助函数：判断是否合法标识符 ----------
    @staticmethod
    def _is_identifier(name: str) -> bool:
        return name.isidentifier() and not keyword.iskeyword(name)

    # ---------- 插入标签 ----------
    def _on_combo_activated(self, index: int) -> None:
        if index < 0:
            return
        col = self.var_combo.currentText()
        token = col if self._is_identifier(col) else f"`{col}`"   # ★
        self._insert_token(token)
        self.var_combo.setCurrentIndex(-1)

    def _insert_token(self, token: str) -> None:
        cur = self.expr_edit.cursorPosition()
        txt = self.expr_edit.text()
        self.expr_edit.setText(txt[:cur] + token + txt[cur:])
        self.expr_edit.setCursorPosition(cur + len(token))

    # --------------------------- 计算 ------------------------------------
    def _apply_expression(self) -> None:
        if self.df is None:
            QMessageBox.warning(self, "错误", "请先注入 DataFrame")
            return

        name = self.name_edit.text().strip()
        expr = self.expr_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "错误", "请输入新增特征名")
            return
        if not expr:
            QMessageBox.warning(self, "错误", "请输入表达式")
            return

        # 预处理符号
        expr = (expr.replace("^", "**")
                .replace("√", "sqrt")
                .replace("ln", "log")
                .replace("log10", "log10"))

        # ② DataFrame.eval，反引号语法天然支持
        try:
            res = self.df.eval(expr,
                               engine="python",
                               local_dict={
                                   "np": np,
                                   "sqrt": np.sqrt,
                                   "log": np.log,
                                   "log10": np.log10,
                                   "abs": np.abs,
                               })
            res = res.replace([np.inf, -np.inf], np.nan).fillna(0)
        except Exception as e:
            QMessageBox.critical(self, "计算失败", str(e))
            return

        #  就地追加新列并通知外部
        self.df[name] = res
        self.var_combo.addItem(name)
        self.new_column.emit(name, res)
        QMessageBox.information(self, "完成", f"已添加新列：{name}")


# -----------------------------------------------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)

    df = pd.DataFrame({"a": [1, 2, 3],
                       "b": [10, 20, 30],
                       "c": [5, 6, 7]})

    w = CalculatorWidget()
    w.setDataFrame(df)
    w.new_column.connect(lambda n, s: print("新列:", n, "\n", s))
    w.show()

    sys.exit(app.exec())