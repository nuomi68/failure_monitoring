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
    """
    计算新特征（改造版）
    - 不再只“保存列值”，而是“记录计算公式（recipe）”，并在需要时可重复计算
    - 仍向外发射 `new_column(name, series)` 以保持兼容；新增 `recipe_added(name, expr)`
    - 提供 `get_recipes()` 暴露当前公式列表；`apply_recipes(df, recipes)` 可批量计算
    """

    new_column = pyqtSignal(str, object)             # 兼容：新增列完成时发射（Series）
    recipe_added = pyqtSignal(str, str)              # 新增：发射（列名, 归一化表达式）

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("计算新特征", parent)
        self.df: pd.DataFrame | None = None
        self._recipes: list[tuple[str, str]] = []    # [(name, expr_norm)] 保留顺序，便于有依赖的链式计算
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
        self.var_combo.setEditable(False)
        self.var_combo.setPlaceholderText("选择特征")
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
        token = col if self._is_identifier(col) else f"`{col}`"   # DataFrame.eval 的反引号语法
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
        expr_raw = self.expr_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "错误", "请输入新增特征名")
            return
        if not expr_raw:
            QMessageBox.warning(self, "错误", "请输入表达式")
            return

        # 统一表达式写法（保存“公式”用的是规范化后的表达式）
        expr = self.normalize_expr(expr_raw)

        # 先计算一遍，给到用户实时预览
        try:
            res = self.eval_expr_on_df(self.df, expr)
        except Exception as e:
            QMessageBox.critical(self, "计算失败", str(e))
            return

        # 就地追加新列以便继续可视化/选择；并记录“公式”
        self.df[name] = res
        self.var_combo.addItem(name)
        self._recipes.append((name, expr))

        # 向外发射两个信号：兼容的列值 + 新的配方
        self.new_column.emit(name, res)
        self.recipe_added.emit(name, expr)
        QMessageBox.information(self, "完成", f"已添加新列：{name}")

    # --------------------------- 对外 API ---------------------------------
    def get_recipes(self) -> list[dict]:
        """以 [{"name":..,"expr":..}] 形式返回“计算配方”。保持录入顺序。"""
        return [{"name": n, "expr": e} for n, e in self._recipes]

    @staticmethod
    def normalize_expr(expr: str) -> str:
        """将常见输入符号转为 pandas.eval 可执行的 Python 表达式。"""
        return (expr.replace("^", "**")
                    .replace("√", "sqrt")
                    .replace("ln", "log")
                    .replace("log10", "log10"))

    @staticmethod
    def eval_expr_on_df(df: pd.DataFrame, expr: str) -> pd.Series:
        """在 DataFrame 上执行单个表达式，返回 Series。
        - 支持反引号列名；
        - 对 inf/NaN 做健壮处理。
        """
        res = df.eval(expr,
                      engine="python",
                      local_dict={
                          "np": np,
                          "sqrt": np.sqrt,
                          "log": np.log,
                          "log10": np.log10,
                          "abs": np.abs,
                      })
        return res.replace([np.inf, -np.inf], np.nan).fillna(0)

    @staticmethod
    def apply_recipes(df: pd.DataFrame, recipes: list[dict]) -> pd.DataFrame:
        """按照给定配方依次在 df 上生成新列；返回 *新的* DataFrame（不修改原 df）。"""
        if not recipes:
            return df.copy()
        out = df.copy()
        for item in recipes:
            try:
                name = str(item.get("name"))
                expr = CalculatorWidget.normalize_expr(str(item.get("expr")))
                out[name] = CalculatorWidget.eval_expr_on_df(out, expr)
            except Exception:
                # 单条失败不终止整体流程，按 NaN 兜底
                out[name] = np.nan
        return out


# -----------------------------------------------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)

    df = pd.DataFrame({"a": [1, 2, 3],
                       "b": [10, 20, 30],
                       "c": [5, 6, 7]})

    w = CalculatorWidget()
    w.setDataFrame(df)
    w.new_column.connect(lambda n, s: print("新列:", n, "\n", s))
    w.recipe_added.connect(lambda n, e: print("新配方:", n, "<=", e))
    w.show()

    sys.exit(app.exec())
