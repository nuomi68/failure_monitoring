# coding: utf-8
"""
独立的参数面板组件：基础参数 + 高级参数弹窗
"""
from typing import Any, Dict
from PyQt6.QtWidgets import (
    QWidget, QFormLayout, QSpinBox, QPushButton, QDialog,
    QDialogButtonBox, QLineEdit, QMessageBox
)

# === 高级参数对话框 =========================================================
class _AdvParamDialog(QDialog):
    def __init__(self, init_params: Dict[str, Any], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("高级参数")
        self._editors: Dict[str, QLineEdit] = {}

        form = QFormLayout(self)
        for name, default in init_params.items():
            editor = QLineEdit(str(default))
            form.addRow(name, editor)
            self._editors[name] = editor

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        form.addRow(btns)

    # 把用户修改后的值取回
    def result_params(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, editor in self._editors.items():
            txt = editor.text().strip()
            # 尝试转成数字；失败就保留字符串
            for caster in (int, float):
                try:
                    out[k] = caster(txt)
                    break
                except ValueError:
                    continue
            else:
                out[k] = txt
        return out


# === 基础 + 高级参数面板 ====================================================
class ParamPanel(QWidget):
    """
    * 批大小 / 轮数 / 滑窗长度  →  SpinBox
    * 「高级参数…」           →  弹窗
    """
    def __init__(self, manager, parent=None) -> None:
        super().__init__(parent)
        self._manager = manager
        self._adv_params: Dict[str, Any] = {}

        self.batch = QSpinBox(minimum=1, maximum=2048, value=32)
        self.epochs = QSpinBox(minimum=1, maximum=2000, value=50)
        self.look_back = QSpinBox(minimum=1, maximum=1024, value=32)

        btn_adv = QPushButton("高级参数…")
        btn_adv.clicked.connect(self._open_adv_dialog)

        form = QFormLayout(self)
        form.addRow("批大小", self.batch)
        form.addRow("轮数", self.epochs)
        form.addRow("滑窗长度", self.look_back)
        form.addRow(btn_adv)

    # 给外部调用，拿到完整参数 dict
    def params(self) -> Dict[str, Any]:
        base = {
            "batch_size": self.batch.value(),
            "epochs": self.epochs.value(),
            "look_back": self.look_back.value(),
        }
        base.update(self._adv_params)
        return base

    # ------------------------------------------------------------------ #
    #                         内部：高级参数弹窗                            #
    # ------------------------------------------------------------------ #
    def _open_adv_dialog(self) -> None:
        # 从主窗口拿当前模型类型
        main = self.window()
        model_type = self.parent().model_type_combo
        if model_type is None or not model_type.currentText().strip():
            QMessageBox.warning(self, "提示", "请先选择模型类型")
            return
        model_type = model_type.currentText().strip()

        # ❶ 后端拉取高级参数模版（需在 ModelManager 实现 get_advanced_params）
        try:
            init_params = self._manager.get_advanced_params(model_type)
        except Exception as exc:
            QMessageBox.warning(self, "错误", f"获取高级参数失败：{exc}")
            return

        # 弹窗渲染
        dlg = _AdvParamDialog(init_params, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._adv_params = dlg.result_params()
