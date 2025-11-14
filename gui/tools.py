
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtWidgets import QMessageBox, QLabel
from PyQt6.QtGui import QFontMetrics
import html
import logging

logger = logging.getLogger("failure_monitoring")
if not logger.handlers:
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"
    )

    file_handler = logging.FileHandler("log.txt", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

logger.propagate = False


def front_log(msg: str, level: str = "info") -> None:
    getattr(logger, level, logger.info)(msg)


def show_save_success(parent, model_id: str, model_name: str) -> None:
    """统一展示模型保存成功的弹窗。"""

    dlg = QMessageBox(parent)
    dlg.setIcon(QMessageBox.Icon.Information)
    dlg.setWindowTitle("已保存")
    dlg.setTextFormat(Qt.TextFormat.RichText)
    safe_id = html.escape(model_id or "-")
    safe_name = html.escape(model_name or "-")
    dlg.setText(
        """
        <div style="font-size:14px;">
          <p style="margin:6px 0 10px 0;">模型已保存为 <b>{mid}</b></p>
          
          <p style="margin:0;">模型名：{name}</p>
          <p style="margin:0;">模型ID：{mid}</p>
          
        </div>
        """.format(mid=safe_id, name=safe_name)
    )
    dlg.setStandardButtons(QMessageBox.StandardButton.Ok)
    dlg.exec()


class TrainWorker(QThread):
    log_sig  = pyqtSignal(str)           # 实时日志
    done_sig = pyqtSignal(dict)          # 训练返回值或异常

    def __init__(self, mgr, ds_id, m_type, params, parent=None):
        super().__init__(parent)
        self.mgr, self.ds_id, self.m_type, self.params = mgr, ds_id, m_type, params

    def run(self):
        def _log(msg: str):              # 供后端回调
            self.log_sig.emit(msg)
        try:
            res = self.mgr.train(
                self.ds_id, self.m_type, self.params, log_callback=_log, use_color=False
            )
            self.done_sig.emit({"ok": True, "res": res})
        except Exception as e:
            self.done_sig.emit({"ok": False, "err": str(e)})

    def stop(self):
        """Forcefully terminate the worker thread."""
        self.terminate()


def make_section_label(text: str, scale: float = 1.7) -> QLabel:
    lbl = QLabel(text)
    lbl.setObjectName("SectionTitle")
    return lbl
