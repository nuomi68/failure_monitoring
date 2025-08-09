
from PyQt6.QtCore import QThread, pyqtSignal
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

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.propagate = False


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
            res = self.mgr.train(self.ds_id, self.m_type, self.params, log_callback=_log)
            self.done_sig.emit({"ok": True, "res": res})
        except Exception as e:
            self.done_sig.emit({"ok": False, "err": str(e)})

    def stop(self):
        """Forcefully terminate the worker thread."""
        self.terminate()
