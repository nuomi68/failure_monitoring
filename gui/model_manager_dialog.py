from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QVBoxLayout,
    QTableWidget,
    QTableWidgetItem,
    QPushButton,
    QLineEdit,
    QLabel,
    QFileDialog,
    QMessageBox,
    QHeaderView,
    QAbstractItemView,
)
from gui.tools import logger

class ModelManagerDialog(QDialog):
    """通用模型管理对话框，支持可选多选模式。"""

    model_loaded = pyqtSignal(str)
    models_loaded = pyqtSignal(list)

    def __init__(self, manager, parent=None, *, multi_select: bool = False):
        super().__init__(parent)
        self.setWindowTitle("模型管理")
        self.resize(880, 560)
        self.manager = manager
        self._multi = multi_select

        # top: search
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("搜索名称 / 类型 / 数据集…")
        self.search_edit.textChanged.connect(self._refresh)

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["名称", "ID", "类型", "数据集", "创建时间", "备注"]
        )
        self.table.setSelectionBehavior(self.table.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(self.table.EditTrigger.NoEditTriggers)
        if multi_select:
            self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        else:
            self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.table.doubleClicked.connect(self._load_clicked)

        right = QVBoxLayout()
        self.btn_load = QPushButton("加载为当前")
        self.btn_rename = QPushButton("重命名")
        self.btn_delete = QPushButton("删除")
        self.btn_export = QPushButton("导出…")
        self.btn_import = QPushButton("导入…")
        for b in (self.btn_load, self.btn_rename, self.btn_delete, self.btn_export, self.btn_import):
            b.setMinimumHeight(34)
            right.addWidget(b)
        right.addStretch(1)

        self.btn_load.clicked.connect(self._load_clicked)
        self.btn_rename.clicked.connect(self._rename_clicked)
        self.btn_delete.clicked.connect(self._delete_clicked)
        self.btn_export.clicked.connect(self._export_clicked)
        self.btn_import.clicked.connect(self._import_clicked)

        left = QVBoxLayout()
        left.addWidget(QLabel("已保存模型"))
        left.addWidget(self.search_edit)
        left.addWidget(self.table, 1)

        root = QHBoxLayout(self)
        root.addLayout(left, 1)
        root.addLayout(right)

        self._refresh()

    # ---------- helpers ----------
    def _selected_model_id(self):
        r = self.table.currentRow()
        if r < 0:
            return None
        return self.table.item(r, 1).text()

    def _refresh(self):
        kw = self.search_edit.text().strip().lower()
        models = self.manager.refresh_models() or []
        if kw:
            def match(m):
                s = " ".join([
                    m.get("name",""), m.get("model_id",""), m.get("model_type",""),
                    m.get("dataset_id",""), m.get("created_at",""), str(m.get("metrics",""))
                ]).lower()
                return kw in s
            models = [m for m in models if match(m)]

        self.table.setRowCount(0)
        for meta in models:
            r = self.table.rowCount()
            self.table.insertRow(r)
            self.table.setItem(r, 0, QTableWidgetItem(meta.get("name","")))
            self.table.setItem(r, 1, QTableWidgetItem(meta.get("model_id","")))
            self.table.setItem(r, 2, QTableWidgetItem(meta.get("model_type","")))
            self.table.setItem(r, 3, QTableWidgetItem(meta.get("dataset_id","")))
            self.table.setItem(r, 4, QTableWidgetItem(meta.get("created_at","")))
            metrics = meta.get("metrics")

            if isinstance(metrics, dict):
                metrics_text = ", ".join(f"{k}: {v}" for k, v in metrics.items())
            else:
                metrics_text = str(metrics or "")
            self.table.setItem(r, 5, QTableWidgetItem(metrics_text))

    def _cell_text(self, row: int, col: int) -> str:
        item = self.table.item(row, col)
        return item.text().strip() if item else ""

    def _row_meta(self, r: int) -> dict:
        return {
            "name": self._cell_text(r, 0),
            "model_id": self._cell_text(r, 1),
            "model_type": self._cell_text(r, 2),
            "dataset_id": self._cell_text(r, 3),
            "created_at": self._cell_text(r, 4),
            "remarks": self._cell_text(r, 5),  # 你现在第5列填的是“备注/指标”的文本
        }

    # ---------- actions ----------
    def _load_clicked(self):
        if self._multi:
            rows = {idx.row() for idx in self.table.selectedIndexes()}
            ids = [self.table.item(r, 1).text() for r in rows if self.table.item(r, 1)]
            if not ids:
                QMessageBox.information(self, "提示", "请先选中一个模型。")
                return
            self.models_loaded.emit(ids)
            self.accept()
            return

        # ---- 单选 ----
        r = self.table.currentRow()
        mid = self._selected_model_id()
        m = self._row_meta(r)
        remark = (m["remarks"][:100] + "…") if len(m["remarks"]) > 100 else m["remarks"]
        logger.info(
            "加载模型：%s (ID:%s, 类型:%s, 数据集:%s, 创建时间:%s%s)",
            m["name"], m["model_id"], m["model_type"], m["dataset_id"], m["created_at"],
            (", 备注=" + remark) if remark else ""
        )
        if not mid:
            QMessageBox.information(self, "提示", "请先选中一个模型。")
            return
        self.model_loaded.emit(mid)
        self.accept()

    def _rename_clicked(self):
        mid = self._selected_model_id()
        if not mid:
            return
        from PyQt6.QtWidgets import QInputDialog
        new, ok = QInputDialog.getText(self, "重命名", "新名称：")
        if ok and new.strip():
            ok2 = self.manager.rename_model(mid, new.strip())
            if not ok2:
                QMessageBox.warning(self, "失败", "重命名失败。")
        self._refresh()

    def _delete_clicked(self):
        mid = self._selected_model_id()
        if not mid:
            return
        if QMessageBox.question(self, "删除确认", "确定删除该模型及其文件？") == QMessageBox.StandardButton.Yes:
            ok = self.manager.delete_model(mid)
            if not ok:
                QMessageBox.warning(self, "失败", "删除失败。")
        self._refresh()

    def _export_clicked(self):
        mid = self._selected_model_id()
        if not mid:
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出模型为…", f"{mid}.zip", "Zip (*.zip)")
        if not path:
            return
        ok = self.manager.export_model(mid, path)
        QMessageBox.information(self, "导出", "成功" if ok else "失败")

    def _import_clicked(self):
        path, _ = QFileDialog.getOpenFileName(self, "导入模型包…", "", "Zip (*.zip)")
        if not path:
            return
        ok = self.manager.import_model(path)
        QMessageBox.information(self, "导入", "成功" if ok else "失败")
        self._refresh()
