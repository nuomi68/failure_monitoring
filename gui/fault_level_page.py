from __future__ import annotations

import pandas as pd
import numpy as np

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox, QMessageBox, QSplitter
)

from sklearn.preprocessing import StandardScaler
from backend.fault_level_estimator import FaultLevelEstimator
from gui.smart_table import SmartTable, SmartTableConfig


class FaultLevelPage(QWidget):
    """故障等级估计器页面，使用 SmartTable 统一表格展示"""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("故障等级估计器")

        self._use_scaler: bool = True
        self._estimator: FaultLevelEstimator | None = None

        top = QHBoxLayout()
        top.addStretch()
        top.addWidget(QLabel("特征标准化："))
        self.cb_scaler = QComboBox()
        self.cb_scaler.addItems(["启用（StandardScaler）", "不启用"])
        self.cb_scaler.setCurrentIndex(0)
        self.cb_scaler.currentIndexChanged.connect(
            lambda _: setattr(self, "_use_scaler", self.cb_scaler.currentIndex() == 0)
        )
        top.addWidget(self.cb_scaler)

        splitter = QSplitter(Qt.Orientation.Vertical)

        upper = QWidget()
        up_lay = QVBoxLayout(upper)
        up_lay.addWidget(QLabel("故障等级样本表"))
        self.tbl_labelled = SmartTable(SmartTableConfig(show_label_selector=True))
        up_lay.addWidget(self.tbl_labelled)

        lower = QWidget()
        lo_lay = QVBoxLayout(lower)
        lo_lay.addWidget(QLabel("待预测样本表"))
        self.tbl_unlabelled = SmartTable(SmartTableConfig())
        lo_lay.addWidget(self.tbl_unlabelled)

        splitter.addWidget(upper)
        splitter.addWidget(lower)

        bottom = QHBoxLayout()
        self.btn_predict = QPushButton("计算等级")
        self.btn_predict.clicked.connect(self._on_predict)
        bottom.addStretch()
        bottom.addWidget(self.btn_predict)

        root = QVBoxLayout(self)
        root.addLayout(top)
        root.addWidget(splitter)
        root.addLayout(bottom)

        self.tbl_labelled.bind_features_sink(self.tbl_unlabelled)

        demo = pd.DataFrame({
            "feat1": [0.2, 1.0, 0.1],
            "feat2": [0.5, 0.9, 0.2],
            "fault_level": [0, 2, 1],
        })
        self.tbl_labelled.set_dataframe(demo)
        self.tbl_labelled.set_label_column("fault_level")

    def _on_predict(self):
        label_col = self.tbl_labelled.label_column()
        if not label_col:
            QMessageBox.warning(self, "提示", "请选择故障等级列。")
            return

        df_lab = self.tbl_labelled.dataframe()
        if df_lab.empty:
            QMessageBox.warning(self, "提示", "故障等级样本表为空。")
            return
        if label_col not in df_lab.columns:
            QMessageBox.warning(self, "提示", f"等级列“{label_col}”不在样本表中。")
            return

        feat_cols = [c for c in df_lab.columns if c != label_col]
        X_lab = df_lab[feat_cols].apply(pd.to_numeric, errors="coerce")
        y_lab = df_lab[label_col]
        nan_rows_lab = X_lab.isna().any(axis=1).to_numpy().nonzero()[0].tolist()
        if nan_rows_lab:
            QMessageBox.information(
                self,
                "数据清洗",
                f"样本表中有 {len(nan_rows_lab)} 行包含非数值特征，已自动剔除：\n{[int(i) for i in nan_rows_lab]}"
            )
        keep_lab = ~X_lab.isna().any(axis=1)
        X_lab = X_lab[keep_lab].to_numpy(dtype=float)
        y_lab = y_lab[keep_lab].to_numpy()

        if X_lab.size == 0:
            QMessageBox.critical(self, "无有效样本", "清洗后样本为空，请检查数据。")
            return

        df_un = self.tbl_unlabelled.dataframe()
        if df_un.empty:
            QMessageBox.warning(self, "提示", "待预测样本表为空，请先填写。")
            return

        use_cols = [c for c in feat_cols if c in df_un.columns]
        if not use_cols:
            QMessageBox.critical(self, "列不匹配", "待预测表与样本表的特征列不匹配。")
            return

        X_un = df_un[use_cols].apply(pd.to_numeric, errors="coerce")
        nan_rows_un = X_un.isna().any(axis=1).to_numpy().nonzero()[0].tolist()
        if nan_rows_un:
            QMessageBox.information(
                self,
                "数据清洗",
                f"待预测表中有 {len(nan_rows_un)} 行包含非数值或缺失，已自动剔除：\n{[int(i) for i in nan_rows_un]}"
            )
        keep_un = ~X_un.isna().any(axis=1)
        X_un_valid = X_un[keep_un].to_numpy(dtype=float)

        scaler = (StandardScaler().fit(X_lab) if self._use_scaler else None)
        self._estimator = FaultLevelEstimator(X_lab, y_lab, scaler=scaler)
        if X_un_valid.shape[0] == 0:
            QMessageBox.warning(self, "提示", "清洗后待预测表无有效行。")
            return
        preds = self._estimator.predict(X_un_valid)

        df_un["预测等级"] = ""
        df_un.loc[keep_un.to_numpy().nonzero()[0], "预测等级"] = preds
        self.tbl_unlabelled.set_dataframe(df_un)
        pred_col = df_un.columns.get_loc("预测等级")
        for r in range(self.tbl_unlabelled.table.rowCount()):
            item = self.tbl_unlabelled.table.item(r, pred_col)
            if item:
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)

        QMessageBox.information(self, "完成", f"已为 {preds.shape[0]} 行写入预测等级。")
