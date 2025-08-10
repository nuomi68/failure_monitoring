from __future__ import annotations

import pandas as pd
import numpy as np

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox, QMessageBox, QSplitter, QFileDialog
)

from backend.fault_level_estimator import FaultLevelEstimator

from gui.smart_table import SmartTable, SmartTableConfig


class FaultLevelPage(QWidget):
    """故障等级估计器页面，使用 SmartTable 统一表格展示
    - 前端可选择算法方法
    - 可保存/加载模型（包含方法、参数、特征名）
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("故障等级估计器")

        self._scaler_code: str = "standard"
        self._method_code: str = 'wknn'  # 默认更稳健的距离加权 kNN
        self._estimator: FaultLevelEstimator | None = None

        # ---------------- Top Controls ----------------
        top = QHBoxLayout()
        top.addStretch()

        top.addWidget(QLabel("算法方法："))
        self.cb_method = QComboBox()

        self._methods = FaultLevelEstimator.available_methods()
        for k, v in self._methods.items():
            self.cb_method.addItem(v, userData=k)
        # 默认选中 self._method_code
        idx_default = list(self._methods.keys()).index(self._method_code)
        self.cb_method.setCurrentIndex(idx_default)
        self.cb_method.currentIndexChanged.connect(self._on_method_changed)
        top.addWidget(self.cb_method)

        top.addSpacing(16)
        top.addWidget(QLabel("特征缩放："))
        self.cb_scaler = QComboBox()
        for name, spec in FaultLevelEstimator.available_scalers():
            self.cb_scaler.addItem(name, spec)
        self.cb_scaler.setCurrentIndex(0)
        self.cb_scaler.currentIndexChanged.connect(
            lambda _: setattr(self, "_scaler_code", self.cb_scaler.currentData())
        )
        top.addWidget(self.cb_scaler)

        # ---------------- Splitter (Tables) ----------------
        splitter = QSplitter(Qt.Orientation.Vertical)

        upper = QWidget()
        up_lay = QVBoxLayout(upper)
        up_lay.addWidget(QLabel("故障等级样本表"))
        self.tbl_labelled = SmartTable(
            SmartTableConfig(
                show_label_selector=True,
                default_headers=["feat1", "feat2", "fault_level"],
            )
        )
        up_lay.addWidget(self.tbl_labelled)

        lower = QWidget()
        lo_lay = QVBoxLayout(lower)
        lo_lay.addWidget(QLabel("待预测样本表"))
        self.tbl_unlabelled = SmartTable(SmartTableConfig())
        lo_lay.addWidget(self.tbl_unlabelled)

        splitter.addWidget(upper)
        splitter.addWidget(lower)

        # ---------------- Bottom Buttons ----------------
        bottom = QHBoxLayout()
        self.btn_predict = QPushButton("计算等级")
        self.btn_predict.clicked.connect(self._on_predict)
        self.btn_save = QPushButton("保存模型…")
        self.btn_save.clicked.connect(self._on_save_model)
        self.btn_load = QPushButton("加载模型…")
        self.btn_load.clicked.connect(self._on_load_model)

        bottom.addStretch()
        bottom.addWidget(self.btn_predict)
        bottom.addSpacing(12)
        bottom.addWidget(self.btn_save)
        bottom.addWidget(self.btn_load)

        # ---------------- Root Layout ----------------
        root = QVBoxLayout(self)
        root.addLayout(top)
        root.addWidget(splitter)
        root.addLayout(bottom)

        self.tbl_labelled.bind_features_sink(self.tbl_unlabelled)

        # Demo 数据
        demo = pd.DataFrame({
            "feat1": [0.2, 1.0, 0.1],
            "feat2": [0.5, 0.9, 0.2],
            "fault_level": [0, 2, 1],
        })
        self.tbl_labelled.set_dataframe(demo)
        self.tbl_labelled.set_label_column("fault_level")

    # ---------------- Slots ----------------
    def _on_method_changed(self, _: int):
        self._method_code = self.cb_method.currentData()

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

        # 优先使用保存/加载的特征顺序
        feat_for_predict = feat_cols
        if self._estimator and self._estimator.feature_names:
            feat_for_predict = list(self._estimator.feature_names)
        use_cols = [c for c in feat_for_predict if c in df_un.columns]
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

        self._estimator = FaultLevelEstimator(
            X_lab,
            y_lab,
            method=self._method_code,
            metric="euclidean",
            scaler=self._scaler_code,
            feature_names=feat_cols,
        )

        if X_un_valid.shape[0] == 0:
            QMessageBox.warning(self, "提示", "清洗后待预测表无有效行。")
            return

        preds = self._estimator.predict(X_un_valid)

        df_un["预测等级"] = ""
        df_un.loc[keep_un.to_numpy().nonzero()[0], "预测等级"] = preds
        with self.tbl_unlabelled.no_record():
            self.tbl_unlabelled.set_dataframe(df_un, record_state=False)
            pred_col = df_un.columns.get_loc("预测等级")
            for r in range(self.tbl_unlabelled.table.rowCount()):
                item = self.tbl_unlabelled.table.item(r, pred_col)
                if item:
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)

        QMessageBox.information(self, "完成", f"已为 {preds.shape[0]} 行写入预测等级。\n方法：{self._methods[self._method_code]}")

    def _on_save_model(self):
        # 需要有样本才可保存
        df_lab = self.tbl_labelled.dataframe()
        label_col = self.tbl_labelled.label_column()
        if df_lab.empty or not label_col or label_col not in df_lab.columns:
            QMessageBox.warning(self, "提示", "请先准备好样本表并指定等级列，再保存模型。")
            return

        feat_cols = [c for c in df_lab.columns if c != label_col]
        X_lab = df_lab[feat_cols].apply(pd.to_numeric, errors="coerce")
        y_lab = df_lab[label_col]
        keep_lab = ~X_lab.isna().any(axis=1)
        X_lab = X_lab[keep_lab].to_numpy(dtype=float)
        y_lab = y_lab[keep_lab].to_numpy()
        if X_lab.size == 0:
            QMessageBox.critical(self, "无有效样本", "清洗后样本为空，无法保存。")
            return

        # 确保有一个估计器实例
        self._estimator = FaultLevelEstimator(
            X_lab,
            y_lab,
            method=self._method_code,
            metric="euclidean",
            scaler=self._scaler_code,
            feature_names=feat_cols,
        )

        path, _ = QFileDialog.getSaveFileName(self, "保存模型", filter="Joblib (*.joblib);;All Files (*)")
        if not path:
            return
        try:
            self._estimator.save(path)
            QMessageBox.information(self, "已保存", f"模型已保存：{path}\n方法：{self._methods[self._method_code]}")
        except Exception as e:
            QMessageBox.critical(self, "保存失败", f"保存模型失败：{e}")

    def _on_load_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "加载模型", filter="Joblib (*.joblib);;All Files (*)")
        if not path:
            return
        try:
            est = FaultLevelEstimator.load(path)
            self._estimator = est
            # 同步 UI：方法、标准化开关
            keys = list(self._methods.keys())
            if est.method in keys:
                self.cb_method.setCurrentIndex(keys.index(est.method))
                self._method_code = est.method
            # 缩放器状态
            s_keys = list(FaultLevelEstimator.available_scalers().keys())
            code = est.scaler_spec if est.scaler_spec in s_keys else "none"
            self.cb_scaler.setCurrentIndex(s_keys.index(code))
            self._scaler_code = code

            # 给出提示信息
            fnames = est.feature_names or []
            QMessageBox.information(
                self, "已加载",
                f"已加载模型：{path}\n方法：{est.method}\n特征列：{fnames if fnames else '（未记录）'}"
            )
        except Exception as e:
            QMessageBox.critical(self, "加载失败", f"加载模型失败：{e}")
