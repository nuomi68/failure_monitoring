from __future__ import annotations

import pandas as pd
import numpy as np

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QMessageBox, QSplitter, QInputDialog
)

from backend.fault_level_estimator import FaultLevelEstimator
from backend.fault_model_manager import FaultModelManager
from gui.model_manager_dialog import ModelManagerDialog

from gui.smart_table import SmartTable, SmartTableConfig


class FaultLevelPage(QWidget):
    """破口大小估计器页面，使用 SmartTable 统一表格展示
    - 前端可选择算法方法
    - 可保存/加载模型（包含方法、参数、特征名）
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("破口大小估计器")

        self._scaler_code: str = "standard"
        self._method_code: str = 'wknn'  # 默认更稳健的距离加权 kNN
        self._estimator: FaultLevelEstimator | None = None
        self.manager = FaultModelManager()

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
        up_lay.addWidget(QLabel("损伤评估样本表"))
        self.tbl_labelled = SmartTable(
            SmartTableConfig(
                show_label_selector=True,
                default_headers=["feat1", "feat2", "break_size"],
            )
        )
        up_lay.addWidget(self.tbl_labelled)

        lower = QWidget()
        lo_lay = QVBoxLayout(lower)
        lo_lay.addWidget(QLabel("损伤评估输入表"))
        self.tbl_unlabelled = SmartTable(SmartTableConfig())
        lo_lay.addWidget(self.tbl_unlabelled)

        splitter.addWidget(upper)
        splitter.addWidget(lower)

        # ---------------- Bottom Buttons ----------------
        bottom = QHBoxLayout()
        self.btn_predict = QPushButton("计算破口大小")
        self.btn_predict.clicked.connect(self._on_predict)
        self.btn_save = QPushButton("保存模型…")
        self.btn_save.clicked.connect(self._on_save_model)
        self.btn_save.setEnabled(False)
        self.btn_load = QPushButton("加载模型…")
        self.btn_load.clicked.connect(self._open_model_manager)

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
            "break_size": [0, 2, 1],
        })
        self.tbl_labelled.set_dataframe(demo)
        self.tbl_labelled.set_label_column("break_size")

    # ---------------- Slots ----------------
    def _on_method_changed(self, _: int):
        self._method_code = self.cb_method.currentData()

    def _on_predict(self):
        label_col = self.tbl_labelled.label_column()
        if not label_col:
            QMessageBox.warning(self, "提示", "请选择破口大小列。")
            return

        df_lab = self.tbl_labelled.dataframe()
        if df_lab.empty:
            QMessageBox.warning(self, "提示", "损伤评估样本表为空。")
            return
        if label_col not in df_lab.columns:
            QMessageBox.warning(self, "提示", f"破口大小列“{label_col}”不在损伤评估样本表中。")
            return

        feat_cols = [c for c in df_lab.columns if c != label_col]
        X_lab = df_lab[feat_cols].apply(pd.to_numeric, errors="coerce")
        y_lab = df_lab[label_col]
        nan_rows_lab = X_lab.isna().any(axis=1).to_numpy().nonzero()[0].tolist()
        if nan_rows_lab:
            QMessageBox.information(
                self,
                "数据清洗",
                f"损伤评估样本表中有 {len(nan_rows_lab)} 行包含非数值特征，已自动剔除：\n{[int(i) for i in nan_rows_lab]}"
            )
        keep_lab = ~X_lab.isna().any(axis=1)
        X_lab = X_lab[keep_lab].to_numpy(dtype=float)
        y_lab = y_lab[keep_lab].to_numpy()

        if X_lab.size == 0:
            QMessageBox.critical(self, "无有效样本", "清洗后样本为空，请检查数据。")
            return

        df_un = self.tbl_unlabelled.dataframe()
        if df_un.empty:
            QMessageBox.warning(self, "提示", "损伤评估输入表为空，请先填写。")
            return

        # 优先使用保存/加载的特征顺序
        feat_for_predict = feat_cols
        if self._estimator and self._estimator.feature_names:
            feat_for_predict = list(self._estimator.feature_names)
        use_cols = [c for c in feat_for_predict if c in df_un.columns]
        if not use_cols:
            QMessageBox.critical(self, "列不匹配", "损伤评估输入表与损伤评估样本表的特征列不匹配。")
            return

        X_un = df_un[use_cols].apply(pd.to_numeric, errors="coerce")
        nan_rows_un = X_un.isna().any(axis=1).to_numpy().nonzero()[0].tolist()
        if nan_rows_un:
            QMessageBox.information(
                self,
                "数据清洗",
                f"损伤评估输入表中有 {len(nan_rows_un)} 行包含非数值或缺失，已自动剔除：\n{[int(i) for i in nan_rows_un]}"
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
            QMessageBox.warning(self, "提示", "清洗后损伤评估输入表无有效行。")
            return

        preds = self._estimator.predict(X_un_valid)

        df_un["预测破口大小"] = ""
        df_un.loc[keep_un.to_numpy().nonzero()[0], "预测破口大小"] = preds
        with self.tbl_unlabelled.no_record():
            self.tbl_unlabelled.set_dataframe(df_un, record_state=False)
            pred_col = df_un.columns.get_loc("预测破口大小")
            for r in range(self.tbl_unlabelled.table.rowCount()):
                item = self.tbl_unlabelled.table.item(r, pred_col)
                if item:
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)

        self.btn_save.setEnabled(True)
        QMessageBox.information(self, "完成", f"已为 {preds.shape[0]} 行写入预测破口大小。\n方法：{self._methods[self._method_code]}")

    def _on_save_model(self):
        # 需要有样本才可保存
        df_lab = self.tbl_labelled.dataframe()
        label_col = self.tbl_labelled.label_column()
        if df_lab.empty or not label_col or label_col not in df_lab.columns:
            QMessageBox.warning(self, "提示", "请先准备好损伤评估样本表并指定破口大小列，再保存模型。")
            return

        feat_cols = [c for c in df_lab.columns if c != label_col]
        X_lab = df_lab[feat_cols].apply(pd.to_numeric, errors="coerce")
        y_lab = df_lab[label_col]
        keep_lab = ~X_lab.isna().any(axis=1)
        X_lab = X_lab[keep_lab].to_numpy(dtype=float)
        y_lab = y_lab[keep_lab].to_numpy()
        df_clean = df_lab.loc[keep_lab].reset_index(drop=True)
        if X_lab.size == 0:
            QMessageBox.critical(self, "无有效样本", "清洗后样本为空，无法保存。")
            return

        # 构建估计器
        self._estimator = FaultLevelEstimator(
            X_lab,
            y_lab,
            method=self._method_code,
            metric="euclidean",
            scaler=self._scaler_code,
            feature_names=feat_cols,
        )

        name, ok = QInputDialog.getText(self, "模型名称", "请输入模型名称：")
        if not ok or not name.strip():
            return

        try:
            mid = self.manager.save_model(
                self._estimator,
                name.strip(),
                label_col=label_col,
                df=df_clean,
            )
            QMessageBox.information(self, "已保存", f"模型已保存：{mid}")
        except Exception as e:
            QMessageBox.critical(self, "保存失败", f"保存模型失败：{e}")

    # ------------------------------------------------------------------
    def _open_model_manager(self):
        dlg = ModelManagerDialog(self.manager, self)
        dlg.model_loaded.connect(self._load_model_by_id)
        dlg.exec()

    def _load_model_by_id(self, model_id: str):
        try:
            est, df, meta = self.manager.load_model(model_id)
        except Exception as e:
            QMessageBox.critical(self, "加载失败", f"加载模型失败：{e}")
            return

        self._estimator = est
        # 同步 UI：方法、缩放器
        keys = list(self._methods.keys())
        if est.method in keys:
            self.cb_method.setCurrentIndex(keys.index(est.method))
            self._method_code = est.method
        s_specs = [spec for _, spec in FaultLevelEstimator.available_scalers()]
        if est.scaler_spec in s_specs:
            idx = s_specs.index(est.scaler_spec)
        else:
            idx = s_specs.index("none") if "none" in s_specs else 0
        self.cb_scaler.setCurrentIndex(idx)
        self._scaler_code = s_specs[idx]
        self.btn_save.setEnabled(True)

        # 载入损伤评估样本表并同步特征列
        if df is not None:
            with self.tbl_labelled.no_record():
                self.tbl_labelled.set_dataframe(df, record_state=False)
                lbl = meta.get("label_col")
                if lbl and lbl in df.columns:
                    self.tbl_labelled.set_label_column(lbl)
            # 清空损伤评估输入表但保留特征列
            cols = est.feature_names or list(df.columns)
            self.tbl_unlabelled.set_dataframe(pd.DataFrame(columns=cols), record_state=False)
        else:
            self.tbl_unlabelled.set_headers(est.feature_names or [])

        fnames = est.feature_names or []
        QMessageBox.information(
            self,
            "已加载",
            f"已加载模型：{model_id}\n方法：{est.method}\n特征列：{fnames if fnames else '（未记录）'}",
        )
