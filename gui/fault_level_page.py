from __future__ import annotations

import os
import pandas as pd
import numpy as np

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox, QFileDialog,
    QTableWidget, QTableWidgetItem, QMessageBox, QSplitter, QSpinBox
)

from sklearn.preprocessing import StandardScaler
from backend.fault_level_estimator import FaultLevelEstimator


def _df_from_table(tbl: QTableWidget) -> pd.DataFrame:
    cols = [tbl.horizontalHeaderItem(j).text() for j in range(tbl.columnCount())]
    rows = []
    for i in range(tbl.rowCount()):
        row = {}
        empty_row = True
        for j, c in enumerate(cols):
            item = tbl.item(i, j)
            val = "" if item is None else item.text()
            if val != "":
                empty_row = False
            row[c] = val
        if not empty_row:
            rows.append(row)
    return pd.DataFrame(rows, columns=cols)


def _fill_table_from_df(tbl: QTableWidget, df: pd.DataFrame, editable: bool = True):
    tbl.clear()
    tbl.setColumnCount(len(df.columns))
    tbl.setRowCount(len(df))
    tbl.setHorizontalHeaderLabels(list(df.columns))
    for i in range(len(df)):
        for j, col in enumerate(df.columns):
            val = "" if pd.isna(df.iloc[i, j]) else str(df.iloc[i, j])
            it = QTableWidgetItem(val)
            if not editable:
                it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
            tbl.setItem(i, j, it)
    tbl.resizeColumnsToContents()


def _ensure_cols(tbl: QTableWidget, columns: list[str], keep_data=False):
    if keep_data and tbl.columnCount() == len(columns) and \
            [tbl.horizontalHeaderItem(j).text() for j in range(tbl.columnCount())] == columns:
        return
    data = _df_from_table(tbl) if keep_data and tbl.columnCount() > 0 else pd.DataFrame(columns=columns)
    # 只保留、重排匹配列，多余列丢弃，缺列补空
    data = data.reindex(columns=columns)
    _fill_table_from_df(tbl, data, editable=True)


class FaultLevelPage(QWidget):
    """
    顶部：故障等级样本表（可加载CSV；必须选择“等级列”）
    下方：待预测样本表（列名跟随样本表；去除等级列，支持手工填充）
    右下：计算等级（在待预测表尾部写入“预测等级”列）
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("故障等级估计器")

        self._label_col: str | None = None
        self._use_scaler: bool = True
        self._estimator: FaultLevelEstimator | None = None

        # ====== 顶部区域：加载 + 等级列选择 ======
        top = QHBoxLayout()
        self.btn_load = QPushButton("加载带等级样本 CSV")
        self.btn_load.clicked.connect(self._on_load_labelled)
        top.addWidget(self.btn_load)

        top.addWidget(QLabel("等级列："))
        self.cb_label_col = QComboBox()
        self.cb_label_col.currentIndexChanged.connect(self._on_label_col_changed)
        top.addWidget(self.cb_label_col)

        top.addStretch()

        top.addWidget(QLabel("特征标准化："))
        self.cb_scaler = QComboBox()
        self.cb_scaler.addItems(["启用（StandardScaler）", "不启用"])
        self.cb_scaler.setCurrentIndex(0)
        self.cb_scaler.currentIndexChanged.connect(
            lambda _: setattr(self, "_use_scaler", self.cb_scaler.currentIndex() == 0)
        )
        top.addWidget(self.cb_scaler)

        # ====== 中间：上下两个表格 ======
        splitter = QSplitter(Qt.Orientation.Vertical)

        # 顶部：样本表
        upper = QWidget()
        up_lay = QVBoxLayout(upper)
        up_lay.addWidget(QLabel("故障等级样本表"))
        self.tbl_labelled = QTableWidget(0, 0)
        up_lay.addWidget(self.tbl_labelled)

        # 下方：待预测表
        lower = QWidget()
        lo_lay = QVBoxLayout(lower)
        header_line = QHBoxLayout()
        header_line.addWidget(QLabel("待预测样本表"))
        self.btn_add_row = QPushButton("新增一行")
        self.btn_add_row.clicked.connect(lambda: self.tbl_unlabelled.insertRow(self.tbl_unlabelled.rowCount()))
        self.btn_del_row = QPushButton("删除选中行")
        self.btn_del_row.clicked.connect(self._del_selected_rows)
        header_line.addWidget(self.btn_add_row)
        header_line.addWidget(self.btn_del_row)
        header_line.addStretch()
        lo_lay.addLayout(header_line)

        self.tbl_unlabelled = QTableWidget(0, 0)
        lo_lay.addWidget(self.tbl_unlabelled)

        splitter.addWidget(upper)
        splitter.addWidget(lower)

        # ====== 底部：计算按钮 ======
        bottom = QHBoxLayout()
        self.btn_predict = QPushButton("计算等级")
        self.btn_predict.clicked.connect(self._on_predict)
        bottom.addStretch()
        bottom.addWidget(self.btn_predict)

        # ====== 主布局 ======
        root = QVBoxLayout(self)
        root.addLayout(top)
        root.addWidget(splitter)
        root.addLayout(bottom)

        # ====== 放入演示数据（初始实例表） ======
        demo = pd.DataFrame({
            "feat1": [0.2, 1.0, 0.1],
            "feat2": [0.5, 0.9, 0.2],
            "fault_level": [0, 2, 1],
        })
        _fill_table_from_df(self.tbl_labelled, demo, editable=True)
        self.cb_label_col.clear()
        self.cb_label_col.addItems(list(demo.columns))
        # 默认把“fault_level”当作等级列
        idx = self.cb_label_col.findText("fault_level")
        self.cb_label_col.setCurrentIndex(idx if idx >= 0 else 0)
        # 初始化下方表结构
        self._sync_unlabelled_headers()

    # ---- 事件：加载CSV ----
    def _on_load_labelled(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择带等级样本",
            "",
            "表格文件 (*.csv *.xlsx *.xls);;CSV 文件 (*.csv);;Excel 文件 (*.xlsx *.xls);;所有文件 (*)"
        )
        if not path:
            return

        try:
            if path.endswith('.csv'):
                df = pd.read_csv(path)
            elif path.endswith('.xlsx') or path.endswith('.xls'):
                df = pd.read_excel(path)
            else:
                # 尝试自动检测文件类型
                try:
                    df = pd.read_csv(path)
                except:
                    try:
                        df = pd.read_excel(path)
                    except:
                        raise ValueError("无法识别的文件格式")
        except Exception as e:
            QMessageBox.critical(self, "读取失败", f"无法读取文件：\n{path}\n\n{e}")
            return

        if df.shape[1] == 0:
            QMessageBox.warning(self, "提示", "表格文件没有列。")
            return

        _fill_table_from_df(self.tbl_labelled, df, editable=True)
        # 重新填充等级列下拉
        self.cb_label_col.clear()
        self.cb_label_col.addItems(list(df.columns))
        # 尝试猜测等级列（名字里含 level/label/fault）
        guess = next((c for c in df.columns if str(c).lower() in ("level", "label", "fault", "fault_level")), df.columns[-1])
        idx = self.cb_label_col.findText(guess)
        self.cb_label_col.setCurrentIndex(idx if idx >= 0 else 0)
        self._sync_unlabelled_headers()

    # ---- 事件：选择等级列 ----
    def _on_label_col_changed(self, _idx: int):
        self._label_col = self.cb_label_col.currentText()
        self._sync_unlabelled_headers()

    # ---- 同步：待预测表的表头（等于样本表表头去掉等级列） ----
    def _sync_unlabelled_headers(self):
        labelled_cols = [self.tbl_labelled.horizontalHeaderItem(j).text() for j in range(self.tbl_labelled.columnCount())] \
            if self.tbl_labelled.columnCount() > 0 else []
        label_col = self.cb_label_col.currentText() if self.cb_label_col.count() > 0 else None
        feature_cols = [c for c in labelled_cols if c and c != label_col]
        if not feature_cols:
            # 兜底：给出一个空表结构
            feature_cols = ["feat1", "feat2"]
        _ensure_cols(self.tbl_unlabelled, feature_cols, keep_data=False)

    # ---- 删除选中行 ----
    def _del_selected_rows(self):
        rows = sorted({i.row() for i in self.tbl_unlabelled.selectedIndexes()}, reverse=True)
        if not rows:
            return
        for r in rows:
            self.tbl_unlabelled.removeRow(r)

    # ---- 计算等级 ----
    def _on_predict(self):
        if self.cb_label_col.count() == 0:
            QMessageBox.warning(self, "提示", "请先加载样本，并选择故障等级列。")
            return

        label_col = self.cb_label_col.currentText()
        if not label_col:
            QMessageBox.warning(self, "提示", "请选择故障等级列。")
            return

        df_lab = _df_from_table(self.tbl_labelled)
        if df_lab.empty:
            QMessageBox.warning(self, "提示", "故障等级样本表为空。")
            return
        if label_col not in df_lab.columns:
            QMessageBox.warning(self, "提示", f"等级列“{label_col}”不在样本表中。")
            return

        # 拆分标签与特征，并把特征转为数值（非数值->NaN）
        feat_cols = [c for c in df_lab.columns if c != label_col]
        # 允许非数值列存在（例如 device 型号），但会被强制转换为 NaN 然后剔除
        X_lab = df_lab[feat_cols].apply(pd.to_numeric, errors="coerce")
        y_lab = df_lab[label_col]

        # 提示并剔除含 NaN 的行
        nan_rows_lab = X_lab.isna().any(axis=1).to_numpy().nonzero()[0].tolist()
        if nan_rows_lab:
            QMessageBox.information(
                self, "数据清洗",
                f"样本表中有 {len(nan_rows_lab)} 行包含非数值特征，已自动剔除：\n{[int(i) for i in nan_rows_lab]}"
            )
        keep_lab = ~X_lab.isna().any(axis=1)
        X_lab = X_lab[keep_lab].to_numpy(dtype=float)
        y_lab = y_lab[keep_lab].to_numpy()

        if X_lab.size == 0:
            QMessageBox.critical(self, "无有效样本", "清洗后样本为空，请检查数据。")
            return

        # 读取待预测表，并按相同特征列构建矩阵
        df_un = _df_from_table(self.tbl_unlabelled)
        if df_un.empty:
            QMessageBox.warning(self, "提示", "待预测样本表为空，请先填写。")
            return

        # 只取 feat_cols 的交集顺序
        use_cols = [c for c in feat_cols if c in df_un.columns]
        if not use_cols:
            QMessageBox.critical(self, "列不匹配", "待预测表与样本表的特征列不匹配。")
            return

        X_un = df_un[use_cols].apply(pd.to_numeric, errors="coerce")
        nan_rows_un = X_un.isna().any(axis=1).to_numpy().nonzero()[0].tolist()
        if nan_rows_un:
            QMessageBox.information(
                self, "数据清洗",
                f"待预测表中有 {len(nan_rows_un)} 行包含非数值或缺失，已自动剔除：\n{[int(i) for i in nan_rows_un]}"
            )
        keep_un = ~X_un.isna().any(axis=1)
        X_un_valid = X_un[keep_un].to_numpy(dtype=float)

        # 构建估计器并预测
        scaler = (StandardScaler().fit(X_lab) if self._use_scaler else None)
        self._estimator = FaultLevelEstimator(X_lab, y_lab, scaler=scaler)
        if X_un_valid.shape[0] == 0:
            QMessageBox.warning(self, "提示", "清洗后待预测表无有效行。")
            return

        preds = self._estimator.predict(X_un_valid)

        # 把“预测等级”列写回到待预测表（对于被剔除的行留空）
        self._write_predictions_to_unlabelled(preds, keep_un)

        QMessageBox.information(self, "完成", f"已为 {preds.shape[0]} 行写入预测等级。")

    def _write_predictions_to_unlabelled(self, preds: np.ndarray, keep_mask: pd.Series):
        # 确保存在“预测等级”列；无则新增至末列
        pred_col_name = "预测等级"
        headers = [self.tbl_unlabelled.horizontalHeaderItem(j).text() for j in range(self.tbl_unlabelled.columnCount())]
        if pred_col_name not in headers:
            self.tbl_unlabelled.setColumnCount(self.tbl_unlabelled.columnCount() + 1)
            self.tbl_unlabelled.setHorizontalHeaderItem(self.tbl_unlabelled.columnCount() - 1, QTableWidgetItem(pred_col_name))
            headers.append(pred_col_name)
        pred_col = headers.index(pred_col_name)

        r = 0
        for i in range(self.tbl_unlabelled.rowCount()):
            if i < len(keep_mask) and bool(keep_mask.iloc[i]):
                it = QTableWidgetItem(str(preds[r]))
                it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.tbl_unlabelled.setItem(i, pred_col, it)
                r += 1
            else:
                # 被剔除的行置空
                self.tbl_unlabelled.setItem(i, pred_col, QTableWidgetItem(""))

        self.tbl_unlabelled.resizeColumnsToContents()
