import sys
from pathlib import Path
from typing import List, Any

import numpy as np


import matplotlib
matplotlib.use("QtAgg")                     # 明确后端


from PyQt6.QtWidgets import ( QListWidgetItem, QMessageBox,)
from PyQt6.QtCore import Qt

from ml_page import MLWindow,scale_features,train_knn,train_iforest

class UnsupervisedPage(MLWindow):
    """ML page focused purely on unsupervised anomaly detection."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("异常检测")

    # ------------------------------ interface ------------------------------
    def set_data(self, df, checked):
        self.df = df
        self.target_col = None
        self.column_list.clear()
        for col in df.columns:
            item = QListWidgetItem(col)
            item.setCheckState(Qt.CheckState.Checked if col in checked else Qt.CheckState.Unchecked)
            self.column_list.addItem(item)

        self.alg_combo.clear()
        self.alg_combo.addItem("KNN", "knn")
        self.alg_combo.addItem("孤立森林", "iforest")
        self._reset_viz_mode()

    # ------------------------------------------------------------------
    def train_model(self):
        if self.df is None:
            QMessageBox.warning(self, "提示", "请先加载数据")
            return
        cols = self.selected_columns()
        if not cols:
            QMessageBox.warning(self, "提示", "请选择特征")
            return

        X = self.df[cols].astype(np.float32).values
        self.X_scaled, self.scaler = scale_features(X)

        alg = self.alg_combo.currentData()
        if alg == "knn":
            self.model, tau = train_knn(self.X_scaled, k=self.k_spin.value())
            dists, _ = self.model.kneighbors(self.X_scaled)
            self.scores = dists[:, -1]
        else:
            self.model, tau = train_iforest(
                self.X_scaled,
                n_estimators=self.k_spin.value(),
                contamination=self.contam_spin.value(),
            )
            self.scores = -self.model.decision_function(self.X_scaled)

        self.y_true = self.y_pred = None
        self.meta = {"model_type": alg, "tau": tau}
        self.save_btn.setEnabled(True)
        self.canvas.slider.setValue(int(tau * 1000))
        self.refresh_plot()

    # ------------------------------------------------------------------
    def refresh_plot(self):
        if self.scores is None:
            return
        tau = self.meta.get("tau", 0.95)
        viz = self.viz_combo.currentText()

        if viz == "分数直方图":
            self.canvas.plot_hist(self.scores, tau)
        elif viz == "PCA 散点":
            self.canvas.plot_pca(self.X_scaled, self.scores, tau)
        elif viz == "时序折线":
            xs = np.arange(len(self.scores))
            if "TIME" in self.df.columns:
                xs = self.df["TIME"].values
            self.canvas.plot_series(xs, self.scores, tau)
        self.update_abnormal_table(tau)

    def _reset_viz_mode(self):
        self.viz_combo.blockSignals(True)
        self.viz_combo.clear()
        self.viz_combo.addItems(["分数直方图", "PCA 散点", "时序折线"])
        self.canvas.slider.show()
        self.canvas.lbl_tau.show()
        self.tbl_abn.show()
        self.viz_combo.blockSignals(False)



