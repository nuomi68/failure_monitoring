from ml_page import MLWindow, scale_features
from PyQt6.QtWidgets import QListWidgetItem, QMessageBox
from PyQt6.QtCore import Qt
import numpy as np


class SupervisedPage(MLWindow):
    """ML page dedicated to supervised learning."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("监督学习")

    # ------------------------------------------------------------------
    def set_data(self, df, checked, target):
        self.df = df
        self.target_col = target
        self.column_list.clear()
        for col in df.columns:
            item = QListWidgetItem(col)
            item.setCheckState(Qt.CheckState.Checked if col in checked else Qt.CheckState.Unchecked)
            self.column_list.addItem(item)

        self.alg_combo.clear()
        self.alg_combo.addItem("KNN", "knn_clf")
        self.alg_combo.addItem("随机森林", "rf")
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

        y = self.df[self.target_col].values
        from sklearn.utils.multiclass import type_of_target
        if type_of_target(y) in ("continuous", "continuous-multioutput"):
            QMessageBox.warning(self, "错误", "标签列必须为离散类别，当前为连续值")
            return

        alg = self.alg_combo.currentData()
        if alg == "rf":
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(
                n_estimators=self.k_spin.value(), random_state=0
            ).fit(self.X_scaled, y)
            proba = self.model.predict_proba(self.X_scaled)
            tau = 0.5
        else:  # knn_clf
            from sklearn.neighbors import KNeighborsClassifier
            self.model = KNeighborsClassifier(n_neighbors=self.k_spin.value())
            self.model.fit(self.X_scaled, y)
            proba = self.model.predict_proba(self.X_scaled)
            tau = 0.5

        if proba.shape[1] == 1:
            self.scores = proba[:, 0]
        else:
            cls_idx = list(self.model.classes_).index(1)
            self.scores = proba[:, cls_idx]

        self.y_true = y
        self.y_pred = self.model.predict(self.X_scaled)
        self.meta = {"model_type": alg, "tau": tau}
        self.save_btn.setEnabled(True)
        self.canvas.slider.setValue(int(tau * 1000))
        self.refresh_plot()

    # ------------------------------------------------------------------
    def refresh_plot(self):
        if self.scores is None:
            return
        viz = self.viz_combo.currentText()
        if viz == "ROC 曲线":
            self.canvas.plot_roc(self.y_true, self.scores)
        elif viz == "混淆矩阵":
            self.canvas.plot_confmat(self.y_true, self.y_pred)
        elif viz == "PCA 彩色散点":
            self.canvas.plot_pca(self.X_scaled, self.y_true, 1)

    def _reset_viz_mode(self):
        self.viz_combo.blockSignals(True)
        self.viz_combo.clear()
        self.viz_combo.addItems(["ROC 曲线", "混淆矩阵", "PCA 彩色散点"])
        self.canvas.slider.hide()
        self.canvas.lbl_tau.hide()
        self.tbl_abn.hide()
        self.viz_combo.blockSignals(False)

