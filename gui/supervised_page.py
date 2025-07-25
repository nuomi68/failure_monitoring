import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from pathlib import Path
from typing import List, Any

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QListWidget, QListWidgetItem,
    QSplitter, QHBoxLayout, QMessageBox, QLabel, QComboBox, QSpinBox,
    QFileDialog, QTableWidget, QTableWidgetItem, QSlider
)
from PyQt6.QtCore import Qt, pyqtSignal

from tools import scale_features, save_model


class PlotCanvas(FigureCanvas):
    """Matplotlib canvas used by the supervised page."""

    threshold_changed = pyqtSignal(float)

    def __init__(self, parent=None) -> None:
        self.fig, self.ax = plt.subplots()
        super().__init__(self.fig)
        self.setParent(parent)

        self.slider = QSlider(Qt.Orientation.Horizontal, parent)
        self.slider.setRange(0, 999)
        self.slider.setValue(500)
        self.slider.valueChanged.connect(self._emit_threshold)
        self.lbl_tau = QLabel("阈值 τ = 0.500")

    def _emit_threshold(self, v: int) -> None:
        tau = v / 1000.0
        self.lbl_tau.setText(f"阈值 τ = {tau:.3f}")
        self.threshold_changed.emit(tau)

    def plot_pca(self, X: np.ndarray, labels: np.ndarray, _tau: float) -> None:
        self.ax.clear()
        pca = PCA(n_components=2, random_state=0)
        XY = pca.fit_transform(X)
        self.ax.scatter(XY[:, 0], XY[:, 1], c=labels, cmap="coolwarm", s=15)
        self.ax.set_xlabel("PC1"); self.ax.set_ylabel("PC2")
        self.fig.tight_layout(); self.draw()

    def plot_roc(self, y_true, y_score) -> None:
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, _ = roc_curve(y_true, y_score)
        self.ax.clear()
        self.ax.plot(fpr, tpr, lw=2, label=f"AUC={auc(fpr,tpr):.3f}")
        self.ax.plot([0,1], [0,1], "--", lw=1, color="gray")
        self.ax.set_xlabel("False Positive Rate"); self.ax.set_ylabel("True Positive Rate")
        self.ax.legend(); self.fig.tight_layout(); self.draw()

    def plot_confmat(self, y_true, y_pred) -> None:
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)
        self.ax.clear()
        im = self.ax.imshow(cm, cmap="Blues")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                self.ax.text(j, i, cm[i, j], ha="center", va="center")
        self.ax.set_xlabel("Predicted"); self.ax.set_ylabel("True")
        self.fig.colorbar(im, ax=self.ax); self.fig.tight_layout(); self.draw()


class SupervisedPage(QWidget):
    """Supervised learning interface similar to ``UnsupervisedPage``."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("监督学习")
        self.resize(1100, 650)

        # 数据与模型
        self.df: pd.DataFrame | None = None
        self.model: Any | None = None
        self.scaler: Any | None = None
        self.meta: dict[str, Any] = {}
        self.scores: np.ndarray | None = None
        self.X_scaled: np.ndarray | None = None
        self.y_true: np.ndarray | None = None
        self.y_pred: np.ndarray | None = None
        self.target_col: str | None = None

        # 顶部参数栏
        top = QHBoxLayout()
        top.addWidget(QLabel("算法："))
        self.alg_combo = QComboBox()
        self.alg_combo.addItem("KNN", "knn_clf")
        self.alg_combo.addItem("随机森林", "rf")
        self.alg_combo.currentIndexChanged.connect(self._on_alg_changed)
        top.addWidget(self.alg_combo)
        top.addStretch()

        self.k_label = QLabel("邻居数：")
        self.k_spin = QSpinBox()
        self.k_spin.setRange(1, 10000)
        self.k_spin.setValue(5)
        self.k_spin.setMinimumWidth(80)
        top.addWidget(self.k_label)
        top.addWidget(self.k_spin)

        top.addStretch()
        self.train_btn = QPushButton("训练")
        self.train_btn.clicked.connect(self.train_model)
        top.addWidget(self.train_btn)

        self.save_btn = QPushButton("保存模型")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self.save_model)
        top.addWidget(self.save_btn)

        # 主体：左特征 + 右可视化
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.column_list = QListWidget(); splitter.addWidget(self.column_list)

        right_panel = QWidget(); right_v = QVBoxLayout(right_panel)
        self.viz_combo = QComboBox()
        self.viz_combo.addItems(["ROC 曲线", "混淆矩阵", "PCA 彩色散点"])
        self.viz_combo.currentIndexChanged.connect(lambda _: self.refresh_plot())
        right_v.addWidget(self.viz_combo, alignment=Qt.AlignmentFlag.AlignLeft)

        self.canvas = PlotCanvas()
        self.canvas.threshold_changed.connect(self.on_tau_changed)
        right_v.addWidget(self.canvas)
        right_v.addWidget(self.canvas.lbl_tau)
        right_v.addWidget(self.canvas.slider)
        self.canvas.slider.hide(); self.canvas.lbl_tau.hide()

        self.tbl_abn = QTableWidget(0, 2)
        self.tbl_abn.setHorizontalHeaderLabels(["索引", "分数"])
        self.tbl_abn.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.tbl_abn.hide()
        right_v.addWidget(QLabel("异常样本："))
        right_v.addWidget(self.tbl_abn)

        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1); splitter.setStretchFactor(1, 2)

        main = QVBoxLayout(self)
        main.addLayout(top)
        main.addWidget(splitter)

    # ---------------- 外部载入数据 ----------------
    def set_data(self, df: pd.DataFrame, checked: List[str], target: str) -> None:
        self.df = df
        self.target_col = target
        self.column_list.clear()
        for col in df.columns:
            item = QListWidgetItem(col)
            item.setCheckState(Qt.CheckState.Checked if col in checked else Qt.CheckState.Unchecked)
            self.column_list.addItem(item)
        self._reset_viz_mode()

    def selected_columns(self) -> List[str]:
        cols = []
        for i in range(self.column_list.count()):
            it = self.column_list.item(i)
            if it.checkState() == Qt.CheckState.Checked:
                cols.append(it.text())
        return cols

    # ---------------- 算法切换 ----------------
    def _on_alg_changed(self, _index: int) -> None:
        alg = self.alg_combo.currentData()
        if alg == "rf":
            self.k_label.setText("树数：")
        else:
            self.k_label.setText("邻居数：")

    # ---------------- 训练模型 ----------------
    def train_model(self) -> None:
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
        else:
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

    def on_tau_changed(self, tau: float) -> None:
        self.meta["tau"] = tau
        self.refresh_plot()

    # ---------------- 刷新可视化 ----------------
    def refresh_plot(self) -> None:
        if self.scores is None:
            return
        viz = self.viz_combo.currentText()
        if viz == "ROC 曲线":
            self.canvas.plot_roc(self.y_true, self.scores)
        elif viz == "混淆矩阵":
            self.canvas.plot_confmat(self.y_true, self.y_pred)
        elif viz == "PCA 彩色散点":
            self.canvas.plot_pca(self.X_scaled, self.y_true, 1)

    def _reset_viz_mode(self) -> None:
        self.viz_combo.blockSignals(True)
        self.viz_combo.clear()
        self.viz_combo.addItems(["ROC 曲线", "混淆矩阵", "PCA 彩色散点"])
        self.canvas.slider.hide()
        self.canvas.lbl_tau.hide()
        self.tbl_abn.hide()
        self.viz_combo.blockSignals(False)

    # ---------------- 保存模型 ----------------
    def save_model(self) -> None:
        if self.model is None:
            return
        name, _ = QFileDialog.getSaveFileName(self, "保存模型", "", "Joblib Files (*.joblib)")
        if name:
            save_model(Path(name), self.model, self.scaler, self.meta)
