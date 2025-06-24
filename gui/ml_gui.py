import sys
from pathlib import Path
from typing import List, Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("QtAgg")                     # 明确后端
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QListWidget, QListWidgetItem,
    QSplitter, QHBoxLayout, QMessageBox, QLabel, QComboBox, QSpinBox,
    QDoubleSpinBox, QTextEdit, QFileDialog, QSlider, QTableWidget, QTableWidgetItem
)
from PyQt6.QtCore import Qt, pyqtSignal


# ======== 纯工具函数======================
def scale_features(X: np.ndarray):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    return Xs, scaler

def save_model(path: Path, model, scaler, meta):
    import joblib
    joblib.dump({"model": model, "scaler": scaler, "meta": meta}, path)

def train_knn(X, k=5):
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=k)
    nbrs.fit(X)
    # tau 取最后一列距离 (k-th) 的 95 分位作为默认阈值
    dist = nbrs.kneighbors(X)[0][:, -1]
    tau = np.quantile(dist, 0.95)
    return nbrs, tau

def train_iforest(X, n_estimators=100, contamination=0.01):
    from sklearn.ensemble import IsolationForest
    clf = IsolationForest(n_estimators=n_estimators,
                          contamination=contamination,
                          random_state=0)
    clf.fit(X)
    score = -clf.decision_function(X)
    tau = np.quantile(score, 0.95)
    return clf, tau


# ======== Matplotlib 统一画布 =========================================
class PlotCanvas(FigureCanvas):
    threshold_changed = pyqtSignal(float)          # 发射新的 τ

    def __init__(self, parent=None):
        self.fig, self.ax = plt.subplots()
        super().__init__(self.fig)
        self.setParent(parent)

        # 滑块 + 标签
        self.slider = QSlider(Qt.Orientation.Horizontal, parent)
        self.slider.setRange(0, 999)               # 0.0 ~ 0.999
        self.slider.setValue(950)                  # 缺省 0.95
        self.slider.valueChanged.connect(self._emit_threshold)
        self.lbl_tau = QLabel("τ = 0.950")

    def _emit_threshold(self, v: int):
        tau = v / 1000.0
        self.lbl_tau.setText(f"阈值 τ = {tau:.3f}")
        self.threshold_changed.emit(tau)

    # ------- 三种绘图方法 -------------------------------------------------
    def plot_hist(self, scores: np.ndarray, tau: float):
        self.ax.clear()
        self.ax.hist(scores, bins=30, alpha=0.7, label="scores")
        self.ax.axvline(tau, color="red", linestyle="--", label=f"τ={tau:.3f}")
        self.ax.legend()
        self.fig.tight_layout()
        self.draw()

    def plot_pca(self, X: np.ndarray, scores: np.ndarray, tau: float):
        self.ax.clear()
        pca = PCA(n_components=2, random_state=0)
        XY = pca.fit_transform(X)
        mask = scores >= tau
        self.ax.scatter(XY[~mask, 0], XY[~mask, 1], c="lightgray", s=10, label="normal")
        self.ax.scatter(XY[mask, 0],  XY[mask, 1],  c="red",       s=15, label="abnormal")
        self.ax.set_xlabel("PC1"); self.ax.set_ylabel("PC2")
        self.ax.legend()
        self.fig.tight_layout()
        self.draw()

    def plot_series(self, xs: np.ndarray, scores: np.ndarray, tau: float):
        self.ax.clear()
        self.ax.plot(xs, scores, label="score")
        mask = scores >= tau
        self.ax.scatter(xs[mask], scores[mask], c="red", zorder=5, label="abnormal")
        self.ax.axhline(tau, color="red", linestyle="--")

        # 等距选择 4~5 个横坐标
        num_ticks = min(5, len(xs))  # 不超过总数
        tick_indices = np.linspace(0, len(xs) - 1, num=num_ticks, dtype=int)
        tick_values = xs[tick_indices]
        self.ax.set_xticks(tick_values)

        # 如果 xs 是文本（如时间标签），也要设 labels
        if np.issubdtype(xs.dtype, np.str_) or xs.dtype == object:
            self.ax.set_xticklabels(tick_values, rotation=45)

        self.ax.set_xlabel("Index")
        self.ax.set_ylabel("score")
        self.ax.legend()
        self.fig.tight_layout()
        self.draw()


# ======== 主窗口 ======================================================
class MLWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ML Frontend")
        self.resize(1000, 600)

        # 数据与模型
        self.df: pd.DataFrame | None = None
        self.model = None
        self.scaler = None
        self.meta: dict[str, Any] = {}
        self.scores: np.ndarray | None = None
        self.X_scaled: np.ndarray | None = None
        self.target_col: str | None = None

        # ===== 顶部参数栏 =================================================
        top = QHBoxLayout()
        top.addWidget(QLabel("算法："))  #
        self.alg_combo = QComboBox()
        self.alg_combo.addItem("KNN", "knn")
        self.alg_combo.addItem("孤立森林", "iforest")
        top.addWidget(self.alg_combo)
        top.addStretch()

        self.k_spin = QSpinBox()
        self.k_spin.setValue(5)
        self.k_spin.setMinimumWidth(80)
        top.addWidget(QLabel("邻居数 / 树数："))
        top.addWidget(self.k_spin)

        self.contam_spin = QDoubleSpinBox()
        self.contam_spin.setDecimals(3)
        self.contam_spin.setSingleStep(0.001)
        self.contam_spin.setRange(0.0, 1.0)
        self.contam_spin.setValue(0.01)
        self.contam_spin.setMinimumWidth(100)
        top.addStretch()
        top.addWidget(QLabel("污染率："))
        top.addWidget(self.contam_spin)
        top.addStretch()

        self.train_btn = QPushButton("训练")
        top.addWidget(self.train_btn)
        self.train_btn.clicked.connect(self.train_model)

        self.save_btn =  QPushButton("保存模型")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self.save_model)
        top.addWidget(self.save_btn)

        # ===== 主体：左特征 + 右可视化 ======================================
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # 左：可勾选特征
        self.column_list = QListWidget()
        splitter.addWidget(self.column_list)

        # 右：可视化区域（垂直 组合）
        right_panel = QWidget(); right_v = QVBoxLayout(right_panel)

        # --- 右上：图形类型切换
        self.viz_combo = QComboBox()
        self.viz_combo.addItems(["分数直方图", "PCA 散点", "时序折线"])
        self.viz_combo.currentIndexChanged.connect(lambda _: self.refresh_plot())
        right_v.addWidget(self.viz_combo, alignment=Qt.AlignmentFlag.AlignLeft)

        # --- 右中：Matplotlib 画布 + 滑块
        self.canvas = PlotCanvas()
        self.canvas.threshold_changed.connect(self.on_tau_changed)
        right_v.addWidget(self.canvas)
        right_v.addWidget(self.canvas.lbl_tau)
        right_v.addWidget(self.canvas.slider)

        # --- 右下：异常样本列表
        self.tbl_abn = QTableWidget(0, 2)
        self.tbl_abn.setHorizontalHeaderLabels(["索引", "分数"])
        self.tbl_abn.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        right_v.addWidget(QLabel("异常样本："))
        right_v.addWidget(self.tbl_abn)

        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1); splitter.setStretchFactor(1, 2)

        # ===== 主布局 =====================================================
        main = QVBoxLayout(self)
        main.addLayout(top)
        main.addWidget(splitter)

    # ---------------------------------------------------------------------
    # 外部注入数据
    def set_data(self, df: pd.DataFrame, checked: List[str], target: str | None = None):
        """载入数据并根据是否有标签切换算法列表"""
        self.df = df
        self.target_col = target
        self.column_list.clear()
        for col in df.columns:
            item = QListWidgetItem(col)
            item.setCheckState(Qt.CheckState.Checked if col in checked else Qt.CheckState.Unchecked)
            self.column_list.addItem(item)

        self.alg_combo.clear()
        if target:
            self.alg_combo.addItem("带标签KNN", "knn_clf")
            self.alg_combo.addItem("随机森林", "rf")
        else:
            self.alg_combo.addItem("KNN", "knn")
            self.alg_combo.addItem("孤立森林", "iforest")

    def selected_columns(self) -> List[str]:
        cols = []
        for i in range(self.column_list.count()):
            it = self.column_list.item(i)
            if it.checkState() == Qt.CheckState.Checked:
                cols.append(it.text())
        return cols

    # ---------------------------------------------------------------------
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
        if self.target_col:
            y = self.df[self.target_col].values
            if alg == "rf":
                from sklearn.ensemble import RandomForestClassifier
                self.model = RandomForestClassifier(
                    n_estimators=self.k_spin.value(), random_state=0
                ).fit(self.X_scaled, y)
                self.scores = self.model.predict_proba(self.X_scaled)[:, 1]
                tau = 0.5
            else:  # knn_clf
                from sklearn.neighbors import KNeighborsClassifier
                self.model = KNeighborsClassifier(n_neighbors=self.k_spin.value())
                self.model.fit(self.X_scaled, y)
                self.scores = self.model.predict_proba(self.X_scaled)[:, 1]
                tau = 0.5
        else:
            if alg == "knn":
                self.model, tau = train_knn(self.X_scaled, k=self.k_spin.value())
                self.scores = self.model.kneighbors(self.X_scaled)[0][:, -1]
            else:
                self.model, tau = train_iforest(
                    self.X_scaled,
                    n_estimators=self.k_spin.value(),
                    contamination=self.contam_spin.value()
                )
                self.scores = -self.model.decision_function(self.X_scaled)

        self.meta = {"model_type": alg, "tau": tau}
        self.save_btn.setEnabled(True)

        # 设置滑块到 tau 位置
        self.canvas.slider.setValue(int(tau * 1000))
        self.refresh_plot()

    # ---------------------------------------------------------------------
    def on_tau_changed(self, tau: float):
        self.meta["tau"] = tau
        self.refresh_plot()

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

        # 更新异常表
        self.update_abnormal_table(tau)

    def update_abnormal_table(self, tau: float):
        mask = self.scores >= tau
        idxs = np.where(mask)[0]
        self.tbl_abn.setRowCount(len(idxs))
        for r, i in enumerate(idxs):
            self.tbl_abn.setItem(r, 0, QTableWidgetItem(str(i)))
            self.tbl_abn.setItem(r, 1, QTableWidgetItem(f"{self.scores[i]:.4f}"))

    # ---------------------------------------------------------------------
    def save_model(self):
        if self.model is None:  # 未训练
            return
        name, _ = QFileDialog.getSaveFileName(self, "Save Model", "", "Joblib Files (*.joblib)")
        if name:
            save_model(Path(name), self.model, self.scaler, self.meta)


# ================== DEMO 入口 ============================================
def main():
    # 随机生成 demo 数据
    rng = np.random.default_rng(0)
    df_demo = pd.DataFrame(rng.normal(size=(500, 6)), columns=[f"f{i}" for i in range(6)])
    df_demo["TIME"] = pd.date_range("2023-01-01", periods=len(df_demo), freq="H")

    app = QApplication(sys.argv)
    win = MLWindow()
    win.set_data(df_demo, checked=["f0", "f1", "f2"])
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
