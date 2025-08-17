
"""
unsupervised_page_rewired.py

无监督学习页（简化示例）：
- 前端不做规范化，把原始 X 传给后端
- 可选规范化器；训练后使用 ML.transform(X) 供 PCA/时序等绘图
- 阈值 τ 取自后端 meta["tau"]（默认 95% 分位）
"""

import numpy as np, pandas as pd, matplotlib.pyplot as plt
from typing import List, Dict, Any
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QListWidget, QListWidgetItem, QHBoxLayout,
    QLabel, QComboBox, QSpinBox, QSlider, QFileDialog, QMessageBox, QDoubleSpinBox,
    QLineEdit, QCheckBox, QFormLayout, QDialog, QDialogButtonBox, QTableWidget,
    QTableWidgetItem, QSplitter
)
from PyQt6.QtCore import Qt, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from sklearn.decomposition import PCA

from backend.ml_interface import ML

# ============== 画布 ==============
class UnCanvas(FigureCanvas):
    tau_changed = pyqtSignal(float)
    def __init__(self, parent=None):
        self.fig, self.ax = plt.subplots()
        super().__init__(self.fig)
        self.setParent(parent)
        self.slider = QSlider(Qt.Orientation.Horizontal, parent)
        self.slider.setRange(0,999)
        self.slider.valueChanged.connect(self._emit)
        self.lbl_tau = QLabel("τ = 0.950")
    def _emit(self, v:int):
        tau = v/1000.0
        self.lbl_tau.setText(f"τ = {tau:.3f}")
        self.tau_changed.emit(tau)

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
        self.ax.set_xlabel("PC1")
        self.ax.set_ylabel("PC2")
        self.ax.legend()
        self.fig.tight_layout()
        self.draw()

    def plot_timeseries(self, X: np.ndarray, scores: np.ndarray, tau: float):
        """把样本按原始顺序，画分数随样本 index 的折线图。"""

        self.ax.clear()
        self.ax.plot(np.arange(len(scores)), scores, lw=1)
        self.ax.axhline(tau, color="red", linestyle="--", label=f"τ={tau:.3f}")
        self.ax.set_xlabel("样本序号")
        self.ax.set_ylabel("打分")
        self.ax.legend()
        self.fig.tight_layout()
        self.draw()

# ================= 高级参数对话框 =================
class AdvancedParamsDialog(QDialog):
    def __init__(self, parent: QWidget, alg: str, params: Dict[str, Any]):
        super().__init__(parent)
        self.alg = alg
        self.params = params  # 传入的字典，会直接修改
        self.setWindowTitle("高级参数")

        form = QFormLayout(self)

        if alg == "knn":
            # algorithm
            self.cb_algorithm = QComboBox()
            self.cb_algorithm.addItems(["auto", "ball_tree", "kd_tree", "brute"])
            self.cb_algorithm.setCurrentText(params.get("algorithm", "auto"))
            form.addRow("algorithm", self.cb_algorithm)

            # leaf_size
            self.sp_leaf = QSpinBox()
            self.sp_leaf.setRange(1, 1000)
            self.sp_leaf.setValue(int(params.get("leaf_size", 30)))
            form.addRow("leaf_size", self.sp_leaf)

            # metric
            self.cb_metric = QComboBox()
            self.cb_metric.addItems(["minkowski", "euclidean", "manhattan", "chebyshev"])
            self.cb_metric.setCurrentText(params.get("metric", "minkowski"))
            form.addRow("metric", self.cb_metric)

            # p
            self.p_row = QWidget()
            p_layout = QHBoxLayout(self.p_row)
            p_layout.setContentsMargins(0, 0, 0, 0)
            lbl_p = QLabel("p:")
            self.sp_p = QSpinBox()
            self.sp_p.setRange(1, 10)
            # 读取并保护 p 的初始值
            raw_p = params.get("p", 2)
            try:
                if raw_p is None:
                    raw_p = 2
                raw_p = int(raw_p)
            except Exception:
                raw_p = 2
            self.sp_p.setValue(raw_p)
            p_layout.addWidget(lbl_p)
            p_layout.addWidget(self.sp_p)
            form.addRow(self.p_row)  # 整行插入
            # 根据 metric 自动显示／隐藏
            self.cb_metric.currentTextChanged.connect(self._toggle_p_row)
            self._toggle_p_row(self.cb_metric.currentText())

            # n_jobs
            self.sp_jobs = QSpinBox()
            self.sp_jobs.setRange(-1, 64)
            self.sp_jobs.setValue(int(params.get("n_jobs", -1)))
            form.addRow("n_jobs", self.sp_jobs)

        elif alg == "iforest":  #
            # max_samples
            self.le_max_samples = QLineEdit(str(params.get("max_samples", "auto")))
            form.addRow("max_samples", self.le_max_samples)

            # max_features
            self.sp_max_features = QDoubleSpinBox()
            self.sp_max_features.setRange(0.0, 1.0)
            self.sp_max_features.setDecimals(3)
            self.sp_max_features.setSingleStep(0.05)
            self.sp_max_features.setValue(float(params.get("max_features", 1.0) or 1.0))
            form.addRow("max_features", self.sp_max_features)

            # bootstrap
            self.ck_bootstrap = QCheckBox()
            self.ck_bootstrap.setChecked(bool(params.get("bootstrap", False)))
            form.addRow("bootstrap", self.ck_bootstrap)

            # random_state
            self.sp_rs = QSpinBox()
            self.sp_rs.setRange(-1, 999999)
            self.sp_rs.setValue(int(params.get("random_state", 0) or 0))
            form.addRow("random_state", self.sp_rs)

            # warm_start
            self.ck_ws = QCheckBox()
            self.ck_ws.setChecked(bool(params.get("warm_start", False)))
            form.addRow("warm_start", self.ck_ws)

            # n_jobs
            self.sp_jobs = QSpinBox()
            self.sp_jobs.setRange(-1, 64)
            self.sp_jobs.setValue(int(params.get("n_jobs", -1)))
            form.addRow("n_jobs", self.sp_jobs)
        elif alg == "autoencoder":
            # 隐藏层
            self.le_hidden = QLineEdit(
                ",".join(str(x) for x in params.get("hidden", [64, 32]))
            )
            form.addRow("隐藏层(逗号分隔)", self.le_hidden)

            self.sp_latent = QSpinBox()
            self.sp_latent.setRange(1, 1024)
            self.sp_latent.setValue(int(params.get("latent_dim", 16)))
            form.addRow("latent_dim", self.sp_latent)

            self.sp_epochs = QSpinBox()
            self.sp_epochs.setRange(1, 10000)
            self.sp_epochs.setValue(int(params.get("epochs", 50)))
            form.addRow("epochs", self.sp_epochs)

            self.sp_batch = QSpinBox()
            self.sp_batch.setRange(1, 100000)
            self.sp_batch.setValue(int(params.get("batch_size", 128)))
            form.addRow("batch_size", self.sp_batch)

            self.sp_lr = QDoubleSpinBox()
            self.sp_lr.setDecimals(6)
            self.sp_lr.setRange(1e-6, 1.0)
            self.sp_lr.setSingleStep(0.0001)
            self.sp_lr.setValue(float(params.get("lr", 1e-3)))
            form.addRow("learning_rate", self.sp_lr)

            self.sp_dropout = QDoubleSpinBox()
            self.sp_dropout.setRange(0.0, 0.9)
            self.sp_dropout.setSingleStep(0.05)
            self.sp_dropout.setValue(float(params.get("dropout", 0.0)))
            form.addRow("dropout", self.sp_dropout)

            buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
            buttons.accepted.connect(self.accept)
            buttons.rejected.connect(self.reject)
            form.addRow(buttons)

    def _toggle_p_row(self, metric: str):
        """只有 minkowski 时显示 p，否则隐藏这一行"""
        is_minko = (metric == "minkowski")
        self.p_row.setVisible(is_minko)

    def accept(self):
        if self.alg == "knn":
            self.params["algorithm"] = self.cb_algorithm.currentText()
            self.params["leaf_size"] = self.sp_leaf.value()
            self.params["metric"] = self.cb_metric.currentText()
            # 只有 minkowski 时更新 p，其它 metric 保留原值，避免出现 None
            if self.cb_metric.currentText() == "minkowski":
                self.params["p"] = self.sp_p.value()
            elif "p" not in self.params or self.params["p"] is None:
                # 若之前没有 p，设一个默认值
                self.params["p"] = 2
            self.params["n_jobs"] = self.sp_jobs.value()
        elif self.alg == "iforest":
            self.params["max_samples"] = self.le_max_samples.text().strip()
            self.params["max_features"] = self.sp_max_features.value()
            self.params["bootstrap"] = self.ck_bootstrap.isChecked()
            self.params["random_state"] = self.sp_rs.value()
            self.params["warm_start"] = self.ck_ws.isChecked()
            self.params["n_jobs"] = self.sp_jobs.value()
        elif self.alg == "autoencoder":
            # 解析隐藏层
            raw = self.le_hidden.text().strip()
            try:
                hidden = [int(x) for x in raw.split(",") if x.strip()]
                if not hidden: hidden = [64, 32]
            except Exception:
                hidden = [64, 32]
            self.params["hidden"] = hidden
            self.params["latent_dim"] = self.sp_latent.value()
            self.params["batch_size"] = self.sp_batch.value()
            self.params["lr"] = self.sp_lr.value()
            self.params["dropout"] = self.sp_dropout.value()
        super().accept()


# ============== 主界面 ==============
class UnsupervisedPage(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("异常检测")
        self.resize(1100, 650)

        # 数据与模型
        self.df: pd.DataFrame | None = None
        self.model = None
        self.scaler = None
        self.meta: dict[str, Any] = {}
        self.scores: np.ndarray | None = None
        self.X_scaled: np.ndarray | None = None

        # 每种算法的高级参数
        self.knn_params: Dict[str, Any] = {
            "n_neighbors": 5,
            "algorithm": "auto",
            "leaf_size": 30,
            "metric": "minkowski",
            "p": 2,
            "n_jobs": -1,
        }
        self.if_params: Dict[str, Any] = {
            "n_estimators": 100,
            "max_samples": "auto",
            "max_features": 1.0,
            "bootstrap": False,
            "random_state": 0,
            "warm_start": False,
            "n_jobs": -1,
        }
        self.ae_params: Dict[str, Any] = {
            "hidden": [64, 32],
            "latent_dim": 16,
            "epochs": 50,
            "batch_size": 128,
            "lr": 1e-3,
            "dropout": 0.0,
        }
        # 顶部参数栏
        top = QHBoxLayout()
        top.addWidget(QLabel("算法："))
        self.alg_combo = QComboBox()
        self.alg_combo.addItem("KNN", "knn")
        self.alg_combo.addItem("孤立森林", "iforest")
        self.alg_combo.addItem("自编码器", "autoencoder")
        self.alg_combo.currentIndexChanged.connect(self._on_alg_changed)
        top.addWidget(self.alg_combo)
        top.addWidget(QLabel("规范化："))
        self.scaler_combo = QComboBox()
        for name, spec in ML.available_scalers():
            self.scaler_combo.addItem(name, spec)
        self.scaler_combo.setCurrentIndex(0)
        top.addWidget(self.scaler_combo)
        top.addStretch()

        # 复用这个 label + spinbox，三种算法下显示不同含义
        self.k_label = QLabel("邻居数：")
        self.k_spin = QSpinBox()
        self.k_spin.setRange(1, 10000)
        self.k_spin.setValue(5)
        self.k_spin.setMinimumWidth(80)
        top.addWidget(self.k_label)
        top.addWidget(self.k_spin)

        # 污染率控件，也要引用到成员，方便 hide/show
        self.contam_label = QLabel("污染率：")
        self.contam_spin = QDoubleSpinBox()
        self.contam_spin.setDecimals(3)
        self.contam_spin.setSingleStep(0.001)
        self.contam_spin.setRange(0.0, 1.0)
        self.contam_spin.setValue(0.01)
        self.contam_spin.setMinimumWidth(100)
        top.addWidget(self.contam_label)
        top.addWidget(self.contam_spin)

        self.btn_advanced = QPushButton("高级参数")
        self.btn_advanced.clicked.connect(self.open_advanced_dialog)
        top.addWidget(self.btn_advanced)

        top.addStretch()

        self.train_btn = QPushButton("训练")
        self.train_btn.clicked.connect(self.train_model)
        top.addWidget(self.train_btn)

        # 主体：左特征 + 右可视化
        splitter = QSplitter(Qt.Orientation.Horizontal)

        self.column_list = QListWidget()
        splitter.addWidget(self.column_list)

        right_panel = QWidget()
        right_v = QVBoxLayout(right_panel)

        self.viz_combo = QComboBox()
        self.viz_combo.addItems(["分数直方图", "PCA 散点", "时序折线"])
        self.viz_combo.currentIndexChanged.connect(lambda _: self.refresh_plot())
        right_v.addWidget(self.viz_combo, alignment=Qt.AlignmentFlag.AlignLeft)

        self.canvas = UnCanvas()
        self.canvas.tau_changed.connect(self.on_tau_changed)
        right_v.addWidget(self.canvas)
        right_v.addWidget(self.canvas.lbl_tau)
        right_v.addWidget(self.canvas.slider)

        self.tbl_abn = QTableWidget(0, 2)
        self.tbl_abn.setHorizontalHeaderLabels(["样本", "分数"])
        self.tbl_abn.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        right_v.addWidget(QLabel("异常样本："))
        right_v.addWidget(self.tbl_abn)

        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)

        main = QVBoxLayout(self)
        main.addLayout(top)
        main.addWidget(splitter)

    def set_data(self, df: pd.DataFrame, checked: List[str]):
        self.df = df
        self.column_list.clear()
        for col in df.columns:
            it = QListWidgetItem(col)
            it.setCheckState(Qt.CheckState.Checked if col in checked else Qt.CheckState.Unchecked)
            self.column_list.addItem(it)

    def selected_columns(self) -> List[str]:
        return [self.column_list.item(i).text() for i in range(self.column_list.count()) if self.column_list.item(i).checkState()==Qt.CheckState.Checked]

    # ---------------- 算法切换 ----------------
    def _on_alg_changed(self, _index: int):
        alg = self.alg_combo.currentData()
        if alg == "knn":
            # KNN：邻居数 + 显示污染率
            self.k_label.setText("邻居数：")
            self.k_spin.setValue(int(self.knn_params.get("n_neighbors", 5)))
            self.contam_label.show()
            self.contam_spin.show()

        elif alg == "iforest":
            # 孤立森林：树数 + 显示污染率
            self.k_label.setText("树数：")
            self.k_spin.setValue(int(self.if_params.get("n_estimators", 100)))
            self.contam_spin.setValue(float(self.if_params.get("contamination", 0.01)))
            self.contam_label.show()
            self.contam_spin.show()

        else:  # autoencoder
            # 自编码器：Epochs + 隐藏污染率
            self.k_label.setText("Epochs：")
            self.k_spin.setValue(int(self.ae_params.get("epochs", 50)))
            self.contam_label.hide()
            self.contam_spin.hide()
    # ---------------- 打开高级参数对话框 ----------------
    def open_advanced_dialog(self):
        alg = self.alg_combo.currentData()
        params = (
            self.knn_params if alg == "knn"
            else self.if_params if alg == "iforest"
            else self.ae_params
        )
        dlg = AdvancedParamsDialog(self, alg, params)
        if dlg.exec():
            # 写回已经在对话框里做了；如果需要可以在此处刷新
            pass

    def train_model(self):
        if self.df is None:
            QMessageBox.warning(self,"提示","请先加载数据")
            return
        cols = self.selected_columns()
        if not cols:
            QMessageBox.warning(self,"提示","请选择特征")
            return

        X = self.df[cols].astype(np.float32).values
        alg = self.alg_combo.currentData()
        scaler_spec = self.scaler_combo.currentData()
        # 主参数：为不同算法映射
        if alg == "knn":
            self.knn_params["n_neighbors"] = self.k_spin.value()
            params = self.knn_params.copy()
        elif alg == "autoencoder":
            self.ae_params["epochs"] = self.k_spin.value()
            params = self.ae_params.copy()
        else:
            self.if_params["n_estimators"] = self.k_spin.value()
            self.if_params["contamination"] = self.contam_spin.value()
            params = self.if_params.copy()
            # 解析 max_samples
            max_samples = self._parse_max_samples(params.get("max_samples", "auto"), len(X))
            params["max_samples"] = max_samples

        rep = ML.train(
            alg=alg,
            X=X,
            params=params,
            scaler=scaler_spec,
            feature_names=cols,
        )
        self.meta = ML.get_meta()                    # meta["tau"] 是 0~1
        self.scores = rep.scores                     # 已经是 0~1
        self.X_scaled = ML.transform(X)
        tau = float(self.meta.get("tau", 0.95))      # 若无则给个默认
        self.canvas.slider.setValue(int(tau * 1000))
        self.refresh_plot()

    def on_tau_changed(self, tau: float):
        self.meta["tau"] = tau
        try:
            from backend.ml_interface import ML
            ML.set_tau(tau, normalized=True)     # ★ 回写给后端
        except Exception:
            pass
        self.refresh_plot()

    def refresh_plot(self):
        if self.scores is None:
            return
        tau = float(self.meta.get("tau", 0.5))
        self._update_abn_table(tau)
        mode = self.viz_combo.currentText()

        if mode == "分数直方图":
            self.canvas.plot_hist(self.scores, tau)
        elif mode == "PCA 散点":
            self.canvas.plot_pca(self.X_scaled, self.scores, tau)
        elif mode == "时序折线":
            self.canvas.plot_timeseries(self.X_scaled, self.scores, tau)
        else:
            # 万一路由到其它，默认直方图
            self.canvas.plot_hist(self.scores, tau)

    def _update_abn_table(self, tau: float):
        if self.scores is None:
            return
        mask = self.scores >= tau
        idxs = np.where(mask)[0]
        # 按分数从高到低排序
        order = np.argsort(self.scores[idxs])[::-1]
        idxs = idxs[order]
        self.tbl_abn.setRowCount(len(idxs))
        for row, idx in enumerate(idxs):
            if self.df is not None and len(self.df.index) > int(idx):
                name = str(self.df.index[int(idx)])
            else:
                name = str(int(idx))
            self.tbl_abn.setItem(row, 0, QTableWidgetItem(name))
            self.tbl_abn.setItem(row, 1, QTableWidgetItem(f"{self.scores[idx]:.3f}"))

    def _parse_max_samples(self,val, n_samples: int):
        """Utility to parse IsolationForest max_samples value."""
        if val == "" or str(val).lower() == "auto":
            return "auto"
        try:
            if "." in str(val):
                f = float(val)
                if 0 < f <= 1:
                    return f
            i = int(val)
            if i > 0:
                return min(i, n_samples)
        except Exception:
            pass
        return "auto"
