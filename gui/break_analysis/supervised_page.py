
"""
supervised_page_rewired.py

监督学习页（简化示例）：
- 前端不做规范化；把原始 X/y 传给后端
- 可选择规范化器；绘图使用 ML.transform(X_test) 取得与训练一致的规范化数据
"""

import numpy as np, pandas as pd, matplotlib.pyplot as plt
from typing import Any, Dict, List, Optional
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QListWidget, QListWidgetItem, QSplitter, QHBoxLayout,
    QMessageBox, QLabel, QComboBox, QSpinBox, QPushButton, QFileDialog, QLineEdit,
    QTableWidget, QTableWidgetItem, QSlider, QDialog, QFormLayout,
    QDialogButtonBox, QDoubleSpinBox, QCheckBox, QSizePolicy, QAbstractItemView
)
from PyQt6.QtCore import Qt, pyqtSignal
from sklearn.decomposition import PCA
from sklearn.utils.multiclass import type_of_target
from sklearn.model_selection import train_test_split

from backend.ml_interface import ML
from gui.smart_table import SmartTable, SmartTableConfig
from gui.tools import logger
from gui.mpl_preview import MatplotlibPreviewDialog, InteractiveMplCanvas

SPLITTER_HANDLE_STYLE = """
QSplitter::handle {
    background-color: #000000;
    border: none;
    margin: 0;
}
"""
SPLITTER_HANDLE_WIDTH = 2
# ============== 画布 ==============
class PlotCanvas(InteractiveMplCanvas):
    threshold_changed = pyqtSignal(float)
    def __init__(self, parent=None, show_controls: bool = True):
        super().__init__(parent)
        self.cbar = None
        self.slider = None
        self.lbl_tau = None
        if show_controls:
            self.slider = QSlider(Qt.Orientation.Horizontal, parent)
            self.slider.setRange(0, 999)
            self.slider.setValue(500)
            self.slider.valueChanged.connect(self._emit_threshold)
            self.lbl_tau = QLabel("阈值 τ = 0.500")
        else:
            self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    def _emit_threshold(self, v: int):
        tau = v / 1000.0
        if self.lbl_tau:
            self.lbl_tau.setText(f"阈值 τ = {tau:.3f}")
        self.threshold_changed.emit(tau)
    def _clear_cbar(self):
        if self.cbar:
            try:
                self.cbar.remove()
            except Exception:
                pass
            self.cbar = None
    def clear(self):
        self._clear_cbar()
        self.ax.clear()
        self.draw()
    def show_text(self, text: str):
        self._clear_cbar()
        self.ax.clear()
        self.ax.text(0.5, 0.5, text, ha="center", va="center")
        self.ax.set_axis_off()
        self.fig.tight_layout()
        self.draw()

    def _set_square_axes(self, ax=None):
        target_ax = ax if ax is not None else self.ax

        # 先根据现有数据得到范围
        x0, x1 = target_ax.get_xlim()
        y0, y1 = target_ax.get_ylim()

        dx = x1 - x0
        dy = y1 - y0
        side = max(dx, dy)  # 取更大的那个跨度

        # 以当前中心为基准，把两个轴都调成同样的跨度
        x_c = (x0 + x1) / 2
        y_c = (y0 + y1) / 2
        target_ax.set_xlim(x_c - side / 2, x_c + side / 2)
        target_ax.set_ylim(y_c - side / 2, y_c + side / 2)

        # 再保证单位比例一致
        target_ax.set_aspect("equal", adjustable="box")

    def _resize_colorbar(self, ratio: float = 0.6):
        if not self.cbar:
            return
        try:
            ratio = float(ratio)
        except Exception:
            ratio = 0.6
        ratio = max(0.0, min(1.0, ratio))
        ax_box = self.ax.get_position()
        cb_ax = self.cbar.ax
        cb_box = cb_ax.get_position()
        target_h = ax_box.height * ratio
        y0 = ax_box.y0 + (ax_box.height - target_h) / 2
        cb_ax.set_position([cb_box.x0, y0, cb_box.width, target_h])
    def plot_pca(self, X: np.ndarray, labels: np.ndarray):
        self._clear_cbar()
        self.ax.clear()
        XY = PCA(n_components=2, random_state=0).fit_transform(X)
        classes = np.unique(labels)
        for cls in classes:
            mask = labels == cls
            self.ax.scatter(XY[mask, 0], XY[mask, 1], s=15, label=str(cls))
        self.ax.set_xlabel("PC1")
        self.ax.set_ylabel("PC2")
        self.ax.set_title("PCA 彩色散点")
        self.ax.legend(title="类别")
        self._set_square_axes(self.ax)
        self.fig.tight_layout()
        self.draw()
    def plot_confmat(self, y_true, y_pred, labels=None):
        from sklearn.metrics import confusion_matrix
        # 若后端提供了原始类名，而 y_true/y_pred 是编码后的整数，
        # 则用整数索引计算矩阵，但坐标轴显示原始标签。
        if labels is not None and np.issubdtype(np.asarray(y_true).dtype, np.number):
            cm_labels = list(range(len(labels)))
            tick_labels = labels
        else:
            cm_labels = np.unique(list(y_true) + list(y_pred)) if labels is None else labels
            tick_labels = cm_labels
        cm = confusion_matrix(y_true, y_pred, labels=cm_labels)
        self._clear_cbar()
        self.ax.clear()
        im = self.ax.imshow(cm, cmap="Blues")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                self.ax.text(j, i, cm[i, j], ha="center", va="center")
        ticks = range(len(cm_labels))
        self.ax.set_xticks(ticks)
        self.ax.set_yticks(ticks)
        self.ax.set_xticklabels(tick_labels, rotation=45, ha="right")
        self.ax.set_yticklabels(tick_labels)
        self.ax.set_xlabel("预测")
        self.ax.set_ylabel("真实")
        self.ax.set_title("混淆矩阵")
        self._set_square_axes(self.ax)
        self.cbar = self.fig.colorbar(im, ax=self.ax)
        self.fig.tight_layout()
        self.draw()

    def plot_scores(self, y_true, y_pred, labels=None):
        """
        仅按“真实标签 y_true 中出现过的类”计算并绘制每类的
        精确率/召回率/F1 柱状图；不绘制平均。
        兼容 y_true/y_pred 为数字编码而 labels 为字符串标签的情况。
        """
        import numpy as np
        from sklearn.metrics import precision_recall_fscore_support

        self._clear_cbar()
        self.ax.clear()

        yt = np.asarray(y_true)
        yp = np.asarray(y_pred)

        # 1) 真实标签中出现过的类（顺序：按出现顺序去重；若想排序可换成 np.unique）
        #   用 dict.fromkeys 保持首次出现顺序
        classes_true = np.array(list(dict.fromkeys(yt.tolist())))

        # 2) 对齐“编码/原始标签”
        #   - y_true 是数字编码 & labels 是字符串：计算时用编码，显示用原始字符串
        #   - y_true 是字符串：直接按字符串类名计算与显示
        #   - 其余情况：直接用 classes_true 作为计算与显示标签
        if np.issubdtype(yt.dtype, np.number) and labels is not None and len(labels) > 0 and not np.issubdtype(
                np.asarray(labels).dtype, np.number):
            score_labels = classes_true.astype(int).tolist()
            tick_labels = [str(labels[i]) for i in score_labels]  # 只取真实里出现过的类名
        else:
            score_labels = classes_true.tolist()
            tick_labels = [str(x) for x in score_labels]

        # 3) 逐类计算（缺失/无预测用 0 兜底）
        p, r, f1, _ = precision_recall_fscore_support(
            yt, yp, labels=score_labels, zero_division=0
        )

        # 4) 绘图（每类三根柱）
        x = np.arange(len(score_labels))
        w = 0.25
        self.ax.bar(x - w, p, w, label="精确率")
        self.ax.bar(x, r, w, label="召回率")
        self.ax.bar(x + w, f1, w, label="F1")

        # 数值标注（可选，清晰一些）
        def _annot(vals, xoff):
            for i, v in enumerate(vals):
                self.ax.text(i + xoff, float(v) + 0.02, f"{v:.2f}",
                             ha="center", va="bottom", fontsize=8)

        _annot(p, -w)
        _annot(r, 0)
        _annot(f1, +w)

        self.ax.set_xticks(x)
        self.ax.set_xticklabels(tick_labels, rotation=45, ha="right")
        self.ax.set_ylim(0, 1.05)
        self.ax.set_ylabel("得分")
        self.ax.set_title("各类别指标")
        self.ax.legend()
        self.fig.tight_layout()
        self.draw()


# ============== 放大窗口 ==============
class SquareCanvasHolder(QWidget):
    """Keeps an embedded canvas square and centered within its container."""

    def __init__(self, canvas: PlotCanvas, parent: QWidget | None = None):
        super().__init__(parent)
        self.canvas = canvas
        self.canvas.setParent(self)
        self.setMinimumSize(200, 200)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def resizeEvent(self, event):
        w = event.size().width()
        h = event.size().height()
        side = max(min(w, h), 200)
        x = (w - side) // 2
        y = (h - side) // 2
        self.canvas.setGeometry(x, y, side, side)
        super().resizeEvent(event)


class SliderHolder(QWidget):
    """Keep the τ slider at ~90% of the available width while staying centered."""

    def __init__(self, slider: QSlider, parent: QWidget | None = None, ratio: float = 0.9):
        super().__init__(parent)
        self.slider = slider
        self.slider.setParent(self)
        self._ratio = max(0.0, min(1.0, float(ratio)))
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumHeight(slider.sizeHint().height() + 12)

    def resizeEvent(self, event):
        w = event.size().width()
        h = event.size().height()
        target_w = max(1, min(w, int(w * self._ratio)))
        slider_h = min(h, self.slider.sizeHint().height())
        x = (w - target_w) // 2
        y = (h - slider_h) // 2
        self.slider.setGeometry(x, y, target_w, slider_h)
        super().resizeEvent(event)


# ============== 高级参数对话框 ==============
class AdvParamsDlg(QDialog):
    def __init__(self, parent: QWidget, code: str, params: Dict[str, Any], is_classification: bool):
        super().__init__(parent)
        self.code = code
        self.params = params
        self.is_clf = is_classification
        self.setWindowTitle("高级参数")
        form = QFormLayout(self)

        if "knn" in code:
            self.cb_w = QComboBox()
            self.cb_w.addItems(["uniform", "distance"])
            self.cb_w.setCurrentText(params.get("weights", "uniform"))
            form.addRow("weights", self.cb_w)

            self.cb_alg = QComboBox()
            self.cb_alg.addItems(["auto", "ball_tree", "kd_tree", "brute"])
            self.cb_alg.setCurrentText(params.get("algorithm", "auto"))
            form.addRow("algorithm", self.cb_alg)

            self.sp_leaf = QSpinBox()
            self.sp_leaf.setRange(1, 1000)
            self.sp_leaf.setValue(int(params.get("leaf_size", 30)))
            form.addRow("leaf_size", self.sp_leaf)

            self.cb_metric = QComboBox()
            self.cb_metric.addItems(["minkowski", "euclidean", "manhattan", "chebyshev"])
            self.cb_metric.setCurrentText(params.get("metric", "minkowski"))
            form.addRow("metric", self.cb_metric)

            self.p_row = QWidget()
            p_layout = QHBoxLayout(self.p_row)
            p_layout.setContentsMargins(0, 0, 0, 0)
            self.sp_p = QSpinBox()
            self.sp_p.setRange(1, 10)
            self.sp_p.setValue(int(params.get("p", 2)))
            p_layout.addWidget(QLabel("p"))
            p_layout.addWidget(self.sp_p)
            form.addRow(self.p_row)

            self.cb_metric.currentTextChanged.connect(self._toggle_p_row)
            self._toggle_p_row(self.cb_metric.currentText())

            self.sp_jobs = QSpinBox()
            self.sp_jobs.setRange(-1, 64)
            self.sp_jobs.setValue(int(params.get("n_jobs", -1)))
            form.addRow("n_jobs", self.sp_jobs)

        elif "rf" in code:
            self.cb_criterion = QComboBox()
            if self.is_clf:
                self.cb_criterion.addItems(["gini", "entropy", "log_loss"])
            else:
                self.cb_criterion.addItems(["squared_error", "absolute_error", "friedman_mse", "poisson"])
            self.cb_criterion.setCurrentText(params.get("criterion", self.cb_criterion.itemText(0)))
            form.addRow("criterion", self.cb_criterion)

            self.sp_depth = QSpinBox()
            self.sp_depth.setRange(0, 1000)
            depth_val = params.get("max_depth", None)
            self.sp_depth.setValue(0 if depth_val in (None, 0) else int(depth_val))
            form.addRow("max_depth(0=∞)", self.sp_depth)

            self.sp_mss = QSpinBox()
            self.sp_mss.setRange(2, 1000)
            self.sp_mss.setValue(int(params.get("min_samples_split", 2)))
            form.addRow("min_samples_split", self.sp_mss)

            self.sp_msl = QSpinBox()
            self.sp_msl.setRange(1, 1000)
            self.sp_msl.setValue(int(params.get("min_samples_leaf", 1)))
            form.addRow("min_samples_leaf", self.sp_msl)

            self.le_mf = QLineEdit(str(params.get("max_features", "sqrt")))
            form.addRow("max_features", self.le_mf)

            self.ck_boot = QCheckBox()
            self.ck_boot.setChecked(params.get("bootstrap", True))
            form.addRow("bootstrap", self.ck_boot)

            self.sp_jobs = QSpinBox()
            self.sp_jobs.setRange(-1, 64)
            self.sp_jobs.setValue(int(params.get("n_jobs", -1)))
            form.addRow("n_jobs", self.sp_jobs)

        else:
            def _dbl(min_v, max_v, step, value, decimals=3):
                sp = QDoubleSpinBox()
                sp.setDecimals(decimals)
                sp.setRange(min_v, max_v)
                sp.setSingleStep(step)
                sp.setValue(value)
                return sp

            self.sp_depth = QSpinBox()
            self.sp_depth.setRange(1, 64)
            self.sp_depth.setValue(int(params.get("max_depth", 6)))
            form.addRow("max_depth", self.sp_depth)

            self.sp_lr = _dbl(0.001, 1.0, 0.01, float(params.get("learning_rate", 0.1)))
            form.addRow("learning_rate", self.sp_lr)

            self.sp_subsample = _dbl(0.1, 1.0, 0.05, float(params.get("subsample", 0.8)), decimals=2)
            form.addRow("subsample", self.sp_subsample)

            self.sp_colsample = _dbl(0.1, 1.0, 0.05, float(params.get("colsample_bytree", 0.8)), decimals=2)
            form.addRow("colsample_bytree", self.sp_colsample)

            self.sp_mcw = _dbl(0.0, 100.0, 0.5, float(params.get("min_child_weight", 1.0)), decimals=2)
            form.addRow("min_child_weight", self.sp_mcw)

            self.sp_gamma = _dbl(0.0, 10.0, 0.1, float(params.get("gamma", 0.0)), decimals=2)
            form.addRow("gamma", self.sp_gamma)

            self.sp_reg_lambda = _dbl(0.0, 20.0, 0.1, float(params.get("reg_lambda", 1.0)), decimals=2)
            form.addRow("reg_lambda", self.sp_reg_lambda)

            self.sp_reg_alpha = _dbl(0.0, 20.0, 0.1, float(params.get("reg_alpha", 0.0)), decimals=2)
            form.addRow("reg_alpha", self.sp_reg_alpha)

            self.sp_jobs = QSpinBox()
            self.sp_jobs.setRange(-1, 64)
            self.sp_jobs.setValue(int(params.get("n_jobs", -1)))
            form.addRow("n_jobs", self.sp_jobs)

            self.cb_objective = QComboBox()
            if self.is_clf:
                options = [
                    ("binary:logistic", "binary:logistic"),
                    ("binary:logitraw", "binary:logitraw"),
                    ("multi:softprob", "multi:softprob"),
                ]
                self.sp_spw = _dbl(0.0, 1000.0, 0.5, float(params.get("scale_pos_weight", 1.0)), decimals=2)
                form.addRow("scale_pos_weight", self.sp_spw)
            else:
                options = [
                    ("reg:squarederror", "reg:squarederror"),
                    ("reg:gamma", "reg:gamma"),
                    ("reg:pseudohubererror", "reg:pseudohubererror"),
                ]
            for text, data in options:
                self.cb_objective.addItem(text, data)
            cur = params.get("objective", options[0][1])
            idx = self.cb_objective.findData(cur)
            self.cb_objective.setCurrentIndex(idx if idx >= 0 else 0)
            form.addRow("objective", self.cb_objective)

        self.sp_test_size = QDoubleSpinBox()
        self.sp_test_size.setRange(0.05,0.95)
        self.sp_test_size.setSingleStep(0.05)
        self.sp_test_size.setValue(float(params.get("test_size",0.2)))
        form.addRow("\u6d4b\u8bd5\u96c6\u6bd4\u4f8b", self.sp_test_size)
        self.sp_rs = QSpinBox()
        self.sp_rs.setRange(-1,999999)
        self.sp_rs.setValue(int(params.get("random_state",0)))
        form.addRow("\u968f\u673a\u79cd\u5b50", self.sp_rs)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        form.addRow(btns)

    def _toggle_p_row(self, metric: str):
        if hasattr(self, "p_row"): self.p_row.setVisible(metric == "minkowski")

    def accept(self):
        self.params["test_size"] = self.sp_test_size.value()
        self.params["random_state"] = self.sp_rs.value()
        if "knn" in self.code:
            self.params["weights"] = self.cb_w.currentText()
            self.params["algorithm"] = self.cb_alg.currentText()
            self.params["leaf_size"] = self.sp_leaf.value()
            self.params["metric"] = self.cb_metric.currentText()
            self.params["n_jobs"] = self.sp_jobs.value()
            if self.cb_metric.currentText()=="minkowski": self.params["p"] = self.sp_p.value()
            elif "p" in self.params: del self.params["p"]
        elif "rf" in self.code:
            self.params["criterion"] = self.cb_criterion.currentText()
            self.params["max_depth"] = None if self.sp_depth.value()==0 else self.sp_depth.value()
            self.params["min_samples_split"] = self.sp_mss.value()
            self.params["min_samples_leaf"] = self.sp_msl.value()
            self.params["max_features"] = self._parse_max_features(self.le_mf.text().strip())
            self.params["bootstrap"] = self.ck_boot.isChecked()
            self.params["n_jobs"] = self.sp_jobs.value()
        else:
            self.params["max_depth"] = self.sp_depth.value()
            self.params["learning_rate"] = self.sp_lr.value()
            self.params["subsample"] = self.sp_subsample.value()
            self.params["colsample_bytree"] = self.sp_colsample.value()
            self.params["min_child_weight"] = self.sp_mcw.value()
            self.params["gamma"] = self.sp_gamma.value()
            self.params["reg_lambda"] = self.sp_reg_lambda.value()
            self.params["reg_alpha"] = self.sp_reg_alpha.value()
            self.params["n_jobs"] = self.sp_jobs.value()
            self.params["objective"] = self.cb_objective.currentData()
            if hasattr(self, "sp_spw"):
                self.params["scale_pos_weight"] = self.sp_spw.value()
            else:
                self.params.pop("scale_pos_weight", None)
        super().accept()
        super().accept()

    @staticmethod
    def _parse_max_features(text: str):
        if text.lower() in ("","none"): return None
        if text in ("sqrt","log2"): return text
        try:
            if "." in text:
                f = float(text)
                if 0 < f <= 1: return f
            i = int(text)
            if i>0: return i
        except Exception: pass
        return "sqrt"


# ============== 主界面 ==============
class SupervisedPage(QWidget):
    def __init__(self):
        super().__init__()
        self.df: pd.DataFrame | None = None
        self.target_col: str | None = None
        self.X_test: np.ndarray | None = None
        self.X_test_raw: np.ndarray | None = None
        self.y_true: np.ndarray | None = None
        self.y_pred: np.ndarray | None = None
        self.test_scores: np.ndarray | None = None
        self.meta: Dict[str, Any] = {}
        self.is_clf = True
        self.binary = False
        self.metrics_text = ""
        self._preview_dialogs: list[MatplotlibPreviewDialog] = []
        self._test_feature_names: List[str] = []
        self._test_indices: np.ndarray | None = None
        self._viz_options: list[tuple[str, str]] = []
        self._scatter_points: dict[str, Any] | None = None
        self.slider_holder: SliderHolder | None = None
        self.X_train_raw: np.ndarray | None = None
        self.y_train: np.ndarray | None = None
        self.y_pred_train: np.ndarray | None = None
        self.train_scores: np.ndarray | None = None
        self._train_indices: np.ndarray | None = None

        self.knn_params: Dict[str, Any] = {
            "n_neighbors": 5,
            "weights": "uniform",
            "algorithm": "auto",
            "leaf_size": 30,
            "metric": "minkowski",
            "p": 2,
            "n_jobs": -1,
            "test_size": 0.2,
            "random_state": 0,
        }
        self.rf_params: Dict[str, Any] = {
            "n_estimators": 100,
            "criterion": "gini",
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "bootstrap": True,
            "n_jobs": -1,
            "test_size": 0.2,
            "random_state": 0,
        }
        self.xgb_params_clf: Dict[str, Any] = {
            "n_estimators": 200,
            "learning_rate": 0.1,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "gamma": 0.0,
            "reg_lambda": 1.0,
            "reg_alpha": 0.0,
            "min_child_weight": 1.0,
            "scale_pos_weight": 1.0,
            "objective": "binary:logistic",
            "n_jobs": -1,
            "test_size": 0.2,
            "random_state": 0,
        }
        self.xgb_params_reg: Dict[str, Any] = {
            "n_estimators": 200,
            "learning_rate": 0.1,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "gamma": 0.0,
            "reg_lambda": 1.0,
            "reg_alpha": 0.0,
            "min_child_weight": 1.0,
            "objective": "reg:squarederror",
            "n_jobs": -1,
            "test_size": 0.2,
            "random_state": 0,
        }

        top = QHBoxLayout()
        top.addWidget(QLabel("算法："))
        self.alg_combo = QComboBox()
        self.alg_combo.currentIndexChanged.connect(self._on_alg_changed)
        top.addWidget(self.alg_combo)

        top.addWidget(QLabel("规范化："))
        self.scaler_combo = QComboBox()
        for name, spec in ML.available_scalers():
            self.scaler_combo.addItem(name, spec)
        self.scaler_combo.setCurrentIndex(0)
        top.addWidget(self.scaler_combo)
        top.addStretch()

        self.k_label = QLabel("邻居数：")
        self.k_spin  = QSpinBox()
        self.k_spin.setRange(1,10000)
        self.k_spin.setValue(5)
        top.addWidget(self.k_label)
        top.addWidget(self.k_spin)

        self.btn_adv = QPushButton("高级参数")
        self.btn_adv.clicked.connect(self.open_adv_dialog)
        top.addWidget(self.btn_adv)
        top.addStretch()
        self.train_btn = QPushButton("训练")
        self.train_btn.clicked.connect(self.train_model)
        top.addWidget(self.train_btn)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(SPLITTER_HANDLE_WIDTH)
        splitter.setStyleSheet(SPLITTER_HANDLE_STYLE)

        column_panel = QWidget()
        column_layout = QVBoxLayout(column_panel)
        column_layout.setContentsMargins(0, 0, 0, 0)
        column_layout.setSpacing(6)
        column_label = QLabel("特征选择")
        column_layout.addWidget(column_label)
        self.column_list = QListWidget()
        column_layout.addWidget(self.column_list, 1)
        splitter.addWidget(column_panel)
        splitter.setStretchFactor(0, 1)

        plot_panel = QWidget()
        plot_panel_layout = QVBoxLayout(plot_panel)
        plot_panel_layout.setContentsMargins(0, 0, 0, 0)
        plot_panel_layout.setSpacing(8)

        viz_bar = QHBoxLayout()
        viz_bar.setContentsMargins(0, 0, 0, 0)
        viz_bar.addWidget(QLabel("图形:"))
        self.viz_combo = QComboBox()
        self.viz_combo.currentIndexChanged.connect(lambda: self._plot())
        viz_bar.addWidget(self.viz_combo)

        self.btn_plot_enlarge = QPushButton("放大查看")
        self.btn_plot_enlarge.setFixedHeight(28)
        self.btn_plot_enlarge.clicked.connect(self._open_plot_preview)
        viz_bar.addWidget(self.btn_plot_enlarge)
        viz_bar.addStretch()
        plot_panel_layout.addLayout(viz_bar)

        self.canvas = PlotCanvas()
        self.canvas.threshold_changed.connect(self._on_tau_change)
        self.canvas.mpl_connect("button_press_event", self._on_plot_click)

        self.metrics_label = QLabel("")
        self.metrics_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.metrics_label.setVisible(False)

        plot_container = QWidget()
        plot_layout = QVBoxLayout(plot_container)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.setSpacing(6)
        self.canvas_holder = SquareCanvasHolder(self.canvas)
        plot_layout.addWidget(self.canvas_holder, 1)
        if self.canvas.lbl_tau:
            plot_layout.addWidget(self.canvas.lbl_tau, alignment=Qt.AlignmentFlag.AlignCenter)
        if self.canvas.slider:
            self.slider_holder = SliderHolder(self.canvas.slider, parent=plot_container)
            plot_layout.addWidget(self.slider_holder)
        # 默认隐藏阈值控件
        self._set_tau_slider_visible(False)
        if self.canvas.lbl_tau:
            self.canvas.lbl_tau.setVisible(False)
        plot_layout.addWidget(self.metrics_label)
        plot_panel_layout.addWidget(plot_container, 1)

        table_panel = QWidget()
        table_layout = QVBoxLayout(table_panel)
        table_layout.setContentsMargins(0, 0, 0, 0)
        table_layout.setSpacing(6)
        table_layout.addWidget(QLabel("测试样本详情"))
        self.tbl_test = SmartTable(SmartTableConfig(show_toolbar=False, editable=False, min_rows=0))
        self.tbl_test.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table_layout.addWidget(self.tbl_test)
        self.btn_export = QPushButton("导出结果")
        self.btn_export.clicked.connect(self._export_results)
        table_layout.addWidget(self.btn_export)

        splitter.addWidget(plot_panel)
        splitter.addWidget(table_panel)
        splitter.setStretchFactor(1, 2)
        splitter.setStretchFactor(2, 1)
        main = QVBoxLayout(self)
        main.addLayout(top)
        main.addWidget(splitter)

    def set_data(self, df: pd.DataFrame, checked: List[str], target: str):
        self.df = df
        self.target_col = target
        y = df[target].values
        self.is_clf = type_of_target(y) not in ("continuous","continuous-multioutput")
        self.alg_combo.clear()
        if self.is_clf:
            self.alg_combo.addItem("KNN 分类","knn_clf")
            self.alg_combo.addItem("随机森林分类","rf_clf")
            self.alg_combo.addItem("XGBoost 分类","xgb_clf")
            self.rf_params["criterion"] = "gini"
            idx = self.alg_combo.findData("rf_clf")
            if idx >= 0:
                self.alg_combo.setCurrentIndex(idx)
        else:
            self.alg_combo.addItem("KNN 回归","knn_reg")
            self.alg_combo.addItem("随机森林回归","rf_reg")
            self.alg_combo.addItem("XGBoost 回归","xgb_reg")
            self.rf_params["criterion"] = "squared_error"
            idx = self.alg_combo.findData("rf_reg")
            if idx >= 0:
                self.alg_combo.setCurrentIndex(idx)
        self.column_list.clear()
        for col in df.columns:
            it = QListWidgetItem(col)
            it.setCheckState(Qt.CheckState.Checked if col in checked else Qt.CheckState.Unchecked)
            self.column_list.addItem(it)
        self._reset_viz_mode()
        # [LOG]
        self._log_data_overview(self.selected_columns())

    def selected_columns(self) -> List[str]:
        return [self.column_list.item(i).text() for i in range(self.column_list.count()) if self.column_list.item(i).checkState()==Qt.CheckState.Checked]

    def _params_for_code(self, code: Optional[str]) -> Dict[str, Any]:
        if not code:
            return {}
        if "knn" in code:
            return self.knn_params
        if "rf" in code:
            return self.rf_params
        if "xgb" in code:
            return self.xgb_params_clf if "clf" in code else self.xgb_params_reg
        return {}

    def open_adv_dialog(self):
        # [LOG]
        logger.info("打开高级参数：算法:%s", self.alg_combo.currentText())
        code = self.alg_combo.currentData()
        params = self._params_for_code(code)
        if not params:
            QMessageBox.warning(self, "提示", "请先选择算法")
            return
        dlg = AdvParamsDlg(self, code, params, self.is_clf)
        dlg.exec()

    def train_model(self):
        if self.df is None:
            QMessageBox.warning(self,"提示","请先加载数据")
            return
        cols = self.selected_columns()
        if not cols:
            QMessageBox.warning(self,"提示","请选择特征")
            return

        self._test_feature_names = list(cols)
        X = self.df[cols].values
        all_indices = np.arange(len(X))
        y = self.df[self.target_col].values
        code = self.alg_combo.currentData()
        if not code:
            QMessageBox.warning(self, "提示", "请选择算法")
            return
        scaler_spec = self.scaler_combo.currentData()

        params_base = self._params_for_code(code)
        if not params_base:
            QMessageBox.warning(self, "提示", "无法读取算法参数")
            return
        test_size = float(params_base.get("test_size", 0.2))
        rs = int(params_base.get("random_state", 0))
        stratify = y if (self.is_clf and len(np.unique(y)) > 1) else None
        if stratify is not None:
            try:
                _, cnts = np.unique(stratify, return_counts=True)
                if cnts.min() < 2:
                    stratify = None
            except Exception:
                stratify = None

        n_main = self.k_spin.value()
        params = params_base.copy()
        params.pop("test_size", None)
        params["random_state"] = rs
        params["target_name"] = self.target_col
        if "knn" in code:
            params_base["n_neighbors"] = n_main
            params["n_neighbors"] = n_main
            params.pop("random_state", None)
        else:
            params_base["n_estimators"] = n_main
            params["n_estimators"] = n_main
        # [LOG] 训练前信息与切分策略
        self._log_split(X, y, test_size, rs, stratify)
        # 训练（后端会记录 task/n_classes/is_binary/tau 等 meta）
        rep = ML.train(
            alg=code,
            X=X,
            y=y,
            params=params,
            scaler=scaler_spec,
            test_size=test_size,
            random_state=rs,
            stratify=stratify,
            feature_names=cols,
        )

        X_tr_raw, X_te_raw, y_tr, y_te, _idx_tr, idx_te = train_test_split(
            X, y, all_indices, test_size=test_size, random_state=rs, stratify=stratify
        )
        self.X_train_raw = X_tr_raw
        self.y_train = y_tr
        self._train_indices = _idx_tr
        self.X_test_raw = X_te_raw
        self._test_indices = idx_te
        self.X_test = ML.transform(X_te_raw)  # 与训练一致的规范化
        self.y_true = y_te
        pre =  ML.predict(X_te_raw)
        self.y_pred, self.test_scores = pre["labels"], pre["scores"]
        try:
            pre_train = ML.predict(X_tr_raw)
            self.y_pred_train = pre_train.get("labels")
            self.train_scores = pre_train.get("scores")
        except Exception:
            self.y_pred_train = None
            self.train_scores = None
        # [LOG] 预测完成
        self._log_predict_done()

        # 读取后端 meta，决定任务类型
        self.meta = ML.get_meta()
        # [LOG] 训练完成后的 meta 概览
        self._log_meta_after_train()
        # 如果后端给出了任务/类别数，按它；否则回退到本地判断
        task = self.meta.get("task")
        self.is_clf = (task == "supervised_clf") if task is not None else self.is_clf
        self.n_classes = int(self.meta.get("n_classes", 0) or len(np.unique(y)))
        self.binary = (self.is_clf and self.n_classes == 2)

          # 只有二分类才启用阈值
        tau_meta = self.meta.get("tau")

        if self.binary and tau_meta is not None and self.canvas.slider:
            tau = float(tau_meta)
            self.canvas.slider.setValue(int(tau * 1000))
            self._update_pred_by_threshold()
            logger.info("启用阈值：二分类，τ::%.3f", float(tau))  # [LOG]
        else:
            #多分类/回归：隐藏阈值相关控件
            self._set_tau_slider_visible(False)
            if self.canvas.lbl_tau:
                self.canvas.lbl_tau.setVisible(False)
            logger.info("关闭阈值：%s", "回归或多分类")           # [LOG]

        self._compute_metrics()
        self._reset_viz_mode()
        self._plot()
        self._update_test_table()

    def _update_pred_by_threshold(self):
        if not (self.is_clf and self.binary and self.test_scores is not None):
            return
        tau = self.meta.get("tau",0.5)
        classes = list(self.meta.get("classes_", []))
        if len(classes) >= 2:
            pos_label = 1 if 1 in classes else classes[1]
            neg_label = [c for c in classes if c!=pos_label][0]
        else: pos_label, neg_label = 1, 0
        self.y_pred = np.where(self.test_scores >= tau, pos_label, neg_label)

    def _on_tau_change(self, tau: float):
        # [LOG]
        #self._log_tau_changed(tau)
        if self.is_clf and self.binary:
            self.meta["tau"] = tau
            self._update_pred_by_threshold()
            self._compute_metrics()
            self._plot()

    def _compute_metrics(self):
        if self.y_true is None: return
        if self.is_clf:
            self.metrics_text = ""
            self.metrics_label.clear()
            self.metrics_label.setVisible(False)
            # [LOG] 分类指标
            self._log_metrics_cls(self.y_true, self.y_pred)
        else:
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            mae = mean_absolute_error(self.y_true, self.y_pred)
            mse = mean_squared_error(self.y_true, self.y_pred)
            r2 = r2_score(self.y_true, self.y_pred)
            self.metrics_text = f"MAE:{mae:.3f}   MSE:{mse:.3f}   R2:{r2:.3f}"
            self.metrics_label.setText(self.metrics_text)
            self.metrics_label.setVisible(True)
            # [LOG] 回归指标
            logger.info("测试集MAE误差:%.3f   测试集MSE误差:%.3f   测试集R2误差:%.3f", mae, mse, r2)
    def _reset_viz_mode(self):
        self._set_tau_slider_visible(self.is_clf and self.binary)
        if self.canvas.lbl_tau:
            self.canvas.lbl_tau.setVisible(self.is_clf and self.binary)
        self.viz_combo.blockSignals(True)
        self.viz_combo.clear()
        self._viz_options = []

        if self.is_clf:
            options = [
                ("confmat", "混淆矩阵"),
                ("scores", "评分柱状图"),
            ]
        else:
            options = [
                ("pred_vs_true", "预测 vs 真实"),
               # ("residual_hist", "残差直方图"),
                ("residual_scatter", "残差散点"),
                ("pca_scatter", "PCA 彩色散点"),
            ]
        for key, label in options:
            self.viz_combo.addItem(label, userData=key)
        self._viz_options = options
        self.viz_combo.blockSignals(False)

    def _set_tau_slider_visible(self, visible: bool) -> None:
        if self.slider_holder:
            self.slider_holder.setVisible(visible)
        if self.canvas.slider:
            self.canvas.slider.setVisible(visible)



    def _plot(self, target_canvas: PlotCanvas | None = None):
        if self.y_true is None:
            return
        canvas = target_canvas or self.canvas
        if canvas is None:
            return
        mode = self.viz_combo.currentData()
        if mode is None and 0 <= self.viz_combo.currentIndex() < len(self._viz_options):
            mode = self._viz_options[self.viz_combo.currentIndex()][0]
        self._scatter_points = None
        if self.is_clf:
            if mode == "confmat":
                canvas.plot_confmat(self.y_true, self.y_pred, self.meta.get("classes_", None))
            else:
                canvas.plot_scores(self.y_true, self.y_pred, self.meta.get("classes_", None))
        else:
            if mode == "pred_vs_true":
                canvas.clear()
                ax = canvas.ax
                ax.scatter(self.y_true, self.y_pred, s=12)
                y_true_min, y_true_max = self.y_true.min(), self.y_true.max()
                y_pred_min, y_pred_max = self.y_pred.min(), self.y_pred.max()
                lo = min(y_true_min, y_pred_min)
                hi = max(y_true_max, y_pred_max)
                pad = max((hi - lo) * 0.05, 1e-6)
                ax.plot([lo, hi], [lo, hi], "--", color="gray", lw=1)
                ax.set_xlabel("真实值")
                ax.set_ylabel("预测值")
                ax.set_title("预测值 vs 真实值")
                ax.set_xlim(lo - pad, hi + pad)
                ax.set_ylim(lo - pad, hi + pad)
                canvas._set_square_axes(ax)
                canvas.fig.tight_layout()
                canvas.draw()
                self._register_scatter_points("pred_vs_true", self.y_true, self.y_pred)
            elif mode == "residual_hist":
                canvas.clear()
                res = self.y_pred - self.y_true
                ax = canvas.ax
                counts, _, _ = ax.hist(
                    res,
                    bins=min(30, max(5, len(res))),
                    alpha=0.75,
                )
                ax.set_xlabel("残差")
                ax.set_ylabel("频数")
                ax.set_title("残差直方图")
                ymax = float(np.max(counts)) if counts.size else 0.0
                ax.set_ylim(0, max(1.0, ymax * 1.2))
                canvas.fig.tight_layout()
                canvas.draw()
            elif mode == "residual_scatter":
                canvas.clear()
                res = self.y_pred - self.y_true
                ax = canvas.ax
                ax.scatter(self.y_pred, res, s=12)
                ax.axhline(0, color="gray", ls="--")
                ax.set_xlabel("预测值")
                ax.set_ylabel("残差")
                ax.set_title("残差散点图")
                canvas._set_square_axes(ax)
                canvas.fig.tight_layout()
                canvas.draw()
                self._register_scatter_points("residual_scatter", self.y_pred, res)
            elif mode == "pca_scatter":
                canvas.clear()
                XY = PCA(2, random_state=0).fit_transform(self.X_test)
                err = np.abs(self.y_pred - self.y_true)
                ax = canvas.ax
                sc = ax.scatter(XY[:, 0], XY[:, 1], c=err, cmap="viridis", s=15)
                canvas.cbar = canvas.fig.colorbar(sc, ax=ax, label="|残差|")
                ax.set_xlabel("PC1")
                ax.set_ylabel("PC2")
                ax.set_title("PCA 彩色散点")
                canvas._set_square_axes(ax)
                canvas.fig.tight_layout()
                canvas._resize_colorbar(0.6)
                canvas.draw()
                self._register_scatter_points("pca_scatter", XY[:, 0], XY[:, 1])
                canvas.draw()

    def _register_scatter_points(self, mode: str, xs: np.ndarray, ys: np.ndarray) -> None:
        try:
            xs = np.asarray(xs, dtype=float).ravel()
            ys = np.asarray(ys, dtype=float).ravel()
        except Exception:
            self._scatter_points = None
            return
        if xs.size == 0 or ys.size == 0 or xs.size != ys.size:
            self._scatter_points = None
            return
        rows = np.arange(xs.size)
        coords = np.column_stack([xs, ys])
        self._scatter_points = {"mode": mode, "coords": coords, "rows": rows}

    def _update_test_table(self) -> None:
        if not hasattr(self, "tbl_test"):
            return
        df = self._build_result_dataframe(
            self.X_test_raw, self.y_true, self.y_pred, self.test_scores, self._test_indices
        )
        with self.tbl_test.no_record():
            self.tbl_test.set_dataframe(df, editable=False, record_state=False)

    def _export_results(self) -> None:
        if self.df is None:
            QMessageBox.warning(self, "提示", "请先加载并训练模型。")
            return
        if self.X_test_raw is None and self.X_train_raw is None:
            QMessageBox.warning(self, "提示", "暂无可导出的结果。")
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出结果", "", "Excel 文件 (*.xlsx)")
        if not path:
            return
        if not path.lower().endswith(".xlsx"):
            path += ".xlsx"
        test_df = self._build_result_dataframe(
            self.X_test_raw, self.y_true, self.y_pred, self.test_scores, self._test_indices
        )
        train_df = self._build_result_dataframe(
            self.X_train_raw, self.y_train, self.y_pred_train, self.train_scores, self._train_indices
        )
        try:
            with pd.ExcelWriter(path) as writer:
                test_df.to_excel(writer, sheet_name="测试集结果", index=False)
                train_df.to_excel(writer, sheet_name="训练集结果", index=False)
            QMessageBox.information(self, "成功", f"已导出到 {path}")
        except Exception as exc:
            QMessageBox.critical(self, "导出失败", f"写入文件失败：{exc}")

    def _focus_test_row(self, row_idx: int) -> None:
        if not hasattr(self, "tbl_test"):
            return
        table = self.tbl_test.table
        if row_idx < 0 or row_idx >= table.rowCount():
            return
        table.selectRow(row_idx)
        item = table.item(row_idx, 0)
        if item is not None:
            table.scrollToItem(item)

    def _sample_name_column(self) -> str | None:
        if self.df is not None:
            name = self.df.index.name
            if name and name in self.df.columns:
                return str(name)
        return None

    def _sample_name_header(self) -> str:
        col = self._sample_name_column()
        return col if col else "样本"

    def _sample_names_for_indices(self, indices: np.ndarray | list[int]) -> list[str]:
        if self.df is None or indices is None:
            return []
        arr = np.asarray(indices, dtype=int)
        try:
            names = self.df.index.take(arr)
        except Exception:
            names = arr
        return [str(name) for name in names]

    def _with_sample_name_column(self, df: pd.DataFrame, indices: np.ndarray | list[int]) -> pd.DataFrame:
        df = df.copy()
        header = self._sample_name_header()
        names = self._sample_names_for_indices(indices)
        if len(df) != len(names):
            return df
        if header in df.columns:
            df.loc[:, header] = names
            cols = [header] + [c for c in df.columns if c != header]
            df = df[cols]
        else:
            df.insert(0, header, names)
        return df

    def _build_result_dataframe(
        self,
        X_raw: np.ndarray | None,
        y_true: np.ndarray | None,
        y_pred: np.ndarray | None,
        scores: np.ndarray | None,
        indices: np.ndarray | list[int] | None,
    ) -> pd.DataFrame:
        if X_raw is None or y_true is None:
            return pd.DataFrame(columns=self._test_feature_names)
        df = pd.DataFrame(X_raw, columns=self._test_feature_names)
        actual_col = "真实值" if not self.is_clf else "真实标签"
        df[actual_col] = y_true
        if y_pred is not None:
            pred_col = "预测值" if not self.is_clf else "预测标签"
            df[pred_col] = y_pred
        if not self.is_clf and y_pred is not None:
            df["残差"] = y_pred - y_true
        elif self.is_clf and scores is not None:
            df["得分"] = scores
        if indices is not None and len(df):
            df = self._with_sample_name_column(df, indices[: len(df)])
        return df

    def _on_plot_click(self, event):
        if (
            self._scatter_points is None
            or event.inaxes is not self.canvas.ax
            or event.button != 1
            or not getattr(event, "dblclick", False)
            or event.xdata is None
            or event.ydata is None
        ):
            return
        coords = self._scatter_points.get("coords")
        if coords is None or not len(coords):
            return
        pos = np.array([event.xdata, event.ydata], dtype=float)
        dists = np.linalg.norm(coords - pos, axis=1)
        idx = int(np.argmin(dists))
        self._focus_test_row(idx)


    def _open_plot_preview(self):
        if self.y_true is None:
            QMessageBox.information(self, "暂无图形", "请先训练模型或加载预测结果。")
            return
        preview = MatplotlibPreviewDialog(self, canvas_factory=lambda parent: PlotCanvas(parent, show_controls=False))
        preview.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self._preview_dialogs.append(preview)
        self._plot(preview.canvas)
        preview.finished.connect(lambda _res, dlg=preview: self._cleanup_preview_dialog(dlg))
        preview.show()

    def _cleanup_preview_dialog(self, dlg: MatplotlibPreviewDialog):
        if dlg in self._preview_dialogs:
            self._preview_dialogs.remove(dlg)

    def _on_alg_changed(self, _i: int):
        code = self.alg_combo.currentData()
        params = self._params_for_code(code)
        if code and "knn" in code:
            self.k_label.setText("邻居数：")
            self.k_spin.setValue(int(params.get("n_neighbors", 5)))
        else:
            self.k_label.setText("树数量：")
            self.k_spin.setValue(int(params.get("n_estimators", 100)))
        self._reset_viz_mode()
        # [LOG]
        logger.info("已选择算法：%s（主参数标签：%s）", self.alg_combo.currentText(), self.k_label.text())

    # ============== logging helpers ==============
    def _log_data_overview(self, cols):
        total = self.df.shape[1] if self.df is not None else 0
        logger.info("数据集就绪：目标:%s，任务:%s，特征已选:%d/%d",
                    self.target_col, "分类" if self.is_clf else "回归",
                    len(cols), total)

    def _log_split(self, X, y, test_size, rs, stratify):
        logger.info("开始训练：算法:%s，规范化器:%s，样本数:%d，特征数:%d，测试集比例:%.2f，随机种子:%d，分层:%s",
                    self.alg_combo.currentText(), self.scaler_combo.currentText(),
                    X.shape[0], X.shape[1], test_size, rs,
                    "是" if stratify is not None else "否")

    def _log_meta_after_train(self):
        if not self.meta: return
        task = self.meta.get("task", "")
        model = self.meta.get("model_type", "")
        scaler = self.meta.get("scaler", "")
        n_classes = self.meta.get("n_classes", "")
        classes = self.meta.get("classes_", [])
        tau = self.meta.get("tau", None)
        pieces = [f"任务:{task}", f"模型:{model}", f"规范化器:{scaler}"]
        if int(n_classes)>1: pieces.append(f"类别数:{n_classes}")
        if classes is not None:   pieces.append(f"类名预览:{list(classes)[:5]}")
        if tau is not None: pieces.append(f"阈值τ:{tau:.3f}")
        logger.info("训练完成：%s", "，".join(pieces))

    def _log_predict_done(self):
        if self.y_true is None or self.y_pred is None: return
        logger.info("预测完成：测试集大小:%d", len(self.y_true))

    def _log_tau_changed(self, tau: float):
        logger.info("阈值 τ 更新为 %.3f", tau)


    def _log_metrics_cls(self, y_true, y_pred):
        try:
            from sklearn.metrics import accuracy_score, precision_recall_fscore_support
            acc = accuracy_score(y_true, y_pred)
            p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
            logger.info("测试集指标：准确率:%.3f | 精确率:%.3f | 召回率:%.3f | F1:%.3f", acc, p, r, f1)
        except Exception as e:
            logger.warning("分类指标计算失败：%s", e)
