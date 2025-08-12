
"""
supervised_page_rewired.py

监督学习页（简化示例）：
- 前端不做规范化；把原始 X/y 传给后端
- 可选择规范化器；绘图使用 ML.transform(X_test) 取得与训练一致的规范化数据
"""

import numpy as np, pandas as pd, matplotlib.pyplot as plt
from typing import Any, Dict, List
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QListWidget, QListWidgetItem, QSplitter, QHBoxLayout,
    QMessageBox, QLabel, QComboBox, QSpinBox, QPushButton, QFileDialog, QLineEdit,
    QTableWidget, QTableWidgetItem, QSlider, QDialog, QFormLayout,
    QDialogButtonBox, QDoubleSpinBox, QCheckBox
)
from PyQt6.QtCore import Qt, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from sklearn.decomposition import PCA
from sklearn.utils.multiclass import type_of_target
from sklearn.model_selection import train_test_split

from backend.ml_interface import ML

# ============== 画布 ==============
class PlotCanvas(FigureCanvas):
    threshold_changed = pyqtSignal(float)
    def __init__(self, parent=None):
        self.fig, self.ax = plt.subplots()
        super().__init__(self.fig)
        self.setParent(parent)
        self.slider = QSlider(Qt.Orientation.Horizontal, parent)
        self.slider.setRange(0, 999)
        self.slider.setValue(500)
        self.slider.valueChanged.connect(self._emit_threshold)
        self.lbl_tau = QLabel("阈值 τ = 0.500")
    def _emit_threshold(self, v: int):
        tau = v / 1000.0
        self.lbl_tau.setText(f"阈值 τ = {tau:.3f}")
        self.threshold_changed.emit(tau)
    def clear(self):
        self.ax.clear()
        self.draw()
    def show_text(self, text: str):
        self.ax.clear()
        self.ax.text(0.5, 0.5, text, ha="center", va="center")
        self.ax.set_axis_off()
        self.fig.tight_layout()
        self.draw()
    def plot_pca(self, X: np.ndarray, labels: np.ndarray):
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
        self.fig.tight_layout()
        self.draw()
    def plot_confmat(self, y_true, y_pred, labels=None):
        from sklearn.metrics import confusion_matrix
        if labels is None:
            labels = np.unique(list(y_true) + list(y_pred))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        self.ax.clear()
        im = self.ax.imshow(cm, cmap="Blues")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                self.ax.text(j, i, cm[i, j], ha="center", va="center")
        self.ax.set_xticks(range(len(labels)))
        self.ax.set_yticks(range(len(labels)))
        self.ax.set_xticklabels(labels, rotation=45, ha="right")
        self.ax.set_yticklabels(labels)
        self.ax.set_xlabel("预测")
        self.ax.set_ylabel("真实")
        self.ax.set_title("混淆矩阵")
        self.fig.colorbar(im, ax=self.ax)
        self.fig.tight_layout()
        self.draw()

    def plot_scores(self, y_true, y_pred, labels=None):
        from sklearn.metrics import precision_recall_fscore_support
        if labels is None:
            labels = np.unique(list(y_true) + list(y_pred))
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
        x = np.arange(len(labels))
        w = 0.25
        self.ax.clear()
        self.ax.bar(x - w, p, w, label="精确率")
        self.ax.bar(x, r, w, label="召回率")
        self.ax.bar(x + w, f1, w, label="F1")
        self.ax.set_xticks(x)
        self.ax.set_xticklabels(labels, rotation=45, ha="right")
        self.ax.set_ylim(0, 1)
        self.ax.set_ylabel("得分")
        self.ax.set_title("各类别指标")
        self.ax.legend()
        self.fig.tight_layout()
        self.draw()


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
            p_layout.addWidget(QLabel("p:"))
            p_layout.addWidget(self.sp_p)
            form.addRow(self.p_row)

            self.cb_metric.currentTextChanged.connect(self._toggle_p_row)
            self._toggle_p_row(self.cb_metric.currentText())

            self.sp_jobs = QSpinBox()
            self.sp_jobs.setRange(-1, 64)
            self.sp_jobs.setValue(int(params.get("n_jobs", -1)))
            form.addRow("n_jobs", self.sp_jobs)

        else:
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

        self.sp_test_size = QDoubleSpinBox()
        self.sp_test_size.setRange(0.05,0.95)
        self.sp_test_size.setSingleStep(0.05)
        self.sp_test_size.setValue(float(params.get("test_size",0.2)))
        form.addRow("测试集比例", self.sp_test_size)
        self.sp_rs = QSpinBox()
        self.sp_rs.setRange(-1,999999)
        self.sp_rs.setValue(int(params.get("random_state",0)))
        form.addRow("随机种子", self.sp_rs)

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
        else:
            self.params["criterion"] = self.cb_criterion.currentText()
            self.params["max_depth"] = None if self.sp_depth.value()==0 else self.sp_depth.value()
            self.params["min_samples_split"] = self.sp_mss.value()
            self.params["min_samples_leaf"] = self.sp_msl.value()
            self.params["max_features"] = self._parse_max_features(self.le_mf.text().strip())
            self.params["bootstrap"] = self.ck_boot.isChecked()
            self.params["n_jobs"] = self.sp_jobs.value()
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
        self.y_true: np.ndarray | None = None
        self.y_pred: np.ndarray | None = None
        self.test_scores: np.ndarray | None = None
        self.meta: Dict[str, Any] = {}
        self.is_clf = True
        self.binary = False
        self.metrics_text = ""

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
        self.column_list = QListWidget()
        splitter.addWidget(self.column_list)

        right_panel = QWidget()
        right_v = QVBoxLayout(right_panel)
        self.viz_combo = QComboBox()
        self.viz_combo.setCurrentIndex(120)
        self.viz_combo.currentIndexChanged.connect(lambda: self._plot())
        right_v.addWidget(self.viz_combo, alignment=Qt.AlignmentFlag.AlignLeft)

        self.canvas = PlotCanvas()
        self.canvas.threshold_changed.connect(self._on_tau_change)
        right_v.addWidget(self.canvas)
        right_v.addWidget(self.canvas.lbl_tau)
        right_v.addWidget(self.canvas.slider)
        # 默认隐藏 τ 控件
        self.canvas.slider.setVisible(False)
        self.canvas.lbl_tau.setVisible(False)

        self.metrics_label = QLabel("")
        self.metrics_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        right_v.addWidget(self.metrics_label)

        splitter.addWidget(right_panel)
        splitter.setStretchFactor(1,2)
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
            self.rf_params["criterion"] = "gini"
        else:
            self.alg_combo.addItem("KNN 回归","knn_reg")
            self.alg_combo.addItem("随机森林回归","rf_reg")
            self.rf_params["criterion"] = "squared_error"
        self.column_list.clear()
        for col in df.columns:
            it = QListWidgetItem(col)
            it.setCheckState(Qt.CheckState.Checked if col in checked else Qt.CheckState.Unchecked)
            self.column_list.addItem(it)
        self._reset_viz_mode()

    def selected_columns(self) -> List[str]:
        return [self.column_list.item(i).text() for i in range(self.column_list.count()) if self.column_list.item(i).checkState()==Qt.CheckState.Checked]

    def open_adv_dialog(self):
        code = self.alg_combo.currentData()
        params = self.knn_params if "knn" in code else self.rf_params
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

        X = self.df[cols].values
        y = self.df[self.target_col].values
        code = self.alg_combo.currentData()
        scaler_spec = self.scaler_combo.currentData()

        params_base = self.knn_params if "knn" in code else self.rf_params
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
        if "knn" in code:
            self.knn_params["n_neighbors"] = n_main
        else:
            self.rf_params["n_estimators"] = n_main
        params = params_base.copy()
        params.pop("test_size", None)
        params["random_state"] = rs
        params["target_name"] = self.target_col
        if "knn" in code:
            params["n_neighbors"] = n_main
            params.pop("random_state", None)
        else:
            params["n_estimators"] = n_main

          # 训练（后端会记录 task/n_classes/is_binary/tau 等 meta）
        _ = ML.train(
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

        X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=rs, stratify=stratify)
        self.X_test = ML.transform(X_te_raw)  # 与训练一致的规范化
        self.y_true = y_te
        pre =  ML.predict(X_te_raw)
        self.y_pred, self.test_scores = pre["labels"], pre["scores"]

          # 读取后端 meta，决定任务类型
        self.meta = ML.get_meta()
          # 如果后端给出了任务/类别数，按它；否则回退到本地判断
        task = self.meta.get("task")
        self.is_clf = (task == "supervised_clf") if task is not None else self.is_clf
        self.n_classes = int(self.meta.get("n_classes", 0) or len(np.unique(y)))
        self.binary = (self.is_clf and self.n_classes == 2)

          # 只有二分类才启用阈值
        tau_meta = self.meta.get("tau")

        if self.binary and tau_meta is not None:
            tau = float(tau_meta)
            self.canvas.slider.setValue(int(tau * 1000))
            self._update_pred_by_threshold()
        else:
          # 多分类/回归：隐藏阈值相关控件
            self.canvas.slider.setVisible(False)
            self.canvas.lbl_tau.setVisible(False)

        self._compute_metrics()
        self._reset_viz_mode()
        self._plot()

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
        if self.is_clf and self.binary:
            self.meta["tau"] = tau
            self._update_pred_by_threshold()
            self._compute_metrics()
            self._plot()

    def _compute_metrics(self):
        if self.y_true is None: return
        if self.is_clf:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            import numpy as np
            auc = None
            classes = list(self.meta.get("classes_", []))
            mapping = {c: i for i, c in enumerate(classes)} if classes else None
            if self.binary and self.test_scores is not None and mapping is not None:
                try:
                    y_true_num = np.vectorize(mapping.get)(self.y_true)
                    auc = roc_auc_score(y_true_num, self.test_scores)
                except Exception:
                    auc = None
            acc = accuracy_score(self.y_true, self.y_pred)
            avg = "binary" if self.binary else "macro"
            if self.binary and classes:
                pos_label = classes[1] if len(classes) > 1 else classes[0]
                prec = precision_score(self.y_true, self.y_pred, average=avg, pos_label=pos_label, zero_division=0)
                rec = recall_score(self.y_true, self.y_pred, average=avg, pos_label=pos_label, zero_division=0)
                f1 = f1_score(self.y_true, self.y_pred, average=avg, pos_label=pos_label, zero_division=0)
            else:
                prec = precision_score(self.y_true, self.y_pred, average=avg, zero_division=0)
                rec = recall_score(self.y_true, self.y_pred, average=avg, zero_division=0)
                f1 = f1_score(self.y_true, self.y_pred, average=avg, zero_division=0)
            parts = [
                f"准确率={acc:.3f}",
                f"精确率={prec:.3f}",
                f"召回率={rec:.3f}",
                f"F1值={f1:.3f}",
            ]
            if auc is not None:
                parts.append(f"AUC={auc:.3f}")
            self.metrics_text = " | ".join(parts)
        else:
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            mae = mean_absolute_error(self.y_true, self.y_pred)
            mse = mean_squared_error(self.y_true, self.y_pred)
            r2 = r2_score(self.y_true, self.y_pred)
            self.metrics_text = f"MAE={mae:.3f} | MSE={mse:.3f} | R2={r2:.3f}"
        self.metrics_label.setText(self.metrics_text)

    def _reset_viz_mode(self):
        self.canvas.slider.setVisible(self.is_clf and self.binary)
        self.canvas.lbl_tau.setVisible(self.is_clf and self.binary)
        self.viz_combo.blockSignals(True)
        self.viz_combo.clear()

        if self.is_clf:
            self.viz_combo.addItems(["混淆矩阵", "得分条形图"])
        else:
            self.viz_combo.addItems(["预测 vs 真实", "残差直方图", "残差散点", "PCA 彩色散点"])
        self.viz_combo.blockSignals(False)

    def _plot(self):
        if self.y_true is None: return
        choice = self.viz_combo.currentText()
        if self.is_clf:
            if choice == "混淆矩阵":
                self.canvas.plot_confmat(self.y_true, self.y_pred, self.meta.get("classes_", None))
            else:
                self.canvas.plot_scores(self.y_true, self.y_pred, self.meta.get("classes_", None))
        else:
            if choice == "预测 vs 真实":
                ax = self.canvas.ax
                ax.clear()
                ax.scatter(self.y_true, self.y_pred, s=12)
                lo, hi = self.y_true.min(), self.y_true.max()
                ax.plot([lo, hi], [lo, hi], "--", color="gray", lw=1)
                ax.set_xlabel("真实值")
                ax.set_ylabel("预测值")
                self.canvas.fig.tight_layout()
                self.canvas.draw()
            elif choice == "残差直方图":
                res = self.y_pred - self.y_true
                ax = self.canvas.ax
                ax.clear()
                ax.hist(res, bins=30, alpha=0.75)
                ax.set_xlabel("残差")
                ax.set_ylabel("频数")
                self.canvas.fig.tight_layout()
                self.canvas.draw()
            elif choice == "残差散点":
                res = self.y_pred - self.y_true
                ax = self.canvas.ax
                ax.clear()
                ax.scatter(self.y_pred, res, s=12)
                ax.axhline(0, color="gray", ls="--")
                ax.set_xlabel("预测值")
                ax.set_ylabel("残差")
                self.canvas.fig.tight_layout()
                self.canvas.draw()
            elif choice == "PCA 彩色散点":
                XY = PCA(2, random_state=0).fit_transform(self.X_test)
                err = np.abs(self.y_pred - self.y_true)
                ax = self.canvas.ax
                ax.clear()
                sc = ax.scatter(XY[:, 0], XY[:, 1], c=err, cmap="viridis", s=15)
                self.canvas.fig.colorbar(sc, ax=ax, label="|残差|")
                ax.set_xlabel("PC1")
                ax.set_ylabel("PC2")
                self.canvas.fig.tight_layout()
                self.canvas.draw()

    def _on_alg_changed(self, _i: int):
        code = self.alg_combo.currentData()
        if code and "knn" in code:
            self.k_label.setText("邻居数：")
            self.k_spin.setValue(int(self.knn_params.get("n_neighbors", 5)))
        else:
            self.k_label.setText("树数：")
            self.k_spin.setValue(int(self.rf_params.get("n_estimators", 100)))
        self._reset_viz_mode()

