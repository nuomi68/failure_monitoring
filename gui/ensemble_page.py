from typing import Any, Dict, List, Optional, Callable
import traceback

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QComboBox, QPushButton, QLineEdit, QLabel, QTextEdit, QFileDialog,
    QSplitter, QTabWidget, QScrollArea, QFormLayout, QGroupBox, QMessageBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction
import pandas as pd

from backend.ml_interface import ML
from backend.model_registry import list_all as list_models

from gui.validation_page import ValidationPage


class EnsemblePage(QWidget):
    """
    模型集成页（不包含任何样式代码）
    左侧：模型列表 + 集成方式 + 操作
    右侧：Tab(单条输入 / 批量输入 / 结果 / 特征)
    """
    def __init__(self) -> None:
        super().__init__()
        self.is_ensemble = False  # 保留字段，但不再用于判断调用路径
        # ====== 顶层：左右分栏 ======
        root = QVBoxLayout(self)
        splitter = QSplitter(self)
        splitter.setOrientation(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        # ====== 左侧：模型区 ======
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # 搜索条与按钮行
        search_bar = QHBoxLayout()
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("搜索模型（按名字/类型/备注）")
        self.search_edit.textChanged.connect(self._apply_filter)
        search_bar.addWidget(self.search_edit)

        self.refresh_btn = QPushButton("刷新")
        self.refresh_btn.clicked.connect(self._populate)
        search_bar.addWidget(self.refresh_btn)
        left_layout.addLayout(search_bar)

        # 模型表
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["✔", "名字", "时间", "类型", "备注"])
        self.table.setSortingEnabled(True)
        self.table.verticalHeader().setVisible(False)
        # 第一列仅用于复选，宽度稍小可以交给样式表控制
        left_layout.addWidget(self.table)

        # 快捷选择行
        quick_sel = QHBoxLayout()
        self.btn_sel_all = QPushButton("全选")
        self.btn_sel_none = QPushButton("全不选")
        self.btn_sel_inv = QPushButton("反选")
        self.btn_sel_all.clicked.connect(lambda: self._set_all_checks(Qt.CheckState.Checked))
        self.btn_sel_none.clicked.connect(lambda: self._set_all_checks(Qt.CheckState.Unchecked))
        self.btn_sel_inv.clicked.connect(self._invert_checks)
        quick_sel.addWidget(self.btn_sel_all)
        quick_sel.addWidget(self.btn_sel_none)
        quick_sel.addWidget(self.btn_sel_inv)
        quick_sel.addStretch()
        left_layout.addLayout(quick_sel)

        # 控制行（集成方式 + 加载 + 验证）
        ctrl = QHBoxLayout()
        self.method_combo = QComboBox()
        self.method_combo.addItems(["平均", "投票"])
        ctrl.addWidget(QLabel("集成方式："))
        ctrl.addWidget(self.method_combo)

        self.load_btn = QPushButton("加载")
        self.load_btn.clicked.connect(self.on_load)
        ctrl.addWidget(self.load_btn)

        ctrl.addStretch()
        left_layout.addLayout(ctrl)

        # 状态行
        self.status_label = QLabel("未加载")
        left_layout.addWidget(self.status_label)

        splitter.addWidget(left_panel)

        # ====== 右侧：输入/结果/特征 ======
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        self.tabs = QTabWidget()

        # --- 单条输入 Tab ---
        self.single_tab = QWidget()
        st_layout = QVBoxLayout(self.single_tab)

        self.single_form_group = QGroupBox("输入特征")
        self.single_form_layout = QFormLayout(self.single_form_group)
        # 动态表单会在加载模型后构建
        st_layout.addWidget(self.single_form_group)

        single_btns = QHBoxLayout()
        self.btn_single_predict = QPushButton("预测")
        self.btn_single_predict.clicked.connect(self._predict_single)
        self.btn_single_clear = QPushButton("清空")
        self.btn_single_clear.clicked.connect(self._clear_single_inputs)
        single_btns.addWidget(self.btn_single_predict)
        single_btns.addWidget(self.btn_single_clear)
        single_btns.addStretch()
        st_layout.addLayout(single_btns)

        self.single_msg = QLabel("")
        st_layout.addWidget(self.single_msg)

        self.tabs.addTab(self.single_tab, "单条输入")

        # --- 批量输入 Tab ---
        self.batch_tab = QWidget()
        bt_layout = QVBoxLayout(self.batch_tab)

        helper = QLabel("可整体粘贴表格，但需手动对其相应的列")
        bt_layout.addWidget(helper)


        # 嵌入表格编辑器（沿用 ValidationPage，但外部模式）
        self.batch_editor = ValidationPage()
        self.batch_editor.save_btn.setEnabled(False)
        bt_layout.addWidget(self.batch_editor)


        self.tabs.addTab(self.batch_tab, "批量输入")

        # --- 结果 Tab ---
        self.result_tab = QWidget()
        rt_layout = QVBoxLayout(self.result_tab)
        self.result_table = QTableWidget()
        self.result_table.setColumnCount(2)
        self.result_table.setHorizontalHeaderLabels(["索引", "预测值"])
        self.result_table.verticalHeader().setVisible(False)
        self.result_table.setSortingEnabled(False)
        rt_layout.addWidget(self.result_table)

        # 导出按钮（可选）
        export_row = QHBoxLayout()
        self.btn_export_csv = QPushButton("导出结果为 CSV")
        self.btn_export_csv.clicked.connect(self._export_predictions)
        export_row.addWidget(self.btn_export_csv)
        export_row.addStretch()
        rt_layout.addLayout(export_row)

        self.tabs.addTab(self.result_tab, "结果")

        # --- 特征/信息 Tab ---
        self.info_tab = QWidget()
        it_layout = QVBoxLayout(self.info_tab)
        self.info_label = QLabel("尚未加载模型。")
        it_layout.addWidget(self.info_label)
        self.tabs.addTab(self.info_tab, "特征/信息")

        right_layout.addWidget(self.tabs)
        splitter.addWidget(right_panel)

        splitter.setStretchFactor(0, 3)  # 左右分配比例，交给样式表也可
        splitter.setStretchFactor(1, 5)

        # ====== 其他初始化 ======
        self.records: List[Dict[str, Any]] = []
        self._populate()

        # 动态数据
        self.features: List[str] = []  # features_union
        self.feature_inputs: Dict[str, QLineEdit] = {}  # 单条输入控件
        self.predictor: Optional[Callable[[pd.DataFrame], Any]] = None  # 弹性预测入口
        self.last_predictions: Optional[pd.Series] = None

        # 右键菜单（表格快速开/关）
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.ActionsContextMenu)
        act_all = QAction("全选", self.table)
        act_all.triggered.connect(lambda: self._set_all_checks(Qt.CheckState.Checked))
        self.table.addAction(act_all)
        act_none = QAction("全不选", self.table)
        act_none.triggered.connect(lambda: self._set_all_checks(Qt.CheckState.Unchecked))
        self.table.addAction(act_none)
        act_inv = QAction("反选", self.table)
        act_inv.triggered.connect(self._invert_checks)
        self.table.addAction(act_inv)

    # ========================= 左侧：模型列表 =========================
    def _populate(self) -> None:
        self.records = list_models() or []
        self.table.setRowCount(len(self.records))
        for r, rec in enumerate(self.records):
            # 勾选框
            ck = QTableWidgetItem()
            ck.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsSelectable)
            ck.setCheckState(Qt.CheckState.Unchecked)
            self.table.setItem(r, 0, ck)

            self.table.setItem(r, 1, QTableWidgetItem(rec.get("name", "")))
            self.table.setItem(r, 2, QTableWidgetItem(rec.get("created_at", "")))
            self.table.setItem(r, 3, QTableWidgetItem((rec.get("meta") or {}).get("model_type", "")))
            self.table.setItem(r, 4, QTableWidgetItem(str((rec.get("meta") or {}).get("advanced", {}))))

        self._apply_filter()
        self.status_label.setText(f"已加载模型清单：{len(self.records)} 条")

    def _apply_filter(self) -> None:
        key = (self.search_edit.text() or "").strip().lower()
        for r in range(self.table.rowCount()):
            if not key:
                self.table.setRowHidden(r, False)
                continue
            cells = []
            for c in range(1, 5):
                item = self.table.item(r, c)
                cells.append(item.text().lower() if item else "")
            self.table.setRowHidden(r, key not in " ".join(cells))

    def _set_all_checks(self, state: Qt.CheckState) -> None:
        for r in range(self.table.rowCount()):
            item = self.table.item(r, 0)
            if item is not None:
                item.setCheckState(state)

    def _invert_checks(self) -> None:
        for r in range(self.table.rowCount()):
            item = self.table.item(r, 0)
            if item is not None:
                item.setCheckState(
                    Qt.CheckState.Unchecked if item.checkState() == Qt.CheckState.Checked else Qt.CheckState.Checked
                )

    # ========================= 模型加载与验证页 =========================
    def on_load(self) -> None:
        try:
            paths = [
                self.records[r]["path"]
                for r in range(self.table.rowCount())
                if self.table.item(r, 0) and self.table.item(r, 0).checkState() == Qt.CheckState.Checked
            ]
            if not paths:
                QMessageBox.warning(self, "提示", "请先勾选至少一个模型。")
                return
            names = [
                self.records[r]["name"]
                for r in range(self.table.rowCount())
                if self.table.item(r, 0) and self.table.item(r, 0).checkState() == Qt.CheckState.Checked
            ]
            self.selected_models = [{"name": n, "path": p} for n, p in zip(names, paths)]

            method = "mean" if self.method_combo.currentText() == "平均" else "vote"
            info = ML.load_many(paths, method=method)

            # 先拿 features_union 再配置批量表格
            feats = info.get("features_union") if isinstance(info, dict) else None
            if not feats:
                feats = []
            self.features = list(feats)

            meta = ML.get_meta() or {}
            # 统一判定：multi_output 优先；否则 ensemble；否则 single
            self.backend_meta = meta
            self.is_ensemble = bool(meta.get("multi_output", False) or meta.get("ensemble", False))
            # ↑ 仅作展示用途，实际预测路径依据 backend_meta 决定

            # ✅ 现在再启用批量表格的外部模式（用正确的 features 初始化）
            self.batch_editor.enable_external(self.features, self._predict_targets_for_df)

            # 可选 predictor
            self.predictor = None
            if isinstance(info, dict):
                cand = info.get("predict")
                if callable(cand):
                    self.predictor = cand

            self._rebuild_single_form()
            self.info_label.setText(
                f"已加载 {len(paths)} 个模型，集成方式：{method}\n特征（{len(self.features)}）: {', '.join(self.features) if self.features else '(后端未提供)'}"
            )
            self.status_label.setText("已加载集成模型，准备预测。")
            self.tabs.setCurrentWidget(self.single_tab)

        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "错误", f"加载失败：\n{e}")

    # ========================= 右侧：单条输入 =========================
    def _rebuild_single_form(self) -> None:
        # 清空旧的
        for i in reversed(range(self.single_form_layout.count())):
            item = self.single_form_layout.itemAt(i)
            w = item.widget()
            if w:
                w.setParent(None)
        self.feature_inputs.clear()

        # 无特征时给个说明
        if not self.features:
            lbl = QLabel("后端未提供特征列表，仍可在批量输入页导入 CSV 预测。")
            self.single_form_layout.addRow(lbl)
            return

        # 构建输入行
        for f in self.features:
            edit = QLineEdit()
            edit.setPlaceholderText("数值/可解析为数值")
            self.feature_inputs[f] = edit
            self.single_form_layout.addRow(QLabel(f), edit)

    def _clear_single_inputs(self) -> None:
        for e in self.feature_inputs.values():
            e.clear()
        self.single_msg.setText("")

    def _collect_single_row(self) -> pd.DataFrame:
        # 将单条输入拼成 DataFrame（一个样本）
        data = {}
        for f, edit in self.feature_inputs.items():
            txt = edit.text().strip()
            if txt == "":
                data[f] = None
            else:
                try:
                    # 尝试数值化
                    data[f] = float(txt)
                except Exception:
                    data[f] = txt  # 留给后端/预处理
        if not self.features:
            # 若无特征定义，就把 dict 的键作为列
            cols = list(data.keys())
        else:
            cols = self.features
            # 补齐缺失
            for f in cols:
                data.setdefault(f, None)

        df = pd.DataFrame([data], columns=cols)
        return df


    # --------- 辅助：标准化后端返回为 {target: labels_ndarray} ---------
    def _normalize_backend_result(self, ret) -> Dict[str, Any]:
        # MultiOutput: {t: {"labels": y, "scores": s}}
        if isinstance(ret, dict) and all(isinstance(v, dict) and "labels" in v for v in ret.values()):
            return {t: v.get("labels") for t, v in ret.items()}
        # 单模型/旧式集合：{"target": name, "labels": y, "scores": s}
        if isinstance(ret, dict) and "labels" in ret:
            return {str(ret.get("target", "输出")): ret.get("labels")}
        # 兼容极少数旧返回：(y, scores) 或 直接 y
        if isinstance(ret, tuple) and len(ret) >= 1:
            return {"输出": ret[0]}
        return {"输出": ret}

    # ========================= 结果与导出 =========================
    def _fill_result_table(self, y) -> None:
        """
        支持 y 是 ndarray/list/Series/DataFrame(单列)；
        若后端返回 (y_pred, scores) 的元组，则会解包并在可行时增加“分数”列。
        """
        import numpy as np
        import pandas as pd

        scores = None
        # 兼容 (y_pred, scores)
        if isinstance(y, tuple) and len(y) >= 1:
            scores = y[1] if len(y) > 1 else None
            y = y[0]

        # 统一成 Series
        if isinstance(y, pd.DataFrame):
            if y.shape[1] == 1:
                s = y.iloc[:, 0]
            else:
                s = y.apply(lambda row: ",".join(map(str, row.tolist())), axis=1)
        elif isinstance(y, pd.Series):
            s = y
        elif isinstance(y, (list, tuple)):
            s = pd.Series(list(y))
        elif "numpy" in str(type(y)):  # 简单识别 np.ndarray
            s = pd.Series(np.asarray(y).ravel())
        else:
            s = pd.Series([y])

        self.last_predictions = s

        # 是否可以展示 scores
        add_scores = False
        sc_series = None
        if scores is not None:
            try:
                sc = np.asarray(scores).ravel()
                if sc.shape[0] == len(s):
                    sc_series = pd.Series(sc)
                    add_scores = True
            except Exception:
                add_scores = False

        cols = 3 if add_scores else 2
        self.result_table.setRowCount(len(s))
        self.result_table.setColumnCount(cols)
        headers = ["索引", "预测值"] + (["分数"] if add_scores else [])
        self.result_table.setHorizontalHeaderLabels(headers)
        self.result_table.verticalHeader().setVisible(False)

        for i, val in enumerate(s):
            self.result_table.setItem(i, 0, QTableWidgetItem(str(i)))
            self.result_table.setItem(i, 1, QTableWidgetItem("" if val is None else str(val)))
            if add_scores and sc_series is not None:
                v = sc_series.iloc[i]
                self.result_table.setItem(i, 2, QTableWidgetItem("" if (pd.isna(v) if hasattr(pd, 'isna') else v is None) else str(v)))
    def _export_predictions(self) -> None:
        if self.last_predictions is None:
            QMessageBox.information(self, "提示", "当前没有可导出的预测结果。")
            return
        path, _ = QFileDialog.getSaveFileName(self, "保存为 CSV", "predictions.csv", "CSV 文件 (*.csv)")
        if not path:
            return
        try:
            self.last_predictions.to_csv(path, header=["pred"], index_label="index")
            QMessageBox.information(self, "成功", f"已导出到：{path}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"导出失败：\n{e}")

    def _resolve_target_name(self, member_meta: Dict[str, Any], fallback: str) -> str:
        adv = dict(member_meta.get("advanced") or {})
        for key in ["target", "y_name", "output", "label", "task_name", "target_name", "output_name"]:
            if key in member_meta and member_meta[key]:
                return str(member_meta[key])
            if key in adv and adv[key]:
                return str(adv[key])
        return fallback or str(member_meta.get("model_type", "输出"))

    def _predict_single(self) -> None:
        try:
            df = self._collect_single_row()
            res_map = self._predict_targets_for_df(df)  # {目标: 数组}
            parts = []
            for k, v in res_map.items():
                try:
                    val = v[0] if hasattr(v, "__len__") else v
                except Exception:
                    val = v
                parts.append(f"{k}: {val}")
            self.single_msg.setText(" | ".join(parts))
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "错误", f"单条预测失败：\n{e}")

    def _predict_targets_for_df(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        批量表格编辑器的回调：
        - 直接调用后端 ML.predict（后端已完成“按目标分组与同目标集成”）
        - 统一解析返回为 {目标名: 结果数组}，供 ValidationPage 动态生成列
        """
        import numpy as np
        if df.shape[1] == 0:
            return {"输出": np.array([])}

        # 对齐到 features_union
        if getattr(self, "features", None):
            for f in self.features:
                if f not in df.columns:
                    df[f] = np.nan
            df = df[self.features]
        # 只保留“整行非 NaN”的样本做预测
        mask = df.notna().all(axis=1).to_numpy()
        valid_idx = np.flatnonzero(mask)
        if valid_idx.size == 0:
            return {}  # 本次没有完整行，直接返回空
        df_valid = df.iloc[valid_idx].reset_index(drop=True)

        # 按后端 meta 决定输入形态
        meta = getattr(self, "backend_meta", {}) or {}
        is_multi = bool(meta.get("multi_output", False))
        is_ens = bool(meta.get("ensemble", False))

        # 仅把完整行送去预测
        try:
            if is_multi or is_ens:
                X_table = {c: df_valid[c].to_numpy() for c in df_valid.columns}
                ret = ML.predict(X_table)
            else:
                ret = ML.predict(df_valid.to_numpy())
        except Exception:
            try:
                ret = ML.predict({c: df_valid[c].to_numpy() for c in df_valid.columns})
            except Exception:
                ret = ML.predict(df_valid.to_numpy())

        # 归一化子结果，并回填到原长度（未参与预测的行为 NaN）
        sub = self._normalize_backend_result(ret)  # {target: 1d}
        out: Dict[str, Any] = {}
        n = len(df)
        for t, arr in sub.items():
            arr = np.asarray(arr).ravel()
            full = np.full((n,), np.nan)
            full[valid_idx[: len(arr)]] = arr[: len(valid_idx)]
            out[t] = full
        return out
