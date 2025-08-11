
# -*- coding: utf-8 -*-
"""
合并版前端（最终）：
- 弹窗数据清洗（沿用你的外部 data_load_dialog.DataLoadDialog，不改接口）。
- 训练/保存交互采用 time_series_page2 的单模型流程（只针对单一模型，不再一次性训练全部）。
- 后端统一通过 backend.timeseries_interface.ModelManager：
    * register_dataset(df, time_col, time_format) -> manifest{dataset_id,...}
    * list_models() / get_model_meta(model_id) / load_dataset(dataset_id)
    * train(dataset_id, model_type, params) -> {model, metrics, extra?}
    * save_model(model_id|None, name, model_type, dataset_id, params, model_obj, metrics) -> model_id

注意：
- 本前端不再依赖 controller.run_all_models；controller 仅供你对照后端实现参考。
- 主界面左侧表格开启“按内容自适应 + 横向滚动”，便于查看宽表。
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFormLayout,
    QGroupBox, QGridLayout, QSizePolicy, QSpacerItem,
    QHeaderView, QComboBox, QMessageBox, QInputDialog
)

# —— 后端 ——
from backend.timeseries_interface import ModelManager
from gui.tools import logger,TrainWorker

from  gui.data_load_dialog import DataLoadDialog
from gui.smart_table import SmartTable, SmartTableConfig
from .model_manager_dialog import ModelManagerDialog
from .param_panel import ParamPanel
from ..feature_selector_widget import FeatureSelectorWidget

# ========================== 行配色常量 ========================== #
GREEN = QColor(Qt.GlobalColor.green).lighter(160)
BLUE = QColor(Qt.GlobalColor.blue).lighter(160)
PEND = QColor(Qt.GlobalColor.lightGray).lighter(170)

class TimeSeriesPage(QWidget):
    """主页面：左侧数据表 + 右侧模型控制（单模型工作流）。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.manager = ModelManager()

        # 原始数据与当前工作数据
        self._base_df: Optional[pd.DataFrame] = None
        self._df: Optional[pd.DataFrame] = None
        self._time_col: Optional[str] = None
        self._time_fmt: str = "%Y年%m月%d日%H%M"

        # 注册后的数据集 ID（训练/保存均依赖它）
        self._dataset_id: Optional[str] = None

        # 最近一次训练产物（保存时使用）
        self._trained_model: Optional[Any] = None
        self._trained_metrics: Optional[Dict[str, Any]] = None
        self._trained_params: Optional[Dict[str, Any]] = None
        self._trained_model_type: Optional[str] = None

        # 当前选中（或刚保存）的模型 ID（用于“加载已有模型”时）
        self._model_id: Optional[str] = None

        # 训练时已有的行数（用于识别新增观测）
        self._orig_rows: int = 0
        # 当前选择用于训练/预测的特征列（默认全选，不含时间列）
        self._feature_cols: list[str] = []
        self._trained_feature_cols_last: list[str] = []
        self._features_dirty: bool = False

        # ================= 左侧：数据表格 =================
        self.table = SmartTable(SmartTableConfig(show_toolbar=False, min_rows=1))
        hh = self.table.table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        hh.setStretchLastSection(False)
        self.table.table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.table.table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.table.dataframeChanged.connect(self._on_table_df_changed)
        self.table.row_filled.connect(self._on_row_filled)

        left = QVBoxLayout()
        left.addWidget(QLabel("数据预览"))
        left.addWidget(self.table, stretch=1)

        # ================= 右侧：控制面板 =================
        right = QVBoxLayout()

        # —— 顶部状态 ——
        self.dataset_label = QLabel("未加载数据集")
        self.dataset_label.setWordWrap(True)
        right.addWidget(self.dataset_label)
        top_btn_box = QWidget()  # 创建一个容器 Widget
        top_btn_layout = QHBoxLayout(top_btn_box)  # 使用水平布局
        top_btn_layout.setContentsMargins(0, 0, 0, 0)  # 移除默认边距

        def _prep_btn(btn: QPushButton, primary: bool = False):
            btn.setMinimumHeight(36)
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            if primary:
                btn.setStyleSheet("font-weight:600;")
            return btn

        btn_load = _prep_btn(QPushButton("加载数据"))
        btn_load.clicked.connect(self._open_load_dialog)
        top_btn_layout.addWidget(btn_load)

        # 模型管理按钮
        self.btn_model_mgr = _prep_btn(QPushButton("加载模型"))
        self.btn_model_mgr.clicked.connect(self._open_model_manager)
        top_btn_layout.addWidget(self.btn_model_mgr)

        right.addWidget(top_btn_box)

        form_box = QGroupBox("模型与参数")
        form = QFormLayout(form_box)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        # 模型类型
        self.model_type_combo = QComboBox()
        all_model =self.manager.get_all_model_name()
        self.model_type_combo.addItems(all_model)
        self.model_type_combo.setObjectName("model_type_combo")
        form.addRow("模型类型", self.model_type_combo)

        # 训练参数
        self.param_panel = ParamPanel(self.manager, self)
        form.addRow("训练参数", self.param_panel)

        right.addWidget(form_box)

        # —— 特征选择 ——
        self.fs_box = QGroupBox("特征选择")
        fs_v = QVBoxLayout(self.fs_box)
        self.feature_selector = FeatureSelectorWidget(self.fs_box)
        self.feature_selector.selectionChanged.connect(self._on_feature_selection_changed)
        fs_v.addWidget(self.feature_selector)
        right.addWidget(self.fs_box)


        # —— 特征选择刷新防抖 ——
        self._fs_pending_cols: list[str] | None = None
        self._fs_debounce = QTimer(self)
        self._fs_debounce.setSingleShot(True)
        self._fs_debounce.setInterval(200)  # 200ms 合并多次操作
        self._fs_debounce.timeout.connect(self._apply_feature_selection)

        # —— 操作区域：两列按钮网格 ——
        ops_box = QGroupBox("操作")
        ops = QGridLayout(ops_box)
        ops.setHorizontalSpacing(8)
        ops.setVerticalSpacing(8)

        self.btn_save = _prep_btn(QPushButton("保存模型"))
        self.btn_save.clicked.connect(self._on_save_model)

        self.btn_train = _prep_btn(QPushButton("训练模型"), primary=True)
        self.btn_train.clicked.connect(self._on_train)

        self.btn_predict = _prep_btn(QPushButton("预测下一步"), primary=True)
        self.btn_predict.setEnabled(False)
        self.btn_predict.clicked.connect(self._on_predict)

        # 终止训练按钮（仅在训练中显示）
        self.btn_stop_train = _prep_btn(QPushButton("终止训练"))
        self.btn_stop_train.setVisible(False)
        self.btn_stop_train.clicked.connect(self._on_stop_train)

        # 2×2 栅格排布
        ops.addWidget(self.btn_train, 0, 0)
        ops.addWidget(self.btn_save, 0, 1)
        ops.addWidget(self.btn_predict, 1, 0)
        ops.addWidget(self.btn_stop_train, 1, 1)

        right.addWidget(ops_box)

        # —— 状态栏（置底） ——
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        right.addWidget(self.status_label)

        # 填充弹性，推送状态栏到底部
        right.addItem(QSpacerItem(0, 0, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))

        # 总体布局
        root = QHBoxLayout(self)
        root.addLayout(left, stretch=3)
        root.addLayout(right, stretch=2)

    # ============================================================
    # 数据加载（外部弹窗 → 本地注册数据集）
    # ============================================================
    def _open_load_dialog(self):
        """弹出清洗弹窗；确认后注册数据集并在左侧显示。"""
        dlg = DataLoadDialog.from_file_dialog(
            self,
            default_time_fmt=self._time_fmt,
            require_time_column=True,
        )
        if dlg is None:
            return

        df = dlg.loaded_dataframe()
        if df is None:
            QMessageBox.warning(self, "提示", "未获取到清洗后的数据。")
            return

        self._time_col = dlg.time_column()
        self._time_fmt = dlg.time_format()

        # 保留原始数据副本并展示到左侧
        self._base_df = df.copy()
        self._df = self._base_df.copy()
        self._input_rows = None
        self._pend_row = None
        self._pred_row = None
        self._set_table_model(self._df)
        if self._time_col is not None:
            hh = self.table.table.horizontalHeader()
            for col in range(self.table.table.columnCount()):
                header_text = self.table.table.horizontalHeaderItem(col).text()
                if header_text == self._time_col:
                    hh.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
                    self.table.table.resizeColumnsToContents()
                    hh.setSectionResizeMode(col, QHeaderView.ResizeMode.Interactive)
                    break
        # 通过 ModelManager 注册数据集，拿到 dataset_id
        try:
            manifest = self.manager.register_dataset(self._base_df, self._time_col or "", self._time_fmt or "")
        except Exception as exc:
            QMessageBox.critical(self, "错误", f"注册数据集失败：{exc}")
            return
        self._dataset_id = manifest.get("dataset_id")
        self.dataset_label.setText(f"数据集：{self._dataset_id}（行 {manifest.get('n_rows')}, 列 {manifest.get('n_cols')}）")
        self.btn_predict.setEnabled(False)
        self._orig_rows = 0

        # 初始化“特征选择”：默认全选（去掉时间列）
        cols_for_selector = [c for c in self._base_df.columns.tolist() if c != (self._time_col or "")]
        self.feature_selector.set_columns(cols_for_selector)
        self._feature_cols = cols_for_selector
        # 由选择同步一次 _df（虽然现在等于全选）
        self._rebuild_work_df()

        # 提前把时间信息传给后端（若有对应方法）
        if hasattr(self.manager, "set_time_column") and self._time_col:
            try:
                self.manager.set_time_column(self._time_col)
            except Exception:
                pass
        if hasattr(self.manager, "set_time_format") and self._time_fmt:
            try:
                self.manager.set_time_format(self._time_fmt)
            except Exception:
                pass
    # ---------- 模型管理 ----------
    def _open_model_manager(self):
        dlg = ModelManagerDialog(self.manager, self)
        dlg.model_loaded.connect(self._load_model_by_id)
        dlg.exec()

    def _load_model_by_id(self, model_id: str):
        try:
            load_ret = self.manager.load_model(model_id)
            meta = load_ret["meta"]
        except Exception as exc:
            QMessageBox.warning(self, "提示", f"加载模型失败：{exc}")
            return
        self._time_col = meta.get("time_col")
        self._time_fmt = meta.get("time_format")
        dataset_id = meta.get("dataset_id")
        if dataset_id:
            try:
                df = self.manager.load_dataset(dataset_id)
                self._base_df = df.copy()
                self._df = self._base_df.copy()
                self._orig_rows = len(self._df)
                self._dataset_id = dataset_id
                self._input_rows = None
                self._pend_row = None
                self._pred_row = None
                self._set_table_model(self._df)
                self.dataset_label.setText(f"数据集：{dataset_id}（来自模型）")
                self.btn_predict.setEnabled(False)
                self._orig_rows = 0
                self.btn_predict.setEnabled(True)
                # 回填特征选择（若模型保存了 feature_cols）
                feat = meta.get("params", {}).get("feature_cols")
                if hasattr(self.manager, "set_time_column") and self._time_col:
                    try:
                        self.manager.set_time_column(self._time_col)
                    except Exception:
                        pass
                if hasattr(self.manager, "set_time_format") and self._time_fmt:
                    try:
                        self.manager.set_time_format(self._time_fmt)
                    except Exception:
                        pass
                cols_for_selector = [c for c in self._base_df.columns.tolist() if c != (self._time_col or "")]
                self.feature_selector.set_columns(cols_for_selector)
                if isinstance(feat, list) and feat:
                    self.feature_selector.set_selected([c for c in feat if c in cols_for_selector])
                self._feature_cols = self.feature_selector.selected()
                self._rebuild_work_df()
            except Exception as exc:
                QMessageBox.warning(self, "提示", f"加载模型关联数据集失败：{exc}")

        model_type = meta.get("model_type", "")
        if model_type and self.model_type_combo.findText(model_type) == -1:
            self.model_type_combo.addItem(model_type)
        if model_type:
            self.model_type_combo.setCurrentText(model_type)

        params = meta.get("params", {})
        self.param_panel.set_params(params)

        metrics = meta.get("metrics", {})
        for line in [f"{k}: {v}" for k, v in metrics.items()]:
            logger.info(line)
        self.status_label.setText(f"已加载模型 {model_id}")
        self._model_id = model_id

        self._look_back = int(meta.get("params", {}).get("look_back", self._look_back))
        # 记录该模型的训练列（优先用 meta["feature_names"]，退化用 params["feature_cols"]）
        trained_cols = meta.get("feature_names") or meta.get("params", {}).get("feature_cols") or []
        self._trained_feature_cols_last = list(trained_cols)
        self._features_dirty = False

        self._trained_model = None
        self._trained_metrics = None
        self._trained_params = None
        self._trained_model_type = None

        self._ensure_blank_row()
        self._init_input_window()
        self.btn_predict.setEnabled(True)
        self.table.table.scrollToBottom()

    # ============================================================
    # 训练
    # ============================================================
    def _on_train(self):
        if not self._dataset_id:
            QMessageBox.warning(self, "提示", "先加载数据集")
            return
        self.status_label.setText("训练中…")
        self.btn_train.setEnabled(False)
        self.btn_save.setEnabled(False)
        self.btn_predict.setEnabled(False)
        self.btn_stop_train.setVisible(True)
        self.btn_stop_train.setEnabled(True)

        params = {**self.param_panel.params(), "feature_cols": list(self._feature_cols)}
        w = TrainWorker(
            self.manager,
            self._dataset_id,
            self.model_type_combo.currentText().strip(),
            params,
            self
        )
        w.log_sig.connect(lambda m: logger.info(m))  # 日志滚动
        w.done_sig.connect(self._on_train_finished)  # 收尾
        w.start()
        self._worker = w  # 防 GC

    def _on_train_finished(self, payload: dict):
        self.btn_train.setEnabled(True)
        self.btn_save.setEnabled(True)
        self.btn_stop_train.setVisible(False)
        if not payload["ok"]:
            QMessageBox.critical(self, "错误", f"训练失败：{payload['err']}")
            self.status_label.setText("训练失败")
            self.btn_predict.setEnabled(self._trained_model is not None)
            return
        result = payload["res"]
        self._trained_model = result["model"]
        self._trained_metrics = result["metrics"]
        self._trained_params = self.param_panel.params()
        self._trained_model_type = self.model_type_combo.currentText().strip()
        self._look_back = int(result.get("extra", {}).get("look_back", self._look_back))
        for k, v in self._trained_metrics.items():
            logger.info(f"{k}: {v}")
        self.status_label.setText("训练完成（未保存）")
        # 训练使用的是选择后的 df，但 _base_df 保持完整
        self._rebuild_work_df()
        self._orig_rows = len(self._df) if self._df is not None else 0
        self._ensure_blank_row()
        self._init_input_window()
        self.table.table.scrollToBottom()
        self.btn_predict.setEnabled(True)
        self._trained_feature_cols_last = list(self._feature_cols)
        self._features_dirty = False

    def _on_stop_train(self):
        """终止当前训练任务。"""
        if getattr(self, "_worker", None):
            try:
                try:
                    self._worker.done_sig.disconnect(self._on_train_finished)
                except Exception:
                    pass
                self._worker.stop()
                self._worker.wait()
            except Exception:
                pass
            self._worker = None
        self.status_label.setText("训练已终止")
        self.btn_train.setEnabled(True)
        self.btn_stop_train.setVisible(False)
        self.btn_predict.setEnabled(self._trained_model is not None)
        self.btn_save.setEnabled(self._trained_model is not None)

    # ============================================================
    # 保存
    # ============================================================
    def _on_save_model(self):
        """保存当前训练结果为新模型，并刷新模型列表。"""
        if self._trained_model is None or self._trained_metrics is None:
            QMessageBox.warning(self, "提示", "请先训练模型，再保存。")
            return
        if not self._dataset_id:
            QMessageBox.warning(self, "提示", "缺少数据集 ID，无法保存模型。")
            return

        name, ok = QInputDialog.getText(self, "模型名称", "请输入模型名称：")
        if not ok or not name.strip():
            return

        try:
            model_id = self.manager.save_model(
                model_id=None,
                name=name.strip(),
                model_type=self._trained_model_type or self.model_type_combo.currentText(),
                dataset_id=self._dataset_id,
                # 把所选特征一并保存到 params，方便下次加载回填
                params={**(self._trained_params or {}), "feature_cols": list(self._feature_cols)},
                model_obj=self._trained_model,
                metrics=self._trained_metrics or {},
            )
        except Exception as exc:
            QMessageBox.critical(self, "错误", f"保存模型失败：{exc}")
            return

        self.status_label.setText(f"模型已保存：{model_id}")
        QMessageBox.information(self, "成功", f"模型已保存为 {model_id}")


    # ============================================================
    # 预测与表格辅助
    # ============================================================
    def _set_table_model(self, df: pd.DataFrame):
        self._df = df
        self.table.set_dataframe(df, record_state=False)
        self._apply_row_colors()

    # --------- 颜色相关状态 --------- #
    _look_back: int = 14           # 训练完成后会被真实参数覆盖
    _input_rows: list[int] | None = None
    _pend_row: int | None = None
    _pred_row: int | None = None

    # --------- 颜色刷新 --------- #
    def _apply_row_colors(self):
        self.table.clear_row_colors()
        if self._input_rows:
            for r in self._input_rows:
                self.table.set_row_color(r, GREEN)
        if self._pend_row is not None:
            self.table.set_row_color(self._pend_row, PEND)
        if self._pred_row is not None:
            self.table.set_row_color(self._pred_row, BLUE)

    def _on_table_df_changed(self, df: pd.DataFrame):
        """同步 SmartTable 的数据到内部 DataFrame 缓存。"""
        self._df = df

    # --------- 特征选择回调 --------- #
    def _on_feature_selection_changed(self, cols: list[str]):
        """当用户在特征选择器中调整时：防抖 + 去重"""
        cols = list(cols) if cols is not None else []
          # 去重：与当前一致就不处理
        if cols == self._feature_cols:
            return
          # 记录待应用的列集并启动防抖定时器
        self._fs_pending_cols = cols
        self._fs_debounce.start()

    def _apply_feature_selection(self):
        """由防抖计时器触发，真正应用列选择并刷新表格"""
        if self._fs_pending_cols is None:
            return
        self._feature_cols = self._fs_pending_cols
        self._fs_pending_cols = None
        self._rebuild_work_df()

        # 变更后与“训练列”对比；不同则禁止预测、提示需要重新训练
        if sorted(self._feature_cols) != sorted(self._trained_feature_cols_last):
            self._features_dirty = True
            self.btn_predict.setEnabled(False)
            self.status_label.setText("特征已改变，请重新训练后再预测。")
        else:
            self._features_dirty = False
            # 只有在已有模型/训练过的前提下才恢复按钮
            self.btn_predict.setEnabled(bool(self._trained_model_type or self._model_id))
    def _rebuild_work_df(self):
        """根据当前选择的特征列重建 _df 并刷新表格模型"""
        if self._base_df is None:
            self._df = None
            self.table.set_dataframe(pd.DataFrame(), record_state=False)
            self.table.clear_row_colors()
            return

        cols = []
        if self._time_col and self._time_col in self._base_df.columns:
            cols.append(self._time_col)
        if self._feature_cols:
            cols.extend([c for c in self._feature_cols if c in self._base_df.columns and c != self._time_col])
        else:
            cols.extend([c for c in self._base_df.columns if c != self._time_col])
        new_df = self._base_df[cols].copy()
        self._df = new_df
        self.table.set_dataframe(self._df, record_state=False)
        self._apply_row_colors()

    # --------- 训练完成后初始化窗口 --------- #
    def _init_input_window(self):
        win_end = len(self._df) - 2
        win_start = max(0, win_end - self._look_back + 1)
        self._input_rows = list(range(win_start, win_end + 1))
        self._pend_row = len(self._df) - 1
        self._pred_row = None
        self._apply_row_colors()

    # --------- 用户填满尾行 --------- #
    def _on_row_filled(self, row: int):
        if row != self._pend_row:
            return
        self._input_rows.append(row)
        while len(self._input_rows) > self._look_back:
            self._input_rows.pop(0)
        self._ensure_blank_row()
        self._pend_row = len(self._df) - 1
        self._apply_row_colors()

    # --------- 预测辅助 --------- #
    def _advance_window_before_predict(self):
        if self._pred_row is not None:
            self._input_rows.append(self._pred_row)
            while len(self._input_rows) > self._look_back:
                self._input_rows.pop(0)
            self._pred_row = None

    def _register_new_prediction(self):
        self._pred_row = len(self._df) - 1
        self._ensure_blank_row(force=True)
        self._pend_row = len(self._df) - 1
        self._apply_row_colors()
        self._orig_rows = len(self._df) - 1

    def _ensure_blank_row(self, force: bool = False):
        if self._df is None:
            return
        need_new = force or self._df.empty or self._df.iloc[-1].notna().all()
        if need_new:
            self._df = self._df.reset_index(drop=True)
            new_row = {
                col: (pd.NaT if col == self._time_col else pd.NA)
                for col in self._df.columns
            }
            self._df = pd.concat([self._df, pd.DataFrame([new_row])], ignore_index=True)
            self.table.set_dataframe(self._df, record_state=False)

    def _on_predict(self):
        if not self._dataset_id:
            QMessageBox.warning(self, "提示", "请先训练/加载模型后再预测。")
            return

        if self._features_dirty or (sorted(self._feature_cols) != sorted(self._trained_feature_cols_last)):
            QMessageBox.warning(self, "提示", "特征与训练时不一致，请先重新训练。")
            return

        # ① 先把之前的蓝色行并入窗口
        self._advance_window_before_predict()

        new_part = self._df.iloc[self._orig_rows:].copy()
        if self._time_col:
            subset = [c for c in new_part.columns if c != self._time_col]
            new_part = new_part.dropna(subset=subset, how="any")
        else:
            new_part = new_part.dropna(how="any")
        new_part = new_part.apply(pd.to_numeric)
        if not new_part.empty:
            try:
                self.manager.append_observations(new_part)
            except Exception as exc:
                QMessageBox.critical(self, "错误", f"追加观测失败：{exc}")
                return

        try:
            res = self.manager.predict(steps=1)
        except Exception as exc:
            QMessageBox.critical(self, "错误", f"预测失败：{exc}")
            return

        table = res[0]["table"]
        logger.info("\n" + table.to_string())

        # 将预测结果回填到表格尾部（覆盖空白行）
        model_col = self._trained_model_type or self.model_type_combo.currentText().strip()
        pred_series = table.get(model_col)
        if pred_series is None:
            pred_series = table.iloc[:, 0]
        row_vals = []
        for col in self._df.columns:
            if col == self._time_col:
                row_vals.append("")
            else:
                v = pred_series.get(col, pd.NA)
                row_vals.append(float(v) if pd.notna(v) else pd.NA)
        # 用预测结果覆盖最后一行
        self._df.iloc[-1] = row_vals
        self.table.set_dataframe(self._df, record_state=False)

        # 🔑 把 *刚生成的预测行* 也写回运行时 —— 下一次窗口才能用到它
        try:
            df_pred = pd.DataFrame([pred_series.values], columns=pred_series.index)
            self.manager.append_observations(df_pred)
        except Exception as exc:
            logger.warning(f"追加预测观测失败: {exc}")

        self._register_new_prediction()
        self.status_label.setText("✅ 预测已完成，结果见列表。")
        self.table.table.scrollToBottom()
