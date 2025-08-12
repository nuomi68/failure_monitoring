from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional, Dict, List

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QPushButton,
    QMessageBox,
)

from backend.ml_interface import ML, infer_input_features
from backend.ml_model_manager import MLModelManager
from backend.fault_level_estimator import FaultLevelEstimator
from backend.fault_model_manager import FaultModelManager
from backend.timeseries_interface import ModelManager

from gui.smart_table import SmartTable, SmartTableConfig
from gui.nuclide_prediction.model_manager_dialog import ModelManagerDialog


class EnsemblePage(QWidget):
    """模型流水线页面：

    - 管理时间序列、通用 ML 与故障等级模型的加载与推理
    - 仅做模型加载与推理，不涉及训练
    - 时间序列预测结果会自动追加到底部共用表
    - 下游表可再喂给 ML 或故障等级模块进行最终推理
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("模型流水线")

        # 后端管理器 / 模型状态
        self.ts_manager = ModelManager()
        self.ts_model_ids: List[str] = []
        self.ts_names: Dict[str, str] = {}
        self.ts_features: Dict[str, List[str]] = {}
        self.ts_lookbacks: Dict[str, int] = {}

        self.fault_manager = FaultModelManager()
        self.fault_models: Dict[str, FaultLevelEstimator] = {}
        self.fault_names: Dict[str, str] = {}
        self.fault_features: Dict[str, List[str]] = {}

        self.ml_manager = MLModelManager()
        self.ml_model_ids: List[str] = []
        self.ml_names: Dict[str, str] = {}
        self.ml_features: Dict[str, List[str]] = {}
        self.ml_targets: Dict[str, str] = {}

        self.pred_cols: set[str] = set()

        # ----- 四宫格布局 -----
        self.grp_ts = self._build_ts_block()
        self.grp_ml = self._build_ml_block()
        self.grp_fault = self._build_fault_block()

        # 时间序列输入表
        self.ts_input_wrap = QWidget()
        right_top = QVBoxLayout(self.ts_input_wrap)
        right_top.addWidget(QLabel("时间序列输入"))
        self.lbl_ts_hint = QLabel()
        right_top.addWidget(self.lbl_ts_hint)
        self.tbl_ts = SmartTable(SmartTableConfig(default_headers=["feat1", "feat2"]))
        right_top.addWidget(self.tbl_ts)
        right_top.addWidget(self._make_ts_actions())

        # 下游共用表（常驻）
        self.common_wrap = QWidget()
        right_bottom = QVBoxLayout(self.common_wrap)
        right_bottom.addWidget(QLabel("下游共用输入"))
        self.tbl_common = SmartTable(SmartTableConfig())
        right_bottom.addWidget(self.tbl_common)
        right_bottom.addWidget(self._make_final_actions())

        # 左下角：ML 与故障块
        self.bottom_left = QWidget()
        left_bottom = QVBoxLayout(self.bottom_left)
        left_bottom.addWidget(self.grp_ml)
        left_bottom.addWidget(self.grp_fault)
        left_bottom.addStretch(1)

        # 根布局 2x2
        root = QGridLayout(self)
        root.addWidget(self.grp_ts, 0, 0)
        root.addWidget(self.ts_input_wrap, 0, 1)
        root.addWidget(self.bottom_left, 1, 0)
        root.addWidget(self.common_wrap, 1, 1)
        root.setColumnStretch(1, 1)
        root.setRowStretch(1, 1)

    # ------------------------------------------------------------------
    # 工具函数
    # ------------------------------------------------------------------
    def _rebuild_model_chips(self, layout, ids: List[str], names: Dict[str, str], remove_cb) -> None:
        layout.setSpacing(4)
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        for mid in ids:
            w = QWidget()
            l = QHBoxLayout(w)
            l.setContentsMargins(2, 0, 2, 0)
            l.setSpacing(2)
            lbl = QLabel(names.get(mid, mid))
            lbl.setStyleSheet("padding:0 2px")
            l.addWidget(lbl)
            btn = QPushButton("x")
            btn.setFixedSize(14, 14)
            btn.setStyleSheet("border:none;padding:0")
            btn.clicked.connect(lambda _, m=mid: remove_cb(m))
            l.addWidget(btn)
            layout.addWidget(w)

    def _set_table_headers(self, tbl: SmartTable, headers: List[str]) -> None:
        df = tbl.dataframe()
        df_new = pd.DataFrame(columns=headers)
        for c in df.columns:
            if c in df_new.columns:
                df_new[c] = df[c]
        with tbl.no_record():
            tbl.set_dataframe(df_new, record_state=False)

    def _required_ts_features(self) -> List[str]:
        feats: set[str] = set()
        for fs in self.ts_features.values():
            feats.update(fs)
        return sorted(feats)

    def _required_common_features(self) -> List[str]:
        feats: set[str] = set()
        for fs in self.ml_features.values():
            feats.update(fs)
        for fs in self.fault_features.values():
            feats.update(fs)
        return sorted(feats)

    def _refresh_ts_models(self) -> None:
        self._rebuild_model_chips(self.ts_model_layout, self.ts_model_ids, self.ts_names, self._remove_ts_model)
        headers = self._required_ts_features()
        if headers:
            self._set_table_headers(self.tbl_ts, headers)
        else:
            self._set_table_headers(self.tbl_ts, [])
        if self.ts_lookbacks:
            lb = max(self.ts_lookbacks.values())
            self.lbl_ts_hint.setText(f"最少输入行数：{lb}")
        else:
            self.lbl_ts_hint.clear()
        self._update_common_headers()

    def _refresh_ml_models(self) -> None:
        self._rebuild_model_chips(self.ml_model_layout, self.ml_model_ids, self.ml_names, self._remove_ml_model)
        self._update_common_headers()

    def _refresh_fault_models(self) -> None:
        ids = list(self.fault_models.keys())
        self._rebuild_model_chips(self.fault_model_layout, ids, self.fault_names, self._remove_fault_model)
        self._update_common_headers()

    def _update_common_headers(self) -> None:
        headers = self._required_common_features()
        self._set_table_headers(self.tbl_common, headers)

    # ------------------------------------------------------------------
    # 左侧块构建
    # ------------------------------------------------------------------
    def _build_ts_block(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        title = QLabel("时间序列模型")
        title.setStyleSheet("font-weight:bold")
        lay.addWidget(title)

        self.ts_controls = QWidget()
        row = QHBoxLayout(self.ts_controls)
        row.setContentsMargins(0, 0, 0, 0)
        self.btn_ts_load = QPushButton("加载模型…")
        self.btn_ts_clear = QPushButton("清空")
        row.addWidget(self.btn_ts_load)
        row.addWidget(self.btn_ts_clear)
        lay.addWidget(self.ts_controls)

        self.ts_model_wrap = QWidget()
        self.ts_model_layout = QVBoxLayout(self.ts_model_wrap)
        self.ts_model_layout.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.ts_model_wrap)

        self.btn_ts_load.clicked.connect(self._open_ts_dialog)
        self.btn_ts_clear.clicked.connect(self._on_clear_ts)
        return w

    def _build_ml_block(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        title = QLabel("通用 ML 模型")
        title.setStyleSheet("font-weight:bold")
        lay.addWidget(title)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        self.btn_ml_load = QPushButton("加载模型…")
        self.btn_ml_clear = QPushButton("清空")
        row.addWidget(self.btn_ml_load)
        row.addWidget(self.btn_ml_clear)
        lay.addLayout(row)
        self.ml_model_wrap = QWidget()
        self.ml_model_layout = QVBoxLayout(self.ml_model_wrap)
        self.ml_model_layout.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.ml_model_wrap)

        self.btn_ml_load.clicked.connect(self._open_ml_dialog)
        self.btn_ml_clear.clicked.connect(self._on_clear_ml)
        return w

    def _build_fault_block(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        title = QLabel("故障等级模型")
        title.setStyleSheet("font-weight:bold")
        lay.addWidget(title)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        self.btn_fault_load = QPushButton("加载模型…")
        self.btn_fault_clear = QPushButton("清空")
        row.addWidget(self.btn_fault_load)
        row.addWidget(self.btn_fault_clear)
        lay.addLayout(row)
        self.fault_model_wrap = QWidget()
        self.fault_model_layout = QVBoxLayout(self.fault_model_wrap)
        self.fault_model_layout.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.fault_model_wrap)

        self.btn_fault_load.clicked.connect(self._open_fault_dialog)
        self.btn_fault_clear.clicked.connect(self._on_clear_fault)
        return w

    # ------------------------------------------------------------------
    # 右侧动作
    # ------------------------------------------------------------------
    def _make_ts_actions(self) -> QWidget:
        box = QWidget(); lay = QHBoxLayout(box)
        self.btn_ts_to_down = QPushButton("生成预测 ➜ 下游")
        self.btn_ts_to_down.clicked.connect(self._on_ts_to_down_and_run)
        lay.addStretch(1); lay.addWidget(self.btn_ts_to_down)
        return box

    def _on_ts_to_down_and_run(self) -> None:
        self._on_ts_generate()
        # 如果上一部没有因校验提前 return，则执行推理
        self._on_run()

    def _make_final_actions(self) -> QWidget:
        box = QWidget(); lay = QHBoxLayout(box)
        self.btn_clear_results = QPushButton("清空结果列")
        self.btn_run = QPushButton("推理")
        self.btn_clear_results.clicked.connect(self._on_clear_results)
        self.btn_run.clicked.connect(self._on_run)
        lay.addStretch(1); lay.addWidget(self.btn_clear_results); lay.addWidget(self.btn_run)
        return box

    # ------------------------------------------------------------------
    # 时间序列加载/清空
    # ------------------------------------------------------------------
    def _open_ts_dialog(self) -> None:
        dlg = ModelManagerDialog(self.ts_manager, self)
        dlg.model_loaded.connect(self._add_ts_model)
        dlg.exec()

    def _add_ts_model(self, model_id: str) -> None:
        try:
            info = self.ts_manager.load_model(model_id)
            feats = self.ts_manager.current_feature_names() or []
            name = self.ts_manager.models_registry.get(model_id, {}).get("name", model_id)
            lb = int(info.get("meta", {}).get("params", {}).get("look_back", 1))
            if model_id not in self.ts_model_ids:
                self.ts_model_ids.append(model_id)
            self.ts_features[model_id] = feats
            self.ts_names[model_id] = name
            self.ts_lookbacks[model_id] = lb
            self._refresh_ts_models()
        except Exception as e:
            QMessageBox.critical(self, "加载失败", str(e))

    def _remove_ts_model(self, model_id: str) -> None:
        if model_id in self.ts_model_ids:
            self.ts_model_ids.remove(model_id)
        self.ts_features.pop(model_id, None)
        self.ts_names.pop(model_id, None)
        self.ts_lookbacks.pop(model_id, None)
        self._refresh_ts_models()

    def _on_clear_ts(self) -> None:
        from backend.timeseries_interface import _RuntimeSingleton
        _RuntimeSingleton.reset()
        self.ts_manager = ModelManager()
        self.ts_model_ids.clear()
        self.ts_features.clear()
        self.ts_names.clear()
        self.ts_lookbacks.clear()
        self._refresh_ts_models()

    # ------------------------------------------------------------------
    # ML 加载/清空
    # ------------------------------------------------------------------
    def _open_ml_dialog(self) -> None:
        dlg = ModelManagerDialog(self.ml_manager, self, multi_select=True)
        dlg.models_loaded.connect(self._add_ml_models)
        dlg.exec()

    def _add_ml_models(self, model_ids: list[str]) -> None:
        for mid in model_ids:
            try:
                meta = self.ml_manager.load_models([mid])
                name = self.ml_manager.registry.get(mid, {}).get("name", mid)
                feats = infer_input_features(meta.get("features", []), meta.get("calc_recipes", []))
                tgt = meta.get("target", "ml_pred")
                if mid not in self.ml_model_ids:
                    self.ml_model_ids.append(mid)
                self.ml_names[mid] = name
                self.ml_features[mid] = feats
                self.ml_targets[mid] = tgt
            except Exception as e:
                QMessageBox.warning(self, "加载失败", str(e))
        ML.clear()
        self._refresh_ml_models()

    def _remove_ml_model(self, model_id: str) -> None:
        if model_id in self.ml_model_ids:
            self.ml_model_ids.remove(model_id)
        self.ml_names.pop(model_id, None)
        self.ml_features.pop(model_id, None)
        self.ml_targets.pop(model_id, None)
        self._refresh_ml_models()

    def _on_clear_ml(self) -> None:
        try:
            ML.clear()
        except Exception:
            pass
        self.ml_model_ids.clear()
        self.ml_names.clear()
        self.ml_features.clear()
        self.ml_targets.clear()
        self._refresh_ml_models()

    # ------------------------------------------------------------------
    # 故障等级加载/清空
    # ------------------------------------------------------------------
    def _open_fault_dialog(self) -> None:
        dlg = ModelManagerDialog(self.fault_manager, self)
        dlg.model_loaded.connect(self._add_fault_model)
        dlg.exec()

    def _add_fault_model(self, model_id: str) -> None:
        try:
            est, _, meta = self.fault_manager.load_model(model_id)
            name = meta.get("name", model_id)
            feats = list(est.feature_names)
            self.fault_models[model_id] = est
            self.fault_names[model_id] = name
            self.fault_features[model_id] = feats
            self._refresh_fault_models()
        except Exception as e:
            QMessageBox.critical(self, "加载失败", str(e))

    def _remove_fault_model(self, model_id: str) -> None:
        self.fault_models.pop(model_id, None)
        self.fault_names.pop(model_id, None)
        self.fault_features.pop(model_id, None)
        self._refresh_fault_models()

    def _on_clear_fault(self) -> None:
        self.fault_models.clear()
        self.fault_names.clear()
        self.fault_features.clear()
        self._refresh_fault_models()

    # ------------------------------------------------------------------
    # 时间序列预测追加到底表
    # ------------------------------------------------------------------
    def _on_ts_generate(self) -> None:
        if not self.ts_model_ids:
            QMessageBox.information(self, "提示", "请先加载时间序列模型。")
            return
        df_ts = self.tbl_ts.dataframe()
        if df_ts.empty:
            QMessageBox.information(self, "提示", "时间序列表为空。")
            return
        df_ts = df_ts.apply(pd.to_numeric, errors="coerce")
        df_ts = df_ts.dropna()
        if df_ts.empty:
            QMessageBox.information(self, "提示", "时间序列表无有效数值行。")
            return
        required_rows = max(self.ts_lookbacks.values()) if self.ts_lookbacks else 0
        if len(df_ts) < required_rows:
            QMessageBox.information(self, "提示", f"时间序列至少需要 {required_rows} 行有效数据。")
            return

        pred_sum: Dict[str, np.ndarray] = {}
        pred_count: Dict[str, int] = {}
        for mid in self.ts_model_ids:
            feats = self.ts_features.get(mid, [])
            miss = [f for f in feats if f not in df_ts.columns]
            if miss:
                QMessageBox.warning(self, "特征缺失", f"模型 {self.ts_names.get(mid, mid)} 缺少列: {miss}")
            df_use = df_ts.reindex(columns=feats).dropna()
            if df_use.empty:
                continue
            try:
                self.ts_manager.load_model(mid)
                self.ts_manager.append_observations(df_use)
                pred = self.ts_manager.predict(steps=1)[0]["table"].T
                for c in pred.columns:
                    arr = np.asarray(pred[c]).ravel()
                    if c in pred_sum:
                        pred_sum[c] += arr
                        pred_count[c] += 1
                    else:
                        pred_sum[c] = arr
                        pred_count[c] = 1
            except Exception as e:
                QMessageBox.warning(self, "预测失败", str(e))

        if not pred_sum:
            QMessageBox.information(self, "提示", "没有生成任何预测。")
            return

        pred_avg = {c: pred_sum[c] / pred_count[c] for c in pred_sum}
        pred_df = pd.DataFrame(pred_avg)

        # 追加到时间序列表
        df_ts_orig = self.tbl_ts.dataframe()
        new_row_ts = {
            c: pred_df[c].iloc[0] if c in pred_df.columns else np.nan
            for c in df_ts_orig.columns
        }
        df_ts_new = pd.concat([df_ts_orig, pd.DataFrame([new_row_ts])], ignore_index=True)
        with self.tbl_ts.no_record():
            self.tbl_ts.set_dataframe(df_ts_new, record_state=False)

        # 根据共用表头追加行
        headers = list(self.tbl_common.dataframe().columns)
        df_common = self.tbl_common.dataframe()
        if df_common.empty:
            df_common = pd.DataFrame(columns=headers)
        new_row_common = {
            c: pred_df[c].iloc[0] if c in pred_df.columns else np.nan
            for c in headers
        }
        df_common_new = pd.concat([df_common, pd.DataFrame([new_row_common])], ignore_index=True)
        with self.tbl_common.no_record():
            self.tbl_common.set_dataframe(df_common_new, record_state=False)
        QMessageBox.information(self, "完成", "已追加预测结果")

    # ------------------------------------------------------------------
    # 最终推理
    # ------------------------------------------------------------------
    def _on_run(self) -> None:
        df = self.tbl_common.dataframe()
        if df.empty:
            QMessageBox.information(self, "提示", "下游输入表为空。")
            return
        feats_required = self._required_common_features()
        if not feats_required:
            QMessageBox.information(self, "提示", "暂无可用特征。")
            return
        df_raw = df.reindex(columns=feats_required).replace("", np.nan)
        valid = ~df_raw.isna().any(axis=1)
        idx = np.where(valid.to_numpy())[0]
        if idx.size == 0:
            QMessageBox.information(self, "提示", "无完整输入行。")
            return
        X_valid_raw = df_raw.iloc[idx].reset_index(drop=True)
        result_cols: Dict[str, np.ndarray] = {}

        if self.ml_model_ids:
            try:
                self.ml_manager.load_models(self.ml_model_ids)
                X_table = {
                    f: X_valid_raw.get(f, pd.Series([np.nan] * len(X_valid_raw))).to_numpy()
                    for f in X_valid_raw.columns
                }
                ret = ML.predict(X_table)
                if isinstance(ret, dict) and "labels" in ret:
                    target = ret.get("target", "ml_pred")
                    result_cols[target] = np.asarray(ret.get("labels")).ravel()
                elif isinstance(ret, dict):
                    for tgt, info in ret.items():
                        if isinstance(info, dict) and "labels" in info:
                            result_cols[str(tgt)] = np.asarray(info["labels"]).ravel()
            except Exception as e:
                QMessageBox.warning(self, "ML 预测失败", str(e))

        if self.fault_models:
            try:
                X_fault = X_valid_raw.apply(pd.to_numeric, errors="coerce")
                y = self.fault_manager.predict_many(list(self.fault_models.values()), X_fault)
                if y.size:
                    result_cols["fault_pred"] = y
            except Exception as e:
                QMessageBox.warning(self, "故障等级预测失败", str(e))

        if not result_cols:
            QMessageBox.information(self, "提示", "没有生成任何结果。")
            return
        df_out = df.copy()
        for name, arr in result_cols.items():
            arr = np.asarray(arr).ravel()
            if arr.dtype.kind in {"U", "S", "O"}:
                col = np.full(len(df_out), None, dtype=object)
            else:
                col = np.full(len(df_out), np.nan)
            col[idx[: len(arr)]] = arr[: len(idx)]
            df_out[name] = col
        with self.tbl_common.no_record():
            self.tbl_common.set_dataframe(df_out, record_state=False)
        self.pred_cols.update(result_cols.keys())
        QMessageBox.information(self, "完成", f"已写入结果列：{list(result_cols.keys())}")

    def _on_clear_results(self) -> None:
        df = self.tbl_common.dataframe()
        drop = [c for c in df.columns if c in self.pred_cols]
        if not drop:
            return
        df2 = df.drop(columns=drop)
        with self.tbl_common.no_record():
            self.tbl_common.set_dataframe(df2, record_state=False)
        self.pred_cols.clear()
