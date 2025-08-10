from __future__ import annotations

from typing import List, Any, Dict, Callable

import numpy as np
import pandas as pd

from backend.ml_interface import ML
from backend.ml_model_manager import MLModelManager

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QHBoxLayout,
    QMessageBox,
    QInputDialog,
)

from gui.smart_table import SmartTable, SmartTableConfig


class ValidationPage(QWidget):
    """批量样本验证页，改用 :class:`SmartTable` 统一表格组件。
    - ★ 关键改造：若已保存“计算器公式”，预测时优先构造列字典并让后端自动补齐派生特征
    """

    MIN_ROWS = 5

    def __init__(self) -> None:
        super().__init__()

        # 数据属性
        self.meta: Dict[str, Any] = {}
        self.features: List[str] = []
        self._external_mode: bool = False
        self._external_cb: Callable[[pd.DataFrame], Dict[str, np.ndarray]] | None = None
        self.manager = MLModelManager()

        layout = QVBoxLayout(self)
        self.table = SmartTable(SmartTableConfig(min_rows=self.MIN_ROWS))
        layout.addWidget(self.table)

        btn_row = QHBoxLayout()
        self.predict_btn = QPushButton("计算")
        self.predict_btn.clicked.connect(self.on_predict)
        btn_row.addWidget(self.predict_btn)
        self.clear_btn = QPushButton("清空全部")
        self.clear_btn.clicked.connect(self._clear_all)
        btn_row.addWidget(self.clear_btn)
        self.result_lbl = QLabel("结果: ")
        btn_row.addWidget(self.result_lbl)
        btn_row.addStretch(1)
        self.save_btn = QPushButton("保存模型")
        self.save_btn.clicked.connect(self.save_model)
        btn_row.addWidget(self.save_btn)
        layout.addLayout(btn_row)

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------
    def configure(self, features: List[str]) -> None:
        self.features = list(features) or ["X0"]
        self.meta = {}
        df = pd.DataFrame(columns=self.features)
        self.table.set_dataframe(df)
        self.table.clear_history()
        self.save_btn.setEnabled(bool(ML.get_meta()))
        self.result_lbl.setText("结果: ")

    def enable_external(self, features: List[str], predict_cb) -> None:
        self._external_mode = True
        self._external_cb = predict_cb
        self.configure(features)

    # ------------------------------------------------------------------
    # 按钮逻辑
    # ------------------------------------------------------------------
    def _clear_all(self):
        self.table.set_dataframe(pd.DataFrame(columns=self.features))
        self.table.clear_history()
        self.result_lbl.setText("结果: ")

    def on_predict(self):
        df = self.table.dataframe()
        df_feat = df[self.features].apply(pd.to_numeric, errors="coerce")

        # ---------- 外部模式 ----------
        if self._external_mode and self._external_cb is not None:
            res = self._external_cb(df_feat) or {}
            if not isinstance(res, dict) or not res:
                self.result_lbl.setText("结果: 空")
                return
            df_all = pd.concat([df_feat, pd.DataFrame(res)], axis=1)
            with self.table.no_record():
                self.table.set_dataframe(df_all, record_state=False)
                for c in range(len(self.features), df_all.shape[1]):
                    for r in range(self.table.table.rowCount()):
                        item = self.table.table.item(r, c)
                        if item:
                            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.result_lbl.setText(
                f"结果: 共 {self.table.table.rowCount()} 行, 目标列 {df_all.shape[1] - len(self.features)} 个"
            )
            return

        # ---------- 非外部模式 ----------
        meta = ML.get_meta() or {}
        mask = ~df_feat.isna().any(axis=1)
        valid_idx = mask.to_numpy().nonzero()[0]
        if valid_idx.size == 0:
            with self.table.no_record():
                self.table.set_dataframe(df_feat, record_state=False)
            self.result_lbl.setText("结果: 本次没有完整行，已跳过")
            return
        df_valid = df_feat.iloc[valid_idx].reset_index(drop=True)

        # ★ 若模型元信息里含有 calc_recipes，则强制使用“列字典”进入后端，便于其自动补齐派生列
        use_table = False
        calc_recipes = None
        try:
            calc_recipes = meta.get("calc_recipes")
        except Exception:
            calc_recipes = None
        if isinstance(calc_recipes, list) and len(calc_recipes) > 0:
            use_table = True
        else:
            # 次优策略：若模型 features 中存在当前 df 没有的列，也切换为列字典
            need_feats = set(meta.get("features", []))
            has_feats = set(df_valid.columns.tolist())
            if not need_feats.issubset(has_feats):
                use_table = True

        try:
            if use_table:
                X = {c: df_valid[c].to_numpy() for c in df_valid.columns}
                ret = ML.predict(X)
            else:
                ret = ML.predict(df_valid.to_numpy())
        except Exception:
            # 兜底：两种方式都试一遍
            try:
                ret = ML.predict({c: df_valid[c].to_numpy() for c in df_valid.columns})
            except Exception:
                ret = ML.predict(df_valid.to_numpy())

        sub = self._normalize_backend_result(ret)
        if not isinstance(sub, dict) or not sub:
            self.result_lbl.setText("结果: 空")
            return
        n = len(df_feat)
        res: Dict[str, np.ndarray] = {}
        for t, arr in sub.items():
            arr = np.asarray(arr).ravel()
            full = np.full((n,), np.nan)
            full[valid_idx[: len(arr)]] = arr[: len(valid_idx)]
            res[t] = full
        df_all = df_feat.copy()
        for t, full in res.items():
            df_all[t] = full
        with self.table.no_record():
            self.table.set_dataframe(df_all, record_state=False)
            for i in range(len(self.features), len(df_all.columns)):
                for r in range(self.table.table.rowCount()):
                    item = self.table.table.item(r, i)
                    if item:
                        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        dropped = (len(df_feat) - len(valid_idx))
        hint = f"，跳过 {dropped} 行" if dropped > 0 else ""
        self.result_lbl.setText(
            f"结果: 共 {self.table.table.rowCount()} 行, 目标列 {len(res)} 个{hint}"
        )

    def save_model(self):
        if not ML.get_meta():
            QMessageBox.warning(self, "提示", "暂无可保存的模型")
            return
        name, ok = QInputDialog.getText(self, "保存模型", "模型名称：")
        if not ok or not name.strip():
            return
        try:
            mid = self.manager.save_current(name.strip())
            QMessageBox.information(self, "已保存", f"模型已保存为 ID: {mid}")
        except Exception as e:
            QMessageBox.warning(self, "错误", str(e))

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------
    def _normalize_backend_result(self, ret) -> Dict[str, Any]:
        if isinstance(ret, dict) and all(isinstance(v, dict) and "labels" in v for v in ret.values()):
            return {t: v.get("labels") for t, v in ret.items()}
        if isinstance(ret, dict) and "labels" in ret:
            return {ret.get("target", "目标"): ret.get("labels")}
        return {}
