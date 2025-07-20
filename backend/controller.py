from pathlib import Path

import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pandas as pd

from data_utils import load_dataset, build_windows, compute_relative_errors

from backend.models import (
    gru_model,
    tcn_model,
    tsmixer_model,
    random_forest_model,
    xgboost_model,
    timesnet_model
)

CSV_PATH = Path(__file__).resolve().parents[1] / "data/20230510-20240924_merged.xlsx"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def prepare_data(look_back: int):
    data, feature_names, scaler = load_dataset(str(CSV_PATH))
    X, y = build_windows(data[:-5], look_back)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    return (X_train, y_train), (X_val, y_val), feature_names, scaler


# 全局配置：想跑哪些模型就在列表里写上名字，注释掉就不跑
ACTIVE_MODELS = [
    # "gru",
    #"tcn",
    "tsmixer",
    #"rf",
    "timesnet",
    # "xgb",
]

# 把所有模型的 train 和 predict 函数映射到同一个结构
MODEL_REGISTRY = {
    # "gru": {
    #     "train": lambda X_tr, y_tr, X_val, y_val: gru_model.train_gru(X_tr, y_tr, X_val, y_val, device=DEVICE),
    #     "predict": lambda model, seq: gru_model.predict(model, seq, device=DEVICE),
    # },
    # "tcn": {
    #     "train": lambda X_tr, y_tr, X_val, y_val: tcn_model.train_tcn(X_tr, y_tr, X_val, y_val, device=DEVICE),
    #     "predict": lambda model, seq: tcn_model.predict(model, seq, device=DEVICE),
    # },
    "tsmixer": {
        "train": lambda X_tr, y_tr, X_val, y_val: tsmixer_model.train_tsmixer(X_tr, y_tr, X_val, y_val, device=DEVICE),
        "predict": lambda model, seq: tsmixer_model.predict(model, seq, device=DEVICE),
    },
    # "rf": {
    #     "train": lambda X_tr, y_tr, X_val, y_val: random_forest_model.train_rf(
    #         X_tr, y_tr, X_val, y_val, n_estimators=400, random_state=42, n_jobs=-1
    #     ),
    #     "predict": lambda model, seq: random_forest_model.predict(model, seq),
    # },
    # "xgb": {
    #     "train": lambda X_tr, y_tr, X_val, y_val: xgboost_model.train_xgb(X_tr, y_tr, X_val, y_val),
    #     "predict": lambda model, seq: xgboost_model.predict(model, seq),
    # },
    "timesnet": {
        "train": lambda X_tr, y_tr, X_val, y_val: timesnet_model.train_timesnet(
            X_tr, y_tr, X_val, y_val, device=DEVICE, d_model=32, num_blocks=3),
        "predict": lambda model, seq: timesnet_model.predict(model, seq, device=DEVICE),
    },
}
DATA_CFG = {
    "tcn": 14,
    "tsmixer": 32,
    "rf": 5,
    "timesnet": 32,       # 新增
}

def run_all_models():
    # 不同模型需要不同 look_back

    trained_models = {}

    for name in ACTIVE_MODELS:
        look_back = DATA_CFG.get(name,14)
        (X_tr, y_tr), (X_val, y_val), _, feats_scaled = prepare_data(look_back)

        print(f"Training {name.upper()} …")
        train_fn = MODEL_REGISTRY[name]["train"]
        result = train_fn(X_tr, y_tr, X_val, y_val)

        # 解析返回值：神经网返回 (model, loss)，树模型返回 (model, mae)
        if len(result) == 2:
            model_obj, metric = result
            if name in ("gru", "tcn", "tsmixer"):
                print(f"{name.upper()} Val Loss: {metric:.4f}")
            else:
                print(f"{name.upper()} MAE: {metric:.4f}")
        else:
            model_obj = result  # 万一只有 model

        trained_models[name] = model_obj

    # 训练完毕后统一做 multi-step 预测
    predict_next_steps(trained_models,feats_scaled)

def predict_next_steps(trained_models: dict,feats_scaled, steps: int = 5):
    """对所有 ACTIVE_MODELS 做逐步预测，并打印相对误差"""
    data_scaled, feature_names, scaler = load_dataset(str(CSV_PATH))
    features = pd.DataFrame(scaler.inverse_transform(data_scaled), columns=feature_names)

    for i in range(1, steps + 1):
        preds = {}
        for name, model_obj in trained_models.items():
            look_back = DATA_CFG.get(name, 14)
            seq = data_scaled[-look_back - i : -i]

            pred = MODEL_REGISTRY[name]["predict"](model_obj, seq)
            preds[name] = scaler.inverse_transform(pred.reshape(1, -1)).squeeze()

        preds["eval"] = 0.5*(preds["tsmixer"]+preds["timesnet"])
        # 真值
        true_series = features.iloc[-i].astype(float)
        # 构造 DataFrame
        df = pd.DataFrame({name: preds[name] for name in preds}, index=feature_names)
        df.insert(0, "True", true_series)

        # 计算误差
        rel_err_df, max_err, mean_err = compute_relative_errors(df)
        print(f"\n—— 第 {i} 步预测 ——")
        # 合并显示
        conc = pd.concat([df, rel_err_df.add_suffix("_err")], axis=1)
        with pd.option_context(
                'display.max_rows', None,
                'display.max_columns', None,
                'display.width', None
        ):
            print(conc)
        print("最大相对误差：", max_err)
        print("平均相对误差：", mean_err)
if __name__ == "__main__":
    run_all_models()
