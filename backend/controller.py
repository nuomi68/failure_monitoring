from pathlib import Path

import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from .data_utils import load_dataset, build_windows
from .models import (
    gru_model,
    tcn_model,
    tsmixer_model,
    random_forest_model,
    xgboost_model,
)

CSV_PATH = Path(__file__).resolve().parents[1] / "data/20230510-20240924_merged.xlsx"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def prepare_data(look_back: int):
    data, feature_names, scaler = load_dataset(str(CSV_PATH))
    X, y = build_windows(data, look_back)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    return (X_train, y_train), (X_val, y_val), feature_names, scaler


def run_all_models():
    (X_train, y_train), (X_val, y_val), _, _ = prepare_data(14)
    
    print("Training GRU …")
    gru, gru_loss = gru_model.train_gru(
        X_train, y_train, X_val, y_val, device=DEVICE
    )
    preds = gru_model.predict(gru, X_val[-1], device=DEVICE)
    mae_gru = mean_absolute_error(y_val[-1], preds)
    print(f"GRU Val Loss: {gru_loss:.4f}  MAE: {mae_gru:.4f}")

    print("Training TCN …")
    tcn, tcn_loss = tcn_model.train_tcn(
        X_train, y_train, X_val, y_val, device=DEVICE
    )
    preds = tcn_model.predict(tcn, X_val[-1], device=DEVICE)
    mae_tcn = mean_absolute_error(y_val[-1], preds)
    print(f"TCN Val Loss: {tcn_loss:.4f}  MAE: {mae_tcn:.4f}")

    print("Training TSMixer …")
    mix, mix_loss = tsmixer_model.train_tsmixer(
        X_train, y_train, X_val, y_val, device=DEVICE
    )
    preds = tsmixer_model.predict(mix, X_val[-1], device=DEVICE)
    mae_mix = mean_absolute_error(y_val[-1], preds)
    print(f"TSMixer Val Loss: {mix_loss:.4f}  MAE: {mae_mix:.4f}")

    (X_train_t, y_train_t), (X_val_t, y_val_t), _, _ = prepare_data(5)

    print("Training RandomForest …")
    rf, mae_rf = random_forest_model.train_rf(
        X_train_t, y_train_t, X_val_t, y_val_t,
        n_estimators=400, random_state=42, n_jobs=-1
    )
    preds = random_forest_model.predict(rf, X_val_t[-1])
    mae_rf = mean_absolute_error(y_val_t[-1], preds)
    print(f"RandomForest MAE: {mae_rf:.4f}")

    print("Training XGBoost …")
    xgb, mae_xgb = xgboost_model.train_xgb(
        X_train_t, y_train_t, X_val_t, y_val_t
    )
    preds = xgboost_model.predict(xgb, X_val_t[-1])
    mae_xgb = mean_absolute_error(y_val_t[-1], preds)
    print(f"XGBoost MAE: {mae_xgb:.4f}")


if __name__ == "__main__":
    run_all_models()
