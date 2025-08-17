import numpy as np
from typing import Callable, Optional
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor


def train_xgb(
    X_train,
    y_train,
    X_val,
    y_val,
    log_callback: Optional[Callable[[str], None]] = None,
    **params,
):
    """Train an XGBoost model and return it with validation MAE."""
    defaults = dict(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="reg:squarederror",
        tree_method="hist",
    )
    defaults.update(params)
    model = MultiOutputRegressor(XGBRegressor(**defaults))
    model.fit(X_train.reshape(len(X_train), -1), y_train)
    preds = model.predict(X_val.reshape(len(X_val), -1))
    mae = mean_absolute_error(y_val, preds)
    return model, mae


def predict(model, seq):
    seq = np.asarray(seq).reshape(1, -1)
    return model.predict(seq).squeeze()
