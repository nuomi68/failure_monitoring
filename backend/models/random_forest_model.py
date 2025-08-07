import numpy as np
from typing import Callable, Optional
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error


def train_rf(
    X_train,
    y_train,
    X_val,
    y_val,
    log_callback: Optional[Callable[[str], None]] = None,
    **params,
):
    """Train a RandomForest model and return it with validation MAE."""
    model = MultiOutputRegressor(RandomForestRegressor(**params))
    model.fit(X_train.reshape(len(X_train), -1), y_train)
    preds = model.predict(X_val.reshape(len(X_val), -1))
    mae = mean_absolute_error(y_val, preds)
    if log_callback:
        log_callback(f"[RF] val_mae={mae:.4f}")
    return model, mae


def predict(model, seq):
    seq = np.asarray(seq).reshape(1, -1)
    return model.predict(seq).squeeze()
