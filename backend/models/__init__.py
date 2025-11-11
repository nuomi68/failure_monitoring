from importlib import import_module
from typing import Any

__all__ = [
    'gru_model',
    'patchtst_model',
    'random_forest_model',
    'tcn_model',
    'tsmixer_model',
    'xgboost_model',
    'timesnet_model',
    'unsupervised_core',
    'supervised_core',
]


_MODULE_MAP = {name: f"{__name__}.{name}" for name in __all__}


def __getattr__(name: str) -> Any:
    if name not in _MODULE_MAP:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    module = import_module(_MODULE_MAP[name])
    globals()[name] = module
    return module


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_MODULE_MAP.keys()))
