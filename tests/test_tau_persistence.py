import os
from tempfile import TemporaryDirectory
import numpy as np
import sys
from pathlib import Path
import types

sys.path.append(str(Path(__file__).resolve().parents[1]))

# Stub out heavy model modules to avoid extra dependencies
for _name in [
    "backend.models.gru_model",
    "backend.models.patchtst_model",
    "backend.models.tcn_model",
    "backend.models.tsmixer_model",
    "backend.models.timesnet_model",
    "backend.models.xgboost_model",
]:
    sys.modules.setdefault(_name, types.SimpleNamespace())

from backend.ml_interface import ML


def test_tau_persistence_save_load():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 3)).astype(np.float32)
    rep = ML.train(alg="knn", X=X, params={"n_neighbors": 2}, feature_names=["a", "b", "c"])
    meta_before = ML.get_meta()
    assert 0 <= meta_before["tau"] <= 1
    assert meta_before.get("tau_is_normalized", False)
    assert ((rep.scores >= 0) & (rep.scores <= 1)).all()

    res_before = ML.predict(X)
    labels_before = res_before["labels"]
    scores_before = res_before["scores"]

    with TemporaryDirectory() as d:
        path = os.path.join(d, "model.pkl")
        ML.save(path)
        ML.clear()
        ML.load(path)
        res_after = ML.predict(X)

    assert np.array_equal(labels_before, res_after["labels"])
    assert np.allclose(scores_before, res_after["scores"])
