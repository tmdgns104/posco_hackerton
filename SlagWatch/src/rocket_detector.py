"""Frozen multivariate MiniRocket representation with two separate Ridge heads."""
import hashlib
from importlib.metadata import version
from pathlib import Path

import numpy as np

RUN = Path(__file__).resolve().parents[1]/"experiments/rocket_v1"
NAMES = ("R0","R1")
PAIRS = (("J","R0"),("R0","R1"))
PACKAGES = ("aeon","numba","llvmlite","numpy","scikit-learn","scipy","joblib","torch","pandas","streamlit")


def versions():
    return {k:version(k) for k in PACKAGES}


def channel_first(waves):
    x = np.asarray(waves,dtype=np.float32)
    if x.ndim!=3 or x.shape[1:]!=(3200,3):
        raise ValueError("Expected windows of shape (3200, 3)")
    return np.ascontiguousarray(x.transpose(0,2,1))


def transform(bundle,waves):
    features = bundle["rocket"].transform(channel_first(waves))
    return bundle["scaler"].transform(features).astype(np.float64)


def representation_hash(rocket,scaler):
    h=hashlib.sha256()
    for array in (*rocket.parameters,scaler.mean_,scaler.scale_,scaler.var_,scaler.n_samples_seen_):
        x=np.asarray(array)
        h.update(str((str(x.dtype),x.shape)).encode())
        h.update(x.tobytes())
    return h.hexdigest()


def decisions(head,x):
    if not np.array_equal(head.classes_,[0,1]):
        raise ValueError("Unexpected Ridge class direction")
    return head.decision_function(x)
