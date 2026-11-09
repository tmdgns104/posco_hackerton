"""Explicitly trusted local joblib heads for the pinned KSDD2 encoder only."""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import tempfile

import joblib
import numpy as np
from filelock import FileLock
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .inspection_model import BinaryAdapter, ENCODER_SHA, PREP
from . import security

MODEL_ID = re.compile(r"[A-Za-z][A-Za-z0-9_-]{1,39}\Z")
MAX_BYTES = 128 * 1024 * 1024


def _read_model(path):
    path = Path(path)
    if path.suffix.lower() != ".joblib" or not path.is_file() or path.is_symlink():
        raise ValueError("A regular local .joblib file is required.")
    with path.open("rb") as stream:
        payload = stream.read(MAX_BYTES + 1)
    if not payload or len(payload) > MAX_BYTES:
        raise ValueError("Model file must be nonempty and at most 128 MiB.")
    return payload


def _validate(payload):
    # Only called after the operator has explicitly trusted this executable pickle.
    bundle = joblib.load(io.BytesIO(payload))
    if not isinstance(bundle, dict) or not {"model_id", "head", "threshold", "encoder_sha256", "preprocessing"} <= bundle.keys():
        raise ValueError("Expected the existing KSDD2 model bundle format.")
    head = bundle["head"]
    threshold = bundle["threshold"]
    if (not isinstance(bundle["model_id"], str) or not bundle["model_id"].strip()
            or bundle["encoder_sha256"] != ENCODER_SHA or bundle["preprocessing"] != PREP
            or isinstance(threshold, (bool, np.bool_)) or not isinstance(threshold, (int, float, np.number))
            or not math.isfinite(float(threshold))
            or not isinstance(head, Pipeline) or len(head.steps) != 2
            or type(head.steps[0][1]) is not StandardScaler or type(head.steps[1][1]) is not LogisticRegression
            or not np.array_equal(head.classes_, [0, 1]) or head.n_features_in_ != 1536):
        raise ValueError("Model must use the pinned ResNet18 encoder, preprocessing and fitted 1536-feature binary scaler/logistic head.")
    score = np.asarray(head.decision_function(np.zeros((1, 1536), dtype=np.float64)))
    if score.shape != (1,) or not np.isfinite(score).all():
        raise ValueError("Model did not produce one finite binary decision score.")
    return bundle


def list_models(root):
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted(p.stem for p in root.glob("*.json") if MODEL_ID.fullmatch(p.stem) and not p.is_symlink())


def register_model(root, source, model_id, *, trusted, auth_token=None):
    if trusted is not True:
        raise ValueError("Explicit trust acknowledgement is required before loading a joblib model.")
    if not isinstance(model_id, str) or not MODEL_ID.fullmatch(model_id) or model_id.upper() in {"K0", "K1", "V0", "G1"}:
        raise ValueError("Use a unique 2–40 character model ID; built-in IDs are reserved.")
    source = Path(source)
    if not source.is_absolute():
        raise ValueError("Use an absolute local model file path.")
    payload = _read_model(source)
    digest = hashlib.sha256(payload).hexdigest()
    audit = security.begin_action(auth_token, "image_model_registration",
                                  {"model_id": model_id, "model_sha256": digest})
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root / ".registry.lock"), timeout=10):
        existing = list_models(root)
        if model_id.casefold() in {item.casefold() for item in existing}:
            raise ValueError("This model ID is already registered; use a new version ID.")
        bundle = _validate(payload)
        metadata = {
            "schema": 1, "model_id": model_id, "embedded_model_id": bundle["model_id"],
            "artifact_sha256": digest, "encoder_sha256": ENCODER_SHA, "preprocessing": PREP,
            "threshold": float(bundle["threshold"]), "class_mapping": {"0": "normal", "1": "defect"},
            "score_rule": "round(decision_function,6) >= threshold for defect",
            "source": "operator-trusted local joblib; development use only",
        }
        artifact = root / (model_id + ".joblib")
        meta = root / (model_id + ".json")
        if artifact.exists() or meta.exists():
            raise ValueError("Model destination already exists.")
        with tempfile.TemporaryDirectory(prefix=".model-", dir=root) as folder:
            stage = Path(folder)
            (stage / artifact.name).write_bytes(payload)
            (stage / meta.name).write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                                           encoding="utf-8")
            os.replace(stage / artifact.name, artifact)
            try:
                os.replace(stage / meta.name, meta)
            except OSError:
                artifact.unlink()
                raise
    security.finish_action(audit, "completed")
    return metadata


def load_registered(root, model_id):
    if not isinstance(model_id, str) or not MODEL_ID.fullmatch(model_id):
        raise ValueError("Invalid registered model ID.")
    root = Path(root)
    meta = root / (model_id + ".json")
    if not meta.is_file() or meta.is_symlink():
        raise ValueError("Registered model metadata is missing.")
    metadata = json.loads(meta.read_text(encoding="utf-8"))
    payload = _read_model(root / (model_id + ".joblib"))
    if (metadata.get("schema") != 1 or metadata.get("model_id") != model_id
            or hashlib.sha256(payload).hexdigest() != metadata.get("artifact_sha256")):
        raise ValueError("Registered model ID or artifact hash differs.")
    bundle = _validate(payload)
    if (bundle["model_id"] != metadata.get("embedded_model_id")
            or float(bundle["threshold"]) != metadata.get("threshold")
            or metadata.get("encoder_sha256") != ENCODER_SHA or metadata.get("preprocessing") != PREP
            or metadata.get("class_mapping") != {"0": "normal", "1": "defect"}):
        raise ValueError("Registered model metadata differs from the trusted artifact.")
    from .model_contract import ImageModelAdapter
    return ImageModelAdapter(BinaryAdapter(model_id, bundle["head"], metadata),
                             model_id, ["normal", "defect"], "binary")


class ModelView:
    """Keep the original project's data guards while selecting different adapters."""
    def __init__(self, project, models, *, development_only=False):
        self._project = project
        self.models = models
        self.official = False if development_only else project.official
        self._development_only = development_only

    def __getattr__(self, name):
        return getattr(self._project, name)

    def check_scope(self, split):
        if self._development_only and split != "validation":
            raise ValueError("Registered candidates are restricted to development validation.")
        return self._project.check_scope(split)
