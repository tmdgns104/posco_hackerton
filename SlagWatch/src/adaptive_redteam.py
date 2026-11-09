"""SteelTrust G2: validation-only, reproducible, surrogate-guided image red teaming.

This module does NOT train or mutate inspected models. It reports an *exploratory*
search result, not an independent test-set accuracy or a proof of label preservation.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Param order: noise std, brightness offset, contrast factor, blur sigma.
LOW = np.array([0.0, -0.12, 0.76, 0.0], dtype=np.float64)
HIGH = np.array([0.10, 0.12, 1.24, 1.2], dtype=np.float64)
PARAMS = ("noise_sigma", "brightness", "contrast", "blur_sigma")


@dataclass(frozen=True)
class Setting:
    sample_id: str
    trial_seed: int = 2026
    budget: int = 24
    warmup: int = 8
    strategy: str = "adaptive"

    def __post_init__(self):
        if not self.sample_id or not 4 <= self.budget <= 120:
            raise ValueError("sample_id and trial budget 4..120 are required")
        if not 2 <= self.warmup <= self.budget:
            raise ValueError("warmup must be 2..budget")
        if self.strategy not in ("adaptive", "random"):
            raise ValueError("strategy must be adaptive or random")


def seed_of(*args):
    token = json.dumps(args, sort_keys=True, ensure_ascii=True).encode("utf8")
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "big")


def image_change(raw: np.ndarray, parameters, noise_field: np.ndarray):
    """Deterministic camera-like alterations; raw 200x200 float32 [0,1]."""
    par = np.asarray(parameters, dtype=np.float64)
    if par.shape != (4,) or not np.isfinite(par).all() or (par < LOW).any() or (par > HIGH).any():
        raise ValueError("invalid perturbation parameters")
    if raw.shape != (200, 200) or raw.dtype != np.float32 or not np.isfinite(raw).all() or raw.min() < 0 or raw.max() > 1:
        raise ValueError("expected finite native 200x200 float32 grayscale")
    if noise_field.shape != raw.shape or not np.isfinite(noise_field).all():
        raise ValueError("invalid fixed noise field")
    noise, bright, contrast, blur = par
    x = np.asarray((raw - .5) * contrast + .5 + bright + noise * noise_field, dtype=np.float32)
    x = np.clip(x, 0, 1).astype(np.float32)
    if blur > 0.01:
        x = cv2.GaussianBlur(x, (5, 5), sigmaX=float(blur), sigmaY=float(blur), borderType=cv2.BORDER_REFLECT_101)
    x = np.ascontiguousarray(np.clip(x,0.0,1.0), dtype=np.float32)
    rmse = float(np.sqrt(np.mean(np.square(x - raw))))
    clipped = float(((x < .0001) | (x > .9999)).mean())
    return x, dict(rmse=rmse, clipped_fraction=clipped)


def classify_objective(scores: np.ndarray, label: str, classes: list[str], severity: float):
    """Search priority: induce actual error, then prefer small image disturbance.

    For correct predictions, optimize a smooth decreasing true-class margin.
    Decision scores are *not* probabilities.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (len(classes),) or not np.isfinite(scores).all() or label not in classes:
        raise ValueError("invalid class scores or label")
    correct_index = classes.index(label)
    alternative = max(scores[i] for i in range(len(classes)) if i != correct_index)
    margin = float(scores[correct_index] - alternative)
    predicted = classes[int(scores.argmax())]
    wrong = predicted != label
    priority = (2.0 if wrong else 0.0) - float(np.tanh(margin / 6.0)) - 0.1 * severity
    return dict(prediction=predicted, wrong=bool(wrong), true_margin=margin, priority=float(priority))


def suggestion(history: list[dict], rng: np.random.Generator, strategy: str, warmup: int,
               low=LOW, high=HIGH):
    """After shared random warmup, train RF surrogate to choose next trial.

    The surrogate learns a score landscape. No model weights are changed.
    """
    low=np.asarray(low,dtype=np.float64)
    high=np.asarray(high,dtype=np.float64)
    if strategy == "random" or len(history) < warmup:
        return rng.uniform(low, high)
    x = np.stack([r["parameters_vector"] for r in history])
    y = np.array([r["priority"] for r in history])
    rf = RandomForestRegressor(n_estimators=48, min_samples_leaf=2, max_features=.85,
                               random_state=42, n_jobs=1)
    rf.fit((x - low)/(high - low), y)
    candidates = rng.uniform(low, high, (128, len(low)))
    norm = (candidates - low)/(high - low)
    per_tree = np.stack([tree.predict(norm) for tree in rf.estimators_])
    acquisition = per_tree.mean(axis=0) + 0.7*per_tree.std(axis=0)
    # Suppress duplicate or nearly identical settings.
    distance = np.linalg.norm(norm[:, None] - (x - low)[None] / (high - low), axis=2).min(axis=1)
    acquisition += 0.04 * np.minimum(distance, .5)
    return candidates[int(np.argmax(acquisition))]


def search_one(setting: Setting, raw: np.ndarray, label: str,
               predict: Callable[[np.ndarray], dict], baseline: dict | None = None):
    """Caller supplies real model inference returning {'scores':[...], 'classes':[...]}."""
    rng = np.random.default_rng(seed_of("trials", setting.sample_id, setting.trial_seed))
    noise_field = np.random.default_rng(seed_of("noise", setting.sample_id, setting.trial_seed)).normal(size=raw.shape).astype(np.float32)
    clean = predict(raw)
    clean_score = classify_objective(clean["scores"], label, clean["classes"], 0)
    if clean_score["wrong"]:
        raise ValueError("clean input is already misclassified; excluded from induced-failure denominator")
    history = []
    for step in range(setting.budget):
        params = suggestion(history, rng, setting.strategy, setting.warmup)
        img, quality = image_change(raw, params, noise_field)
        # Mild guardrails. Rejections consume query budget and remain in the log.
        valid = quality["rmse"] <= 0.18 and quality["clipped_fraction"] <= 0.18
        rec = dict(trial=step + 1, parameters_vector=params.tolist(),
                   parameters={k:float(v) for k,v in zip(PARAMS,params)},
                   input_sha256=hashlib.sha256(img.tobytes()).hexdigest(), **quality,
                   visually_valid_unreviewed=bool(valid))
        if valid:
            o = predict(img)
            if list(o["classes"]) != list(clean["classes"]):
                raise ValueError("model class order changed during red-team run")
            rec.update(classify_objective(o["scores"], label, o["classes"], quality["rmse"]))
            rec["scores"] = [float(t) for t in o["scores"]]
        else:
            rec.update(prediction=None, wrong=False, true_margin=None, priority=-3.0, scores=None)
        history.append(rec)
    failures = [x for x in history if x["wrong"]]
    best = min(failures, key=lambda x:(x["rmse"],x["trial"])) if failures else None
    return dict(settings=asdict(setting), label=label, original_prediction=clean_score["prediction"],
                clean_scores=[float(t) for t in clean["scores"]], clean_input_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),
                failures=len(failures), first_failure_trial=failures[0]["trial"] if failures else None,
                minimum_failure_rmse=best["rmse"] if best else None,
                best_failure_trial=best["trial"] if best else None,
                trials=history)


def surface_models(root: Path, requested: list[str]):
    """Verify frozen experiments *before* loading user-supplied joblib artifacts."""
    from .surface_detector import Detector
    from .surface_g1 import G1Detector, DEFAULT_G1, file_sha
    from .surface_g1_replay import G1Run
    from .surface_replay import SurfaceRun
    from .surface_data import RUN, sha

    # These check code, model hashes, locked experiment scope and saved selection.
    SurfaceRun(sha((RUN / "manifest.json").read_bytes()))
    if "G1" in requested:
        G1Run(DEFAULT_G1, file_sha(DEFAULT_G1 / "manifest.json"))
    models = {}
    for name in requested:
        if name in ("V0", "V1", "V2"):
            models[name] = Detector(name)
        elif name == "G1":
            models[name] = G1Detector(DEFAULT_G1)
        else:
            raise ValueError("Supported surface model IDs: V0, V1, V2, G1")
    return models


def shared_predictor(model):
    """Reuses exact frozen feature extractor and class order."""
    from .surface_data import CLASSES
    def predict(raw):
        response = model.infer([raw])[0]
        return dict(scores=response["scores"], classes=CLASSES, prediction=response["prediction"])
    return predict


def run_surface(root: Path, sample_ids: list[str], model_id="V0", budget=24, warmup=8, seed=2026,
                compare_random=True, progress=None):
    """Development/validation only: no train/test image usage or model training."""
    import pandas as pd
    from .surface_data import RUN, ROOT
    from .surface_detector import read_raw
    if Path(root).resolve() != ROOT.resolve():
        raise ValueError("the provided project root does not match loaded source")
    if not 1 <= len(sample_ids) <= 25 or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("select 1..25 distinct validation images")
    meta = pd.read_csv(RUN / "split_manifest.csv").set_index("sample_id", drop=False)
    models = surface_models(ROOT, [model_id])
    detector = models[model_id]
    predict = shared_predictor(detector)
    runs = []
    for i, sid in enumerate(sample_ids):
        if sid not in meta.index or meta.loc[sid,"split"] != "validation":
            raise ValueError(f"{sid}: only locked validation population is permitted")
        row = meta.loc[sid]
        raw = read_raw(row)
        clean = predict(raw)
        if classify_objective(clean["scores"],row.label,clean["classes"],0)["wrong"]:
            runs.append(dict(sample_id=sid,label=row.label,skipped="clean_already_wrong"))
            continue
        algorithms=("random","adaptive") if compare_random else ("adaptive",)
        for strategy in algorithms:
            setting=Setting(sid,seed,budget,warmup,strategy)
            one=search_one(setting,raw,row.label,predict)
            runs.append(dict(sample_id=sid, model_id=model_id, model_identity=detector.identity,
                             split="validation", strategy=strategy,**one))
        if progress: progress(i+1,len(sample_ids))
    return dict(schema="steeltrust.adaptive_redteam.v1", generated_utc=datetime.now(timezone.utc).isoformat(),
                kind="exploratory_validation_only", no_test_set_used=True, model_id=model_id,
                warning="Synthetic perturbations are not guaranteed to preserve defect labels; manual review required.",
                rng_seed=seed, runs=runs)


def summary_table(report):
    import pandas as pd
    rows=[]
    for r in report["runs"]:
        if "skipped" in r:
            continue
        rows.append(dict(sample_id=r["sample_id"],model_id=r["model_id"],label=r["label"],
                         strategy=r["strategy"],budget=r["settings"]["budget"],
                         failures=r["failures"],first_failure_trial=r["first_failure_trial"],
                         minimum_failure_rmse=r["minimum_failure_rmse"],
                         best_failure_trial=r["best_failure_trial"]))
    return pd.DataFrame(rows)
