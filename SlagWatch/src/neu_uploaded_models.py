"""Trusted NEU bundle registry and paired V0 evaluation."""
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tempfile

from filelock import FileLock
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from . import inspection_runner, security
from .inspection_projects import load_project
from .model_contract import ImageModelAdapter
from .surface_data import CFG, CLASSES, ROOT, RUN, stable_id
from .surface_detector import extract

REGISTRY = ROOT / "p1_tabular_local" / "neu_uploaded"
MAX_BYTES = 128 * 1024 * 1024
MODEL_ID = re.compile(r"[A-Za-z][A-Za-z0-9_-]{1,39}\Z")
RUN_ID = re.compile(r"\d{8}T\d{12}Z-[0-9a-f]{32}\Z")


def _digest(payload):
    return hashlib.sha256(payload).hexdigest()


def _bundle(payload):
    try:
        bundle = joblib.load(io.BytesIO(payload))
    except Exception as exc:
        raise ValueError(f"NEU 모델 번들을 로드하지 못했습니다: {type(exc).__name__}: {exc}") from exc
    if not isinstance(bundle, dict):
        raise ValueError("NEU 모델 번들 딕셔너리가 필요합니다.")
    policy = bundle.get("policy")
    if (not isinstance(policy, str) or policy not in CFG["preprocessing"]
            or bundle.get("input_policy") != CFG["preprocessing"][policy]
            or bundle.get("encoder_sha") != CFG["weight_sha256"]
            or bundle.get("protocol_sha") != _digest((RUN / "protocol.json").read_bytes())):
        raise ValueError("NEU 전처리·인코더·규약이 일치하지 않습니다.")
    head = bundle.get("head")
    if (not callable(getattr(head, "decision_function", None))
            or list(getattr(head, "classes_", [])) != CLASSES
            or getattr(head, "n_features_in_", None) != 512
            or not isinstance(bundle.get("train_id"), str) or not bundle["train_id"]):
        raise ValueError("512차원 입력과 NEU 6개 클래스 순서를 가진 학습된 분류기가 필요합니다.")
    with threadpool_limits(limits=2, user_api="blas"):
        scores = np.asarray(head.decision_function(np.zeros((1, 512))))
    if scores.shape != (1, len(CLASSES)) or not np.isfinite(scores).all():
        raise ValueError("분류기의 출력은 유한한 6개 클래스 점수여야 합니다.")
    return bundle


def _identity(model_id, bundle, digest):
    return {"model_id": model_id, "head_sha256": digest, "encoder_sha256": bundle["encoder_sha"],
            "train_id": bundle["train_id"],
            "preprocessing_sha256": stable_id([bundle["protocol_sha"], bundle["policy"], bundle["input_policy"]])}


def register(root, model_id, payload, filename, *, trusted, auth_token):
    if not isinstance(model_id, str) or not MODEL_ID.fullmatch(model_id) or model_id.upper() in {"V0", "V1", "V2", "G1"}:
        raise ValueError("모델 ID는 예약된 이름을 제외한 영문·숫자·_·- 2~40자여야 합니다.")
    if trusted is not True:
        raise ValueError("신뢰한 파일이라는 확인이 필요합니다.")
    if (not isinstance(payload, bytes) or not payload or len(payload) > MAX_BYTES
            or not isinstance(filename, str) or Path(filename).suffix.lower() != ".joblib"):
        raise ValueError("128 MiB 이하의 .joblib 파일이 필요합니다.")
    event = security.begin_action(auth_token, "neu_uploaded_registration",
                                  {"model_id": model_id, "sha256": _digest(payload)})
    bundle = _bundle(payload)
    identity = _identity(model_id, bundle, _digest(payload))
    models = Path(root) / "models"
    models.mkdir(parents=True, exist_ok=True)
    with FileLock(str(models / ".lock"), timeout=10):
        target = models / model_id
        if target.exists() or target.is_symlink():
            raise ValueError("이미 등록된 모델 ID입니다.")
        with tempfile.TemporaryDirectory(prefix=".staging-", dir=models) as temp:
            stage = Path(temp)
            (stage / "model.joblib").write_bytes(payload)
            (stage / "metadata.json").write_text(json.dumps({"schema": 1, "model_id": model_id,
                "identity": identity}, ensure_ascii=False, sort_keys=True), encoding="utf-8")
            os.rename(stage, target)
    security.finish_action(event, "completed")
    return identity


def list_models(root):
    folder = Path(root) / "models"
    if not folder.exists():
        return []
    result = []
    for path in sorted(folder.iterdir()):
        if not MODEL_ID.fullmatch(path.name):
            continue
        if path.is_symlink() or not path.is_dir() or (path / "metadata.json").is_symlink():
            raise ValueError("이미지 모델 등록 경로가 올바르지 않습니다.")
        item = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        if item.get("schema") != 1 or item.get("model_id") != path.name:
            raise ValueError("이미지 모델 등록 정보가 올바르지 않습니다.")
        result.append(item)
    return result


class BundleAdapter:
    def __init__(self, bundle, identity):
        self.bundle = bundle
        self.identity = identity

    def predict(self, raws):
        features, _ = extract(raws, self.bundle["policy"])
        with threadpool_limits(limits=2, user_api="blas"):
            scores = np.asarray(self.bundle["head"].decision_function(features))
        if scores.shape != (len(raws), len(CLASSES)) or not np.isfinite(scores).all():
            raise ValueError("모델 출력이 유한한 6개 클래스 점수와 일치하지 않습니다.")
        return [{"status": "ok", "predicted_label": CLASSES[int(np.argmax(row))],
                 "scores": dict(zip(CLASSES, row.tolist(), strict=True)),
                 "score_type": "decision_function"} for row in scores]


def _adapter(root, model_id):
    records = {item["model_id"]: item for item in list_models(root)}
    if model_id not in records:
        raise ValueError("등록된 이미지 모델을 선택하세요.")
    path = Path(root) / "models" / model_id / "model.joblib"
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_BYTES:
        raise ValueError("등록된 모델 파일이 올바르지 않습니다.")
    payload = path.read_bytes()
    if _digest(payload) != records[model_id]["identity"]["head_sha256"]:
        raise ValueError("등록 이후 모델 파일이 변경되었습니다.")
    bundle = _bundle(payload)
    identity = _identity(model_id, bundle, _digest(payload))
    if identity != records[model_id]["identity"]:
        raise ValueError("등록 이후 모델 계약이 변경되었습니다.")
    return ImageModelAdapter(BundleAdapter(bundle, identity), model_id, CLASSES, "multiclass")


class _PairedProject:
    def __init__(self, project, candidate):
        self._project = project
        self.models = {"V0": project.models["V0"], candidate.model_id: candidate}

    def __getattr__(self, key):
        return getattr(self._project, key)


def evaluate(root, model_id, *, auth_token, sample_ids=None, progress=None):
    event = security.begin_action(auth_token, "neu_uploaded_evaluation", {"model_id": model_id})
    candidate = _adapter(root, model_id)
    project = _PairedProject(load_project("neu_cls6"), candidate)
    path = inspection_runner.run_project(project, "validation", Path(root) / "runs",
                                         sample_ids=sample_ids, progress=progress)
    security.finish_action(event, "completed")
    return path


def list_runs(root):
    folder = Path(root) / "runs"
    if not folder.exists():
        return []
    return sorted((p.name for p in folder.iterdir() if RUN_ID.fullmatch(p.name)
                   and p.is_dir() and not p.is_symlink()), reverse=True)


def load_run(root, run_id):
    if not isinstance(run_id, str) or not RUN_ID.fullmatch(run_id):
        raise ValueError("평가 기록 ID가 올바르지 않습니다.")
    path = Path(root) / "runs" / run_id
    cfg = inspection_runner.load_run(path)
    if (cfg["run_id"] != run_id or cfg["project_id"] != "neu_cls6" or cfg["split"] != "validation"
            or cfg["before_model_id"] != "V0"):
        raise ValueError("V0 검증 기록이 아닙니다.")
    inspection_runner.recompute_decision(path)
    pairs = pd.read_csv(path / "pairs.csv", keep_default_na=False)
    return cfg, pairs
