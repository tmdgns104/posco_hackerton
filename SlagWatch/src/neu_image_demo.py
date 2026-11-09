"""Isolated local comparison and selection of the bundled NEU V1/V2 models."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import uuid

from filelock import FileLock

from . import security
from .inspection_projects import SurfaceAdapter, load_project
from .model_contract import ImageModelAdapter
from .surface_data import CLASSES
from .surface_detector import Detector

MODEL_IDS = ("V1", "V2")


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode("utf-8")


def _atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".stage-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(_json(value))
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _adapter(model_id):
    if model_id not in MODEL_IDS:
        raise ValueError("이 시험은 동봉된 V1과 V2만 지원합니다.")
    return ImageModelAdapter(SurfaceAdapter(Detector(model_id)), model_id, CLASSES, "multiclass")


def registered_models(root):
    root = Path(root)
    models = []
    for model_id in MODEL_IDS:
        path = root / "models" / (model_id + ".json")
        if path.is_symlink():
            raise ValueError("모델 등록 기록이 심볼릭 링크입니다.")
        if path.exists():
            item = json.loads(path.read_text(encoding="utf-8"))
            if item.get("schema") != 1 or item.get("model_id") != model_id:
                raise ValueError("이미지 모델 등록 기록이 올바르지 않습니다.")
            models.append(item)
    return models


def register_builtin(root, model_id, *, auth_token):
    if model_id not in MODEL_IDS:
        raise ValueError("이 시험은 동봉된 V1과 V2만 지원합니다.")
    audit = security.begin_action(auth_token, "neu_demo_model_registration", {"model_id": model_id})
    root = Path(root)
    folder = root / "models"
    folder.mkdir(parents=True, exist_ok=True)
    with FileLock(str(folder / ".lock"), timeout=10):
        if model_id in {item["model_id"] for item in registered_models(root)}:
            raise ValueError("이미 등록된 이미지 모델입니다.")
        adapter = _adapter(model_id)
        metadata = {"schema": 1, "model_id": model_id, "model_type": "NEU image classification",
                    "classes": list(CLASSES), "identity": adapter.identity,
                    "source": "bundled surface_v1 model; referenced, not copied"}
        _atomic_json(folder / (model_id + ".json"), metadata)
    security.finish_action(audit, "completed")
    return metadata


def _registered_adapter(root, model_id):
    metadata = {item["model_id"]: item for item in registered_models(root)}.get(model_id)
    if metadata is None:
        raise ValueError(f"{model_id} 모델을 먼저 등록하세요.")
    adapter = _adapter(model_id)
    if metadata["identity"] != adapter.identity or metadata["classes"] != list(CLASSES):
        raise ValueError("등록 이후 이미지 모델 또는 입력 계약이 변경되었습니다.")
    return adapter


def active_model(root):
    path = Path(root) / "active.json"
    if path.is_symlink():
        raise ValueError("이미지 활성 모델 기록이 심볼릭 링크입니다.")
    if not path.exists():
        return "V1"
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("schema") != 1 or state.get("model_id") not in MODEL_IDS:
        raise ValueError("이미지 활성 모델 기록이 올바르지 않습니다.")
    return state["model_id"]


def _evaluate(root, before, after):
    if before == after or before not in MODEL_IDS or after not in MODEL_IDS:
        raise ValueError("서로 다른 V1과 V2를 선택하세요.")
    adapters = [_registered_adapter(root, mid) for mid in (before, after)]
    project = load_project("neu_cls6")
    rows = project.rows("validation").groupby("true_label", group_keys=False).head(2)
    if len(rows) != len(CLASSES) * 2 or set(rows.true_label) != set(CLASSES):
        raise ValueError("각 클래스의 검증 이미지 두 장이 필요합니다.")
    raws = [project.load(row) for _, row in rows.iterrows()]
    predictions = [adapter.predict(raws) for adapter in adapters]
    if any(len(result) != len(rows) for result in predictions):
        raise ValueError("예측 결과의 개수가 이미지 개수와 다릅니다.")
    records = []
    for i, (_, row) in enumerate(rows.iterrows()):
        left, right = predictions[0][i], predictions[1][i]
        if left["status"] != "ok" or right["status"] != "ok":
            raise ValueError("완료되지 않은 이미지 추론이 있어 교체할 수 없습니다.")
        records.append({"sample_id": row["sample_id"], "true_label": row["true_label"],
                        "before": left["predicted_label"], "after": right["predicted_label"]})
    counts = {side: sum(item[side] == item["true_label"] for item in records) for side in ("before", "after")}
    return {"schema": 1, "before_model_id": before, "after_model_id": after, "samples": records,
            "correct_before": counts["before"], "correct_after": counts["after"],
            "model_identities": {adapter.model_id: adapter.identity for adapter in adapters},
            "source_manifest_sha256": project.index_hash, "n_samples": len(records),
            "gate_passed": counts["after"] >= counts["before"],
            "scope": "12 validation images, clean inputs; local demo only, not official approval"}


def compare(root, before, after):
    root = Path(root)
    if active_model(root) != before:
        raise ValueError("기준 모델이 현재 이미지 시험의 활성 모델과 다릅니다.")
    result = _evaluate(root, before, after)
    result["run_id"] = "neu-demo-" + uuid.uuid4().hex
    result["created_utc"] = datetime.now(timezone.utc).isoformat()
    _atomic_json(root / "runs" / (result["run_id"] + ".json"), result)
    return result


def list_runs(root):
    folder = Path(root) / "runs"
    if not folder.exists():
        return []
    return sorted((p.stem for p in folder.glob("neu-demo-*.json")
                   if not p.is_symlink() and re.fullmatch(r"neu-demo-[0-9a-f]{32}", p.stem)), reverse=True)


def load_run(root, run_id):
    if not isinstance(run_id, str) or not re.fullmatch(r"neu-demo-[0-9a-f]{32}", run_id):
        raise ValueError("이미지 비교 실행 ID가 올바르지 않습니다.")
    path = Path(root) / "runs" / (run_id + ".json")
    if path.is_symlink():
        raise ValueError("이미지 비교 기록이 심볼릭 링크입니다.")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema") != 1 or report.get("run_id") != run_id:
        raise ValueError("이미지 비교 기록이 올바르지 않습니다.")
    return report


def activate(root, run_id, *, confirmed, auth_token):
    if confirmed is not True:
        raise ValueError("이미지 시험용 로컬 교체를 확인해 주세요.")
    audit = security.begin_action(auth_token, "neu_demo_model_replacement", {"run_id": run_id})
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root / ".activation.lock"), timeout=10):
        report = load_run(root, run_id)
        before, after = report["before_model_id"], report["after_model_id"]
        if active_model(root) != before:
            raise ValueError("현재 이미지 시험의 활성 모델이 변경되었습니다. 다시 비교하세요.")
        current = _evaluate(root, before, after)
        expected = {key: value for key, value in report.items() if key not in {"run_id", "created_utc"}}
        if current != expected:
            raise ValueError("이미지 비교 기록 또는 모델·데이터가 변경되었습니다. 다시 비교하세요.")
        if not current["gate_passed"]:
            raise ValueError("후보의 표본 정확도가 기준 모델보다 낮아 교체를 차단합니다.")
        report_path = root / "runs" / (run_id + ".json")
        state = {"schema": 1, "model_id": after, "previous_model_id": before, "run_id": run_id,
                 "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
                 "actor_authenticated": audit["actor"], "created_utc": datetime.now(timezone.utc).isoformat(),
                 "scope": "isolated NEU image demo; no change to the original workspace or deployment"}
        _atomic_json(root / "active.json", state)
    security.finish_action(audit, "completed")
    return state


def evaluate_active(root):
    model_id = active_model(root)
    adapter = _registered_adapter(root, model_id)
    project = load_project("neu_cls6")
    rows = project.rows("validation").groupby("true_label", group_keys=False).head(2)
    predictions = adapter.predict([project.load(row) for _, row in rows.iterrows()])
    if len(predictions) != len(rows) or any(result["status"] != "ok" for result in predictions):
        raise ValueError("활성 모델의 이미지 추론이 완료되지 않았습니다.")
    return {"model_id": model_id, "samples": len(rows),
            "correct": sum(result["predicted_label"] == label
                           for result, label in zip(predictions, rows.true_label, strict=True))}
