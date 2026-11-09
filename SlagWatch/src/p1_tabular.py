"""Local, explicitly trusted scikit-learn tabular comparison workspace."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tempfile
import time
import uuid

from filelock import FileLock
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from .inspection_event_log import record_event
from .model_contract import BaseModelAdapter
from . import security

ID = re.compile(r"[A-Za-z][A-Za-z0-9_-]{1,39}\Z")
MAX_MODEL_BYTES = 128 * 1024 * 1024
MAX_DATA_BYTES = 25 * 1024 * 1024
MAX_ROWS = 50000


class ContractError(ValueError):
    def __init__(self, code, message):
        self.code = code
        super().__init__(f"{code}: {message}")


def _digest(payload):
    return hashlib.sha256(payload).hexdigest()


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode("utf-8")


def _id(value):
    if not isinstance(value, str) or not ID.fullmatch(value):
        raise ContractError("INVALID_MODEL_ID", "모델 ID는 영문으로 시작하는 2~40자의 영문·숫자·_·-이어야 합니다.")
    return value


def _event(root, event, outcome, details):
    return record_event(Path(root) / "events", event, project_id="p1_tabular",
                        outcome=outcome, details=details)


def _model(payload):
    try:
        model = joblib.load(io.BytesIO(payload))
    except Exception as exc:
        raise ContractError("MODEL_LOAD_FAILED", f"신뢰한 모델을 읽지 못했습니다: {exc}") from exc
    if type(model) not in (LogisticRegression, RandomForestClassifier):
        raise ContractError("MODEL_UNSUPPORTED", "학습된 scikit-learn LogisticRegression 또는 RandomForestClassifier만 지원합니다.")
    if not hasattr(model, "classes_") or not hasattr(model, "n_features_in_"):
        raise ContractError("MODEL_LOAD_FAILED", "학습되지 않은 모델입니다.")
    labels = [str(x) for x in model.classes_]
    if len(labels) < 2 or len(set(labels)) != len(labels) or any(not x for x in labels):
        raise ContractError("CLASS_MAPPING_MISMATCH", "클래스 이름은 서로 다른 빈 문자열이 아닌 값이어야 합니다.")
    return model, labels


def register_model(root, payload, filename, model_id, name, version, features, *, trusted, auth_token=None):
    if trusted is not True:
        raise ContractError("TRUST_REQUIRED", "신뢰한 joblib 파일임을 명시적으로 확인해야 합니다.")
    _id(model_id)
    if not isinstance(payload, bytes) or not payload or len(payload) > MAX_MODEL_BYTES or Path(filename).suffix.lower() != ".joblib":
        raise ContractError("MODEL_FILE_INVALID", "비어 있지 않은 128 MiB 이하의 .joblib 파일이 필요합니다.")
    if not name.strip() or not version.strip() or len(name) > 120 or len(version) > 80:
        raise ContractError("METADATA_INVALID", "모델 이름과 버전을 입력하세요.")
    if (not features or len(set(features)) != len(features)
            or any(not isinstance(x, str) or not x.strip() or x != x.strip() or x in {"sample_id", "true_label"} for x in features)):
        raise ContractError("FEATURE_INVALID", "중복되지 않는 피처 이름을 순서대로 입력하세요.")
    audit = security.begin_action(auth_token, "tabular_model_registration",
                                  {"model_id": model_id, "model_sha256": _digest(payload)})
    root = Path(root)
    registry = root / "models"
    registry.mkdir(parents=True, exist_ok=True)
    _event(root, "model_registration", "started", {"model_id": model_id})
    try:
        model, labels = _model(payload)
        if model.n_features_in_ != len(features):
            raise ContractError("FEATURE_MISSING", f"모델은 {model.n_features_in_}개 피처를 요구합니다.")
        if hasattr(model, "feature_names_in_") and list(model.feature_names_in_) != features:
            raise ContractError("FEATURE_ORDER_MISMATCH", "등록 피처 순서가 모델 학습 피처 순서와 다릅니다.")
        metadata = {
            "schema": 1, "model_id": model_id, "model_name": name.strip(), "model_version": version.strip(),
            "model_type": "tabular_classification", "framework": "scikit-learn",
            "adapter_type": type(model).__name__, "input_schema": {"type": "ordered_numeric_features", "features": features},
            "output_schema": {"type": "classification", "scores": "per_class"},
            "classes": labels, "score_type": "probability", "score_direction": "larger_is_stronger",
            "required_features": features, "preprocessing_version": "user-supplied-numeric-v1",
            "model_hash": _digest(payload), "registration_timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "ready", "trust": "operator-acknowledged executable joblib",
        }
        with FileLock(str(registry / ".lock"), timeout=10):
            if model_id.casefold() in {item["model_id"].casefold() for item in list_models(root)}:
                raise ContractError("MODEL_ID_EXISTS", "이미 등록된 모델 ID입니다.")
            with tempfile.TemporaryDirectory(prefix=".model-", dir=registry) as folder:
                stage = Path(folder)
                (stage / "artifact.joblib").write_bytes(payload)
                (stage / "metadata.json").write_bytes(_json(metadata))
                os.replace(stage, registry / model_id)
        _event(root, "model_registration", "completed", {"model_id": model_id, "model_hash": metadata["model_hash"]})
        security.finish_action(audit, "completed")
        return metadata
    except Exception as exc:
        _event(root, "model_registration", "failed", {"model_id": model_id, "error": str(exc)[:500]})
        raise


def list_models(root):
    registry = Path(root) / "models"
    if not registry.exists():
        return []
    result = []
    for folder in sorted(registry.iterdir()):
        if folder.is_dir() and not folder.is_symlink() and ID.fullmatch(folder.name):
            meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
            if meta.get("model_id") != folder.name:
                raise ContractError("REGISTRY_INVALID", f"레지스트리 ID 불일치: {folder.name}")
            result.append(meta)
    return result


class TabularAdapter(BaseModelAdapter):
    def __init__(self, root, model_id):
        self.model_id = _id(model_id)
        folder = Path(root) / "models" / model_id
        if folder.is_symlink():
            raise ContractError("REGISTRY_INVALID", "모델 경로가 심볼릭 링크입니다.")
        self.metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
        artifact = folder / "artifact.joblib"
        if artifact.is_symlink() or artifact.stat().st_size > MAX_MODEL_BYTES:
            raise ContractError("MODEL_FILE_INVALID", "모델 파일 크기 또는 경로가 올바르지 않습니다.")
        payload = artifact.read_bytes()
        if (self.metadata.get("schema") != 1 or self.metadata.get("status") != "ready"
                or self.metadata.get("model_id") != model_id or _digest(payload) != self.metadata.get("model_hash")):
            raise ContractError("MODEL_INTEGRITY_ERROR", "모델 등록 상태 또는 해시가 일치하지 않습니다.")
        self.model, labels = _model(payload)
        if (labels != self.metadata["classes"] or self.model.n_features_in_ != len(self.metadata["required_features"])
                or type(self.model).__name__ != self.metadata["adapter_type"]
                or (hasattr(self.model, "feature_names_in_") and
                    list(self.model.feature_names_in_) != self.metadata["required_features"])):
            raise ContractError("MODEL_INTEGRITY_ERROR", "모델 계약이 등록 정보와 일치하지 않습니다.")

    def get_metadata(self):
        return self.metadata

    def validate_model(self):
        return self.metadata

    def validate_input(self, data):
        features = self.metadata["required_features"]
        missing = [x for x in features if x not in data.columns]
        if missing:
            raise ContractError("FEATURE_MISSING", "필수 피처 누락: " + ", ".join(missing))
        if list(data.columns) != features:
            raise ContractError("FEATURE_ORDER_MISMATCH", "피처 순서 또는 목록이 모델 계약과 다릅니다.")
        try:
            values = data.astype(np.float64)
        except (ValueError, TypeError) as exc:
            raise ContractError("FEATURE_TYPE_MISMATCH", "모든 피처는 유한한 숫자여야 합니다.") from exc
        if not np.isfinite(values.to_numpy()).all():
            raise ContractError("FEATURE_TYPE_MISMATCH", "NaN과 Inf는 허용되지 않습니다.")
        return values

    def preprocess(self, data):
        values = self.validate_input(data)
        if not hasattr(self.model, "feature_names_in_"):
            values = values.to_numpy()
        return values

    def normalize_output(self, labels, scores):
        scores = np.asarray(scores, dtype=float)
        if (scores.shape != (len(labels), len(self.metadata["classes"])) or not np.isfinite(scores).all()
                or (scores < 0).any() or (scores > 1).any() or not np.allclose(scores.sum(axis=1), 1)):
            raise ContractError("SCORE_TYPE_UNSUPPORTED", "유효한 클래스별 확률이 아닙니다.")
        return [(str(label), dict(zip(self.metadata["classes"], row.tolist()))) for label, row in zip(labels, scores)]

    def predict(self, data):
        values = self.preprocess(data)
        return self.normalize_output(self.model.predict(values), self.model.predict_proba(values))

    def infer(self, sample_ids, data):
        if len(sample_ids) != len(data):
            raise ContractError("DATASET_INVALID", "sample_id와 입력 행 수가 다릅니다.")
        results = []
        for i, sample_id in enumerate(sample_ids):
            started = time.perf_counter()
            label, scores = self.predict(data.iloc[[i]])[0]
            results.append({"sample_id": sample_id, "model_id": self.model_id, "predicted_class": label,
                            "scores": scores, "score_type": "probability", "inference_status": "ok",
                            "error_code": "", "inference_duration_ms": (time.perf_counter()-started)*1000})
        return results


def validate_dataset(payload, adapters):
    if not isinstance(payload, bytes) or not payload or len(payload) > MAX_DATA_BYTES:
        raise ContractError("DATASET_INVALID", "비어 있지 않은 25 MiB 이하의 CSV가 필요합니다.")
    try:
        frame = pd.read_csv(io.BytesIO(payload), dtype={"sample_id": str, "true_label": str}, keep_default_na=False)
    except (ValueError, UnicodeError, pd.errors.ParserError) as exc:
        raise ContractError("DATASET_INVALID", f"CSV를 읽지 못했습니다: {exc}") from exc
    if not 1 <= len(frame) <= MAX_ROWS:
        raise ContractError("DATASET_INVALID", "데이터는 1~50,000행이어야 합니다.")
    if not {"sample_id", "true_label"}.issubset(frame.columns):
        raise ContractError("DATASET_INVALID", "sample_id와 true_label 열이 필요합니다.")
    if frame.sample_id.eq("").any() or frame.sample_id.duplicated().any():
        raise ContractError("DATASET_INVALID", "sample_id는 비어 있지 않고 고유해야 합니다.")
    if len(adapters) != 2 or adapters[0].model_id == adapters[1].model_id:
        raise ContractError("MODEL_SELECTION_INVALID", "서로 다른 모델 두 개를 선택하세요.")
    features = adapters[0].metadata["required_features"]
    if any(a.metadata["required_features"] != features for a in adapters):
        raise ContractError("FEATURE_ORDER_MISMATCH", "두 모델의 피처 목록 또는 순서가 다릅니다.")
    classes = adapters[0].metadata["classes"]
    if any(set(a.metadata["classes"]) != set(classes) for a in adapters) or not set(frame.true_label).issubset(classes):
        raise ContractError("CLASS_MAPPING_MISMATCH", "모델 또는 데이터 클래스가 서로 다릅니다.")
    if list(frame.columns[:2]) != ["sample_id", "true_label"]:
        raise ContractError("FEATURE_ORDER_MISMATCH", "CSV는 sample_id, true_label 다음에 피처를 순서대로 포함해야 합니다.")
    for adapter in adapters:
        adapter.validate_input(frame.iloc[:, 2:])
    return frame


def _summary(frame, classes, positive):
    rows = []
    for (condition, model), group in frame.groupby(["condition", "model_id"], sort=False):
        done = group[group.status.eq("ok")]
        row = {"condition": condition, "model_id": model, "planned": len(group), "completed": len(done),
               "failed": len(group) - len(done), "accuracy": None, "precision_macro": None,
               "recall_macro": None, "f1_macro": None, "fn": None, "fp": None, "confusion_matrix": None}
        if len(done):
            truth, pred = done.true_label.tolist(), done.predicted_class.tolist()
            precision, recall, f1, _ = precision_recall_fscore_support(truth, pred, labels=classes,
                                                                      average="macro", zero_division=0)
            matrix = confusion_matrix(truth, pred, labels=classes)
            row.update(accuracy=float(accuracy_score(truth, pred)), precision_macro=float(precision),
                       recall_macro=float(recall), f1_macro=float(f1),
                       confusion_matrix=json.dumps(matrix.tolist()))
            if positive is not None:
                row["fn"] = sum(t == positive and p != positive for t, p in zip(truth, pred))
                row["fp"] = sum(t != positive and p == positive for t, p in zip(truth, pred))
        rows.append(row)
    return rows


def run_comparison(root, before, after, payload, *, seed, noise_std, positive=None, progress=None):
    root = Path(root)
    if not isinstance(seed, int) or not 0 <= seed <= 2**32 - 1 or not np.isfinite(noise_std) or noise_std < 0:
        raise ContractError("CONFIG_INVALID", "seed 또는 잡음 표준편차가 올바르지 않습니다.")
    adapters = [TabularAdapter(root, before), TabularAdapter(root, after)]
    frame = validate_dataset(payload, adapters)
    classes = adapters[0].metadata["classes"]
    if positive is not None and (len(classes) != 2 or positive not in classes):
        raise ContractError("CLASS_MAPPING_MISMATCH", "FN/FP를 위해 유효한 이진 양성 클래스를 지정하세요.")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex
    config = {"seed": seed, "noise_std": float(noise_std), "conditions": ["original", "gaussian_noise"],
              "positive_class": positive, "gate": "complete_and_candidate_accuracy_not_below_baseline_in_each_condition"}
    _event(root, "comparison", "started", {"run_id": run_id, "before": before, "after": after})
    try:
        inputs = frame.iloc[:, 2:].astype(float)
        noise = np.random.default_rng(seed).normal(0, noise_std, inputs.shape)
        records = []
        for condition, data in (("original", inputs), ("gaussian_noise", inputs + noise)):
            for adapter in adapters:
                for i in range(len(frame)):
                    started = time.perf_counter()
                    record = {"run_id": run_id, "condition": condition, "seed": seed,
                              "sample_id": frame.sample_id.iloc[i], "true_label": frame.true_label.iloc[i],
                              "model_id": adapter.model_id, "predicted_class": "", "scores_json": "",
                              "score_type": "probability", "status": "ok", "error_code": "", "error": ""}
                    try:
                        prediction = adapter.infer([record["sample_id"]], data.iloc[[i]])[0]
                        predicted, scores = prediction["predicted_class"], prediction["scores"]
                        if predicted not in classes:
                            raise ContractError("CLASS_MAPPING_MISMATCH", "예측 클래스가 계약에 없습니다.")
                        record.update(predicted_class=predicted, scores_json=json.dumps(scores, ensure_ascii=False))
                    except Exception as exc:
                        record.update(status="failed", error_code=getattr(exc, "code", "INFERENCE_FAILED"), error=str(exc)[:500])
                    record["inference_duration_ms"] = round((time.perf_counter() - started) * 1000, 3)
                    records.append(record)
                    if progress and (len(records) % 64 == 0 or len(records) == len(frame) * 4):
                        progress(len(records), len(frame) * 4)
        predictions = pd.DataFrame(records)
        summaries = _summary(predictions, classes, positive)
        left = predictions[predictions.model_id.eq(before)]
        right = predictions[predictions.model_id.eq(after)]
        pairs = left.merge(right, on=["run_id", "condition", "seed", "sample_id", "true_label"],
                           suffixes=("_before", "_after"), validate="one_to_one")
        ok = pairs.status_before.eq("ok") & pairs.status_after.eq("ok")
        before_right = pairs.predicted_class_before.eq(pairs.true_label)
        after_right = pairs.predicted_class_after.eq(pairs.true_label)
        pairs["change"] = np.select(
            [~ok, ~before_right & after_right, before_right & ~after_right, ~before_right & ~after_right],
            ["failed", "fixed", "new_error", "both_wrong"], default="both_right")
        transitions = [{"condition": c, **{name: int(group.change.eq(name).sum()) for name in
                                          ("fixed", "new_error", "both_wrong", "both_right", "failed")}}
                       for c, group in pairs.groupby("condition", sort=False)]
        failed = int(predictions.status.ne("ok").sum())
        passed = not failed and all(
            next(r for r in summaries if r["condition"] == c and r["model_id"] == after)["accuracy"] >=
            next(r for r in summaries if r["condition"] == c and r["model_id"] == before)["accuracy"]
            for c in config["conditions"])
        versions = {a.model_id: a.metadata for a in adapters}
        report = {"run_id": run_id, "status": "PARTIAL" if failed else "COMPLETED", "decision": "PASS" if passed else "FAIL",
                  "purpose": "local tabular development comparison; not production approval",
                  "created_utc": datetime.now(timezone.utc).isoformat(), "before_model_id": before,
                  "after_model_id": after, "models": versions, "dataset_sha256": _digest(payload),
                  "n_originals": len(frame), "config": config, "config_sha256": _digest(_json(config)),
                  "code_sha256": _digest(Path(__file__).read_bytes()), "failed_predictions": failed,
                  "metrics": summaries, "transitions": transitions, "limitations": [
                      "Metrics exclude failed predictions and are not an approval.",
                      "The data is operator-supplied development data; independence is not established.",
                      "Positive class is operator selected; no FN/FP for multiclass.",
                      "No statistical uncertainty or abstention experiment is included."]}
        runs = root / "runs"
        runs.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".run-", dir=runs) as folder:
            stage = Path(folder)
            (stage / "dataset.csv").write_bytes(payload)
            predictions.to_csv(stage / "predictions.csv", index=False, encoding="utf-8")
            pairs.to_csv(stage / "pairs.csv", index=False, encoding="utf-8")
            pd.DataFrame(summaries).to_csv(stage / "summary.csv", index=False, encoding="utf-8")
            (stage / "report.json").write_bytes(_json(report))
            inventory = {name: _digest((stage / name).read_bytes()) for name in
                         ("dataset.csv", "predictions.csv", "pairs.csv", "summary.csv", "report.json")}
            (stage / "manifest.json").write_bytes(_json({"run_id": run_id, "files": inventory}))
            os.replace(stage, runs / run_id)
        _event(root, "comparison", "completed", {"run_id": run_id, "status": report["status"]})
        return report
    except Exception as exc:
        _event(root, "comparison", "failed", {"run_id": run_id, "error": str(exc)[:500]})
        raise


def load_run(root, run_id):
    if not re.fullmatch(r"\d{8}T\d{6}Z-[a-f0-9]{32}", run_id):
        raise ContractError("RUN_INVALID", "실행 ID가 올바르지 않습니다.")
    folder = Path(root) / "runs" / run_id
    if folder.is_symlink():
        raise ContractError("RUN_INVALID", "실행 경로가 심볼릭 링크입니다.")
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("run_id") != run_id or set(manifest.get("files", {})) != {"dataset.csv", "predictions.csv", "pairs.csv", "summary.csv", "report.json"}:
        raise ContractError("RUN_INVALID", "실행 목록이 일치하지 않습니다.")
    for name, digest in manifest["files"].items():
        file = folder / name
        if file.is_symlink() or _digest(file.read_bytes()) != digest:
            raise ContractError("RUN_INTEGRITY_ERROR", f"실행 파일 해시 불일치: {name}")
    report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    if report["run_id"] != run_id:
        raise ContractError("RUN_INTEGRITY_ERROR", "보고서의 실행 ID가 일치하지 않습니다.")
    return report


def deactivate_model(root, model_id, *, auth_token=None):
    root = Path(root)
    _id(model_id)
    audit = security.begin_action(auth_token, "tabular_model_deactivation", {"model_id": model_id})
    registry = root / "models"
    with FileLock(str(registry / ".lock"), timeout=10), FileLock(str(root / ".activation.lock"), timeout=10):
        if active_model(root) == model_id:
            raise ContractError("MODEL_ACTIVE", "활성 모델은 등록 해제할 수 없습니다.")
        folder = registry / model_id
        if folder.is_symlink():
            raise ContractError("REGISTRY_INVALID", "모델 경로가 심볼릭 링크입니다.")
        meta = folder / "metadata.json"
        metadata = json.loads(meta.read_text(encoding="utf-8"))
        if metadata.get("model_id") != model_id or metadata.get("status") != "ready":
            raise ContractError("MODEL_INVALID", "활성 등록 모델이 아닙니다.")
        metadata["status"] = "inactive"
        temp = None
        try:
            with tempfile.NamedTemporaryFile(dir=folder, prefix=".metadata-", delete=False) as stream:
                temp = Path(stream.name)
                stream.write(_json(metadata))
            os.replace(temp, meta)
        finally:
            if temp is not None:
                temp.unlink(missing_ok=True)
    _event(root, "model_deactivation", "completed", {"model_id": model_id})
    security.finish_action(audit, "completed")


def active_model(root):
    path = Path(root) / "active.json"
    if not path.exists():
        return None
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("schema") != 1 or not ID.fullmatch(state.get("model_id", "")):
        raise ContractError("ACTIVE_INVALID", "로컬 활성 모델 정보가 올바르지 않습니다.")
    return state["model_id"]


def replace_model(root, run_id, *, actor, confirmed, auth_token=None):
    root = Path(root)
    if confirmed is not True or not actor.strip() or len(actor.strip()) > 120:
        raise ContractError("CONFIRMATION_REQUIRED", "확인자와 명시적 동의가 필요합니다.")
    principal = security.authorize(auth_token)
    if actor.strip() != principal["username"]:
        raise security.SecurityError("확인자는 로그인한 계정과 일치해야 합니다.")
    audit = security.begin_action(auth_token, "tabular_model_replacement", {"run_id": run_id})
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root / ".activation.lock"), timeout=10):
        report = load_run(root, run_id)
        before, after = report["before_model_id"], report["after_model_id"]
        if active_model(root) not in (None, before):
            raise ContractError("ACTIVE_CHANGED", "현재 로컬 모델이 달라졌습니다. 다시 비교하세요.")
        if report["status"] != "COMPLETED" or report["decision"] != "PASS" or report["failed_predictions"]:
            raise ContractError("EVALUATION_NOT_PASSED", "완료된 통과 비교가 필요합니다.")
        if (report["code_sha256"] != _digest(Path(__file__).read_bytes())
                or report["config_sha256"] != _digest(_json(report["config"]))
                or report["dataset_sha256"] != _digest((root / "runs" / run_id / "dataset.csv").read_bytes())):
            raise ContractError("EVALUATION_CHANGED", "평가 코드, 설정 또는 데이터가 실행 기록과 다릅니다.")
        for model in (before, after):
            if TabularAdapter(root, model).metadata != report["models"][model]:
                raise ContractError("MODEL_INTEGRITY_ERROR", "현재 모델 계약과 평가 당시 계약이 다릅니다.")
        state = {"schema": 1, "model_id": after, "previous_model_id": before, "run_id": run_id,
                 "actor_authenticated": principal["username"], "created_utc": datetime.now(timezone.utc).isoformat(),
                 "scope": "local workspace selection; no production deployment or approval"}
        _event(root, "model_replacement", "started", {"run_id": run_id, "before": before, "after": after})
        temp = None
        try:
            with tempfile.NamedTemporaryFile(dir=root, prefix=".active-", delete=False) as stream:
                temp = Path(stream.name)
                stream.write(_json(state))
            os.replace(temp, root / "active.json")
        finally:
            if temp is not None:
                temp.unlink(missing_ok=True)
        _event(root, "model_replacement", "completed", state)
        security.finish_action(audit, "completed")
        return state
