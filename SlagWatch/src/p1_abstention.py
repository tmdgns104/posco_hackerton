"""Frozen confidence policies evaluated on separately recorded tabular data."""
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import uuid

import pandas as pd

from . import p1_tabular as tabular

TARGETS = (0.5, 0.7, 0.8, 0.9, 1.0)
POLICY_ID = re.compile(r"policy-[a-f0-9]{32}\Z")


def _predictions(root, report, model_id, condition):
    if model_id not in report["models"] or condition not in report["config"]["conditions"]:
        raise tabular.ContractError("POLICY_CONTRACT_MISMATCH", "모델 또는 조건이 실행 기록에 없습니다.")
    if report["models"][model_id]["score_type"] != "probability":
        raise tabular.ContractError("SCORE_TYPE_UNSUPPORTED", "클래스별 확률만 정책에 지원합니다.")
    path = Path(root) / "runs" / report["run_id"] / "predictions.csv"
    data = pd.read_csv(path, dtype=str, keep_default_na=False)
    data = data[data.model_id.eq(model_id) & data.condition.eq(condition)].copy()
    if len(data) != report["n_originals"] or data.sample_id.duplicated().any():
        raise tabular.ContractError("POLICY_DATA_INVALID", "실행의 모델별 사례가 일치하지 않습니다.")
    return data


def _confidence(row):
    try:
        scores = json.loads(row["scores_json"])
        value = float(scores[row["predicted_class"]])
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise tabular.ContractError("SCORE_TYPE_UNSUPPORTED", "예측 클래스의 확률이 올바르지 않습니다.") from exc
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise tabular.ContractError("SCORE_TYPE_UNSUPPORTED", "확률은 0에서 1 사이여야 합니다.")
    return value


def freeze_policy(root, run_id, model_id, condition, target, *, name, version):
    if target not in TARGETS or not name.strip() or not version.strip() or len(name) > 120 or len(version) > 80:
        raise tabular.ContractError("POLICY_INVALID", "정책 이름·버전과 지원되는 목표 coverage가 필요합니다.")
    report = tabular.load_run(root, run_id)
    data = _predictions(root, report, model_id, condition)
    if data.status.ne("ok").any():
        raise tabular.ContractError("POLICY_DEVELOPMENT_INCOMPLETE", "개발 데이터의 해당 모델 추론이 모두 완료되어야 합니다.")
    scores = sorted((_confidence(row) for _, row in data.iterrows()), reverse=True)
    threshold = scores[math.ceil(target * len(scores)) - 1]
    achieved = sum(s >= threshold for s in scores) / len(scores)
    policy_id = "policy-" + uuid.uuid4().hex
    policy = {
        "schema": 1, "policy_id": policy_id, "name": name.strip(), "version": version.strip(),
        "created_utc": datetime.now(timezone.utc).isoformat(), "status": "FROZEN",
        "model_id": model_id, "model_identity": report["models"][model_id], "condition": condition,
        "score_type": "predicted_class_probability", "rule": "automatic if score >= threshold",
        "target_coverage": target, "threshold": threshold, "development_coverage": achieved,
        "development_run_id": run_id, "development_dataset_sha256": report["dataset_sha256"],
        "development_sample_ids_sha256": hashlib.sha256(
            "\n".join(sorted(data.sample_id)).encode("utf-8")).hexdigest(),
    }
    folder = Path(root) / "policies"
    folder.mkdir(parents=True, exist_ok=True)
    payload = tabular._json(policy)
    envelope = {"policy": policy, "sha256": hashlib.sha256(payload).hexdigest()}
    temp = None
    try:
        with tempfile.NamedTemporaryFile(dir=folder, prefix=".policy-", delete=False) as stream:
            temp = Path(stream.name)
            stream.write(tabular._json(envelope))
        os.replace(temp, folder / (policy_id + ".json"))
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)
    tabular._event(root, "policy_freeze", "completed", {"policy_id": policy_id, "run_id": run_id})
    return policy


def load_policy(root, policy_id):
    if not isinstance(policy_id, str) or not POLICY_ID.fullmatch(policy_id):
        raise tabular.ContractError("POLICY_INVALID", "정책 ID가 올바르지 않습니다.")
    path = Path(root) / "policies" / (policy_id + ".json")
    if path.is_symlink():
        raise tabular.ContractError("POLICY_INVALID", "정책이 심볼릭 링크입니다.")
    envelope = json.loads(path.read_text(encoding="utf-8"))
    policy = envelope["policy"]
    if (policy.get("policy_id") != policy_id or policy.get("status") != "FROZEN"
            or hashlib.sha256(tabular._json(policy)).hexdigest() != envelope.get("sha256")):
        raise tabular.ContractError("POLICY_INVALID", "동결된 정책이 변경되었습니다.")
    return policy


def metrics(data, policy, positive):
    eligible = data[data.status.eq("ok")].copy()
    eligible["confidence"] = [_confidence(row) for _, row in eligible.iterrows()]
    eligible["automatic"] = eligible.confidence.ge(policy["threshold"])
    eligible["wrong"] = eligible.predicted_class.ne(eligible.true_label)
    auto = eligible[eligible.automatic]
    held = eligible[~eligible.automatic]
    ratio = lambda n, d: n / d if d else None
    result = {
        "policy_id": policy["policy_id"], "model_id": policy["model_id"],
        "target_coverage": policy["target_coverage"], "threshold": policy["threshold"],
        "eligible": len(eligible), "failed": len(data)-len(eligible), "automatic": len(auto), "held": len(held),
        "coverage": ratio(len(auto), len(eligible)), "abstention_rate": ratio(len(held), len(eligible)),
        "automatic_errors": int(auto.wrong.sum()), "automatic_error_rate": ratio(int(auto.wrong.sum()), len(auto)),
        "unnecessary_abstentions": int((~held.wrong).sum()),
        "unnecessary_abstention_rate": ratio(int((~held.wrong).sum()), len(held)),
        "fn": None, "fn_denominator": None, "fn_rate": None, "fp": None, "fp_denominator": None, "fp_rate": None,
    }
    if positive is not None:
        actual_pos = auto.true_label.eq(positive)
        predicted_pos = auto.predicted_class.eq(positive)
        fn, fp = int((actual_pos & ~predicted_pos).sum()), int((~actual_pos & predicted_pos).sum())
        result.update(fn=fn, fn_denominator=int(actual_pos.sum()), fn_rate=ratio(fn, int(actual_pos.sum())),
                      fp=fp, fp_denominator=int((~actual_pos).sum()), fp_rate=ratio(fp, int((~actual_pos).sum())))
    result["coverage_gap"] = (result["coverage"] - policy["target_coverage"]
                              if result["coverage"] is not None else None)
    eligible["policy_id"] = policy["policy_id"]
    return result, eligible


def evaluate_policies(root, policy_ids, evaluation_run_id):
    if not policy_ids or len(set(policy_ids)) != len(policy_ids):
        raise tabular.ContractError("POLICY_INVALID", "중복 없이 하나 이상의 동결 정책을 선택하세요.")
    report = tabular.load_run(root, evaluation_run_id)
    policies = [load_policy(root, policy_id) for policy_id in policy_ids]
    if len({(p["condition"], p["development_dataset_sha256"]) for p in policies}) != 1:
        raise tabular.ContractError("POLICY_COMPARISON_MISMATCH", "같은 개발 데이터와 조건으로 동결된 정책만 비교할 수 있습니다.")
    output, case_frames = [], []
    for policy in policies:
        development = tabular.load_run(root, policy["development_run_id"])
        if development["dataset_sha256"] != policy["development_dataset_sha256"]:
            raise tabular.ContractError("POLICY_DATA_INVALID", "개발 데이터 해시가 변경되었습니다.")
        if (report["dataset_sha256"] == development["dataset_sha256"]
                or report["models"].get(policy["model_id"]) != policy["model_identity"]):
            raise tabular.ContractError("POLICY_EVALUATION_INVALID", "다른 평가 데이터와 동일한 모델 계약이 필요합니다.")
        dev = _predictions(root, development, policy["model_id"], policy["condition"])
        evaluation = _predictions(root, report, policy["model_id"], policy["condition"])
        if set(dev.sample_id) & set(evaluation.sample_id):
            raise tabular.ContractError("POLICY_EVALUATION_INVALID", "개발·평가 sample_id가 겹칩니다.")
        result, cases = metrics(evaluation, policy, report["config"]["positive_class"])
        result["development_coverage"] = policy["development_coverage"]
        output.append(result)
        case_frames.append(cases)
    experiment_id = "abstention-" + uuid.uuid4().hex
    result = {"experiment_id": experiment_id, "evaluation_run_id": evaluation_run_id,
              "created_utc": datetime.now(timezone.utc).isoformat(), "policies": policies, "metrics": output,
              "limitations": ["Policy evaluation is separate from model replacement.",
                              "Held cases are not assumed to be reviewed correctly.",
                              "Disjoint IDs do not prove independent datasets.",
                              "Thresholds use development scores without evaluation labels."]}
    experiments = Path(root) / "policy_experiments"
    experiments.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".experiment-", dir=experiments) as folder:
        stage = Path(folder)
        (stage / "report.json").write_bytes(tabular._json(result))
        pd.DataFrame(output).to_csv(stage / "metrics.csv", index=False, encoding="utf-8")
        pd.concat(case_frames, ignore_index=True).to_csv(stage / "cases.csv", index=False, encoding="utf-8")
        files = {name: hashlib.sha256((stage / name).read_bytes()).hexdigest()
                 for name in ("report.json", "metrics.csv", "cases.csv")}
        (stage / "manifest.json").write_bytes(tabular._json({"experiment_id": experiment_id, "files": files}))
        os.replace(stage, experiments / experiment_id)
    tabular._event(root, "policy_experiment", "completed", {"experiment_id": experiment_id, "run_id": evaluation_run_id})
    return result


def load_experiment(root, experiment_id):
    if not isinstance(experiment_id, str) or not re.fullmatch(r"abstention-[a-f0-9]{32}", experiment_id):
        raise tabular.ContractError("POLICY_INVALID", "실험 ID가 올바르지 않습니다.")
    folder = Path(root) / "policy_experiments" / experiment_id
    if folder.is_symlink():
        raise tabular.ContractError("POLICY_INVALID", "실험 경로가 심볼릭 링크입니다.")
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("experiment_id") != experiment_id or set(manifest.get("files", {}))
            != {"report.json", "metrics.csv", "cases.csv"}):
        raise tabular.ContractError("POLICY_INVALID", "실험 목록이 일치하지 않습니다.")
    for name, digest in manifest["files"].items():
        path = folder / name
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise tabular.ContractError("POLICY_INVALID", f"실험 파일이 변경되었습니다: {name}")
    result = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    if result.get("experiment_id") != experiment_id:
        raise tabular.ContractError("POLICY_INVALID", "실험 ID가 일치하지 않습니다.")
    return result
