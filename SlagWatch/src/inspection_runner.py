"""Paired image evaluation. No dependency on prior predictions or model-specific preprocessing.

A project supplies labeled rows, a scope guard, transformations and two ordered image
adapters. Adapters see only immutable native image pixels. The runner uses no prediction
cache. CSVs keep one scheduled row even when decoding, transforming or inference fails.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from numbers import Integral, Real
from pathlib import Path
import platform
import uuid

import numpy as np
import pandas as pd

BATCH_SIZE = 16
SCHEMA = 1
DATA_COLS = ("sample_id", "image_path", "true_label", "group_id", "split", "source_id", "file_sha256")
BASE = ["sample_id", "group_id", "true_label", "condition", "seed", "split", "source_id",
        "file_sha256", "original_input_hash", "raw_input_hash"]
LAYOUT = ["original_input_shape", "original_input_dtype", "raw_input_shape", "raw_input_dtype"]
BINARY_CHANGES = ("fixed_fn", "new_fn", "fixed_fp", "new_fp", "both_wrong", "both_right")
MULTI_CHANGES = ("fixed", "new_error", "both_wrong", "both_right")
FILES = {"data_manifest.csv", "predictions.csv", "pairs.csv", "summary.csv", "changes.csv",
         "classes.csv", "changes_by_class.csv", "failures.csv", "decision.json", "protocol.json"}
STATUS_LABEL = {"pass": "시제품 기준 충족", "fail": "기준 미달", "incomplete": "평가 불완전"}


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, indent=2) + "\n"


def _put(path, value):
    path.write_text(_json(value), encoding="utf-8", newline="\n")


def _digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def hash_raw(raw):
    """Protocol hash: C-order native float32 pixel bytes (matches frozen NEU)."""
    if not isinstance(raw, np.ndarray) or raw.dtype != np.float32:
        raise ValueError("Expected native float32 image for hashing")
    return hashlib.sha256(memoryview(np.ascontiguousarray(raw)).cast("B")).hexdigest()


def _immutable(raw, shape=None):
    if (not isinstance(raw, np.ndarray) or raw.dtype != np.float32 or raw.ndim not in (2, 3)
            or not raw.size or (raw.ndim == 3 and raw.shape[-1] not in (1, 3, 4))
            or (shape is not None and raw.shape != shape) or not np.isfinite(raw).all()
            or raw.min() < 0 or raw.max() > 1):
        raise ValueError("Expected finite native float32 image in [0,1], with unchanged spatial shape")
    # Immutable bytes own the buffer: even setflags(write=True) cannot give one adapter
    # access to the other's pixels. Keep a protected original for comparison of each adapter's view metadata.
    return np.frombuffer(raw.tobytes(order="C"), dtype=np.float32).reshape(raw.shape)


def execution_environment():
    versions = {}
    for package in ("numpy", "pandas", "Pillow", "torch", "torchvision", "scikit-learn", "opencv-python-headless"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {"os": platform.platform(), "python": platform.python_version(),
            "machine": platform.machine(), "versions": versions}


def _population(project, split, classes, ids):
    rows = project.rows(split).copy()
    if rows.empty or not set(DATA_COLS) <= set(rows):
        raise ValueError("Labeled manifest required: " + ", ".join(DATA_COLS))
    for col in DATA_COLS:
        if rows[col].isna().any():
            raise ValueError(f"Missing {col} in data manifest")
        rows[col] = rows[col].map(str)
        if rows[col].map(lambda x: not x.strip() or "\x00" in x).any():
            raise ValueError(f"Empty or invalid {col}")
    if rows.sample_id.duplicated().any() or not rows.split.eq(split).all():
        raise ValueError("Duplicate sample_id or wrong split")
    if not rows.true_label.isin(classes).all():
        raise ValueError("Unknown or missing true_label: performance evaluation requires ground truth")
    if not rows.file_sha256.str.fullmatch(r"[0-9a-f]{64}").all():
        raise ValueError("Invalid file_sha256 in data manifest")
    if ids is not None:
        if isinstance(ids, str):
            raise ValueError("sample_ids must be an iterable of sample IDs")
        requested = [str(i) for i in ids]
        if not requested or len(requested) != len(set(requested)) or set(requested) - set(rows.sample_id):
            raise ValueError("Empty, duplicate, or unknown sample_ids")
        rows = rows[rows.sample_id.isin(requested)]
    return rows.reset_index(drop=True)


def _fail(reason):
    return dict(status="failed", error=str(reason)[:1000], predicted_label="", scores="{}", score_type="none")


def _exc(stage, error):
    return f"{stage}: {type(error).__name__}: {error}"[:1000]


def validate_prediction(output, meta, model_id, classes):
    if not isinstance(output, Mapping) or output.get("status") != "ok":
        reason = (output.get("error") or output.get("reason") or output.get("status")
                  if isinstance(output, Mapping) else "invalid output")
        raise ValueError("Adapter did not complete this prediction: " + str(reason))
    for field, expected in (("model_id", model_id), ("sample_id", meta["sample_id"]),
                            ("raw_input_hash", meta["raw_input_hash"])):
        if field in output and output[field] != expected:
            raise ValueError(f"Adapter returned mismatched {field}")
    label = output.get("predicted_label")
    if label not in classes:
        raise ValueError("Unknown or missing predicted_label")
    scores = output.get("scores")
    scores = {} if scores is None else scores
    score_type = output.get("score_type")
    if not isinstance(score_type, str) or not score_type.strip():
        raise ValueError("Missing score_type")
    if not isinstance(scores, Mapping):
        raise ValueError("scores must map class names to numeric scores")
    keys = set(scores)
    if keys and keys != set(classes) and not (set(classes) == {"normal", "defect"} and keys == {"defect"}):
        raise ValueError("Incomplete or invalid score class mapping")
    if not keys and score_type not in ("none", "label_only"):
        raise ValueError("Missing scores for declared score type")
    if keys and score_type in ("none", "label_only"):
        raise ValueError("Declared score type conflicts with scores")
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Real) or not math.isfinite(v)
           for v in scores.values()):
        raise ValueError("Non-finite or non-numeric score")
    return dict(status="ok", error="", predicted_label=label, score_type=score_type,
                scores=json.dumps({k: float(v) for k, v in scores.items()}, allow_nan=False, sort_keys=True))


def _check_view(entry):
    """Restore ndarray metadata if a badly behaved adapter changed its readonly view."""
    _, raw, trial = entry
    if (trial.shape == raw.shape and trial.dtype == raw.dtype and trial.strides == raw.strides
            and not trial.flags.writeable and np.shares_memory(trial, raw)):
        return True
    entry[2] = raw.view()
    return False


def _batch(adapter, entries, model_id, classes):
    """Isolate a defective sample when an adapter rejects the entire batch."""
    if not entries:
        return []
    try:
        outputs = list(adapter.predict([entry[2] for entry in entries]))
        if len(outputs) != len(entries):
            raise ValueError(f"Adapter returned {len(outputs)} rows for {len(entries)} images")
    except Exception as exc:
        retry = []
        for entry in entries:
            if not _check_view(entry):
                retry.append(_fail("predict: adapter modified input metadata"))
            elif len(entries) == 1:
                retry.append(_fail(_exc("predict", exc)))
            else:
                # Never align a shortened batch positionally. Retry originals independently.
                retry.extend(_batch(adapter, [entry], model_id, classes))
        return retry
    out = []
    for result, entry in zip(outputs, entries, strict=True):
        try:
            if not _check_view(entry):
                raise ValueError("Adapter modified input metadata")
            out.append(validate_prediction(result, entry[0], model_id, classes))
        except Exception as exc:
            out.append(_fail(_exc("output", exc)))
    return out


def _change_name(kind, truth, old, new):
    if old["status"] != "ok" or new["status"] != "ok":
        return "failed"
    a, b = old["predicted_label"] == truth, new["predicted_label"] == truth
    if a == b:
        return "both_right" if a else "both_wrong"
    if kind == "binary":
        return (("fixed_fn" if truth == "defect" else "fixed_fp") if b else
                ("new_fn" if truth == "defect" else "new_fp"))
    return "fixed" if b else "new_error"


def _infer(project, rows, conditions, models, classes, progress):
    predictions, pairs = [], []
    total = len(rows) * len(conditions)
    before_id, after_id = models
    if progress:
        progress(0, total)
    for start in range(0, len(rows), BATCH_SIZE):
        sources = []
        for _, row in rows.iloc[start:start+BATCH_SIZE].iterrows():
            try:
                original = _immutable(project.load(row))
                sources.append((row, original, hash_raw(original), ""))
            except Exception as exc:
                sources.append((row, None, "", _exc("load", exc)))
        for condition, seed in conditions:
            planned, ready = [], []
            for row, original, original_hash, error in sources:
                meta = {k: row[k] for k in DATA_COLS if k != "image_path"}
                meta.update(condition=condition, seed=seed, original_input_hash=original_hash, raw_input_hash="",
                            original_input_shape="" if original is None else json.dumps(list(original.shape), separators=(",", ":")),
                            original_input_dtype="" if original is None else "float32",
                            raw_input_shape="", raw_input_dtype="")
                if not error:
                    try:
                        raw = _immutable(project.change(original.view(), row, condition, seed), original.shape)
                        meta["raw_input_hash"] = hash_raw(raw)
                        meta["raw_input_shape"] = json.dumps(list(raw.shape), separators=(",", ":"))
                        meta["raw_input_dtype"] = "float32"
                        ready.append([meta, raw, raw.view()])
                    except Exception as exc:
                        error = _exc("transform", exc)
                planned.append((meta, error))
            results = {key: iter(_batch(model, ready, key, classes)) for key, model in models.items()}
            for meta, error in planned:
                a, b = (_fail(error) if error else next(results[key]) for key in models)
                for key, prediction in ((before_id, a), (after_id, b)):
                    predictions.append({**meta, "model_id": key, **prediction})
                change = _change_name(project.kind, meta["true_label"], a, b)
                pairs.append({**meta, "before_model_id": before_id, "after_model_id": after_id,
                              "before_pred": a["predicted_label"], "after_pred": b["predicted_label"],
                              "before_status": a["status"], "after_status": b["status"],
                              "change": change, "status": "failed" if change == "failed" else "ok"})
            if progress:
                progress(len(pairs), total)
    return pd.DataFrame(predictions), pd.DataFrame(pairs)


def _groups(frame):
    yield {"scope": "all", "condition": "ALL", "seed": -1}, frame
    for (name, seed), group in frame.groupby(["condition", "seed"], sort=False):
        yield {"scope": "cell", "condition": name, "seed": int(seed)}, group


def _rate(n, d):
    return n / d if d else None


def _model_metrics(meta, group, model, classes, binary, side=None, constant=None):
    y = group.true_label.to_numpy()
    p = group[side + "_pred"].to_numpy() if side else np.repeat(constant, len(y))
    ok = group[side + "_status"].eq("ok").to_numpy() if side else np.ones(len(y), dtype=bool)
    right = ok & (p == y)
    count = lambda v: int(np.count_nonzero(v))
    n, done = len(y), count(ok)
    summary = dict(**meta, model_id=model, n_originals=group.sample_id.nunique(), planned=n,
                   completed=done, failed=n-done, correct=count(right),
                   accuracy=_rate(count(right), n) if done == n else None)
    per_class = []
    for label in classes:
        truth = y == label
        called = ok & (p == label)
        planned, finished, correct = count(truth), count(truth & ok), count(truth & right)
        f1 = _rate(2*correct, planned+count(called)) if done == n else None
        per_class.append(dict(**meta, model_id=model, class_label=label,
                              n_originals=group.loc[truth, "sample_id"].nunique(),
                              planned=planned, completed=finished, failed=planned-finished, correct=correct,
                              predicted_as_class=count(called), f1=f1,
                              recall=_rate(correct, planned) if finished == planned else None))
    # Fixed class list. A missing true class cannot qualify an unrepresentative evaluation.
    summary["macro_f1"] = (sum(r["f1"] for r in per_class)/len(classes)
                           if all(r["planned"] and r["f1"] is not None for r in per_class) else None)
    if binary:
        pos, neg = y == "defect", y == "normal"
        tp, fn = count(pos & right), count(pos & ok & ~right)
        tn, fp = count(neg & right), count(neg & ok & ~right)
        nd, nn = count(pos), count(neg)
        summary.update(tp=tp, fn=fn, fp=fp, tn=tn, n_defect=nd, n_normal=nn,
                       completed_defect=tp+fn, completed_normal=fp+tn,
                       failed_defect=nd-tp-fn, failed_normal=nn-fp-tn,
                       recall=_rate(tp, nd) if tp+fn == nd else None,
                       miss_rate=_rate(fn, nd) if tp+fn == nd else None,
                       fpr=_rate(fp, nn) if fp+tn == nn else None)
    return summary, per_class


def _tables(pairs, cfg):
    binary = cfg["kind"] == "binary"
    summaries, classes, changes, by_class, baselines = [], [], [], [], []
    kinds = BINARY_CHANGES if binary else MULTI_CHANGES
    for meta, group in _groups(pairs):
        for side in ("before", "after"):
            result, per_class = _model_metrics(meta, group, cfg[f"{side}_model_id"], cfg["classes"],
                                               binary, side=side)
            summaries.append(result)
            classes.extend(per_class)
        if binary:
            for label in ("normal", "defect"):
                base, _ = _model_metrics(meta, group, f"all-{label}", cfg["classes"], True, constant=label)
                baselines.append(base)
        for label in [None, *cfg["classes"]]:
            subset = group[group.true_label == label] if label is not None else group
            counts = subset.change.value_counts()
            failures = int(counts.get("failed", 0))
            row = dict(**meta, n_originals=subset.sample_id.nunique(), planned=len(subset),
                       completed=len(subset)-failures, failed=failures,
                       **{k: int(counts.get(k, 0)) for k in kinds})
            if sum(row[k] for k in kinds) != row["completed"]:
                raise ValueError("Invalid or missing paired change")
            if label is None:
                changes.append(row)
            else:
                by_class.append({**row, "class_label": label})
    return {"summary": summaries, "classes": classes, "changes": changes,
            "changes_by_class": by_class, "baselines": baselines}


def _criterion(name, metric, value, op, limit, *, cell=("ALL", -1), title=""):
    if value is None:
        passed = None
        explanation = "계산 불가 (분모 없음 또는 미완료)"
    else:
        passed = (value >= limit-1e-12) if op == ">=" else (value <= limit+1e-12)
        explanation = ("충족" if passed else "미달") + f" ({value:.5g} {op} {limit:.5g})"
    return {"id": name, "condition": cell[0], "seed": int(cell[1]), "metric": metric, "value": value,
            "operator": op, "limit": limit, "passed": passed,
            "reason": f"{cell[0]} / {cell[1]} · {title or metric}: {explanation}"}


def _binary_criteria(cfg, tables):
    by_key = {(r["model_id"], r["condition"], r["seed"]): r for r in tables["summary"]}
    before, after = cfg["before_model_id"], cfg["after_model_id"]
    cells = cfg["conditions"]
    result = []
    for name, seed in cells:
        b, a = (by_key[(key, name, seed)] for key in (before, after))
        limit = .02 if name == "clean" else .05
        for metric, op, threshold, title in (
            ("recall", ">=", .80, "후보 불량 재현율"), ("fpr", "<=", .10, "후보 정상 오경보율"),
        ):
            result.append(_criterion("absolute_" + metric, metric, a[metric], op, threshold,
                                     cell=(name, seed), title=title))
            value = None if a[metric] is None or b[metric] is None else (
                b[metric]-a[metric] if metric == "recall" else a[metric]-b[metric])
            result.append(_criterion("relative_" + metric, metric + "_deterioration", value, "<=", limit,
                                     cell=(name, seed), title=title + " 악화 폭"))
    both = [by_key[(k, "ALL", -1)] for k in (before, after)]
    total_cells = len(cells)
    complete = all(r["failed"] == 0 and r["n_defect"] and r["n_normal"] for r in both)
    representative = {"weight_per_cell": 1 / total_cells, "cell_count": total_cells,
                      "n_originals": cfg["n_originals"], "planned_pairs": cfg["planned_pairs"],
                      "rule": "Each original has equal weight; average all declared condition/seed cells within each original.",
                      "models": {}}
    for r in both:
        representative["models"][r["model_id"]] = {
            "fn_cell_count": r["fn"], "fp_cell_count": r["fp"],
            "weighted_fn": r["fn"]/total_cells if complete else None,
            "weighted_fp": r["fp"]/total_cells if complete else None}
    passed = None
    if complete:
        b, a = both
        passed = ((a["fn"] < b["fn"] and a["fp"] <= b["fp"]) or
                  (a["fp"] < b["fp"] and a["fn"] <= b["fn"]))
    representative["passed"] = passed
    result.append({"id": "representative_improvement", "condition": "ALL", "seed": -1,
                   "metric": "fn_fp_pareto", "value": None if passed is None else int(passed),
                   "limit": 1, "operator": ">=", "passed": passed,
                   "reason": "전체 조건 동일 비중 · FN 또는 FP 감소 및 다른 오류 비증가: " +
                             ("계산 불가" if passed is None else "충족" if passed else "미달")})
    return result, representative


def _multi_criteria(cfg, tables):
    """Evaluate live six-class rows using the existing G1 definitions, with no binary gate."""
    goals = cfg["protocol"].get("goals", {})
    required = {"min_baseline_clean", "min_clean", "max_clean_loss", "min_noise_gain",
                "min_noise_mean", "max_class_clean_recall_loss", "max_per_sigma_loss"}
    if not required <= set(goals) or len(cfg["conditions"]) < 2:
        return [], {}
    before, after = cfg["before_model_id"], cfg["after_model_id"]
    # Frozen G1 adapter exposes original first, then one condition name per sigma with its seeds.
    original = cfg["conditions"][0]
    clean, seed = original
    rows = {(r["model_id"], r["condition"], r["seed"]): r for r in tables["summary"] if r["scope"] == "cell"}
    b, a = [rows[(mid, clean, seed)]["macro_f1"] for mid in (before, after)]
    delta = lambda x, y: None if x is None or y is None else x-y
    mean = lambda xs: sum(xs)/len(xs) if xs and None not in xs else None
    result = [_criterion("baseline_clean", "macro_f1", b, ">=", goals["min_baseline_clean"], cell=original),
              _criterion("clean_f1", "macro_f1", a, ">=", goals["min_clean"], cell=original),
              _criterion("clean_drop", "macro_f1_loss", delta(b, a), "<=", goals["max_clean_loss"], cell=original)]
    recalls = {(r["model_id"], r["class_label"]): r["recall"] for r in tables["classes"]
               if r["scope"] == "cell" and r["condition"] == clean and r["seed"] == seed}
    for label in cfg["classes"]:
        loss = delta(recalls[before, label], recalls[after, label])
        result.append(_criterion("class_recall", "recall_loss", loss, "<=", goals["max_class_clean_recall_loss"],
                                 cell=original, title=f"{label} 재현율 하락"))
    perturbed = {}
    for name, s in cfg["conditions"][1:]:
        group = perturbed.setdefault(name, [[], []])
        for vals, mid in zip(group, (before, after)):
            vals.append(rows[(mid, name, s)]["macro_f1"])
    noise, gain = [], []
    for name, (base_vals, cand_vals) in perturbed.items():
        cand = mean(cand_vals)
        improvement = delta(cand, mean(base_vals))
        noise.append(cand)
        gain.append(improvement)
        result.append(_criterion("per_sigma", "macro_f1_gain", improvement, ">=", -goals["max_per_sigma_loss"],
                                 cell=(name, -1), title="시드 평균 F1 차이"))
    result.extend([_criterion("avg_gain", "macro_f1_gain", mean(gain), ">=", goals["min_noise_gain"]),
                   _criterion("avg_noise_f1", "macro_f1", mean(noise), ">=", goals["min_noise_mean"])])
    return result, {"rule": "G1: average seeds within each Gaussian level, then average levels."}


def _decision(cfg, tables):
    binary = cfg["kind"] == "binary"
    criteria, representative = (_binary_criteria if binary else _multi_criteria)(cfg, tables)
    blockers = []
    if cfg["scope"] != "full":
        blockers.append("부분집합 실행: 전체 평가에 대한 채택 판정은 불가합니다.")
    if cfg["failed_pairs"]:
        blockers.append(f"예정 {cfg['planned_pairs']}건 중 {cfg['failed_pairs']}건의 짝지은 판정이 미완료입니다.")
    if binary:
        names = Counter(c[0] for c in cfg["conditions"])
        if names != Counter({"clean": 1, "noise_weak": 2, "noise_strong": 2, "dark": 1, "bright": 1}):
            blockers.append("원본·잡음 2강도/각 2시드·밝기 2조건의 필수 7개 평가 셀이 모두 필요합니다.")
    if not criteria or any(c["passed"] is None for c in criteria):
        blockers.append("판정 기준이나 계산 가능한 클래스별 분모가 부족합니다.")
    failed = [c["reason"] for c in criteria if c["passed"] is False]
    status = "incomplete" if blockers else "fail" if failed else "pass"
    return dict(status=status, label=STATUS_LABEL[status], project_id=cfg["project_id"],
                split=cfg["split"], scope=cfg["scope"], official=cfg["official"],
                decision_scope="해당 프로젝트·모델·분할에서 실행한 시제품 기준; 현장 채택 또는 배포를 의미하지 않습니다.",
                reasons=blockers + failed or ["사전에 정한 모든 시험 기준을 충족했습니다."],
                criteria=criteria, representative=representative, frozen_status=cfg.get("frozen_status"))


def _check_rows(data, predictions, pairs, cfg):
    """Conserve every scheduled key, and bind prediction/status/truth to the same raw input."""
    names = [cfg["before_model_id"], cfg["after_model_id"]]
    cells = {tuple(v) for v in cfg["conditions"]}
    planned = len(data)*len(cells)
    keys = ["sample_id", "condition", "seed"]
    classes = set(cfg["classes"])
    if (not planned or len(data) != cfg["n_originals"] or data.sample_id.duplicated().any()
            or not data.split.eq(cfg["split"]).all() or not data.true_label.isin(classes).all()
            or len(pairs) != planned or planned != cfg["planned_pairs"]
            or pairs.duplicated(keys).any() or set(pairs.sample_id) != set(data.sample_id)
            or set(zip(pairs.condition, pairs.seed)) != cells
            or len(predictions) != 2*planned or predictions.duplicated(keys+["model_id"]).any()
            or set(predictions.model_id) != set(names)):
        raise ValueError("Missing, repeated, or out-of-scope evaluation rows")
    reference = data.set_index("sample_id").loc[pairs.sample_id]
    for field in ("true_label", "file_sha256", "source_id", "group_id", "split"):
        if not np.array_equal(pairs[field], reference[field]):
            raise ValueError("Paired row differs from data manifest: " + field)
    indexed = pairs.set_index(keys)
    for side, model_id in zip(("before", "after"), names, strict=True):
        subset = predictions[predictions.model_id == model_id].set_index(keys)
        if set(subset.index) != set(indexed.index):
            raise ValueError("Missing or extra prediction keys")
        subset = subset.loc[indexed.index]
        input_fields = BASE+LAYOUT if cfg.get("input_layout") == "native-shape-dtype-v1" else BASE
        for field in set(input_fields)-set(keys):
            if not np.array_equal(subset[field], indexed[field]):
                raise ValueError("Unpaired prediction: " + field)
        ok = subset.status == "ok"
        if cfg.get("input_layout") == "native-shape-dtype-v1":
            valid = subset.raw_input_hash.ne("")
            if (not subset.loc[valid, "raw_input_dtype"].eq("float32").all() or
                    not subset.loc[valid, "original_input_dtype"].eq("float32").all() or
                    not subset.loc[valid, "raw_input_shape"].eq(subset.loc[valid, "original_input_shape"]).all() or
                    not subset.loc[valid, "raw_input_shape"].str.fullmatch(r"\[[1-9]\d*,[1-9]\d*(,[134])?\]").all()):
                raise ValueError("Invalid native image layout in run record")
        if (not indexed[side+"_model_id"].eq(model_id).all() or
                not np.array_equal(subset.status, indexed[side+"_status"]) or
                not np.array_equal(subset.predicted_label, indexed[side+"_pred"]) or
                not subset.status.isin(["ok", "failed"]).all() or
                not subset.loc[ok, "predicted_label"].isin(classes).all() or
                not subset.loc[~ok, "predicted_label"].eq("").all() or
                not subset.loc[ok, "raw_input_hash"].str.fullmatch("[0-9a-f]{64}").all() or
                not subset.loc[ok, "original_input_hash"].str.fullmatch("[0-9a-f]{64}").all()):
            raise ValueError("Stored predictions disagree with model, class, hash, or status")
    # Count changes from the two decisions, never trust a stored categorical summary as truth.
    for row in pairs.itertuples():
        a = {"status": row.before_status, "predicted_label": row.before_pred}
        b = {"status": row.after_status, "predicted_label": row.after_pred}
        name = _change_name(cfg["kind"], row.true_label, a, b)
        if name != row.change or row.status != ("failed" if name == "failed" else "ok"):
            raise ValueError("Incorrect change type for paired predictions")
    completed = int(pairs.status.eq("ok").sum())
    if cfg["completed_pairs"] != completed or cfg["failed_pairs"] != planned-completed:
        raise ValueError("Incorrect scheduled/completed/failed pair counts")


def _conditions(project, split):
    cells = []
    for name, seed in project.conditions(split):
        if (not isinstance(name, str) or not name.strip() or name == "ALL" or "\x00" in name
                or isinstance(seed, bool) or not isinstance(seed, Integral)):
            raise ValueError("Invalid condition or seed")
        cells.append((name, int(seed)))
    if not cells or len(cells) != len(set(cells)):
        raise ValueError("Conditions cannot be empty or duplicate")
    declared = project.protocol.get("conditions")
    if project.kind == "binary" and isinstance(declared, list):
        if [(c["condition"], c["seed"]) for c in declared] != cells:
            raise ValueError("Execution conditions differ from preregistered protocol")
    return cells


def _csv(path, rows, columns=None):
    # UTF-8, LF and stable serialization in both Windows and Linux packages.
    if not isinstance(rows, pd.DataFrame):
        rows = pd.DataFrame(rows, columns=columns)
    rows.to_csv(path, index=False, lineterminator="\n")


def run_project(project, split, output_root, *, sample_ids=None, progress=None):
    """Run both registered models, returning one immutable, atomically published result directory.

    Progress reports (scheduled pairs processed, total pairs); completion counters distinguish
    successful and failed model calls. Any explicit sample list is a subset, even if it names all
    rows. An interrupted run is retained under a hidden staging directory and cannot be loaded.
    """
    project.check_scope(split)  # must precede model access, manifest reads and image decoding
    classes = list(project.classes)
    if (project.kind not in ("binary", "multiclass") or len(classes) < 2 or len(set(classes)) != len(classes)
            or not all(isinstance(c, str) and c.strip() for c in classes)
            or (project.kind == "binary" and set(classes) != {"normal", "defect"})):
        raise ValueError("Unsupported project kind or classes")
    rows = _population(project, split, classes, sample_ids)
    cells = _conditions(project, split)
    models = dict(project.models)
    if len(models) != 2 or not all(isinstance(k, str) and k.strip() for k in models):
        raise ValueError("Exactly two ordered registered model names are required")
    before, after = list(models)
    identities = {k: adapter.identity for k, adapter in models.items()}
    if not all(isinstance(i, dict) and i for i in identities.values()):
        raise ValueError("Each model must expose its versioned identity")
    protocol = json.loads(_json(project.protocol))  # immutable snapshot before inference
    if project.kind == "binary":
        for key, value in (("minimum_recall", .8), ("maximum_fpr", .1),
                           ("clean_drop_limit", .02), ("other_drop_limit", .05)):
            if ((key in protocol and protocol[key] != value) or
                    (getattr(project, "official", False) and project.id == "ksdd2" and key not in protocol)):
                raise ValueError("Missing or unsupported official binary prototype gate: " + key)
    score_atol = float(protocol.get("score_atol", .0003))
    if not math.isfinite(score_atol) or score_atol < 0:
        raise ValueError("Invalid score tolerance in protocol")
    full_scope = sample_ids is None and (not hasattr(project, "is_full_split") or project.is_full_split(split))
    started = datetime.now(timezone.utc)
    run_id = started.strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid.uuid4().hex
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    final, stage = root/run_id, root/("."+run_id+".pending")
    stage.mkdir(exist_ok=False)
    cfg = dict(schema=SCHEMA, run_id=run_id, project_id=project.id, kind=project.kind,
               official=bool(getattr(project, "official", False)), split=split,
               scope="full" if full_scope else "subset", started_at=started.isoformat(),
               status="running", classes=classes, models=json.loads(_json(identities)),
               before_model_id=before, after_model_id=after, conditions=[list(v) for v in cells],
               n_originals=len(rows), planned_pairs=len(rows)*len(cells), planned_predictions=len(rows)*len(cells)*2,
               score_atol=score_atol, environment=execution_environment(), protocol=protocol,
               input_layout="native-shape-dtype-v1", model_order=[before, after],
               protocol_file_sha256=getattr(project, "protocol_file_sha256", None),
               source_manifest_sha256=getattr(project, "index_hash", None),
               frozen_status=getattr(project, "frozen_status", None),
               metric_denominators="Binary rates require all scheduled outcomes for that truth class. "
               "If an eligible image failed or support is zero, its rate is unavailable; counts retain failures.")
    try:
        _put(stage/"manifest.json", cfg)
        _put(stage/"protocol.json", protocol)
        cfg["protocol_sha256"] = _digest(stage/"protocol.json")
        _csv(stage/"data_manifest.csv", rows)
        pred, pairs = _infer(project, rows, cells, models, classes, progress)
        failures = pred[pred.status != "ok"]
        cfg["failed_predictions"] = int(len(failures))
        cfg["completed_predictions"] = int(len(pred)-len(failures))
        cfg["failed_pairs"] = int(pairs.status.ne("ok").sum())
        cfg["completed_pairs"] = int(len(pairs)-cfg["failed_pairs"])
        _check_rows(rows, pred, pairs, cfg)
        tables = _tables(pairs, cfg)
        _csv(stage/"predictions.csv", pred)
        _csv(stage/"pairs.csv", pairs)
        _csv(stage/"failures.csv", failures)
        for table, records in tables.items():
            if records:
                _csv(stage/(table+".csv"), records)
        _put(stage/"decision.json", _decision(cfg, tables))
        names = FILES | ({"baselines.csv"} if project.kind == "binary" else set())
        cfg.update(status="complete", finished_at=datetime.now(timezone.utc).isoformat(),
                   files={name: _digest(stage/name) for name in sorted(names)})
        cfg["record_sha256"] = hashlib.sha256(_json(cfg).encode()).hexdigest()
        _put(stage/"manifest.json", cfg)
        stage.rename(final)
    except BaseException as exc:
        if stage.is_dir():
            _put(stage/"manifest.json", {**cfg, "status": "incomplete", "error": _exc("run", exc)})
        raise
    return final


def load_run(run_dir, expected_manifest_sha256=None):
    """Check the published inventory. External digest, when supplied, also anchors its provenance."""
    run_dir = Path(run_dir)
    path = run_dir/"manifest.json"
    if run_dir.is_symlink() or path.is_symlink() or not path.is_file():
        raise ValueError("Missing or invalid run manifest")
    payload = path.read_bytes()
    if expected_manifest_sha256 is not None and hashlib.sha256(payload).hexdigest() != expected_manifest_sha256:
        raise ValueError("Run manifest digest mismatch")
    cfg = json.loads(payload)
    if cfg.get("status") != "complete" or cfg.get("schema") != SCHEMA:
        raise ValueError("Incomplete or unsupported run")
    check = {k: v for k, v in cfg.items() if k != "record_sha256"}
    if hashlib.sha256(_json(check).encode()).hexdigest() != cfg.get("record_sha256"):
        raise ValueError("Run manifest payload digest mismatch")
    names = FILES | ({"baselines.csv"} if cfg["kind"] == "binary" else set())
    if set(cfg["files"]) != names:
        raise ValueError("Run inventory differs from schema")
    for name, digest in cfg["files"].items():
        member = run_dir/name
        if member.is_symlink() or not member.is_file() or _digest(member) != digest:
            raise ValueError("Missing or modified run artifact: " + name)
    if (cfg["protocol_sha256"] != cfg["files"]["protocol.json"]
            or json.loads((run_dir/"protocol.json").read_bytes()) != cfg["protocol"]):
        raise ValueError("Protocol mismatch in run record")
    return cfg


def recompute_decision(run_dir, expected_manifest_sha256=None):
    """Independently re-count actual stored calls, pairs and denominators to check a gate seal."""
    folder = Path(run_dir)
    cfg = load_run(folder, expected_manifest_sha256)
    def read(name):
        return pd.read_csv(folder/(name+".csv"), dtype=str, keep_default_na=False)
    data, pred, pairs = read("data_manifest"), read("predictions"), read("pairs")
    for frame in (pred, pairs):
        frame["seed"] = frame.seed.astype(int)
    _check_rows(data, pred, pairs, cfg)
    decision = _decision(cfg, _tables(pairs, cfg))
    def equivalent(left, right):
        # Python 3.12 improved float summation. Recounts produced by 3.11 can differ
        # at the last bit of macro F1. Integer counts, statuses, IDs and labels stay exact.
        if type(left) is not type(right):
            return False
        if isinstance(left, float):
            return math.isclose(left, right, rel_tol=0, abs_tol=1e-12)
        if isinstance(left, dict):
            return left.keys() == right.keys() and all(equivalent(left[k], right[k]) for k in left)
        if isinstance(left, list):
            return len(left) == len(right) and all(equivalent(a, b) for a, b in zip(left, right))
        return left == right
    if not equivalent(decision, json.loads((folder/"decision.json").read_bytes())):
        raise ValueError("Saved decision differs from recomputed image-level result")
    return decision
