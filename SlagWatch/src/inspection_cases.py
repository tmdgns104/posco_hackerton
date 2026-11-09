"""Portable inspection cases. A historical reproduction and a new trial are separate contracts.

A case archive is data, never executable model code. Trust comes from a locally selected
run and registered project; uploaded paths, model IDs, labels and scores are not trusted.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from hashlib import sha256
import io
import json
import math
import os
from pathlib import Path
import re
import stat
import struct
import sys
import tempfile
import uuid
import zipfile
import zlib

import numpy as np

if __package__ in (None, ""):  # Allow an absolute script path from any working directory.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inspection_runner import execution_environment, hash_raw, load_run, validate_prediction

SCHEMA = "steeltrust.inspection.case/1"
ATOL = 0.0003
MAX_ARRAY = 32 * 1024 * 1024
MAX_ARRAY_PAYLOAD = MAX_ARRAY - 8192  # retain room for the NPY header in each member
MAX_META = 256 * 1024
MAX_ZIP = 2 * MAX_ARRAY + MAX_META + 8192
NAMES = {"case.json", "original.npy", "input.npy"}
HISTORY_FILE = re.compile(r"([0-9a-f]{64})_([0-9a-f]{64})\.zip\Z")


def _digest(data):
    return sha256(data).hexdigest()


def _json(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf8")


def _no_duplicates(items):
    result = {}
    for k, v in items:
        if k in result:
            raise ValueError(f"duplicate JSON key: {k}")
        result[k] = v
    return result


def _parse_json(data):
    def bad(value):
        raise ValueError("nonfinite JSON: " + value)
    return json.loads(data, object_pairs_hook=_no_duplicates, parse_constant=bad)


def _serialized_array(arr):
    arr = np.ascontiguousarray(arr)
    _check_array(arr.shape, arr.dtype)
    if not np.isfinite(arr).all():
        raise ValueError("nonfinite inspection image")
    out = io.BytesIO()
    np.save(out, arr, allow_pickle=False)
    return out.getvalue()


def _check_array(shape, dtype):
    if (len(shape) not in (2, 3) or any(not isinstance(n, int) or n <= 0 for n in shape)
            or dtype.kind not in "uif" or dtype.itemsize not in (1, 2, 4, 8)
            or dtype.fields or dtype.subdtype or math.prod(shape) * dtype.itemsize > MAX_ARRAY_PAYLOAD):
        raise ValueError("unsupported inspection image shape, dtype, or size")


def _read_array(data):
    # np.load may allocate before it notices a tiny file claims a gigantic shape.
    # Validate the header and exact payload length, then use a read-only buffer.
    f = io.BytesIO(data)
    ver = np.lib.format.read_magic(f)
    if ver not in ((1, 0), (2, 0)):
        raise ValueError("unsupported NPY header")
    read = (np.lib.format.read_array_header_1_0 if ver == (1, 0)
            else np.lib.format.read_array_header_2_0)
    shape, fortran, dtype = read(f, max_header_size=4096)
    _check_array(shape, dtype)
    if fortran or f.tell() + math.prod(shape) * dtype.itemsize != len(data):
        raise ValueError("NPY layout/size mismatch")
    arr = np.frombuffer(data, dtype=dtype, offset=f.tell()).reshape(shape)
    if not np.isfinite(arr).all():
        raise ValueError("nonfinite inspection image")
    return arr


def _bytes(data):
    if isinstance(data, (str, os.PathLike)):
        with Path(data).open("rb") as f:
            data = f.read(MAX_ZIP + 1)
    if not isinstance(data, (bytes, bytearray, memoryview)) or len(data) > MAX_ZIP:
        raise ValueError("case ZIP exceeds limit or is not bytes")
    return bytes(data)


def _decode(data):
    data = _bytes(data)
    # Check EOCD before ZipFile allocates central-directory objects. No ZIP64 or comments
    # are needed for this three-member interchange format.
    if (len(data) < 22 or data[-22:-18] != b"PK\x05\x06" or data[-2:] != b"\0\0"
            or struct.unpack_from("<4H", data, len(data) - 18) != (0, 0, 3, 3)):
        raise ValueError("invalid inspection case ZIP directory")
    cd_size, cd_offset = struct.unpack_from("<II", data, len(data) - 10)
    if cd_size > 8192 or cd_offset + cd_size != len(data) - 22:
        raise ValueError("invalid inspection case ZIP directory size")
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            files = z.infolist()
            if len(files) != 3 or {i.filename for i in files} != NAMES:
                raise ValueError("missing, unexpected, or duplicate case ZIP member")
            blobs = {}
            for i in files:
                lim = MAX_META if i.filename == "case.json" else MAX_ARRAY
                mode = stat.S_IFMT(i.external_attr >> 16)
                if (mode not in (0, stat.S_IFREG) or i.flag_bits & 1 or i.file_size > lim
                        or i.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)):
                    raise ValueError("unsafe or oversized case ZIP member")
                with z.open(i) as f:
                    blob = f.read(lim + 1)
                if len(blob) > lim or len(blob) != i.file_size:
                    raise ValueError("invalid case ZIP member size")
                blobs[i.filename] = blob
        meta = _parse_json(blobs["case.json"])
        if not isinstance(meta, dict) or meta.get("schema") != SCHEMA:
            raise ValueError("unknown case schema")
        orig, raw = _read_array(blobs["original.npy"]), _read_array(blobs["input.npy"])
        for field, name in (("original_npy_sha256", "original.npy"), ("input_npy_sha256", "input.npy")):
            if _digest(blobs[name]) != meta.get(field):
                raise ValueError("inspection ZIP image hash mismatch: " + name)
        return meta, orig, raw
    except (zipfile.BadZipFile, zlib.error, RuntimeError, OSError, TypeError, RecursionError, EOFError,
            UnicodeError, OverflowError, struct.error) as exc:
        raise ValueError("invalid inspection case ZIP") from exc


def inspect_inputs(data):
    """Read case for preview only. Use replay_case or save_case to verify historical provenance."""
    return _decode(data)


def inspect_case(data):
    return inspect_inputs(data)[0]


def _rows(run_dir, file, sample, condition=None, seed=None):
    with (Path(run_dir) / file).open(encoding="utf-8-sig", newline="") as f:
        return [r for r in csv.DictReader(f) if r["sample_id"] == sample and
                (condition is None or (r["condition"] == condition and int(r["seed"]) == seed))]


def _one(rows, name):
    if len(rows) != 1:
        raise ValueError(f"missing or ambiguous {name}")
    return rows[0]


def _score_map(value):
    if value is None:
        return {}
    if not isinstance(value, dict) or any(
        not isinstance(k, str) or isinstance(v, (bool, np.bool_))
        or not isinstance(v, (int, float, np.integer, np.floating)) or not math.isfinite(v)
        for k, v in value.items()
    ):
        raise ValueError("invalid prediction scores")
    return {k: float(v) for k, v in value.items()}


def _prediction(output, classes):
    if output.get("status") != "ok" or output.get("predicted_label") not in classes:
        raise ValueError("inference failed or label outside project classes")
    return {"predicted_label": str(output["predicted_label"]), "scores": _score_map(output.get("scores")),
            "score_type": str(output.get("score_type") or ""), "status": "ok"}


def _attribution(project, source_id, condition, seed):
    # Arbitrary CSV source claims are prefixed "user_csv:" on registration. Only a
    # verified original match is minted without that prefix (see inspection_register).
    verified_copy = (project.id.startswith("user_") and getattr(project, "template_id", None) == "ksdd2"
                     and re.fullmatch(r"ksdd2:validation:ksdd2-[0-9]+", str(source_id)) is not None)
    if (project.id == "ksdd2" and getattr(project, "official", False)) or verified_copy:
        return {"title": "Kolektor Surface-Defect Dataset 2 (KolektorSDD2 / KSDD2)",
                "image_and_annotation_credit": "Kolektor Group d.o.o.; distributed by ViCoS Lab, University of Ljubljana",
                "source_url": "https://www.vicos.si/resources/kolektorsdd2/",
                "license": "CC BY-NC-SA 4.0", "license_url": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
                "citation": "J. Božič, D. Tabernik and D. Skočaj, Mixed supervision for surface-defect detection: "
                            "from weakly to fully supervised learning, Computers in Industry (2021).",
                "case_image_terms": "Included original and adapted inspection images: CC BY-NC-SA 4.0.",
                "changes": f"Decoded pixels stored as a normalized array; trial condition={condition}; seed={seed}."}
    return {"source_id": str(source_id), "project_source": getattr(project, "source", str(source_id)),
            "changes": f"Decoded pixels stored as a normalized array; trial condition={condition}; seed={seed}."}


def _check_saved_layout(row, original, raw):
    for prefix, arr in (("original", original), ("raw", raw)):
        expected_shape = _parse_json(row.get(prefix + "_input_shape", ""))
        if (not isinstance(expected_shape, list) or any(type(n) is not int for n in expected_shape)
                or expected_shape != list(arr.shape) or row.get(prefix + "_input_dtype") != str(arr.dtype)):
            raise ValueError("uploaded " + prefix + " image layout differs from trusted run")


def _case_payload(project, run_dir, sample_id, condition, seed, expected_manifest=None, supplied=None):
    run_dir = Path(run_dir)
    manifest = load_run(run_dir, expected_manifest_sha256=expected_manifest)
    manifest_sha = _digest((run_dir / "manifest.json").read_bytes())
    if expected_manifest is not None and manifest_sha != expected_manifest:
        raise ValueError("selected local run does not match historical manifest")
    if manifest["project_id"] != project.id or manifest["kind"] != project.kind:
        raise ValueError("case and selected project differ")
    split = manifest["split"]
    if split not in ("validation", "test"):
        raise ValueError("cases require validation or frozen test scope")
    project.check_scope(split)
    if (not isinstance(sample_id, str) or not isinstance(condition, str)
            or isinstance(seed, bool) or not isinstance(seed, (int, np.integer))):
        raise ValueError("invalid sample, condition or seed")
    seed = int(seed)
    if [condition, seed] not in [list(c) for c in manifest["conditions"]]:
        raise ValueError("condition outside historical run")
    classes = list(manifest["classes"])
    if classes != list(project.classes):
        raise ValueError("class mapping differs from historical project")
    rows = project.rows(split)
    matches = rows[rows["sample_id"].astype(str) == sample_id]
    if len(matches) != 1:
        raise ValueError("sample not uniquely present in registered project")
    row = matches.iloc[0]
    sample = _one(_rows(run_dir, "data_manifest.csv", sample_id), "historical data row")
    pair = _one(_rows(run_dir, "pairs.csv", sample_id, condition, seed), "historical pair")
    lineage = {k: str(row[k]) for k in ("sample_id", "group_id", "true_label", "source_id", "file_sha256", "split")}
    if (lineage["split"] != split or lineage["true_label"] not in classes
            or any(sample.get(k) != v or pair.get(k) != v for k, v in lineage.items())):
        raise ValueError("historical truth/source/split differs from registered project")
    if str(row["image_path"]) != sample["image_path"]:
        raise ValueError("historical image path differs from registered project")
    layout = manifest.get("input_layout")
    if layout not in (None, "native-shape-dtype-v1"):
        raise ValueError("unsupported historical input layout")
    if supplied is not None and layout:
        # Portable path: historical hashes authenticate pixels and historical shapes/dtypes
        # authenticate their interpretation. No source image, current transform, or old
        # preprocessing pipeline is consulted to create a replacement array.
        original, raw = supplied
    else:
        # Building a new archive requires registered source pixels. Legacy runs recorded
        # only flat float32 pixel hashes; regenerate there to authenticate array shape.
        original = np.ascontiguousarray(project.load(row))
        raw = np.ascontiguousarray(project.change(original.copy(), row, condition, seed))
        if supplied is not None:
            for candidate, trusted in zip(supplied, (original, raw), strict=True):
                if (candidate.shape != trusted.shape or candidate.dtype != trusted.dtype
                        or candidate.tobytes(order="C") != trusted.tobytes(order="C")):
                    raise ValueError("uploaded legacy image differs from trusted reconstruction")
    if layout:
        _check_saved_layout(pair, original, raw)
    original_npy, input_npy = _serialized_array(original), _serialized_array(raw)
    common_hash = hash_raw(raw)
    original_hash = hash_raw(original)
    if common_hash != pair["raw_input_hash"] or original_hash != pair["original_input_hash"]:
        raise ValueError("input hash differs from historical run")
    before, after = manifest["before_model_id"], manifest["after_model_id"]
    if before == after or before != pair["before_model_id"] or after != pair["after_model_id"]:
        raise ValueError("model IDs disagree in selected run")
    records = _rows(run_dir, "predictions.csv", sample_id, condition, seed)
    if len(records) != 2:
        raise ValueError("missing or ambiguous historical prediction")
    preds = {}
    for pos, model_id in (("before", before), ("after", after)):
        pred = _one([r for r in records if r["model_id"] == model_id], "model prediction")
        if layout:
            _check_saved_layout(pred, original, raw)
        if (pred["raw_input_hash"] != common_hash or pred["original_input_hash"] != original_hash
                or pair[pos + "_status"] != "ok"
                or pred["predicted_label"] != pair[pos + "_pred"]
                or any(pred.get(k) != v for k, v in lineage.items())):
            raise ValueError("historical pair incomplete or inconsistent")
        preds[model_id] = _prediction({**pred, "scores": _parse_json(pred.get("scores") or "{}")}, classes)

    specimen = {"project_id": project.id, "file_sha256": lineage["file_sha256"],
                "true_label": lineage["true_label"], "condition": condition, "seed": seed,
                "original_npy_sha256": _digest(original_npy), "input_npy_sha256": _digest(input_npy)}
    atol = float(manifest.get("score_atol", ATOL))
    if not math.isfinite(atol) or atol < 0:
        raise ValueError("invalid historical score tolerance")
    atol = min(atol, ATOL)
    meta = {**lineage, **specimen, "schema": SCHEMA, "kind": project.kind, "classes": classes,
            "case_id": _digest(_json(specimen)), "run_id": manifest["run_id"], "run_manifest_sha256": manifest_sha,
            "protocol_sha256": manifest["protocol_sha256"], "condition": condition, "seed": seed,
            "label_provenance": {"source_id": lineage["source_id"], "image_path": sample["image_path"],
                                 "manifest": "data_manifest.csv", "manifest_sha256": manifest["files"]["data_manifest.csv"]},
            "attribution": _attribution(project, lineage["source_id"], condition, seed),
            "role": "development_regression" if split == "validation" else "frozen_test_replay_only",
            "original_sha256": original_hash, "input_sha256": common_hash,
            "before_model_id": before, "after_model_id": after, "models": {m: manifest["models"][m] for m in preds},
            "predictions": preds, "change": pair["change"], "score_tolerance": {"atol": atol, "rtol": 0},
            "historical_environment": manifest.get("environment", {})}
    if project.id == "ksdd2" and getattr(project, "official", False):
        # This source index is pinned by the registered protocol. Masks describe the label;
        # neither the mask nor its statistics are ever passed to model inference.
        if project.index_hash != manifest.get("source_manifest_sha256"):
            raise ValueError("official annotated source index differs from historical run")
        annotation = _one(project.index.loc[project.index.sample_id == sample_id].to_dict("records"), "official annotation")
        positives = int(annotation["defect_pixels"])
        if positives < 0 or ("defect" if positives else "normal") != lineage["true_label"]:
            raise ValueError("official annotation pixel count disagrees with case truth")
        meta["label_provenance"].update(source_manifest_sha256=project.index_hash,
                                        official_split=annotation["official_split"],
                                        mask_path=annotation["mask_path"], mask_sha256=annotation["mask_sha256"],
                                        defect_pixels=positives, rule="any nonzero pixel in the official annotation mask = defect")
        meta["condition_settings"] = manifest["protocol"].get("transformations", {})
    meta["history_id"] = _digest(_json(meta))
    return meta, original_npy, input_npy


def build_case(project, run_dir, sample_id, condition, seed):
    return _pack(*_case_payload(project, run_dir, sample_id, condition, seed))


def _pack(meta, original, raw):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, blob in (("case.json", _json(meta)), ("original.npy", original), ("input.npy", raw)):
            # A fixed ZIP header makes repeated exports of one history byte-identical.
            i = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            i.create_system = 3
            i.external_attr = 0o100644 << 16
            z.writestr(i, blob, compress_type=zipfile.ZIP_DEFLATED)
    return buf.getvalue()


def _validate(project, run_dir, data, *, regression=False):
    meta, original, raw = _decode(data)
    needed = ("project_id", "sample_id", "condition", "seed", "run_manifest_sha256")
    if any(k not in meta for k in needed) or meta["project_id"] != project.id:
        raise ValueError("missing case selector or wrong project")
    if regression and meta.get("split") != "validation":
        raise ValueError("only validation cases may be used for new model regression")
    if not isinstance(meta["run_manifest_sha256"], str) or not re.fullmatch("[a-f0-9]{64}", meta["run_manifest_sha256"]):
        raise ValueError("invalid historical manifest hash")
    expected, orig_bytes, raw_bytes = _case_payload(
        project, run_dir, meta["sample_id"], meta["condition"], meta["seed"], meta["run_manifest_sha256"],
        supplied=(original, raw))
    if _json(meta) != _json(expected):
        raise ValueError("case identity, truth, history, or metadata differs from trusted run")
    return expected, raw, orig_bytes, raw_bytes


def _infer(adapter, raw, meta, model_id):
    # Independent (read-only) arrays prevent any adapter changing input for its successor.
    image = np.frombuffer(raw.tobytes(order="C"), dtype=raw.dtype).reshape(raw.shape)
    outputs = adapter.predict([image])
    if len(outputs) != 1:
        raise ValueError("adapter returned wrong prediction count")
    checked = validate_prediction(outputs[0], {"sample_id": meta["sample_id"], "raw_input_hash": meta["input_sha256"]},
                                  model_id, meta["classes"])
    return {k: (_parse_json(v) if k == "scores" else v) for k, v in checked.items() if k != "error"}


def replay_case(project, run_dir, data):
    meta, raw, _, _ = _validate(project, run_dir, data)
    outputs, deviations = {}, {}
    for mid in meta["predictions"]:
        if mid not in project.models or _json(project.models[mid].identity) != _json(meta["models"][mid]):
            raise ValueError("historical model identity unavailable or changed: " + mid)
    for mid, expected in meta["predictions"].items():
        observed = _infer(project.models[mid], raw, meta, mid)
        if (observed["predicted_label"] != expected["predicted_label"]
                or observed["score_type"] != expected["score_type"]
                or observed["scores"].keys() != expected["scores"].keys()):
            raise ValueError("historical prediction or score contract differs: " + mid)
        diff = max((abs(observed["scores"][k] - v) for k, v in expected["scores"].items()), default=0.0)
        if diff > meta["score_tolerance"]["atol"]:
            raise ValueError(f"historical score exceeds tolerance: {mid} (delta={diff:.7g})")
        outputs[mid] = observed
        deviations[mid] = diff
    return {**meta, "operation": "historical_replay", "verified": True,
            "historical_predictions": meta["predictions"], "predictions": outputs, "max_score_delta": deviations,
            "atol": meta["score_tolerance"]["atol"], "rtol": 0.0, "environment": execution_environment()}


def _publish(root, name, data, *, dedup=False):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = root / name
    # Stage in the destination filesystem. Linking a complete temporary file is atomic and
    # exclusive on POSIX/NTFS: no interrupted partial results and no replacing other runs.
    temp = None
    try:
        with tempfile.NamedTemporaryFile(dir=root, prefix=".pending-", delete=False) as f:
            temp = Path(f.name)
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        try:
            os.link(temp, path)
        except FileExistsError:
            if not dedup or path.is_symlink() or not path.is_file() or _bytes(path) != data:
                raise ValueError("existing immutable case/result differs")
            return path, False
        return path, True
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def _summary(meta, path):
    fields = ("case_id", "history_id", "sample_id", "project_id", "run_id", "split", "condition",
              "seed", "true_label", "role", "before_model_id", "after_model_id", "change")
    return {**{k: meta[k] for k in fields}, "path": str(path)}


def save_case(project, run_dir, data, collection_root):
    data = _bytes(data)
    meta, _, original, raw = _validate(project, run_dir, data)
    # Canonical repack allows harmless ZIP-level differences without duplicate histories.
    canonical = _pack(meta, original, raw)
    filename = f"{meta['case_id']}_{meta['history_id']}.zip"
    path, created = _publish(collection_root, filename, canonical, dedup=True)
    return {**_summary(meta, path), "created": created}


def list_cases(collection_root, project_id=None):
    entries = []
    root = Path(collection_root)
    if root.is_dir():
        for p in sorted(root.glob("*.zip")):
            match = HISTORY_FILE.fullmatch(p.name)
            if not match or not p.is_file() or p.is_symlink():
                continue
            meta = inspect_case(p)
            if match.groups() != (meta.get("case_id"), meta.get("history_id")):
                raise ValueError("collection filename does not match case identity")
            if project_id is None or meta["project_id"] == project_id:
                entries.append(_summary(meta, p.resolve()))
    return entries


def retest_case(project, run_dir, data, model_id, output_root):
    meta, raw, _, _ = _validate(project, run_dir, data, regression=True)
    return _retest(project, meta, raw, model_id, output_root)


def _retest(project, meta, raw, model_id, output_root):
    if model_id not in project.models:
        raise ValueError("target model is not registered for this project")
    adapter = project.models[model_id]
    reference_id = meta["before_model_id"]
    prior = meta["predictions"][reference_id]["predicted_label"]
    true = meta["true_label"]
    try:
        now = _infer(adapter, raw, meta, model_id)
        correct = now["predicted_label"] == true
        change = ("unchanged_correct" if correct else "new_error") if prior == true else (
            "fixed" if correct else "still_wrong")
        error = None
    except Exception as exc:
        now = {"status": "failed", "predicted_label": None, "scores": {}, "score_type": None}
        correct, change, error = None, "incomplete", f"{type(exc).__name__}: {exc}"[:512]
    result = {**{k: meta[k] for k in ("case_id", "history_id", "project_id", "run_id", "run_manifest_sha256",
                                     "sample_id", "true_label", "split", "role", "condition", "seed", "input_sha256")},
              "operation": "new_model_regression", "result_id": uuid.uuid4().hex,
              "created_utc": datetime.now(timezone.utc).isoformat(), "model_id": model_id,
              "model_identity": adapter.identity, "reference_model_id": reference_id,
              "reference_identity": meta["models"][reference_id], "reference_prediction": prior,
              "prediction_changed": None if correct is None else prior != now["predicted_label"],
              "correct": correct, "change_from_baseline": change, **now, "error": error,
              "environment": execution_environment()}
    path, _ = _publish(output_root, f"{result['result_id']}.json", _json(result))
    return {**result, "result_path": str(path)}


def retest_collection(project, run_dir, selected_paths, model_id, output_root):
    """One selected historical run; ignore repeated copies of the same specimen."""
    results, seen = [], set()
    for path in selected_paths:
        data = _bytes(path)
        meta, raw, _, _ = _validate(project, run_dir, data, regression=True)
        if meta["case_id"] not in seen:
            results.append(_retest(project, meta, raw, model_id, output_root))
            seen.add(meta["case_id"])
    return results


def main(argv=None):
    from src.inspection_projects import load_project
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="action", required=True)
    for name in ("build", "replay", "retest", "save", "list", "retest-collection"):
        p = subs.add_parser(name)
        if name != "list":
            p.add_argument("--project", required=True)
            p.add_argument("--run", required=True, type=Path, help="trusted historical run directory")
        if name == "build":
            p.add_argument("--sample", required=True)
            p.add_argument("--condition", required=True)
            p.add_argument("--seed", required=True, type=int)
            p.add_argument("--out", required=True, type=Path, help="new ZIP path (never overwritten)")
        if name in ("replay", "retest", "save"):
            p.add_argument("--case", required=True, type=Path)
        if name in ("save", "list"):
            p.add_argument("--collection", required=True, type=Path)
        if name == "list":
            p.add_argument("--project")
        if name.startswith("retest"):
            p.add_argument("--model", required=True)
            p.add_argument("--out", required=True, type=Path, help="directory for immutable result JSON files")
        if name == "retest-collection":
            p.add_argument("--case", nargs="+", required=True, type=Path)
    a = parser.parse_args(argv)
    try:
        if a.action == "list":
            result = list_cases(a.collection, a.project)
        else:
            project = load_project(a.project)
            if a.action == "build":
                data = build_case(project, a.run, a.sample, a.condition, a.seed)
                path, _ = _publish(a.out.parent, a.out.name, data)
                result = _summary(inspect_case(data), path)
            elif a.action == "replay":
                result = replay_case(project, a.run, a.case)
            elif a.action == "retest":
                result = retest_case(project, a.run, a.case, a.model, a.out)
            elif a.action == "save":
                result = save_case(project, a.run, a.case, a.collection)
            else:
                result = retest_collection(project, a.run, a.case, a.model, a.out)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(2, f"case operation refused: {exc}\n")


if __name__ == "__main__":
    main()
