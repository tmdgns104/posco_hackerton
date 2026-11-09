"""Replay a frozen G1 comparison using its own selection, scope and predictions."""
import argparse
import io
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd

from .surface_data import ROOT, RUN, TASK, UNIT, stable_id
from .surface_detector import Detector
from .surface_g1 import (
    CORE, TRAIN_CODE, DEFAULT_G1, G1Detector, check_protocol, file_sha, put_json,
    read_json, source_snapshot,
)
from .surface_g1_study import frozen, verify_selection
from .surface_replay import SurfaceRun, numpy_bytes, png

RUN_CODE = (*CORE, *TRAIN_CODE, "surface_replay.py", "surface_g1_replay.py",
            "surface_ui.py", "surface_g1_ui.py")
BASE_FILES = ("split_manifest.csv", "validation_discoveries.csv.gz", "validation_discovered_inputs.zip")


class G1Run(SurfaceRun):
    def __init__(self, run_dir=DEFAULT_G1, expected_manifest=None):
        self.out = Path(run_dir).resolve()
        manifest = self.out / "manifest.json"
        if expected_manifest and file_sha(manifest) != expected_manifest:
            raise ValueError("G1 manifest hash mismatch")
        self.man = read_json(manifest)
        self.id = self.man["run_id"]
        payload = {k: v for k, v in self.man.items() if k != "run_id"}
        if stable_id(payload) != self.id or self.man["experiment"] != "surface_g1":
            raise ValueError("G1 run identity mismatch")
        for file in RUN_CODE:
            if file_sha(ROOT / "src" / file) != self.man["code_sha256"][file]:
                raise ValueError("G1 runtime code mismatch: " + file)
        self._checked = set()
        self.protocol = check_protocol(self.out)
        self.checked("protocol.json")
        self.meta = pd.read_csv(self.checked("split_manifest.csv")).set_index("sample_id", drop=False)
        self.seal = read_json(self.checked("selection.json"))
        self.status = read_json(self.checked("test_status.json"))
        self.models, self.cache, self.frames, self._lookup = {}, {}, {}, {}
        expected_splits = ["validation", "test"] if self.test_available() else ["validation"]
        if self.man["splits"] != expected_splits:
            raise ValueError("G1 test scope does not match its selection")

    def checked(self, relative):
        if relative in BASE_FILES:
            path, digest = RUN / relative, self.man["baseline_files"][relative]
        else:
            path, digest = self.out / relative, self.man["files"][relative]
        if relative not in self._checked:
            if file_sha(path) != digest:
                raise ValueError("G1 artifact hash mismatch: " + relative)
            self._checked.add(relative)
        return path

    def test_available(self):
        return (self.seal["selected"] == "G1" and self.seal["assessment"]["eligible"]
                and self.status["phase"] == "complete" and self.status["evaluated"] is True)

    def ensure_scope(self, split, after, sigma, seed):
        if (after != "G1" or split not in self.man["splits"] or split == "test" and not self.test_available()
                or [sigma, seed] not in self.protocol["conditions"][split]):
            raise ValueError("G1 split, condition or candidate outside evaluated scope")

    def detector(self, name):
        if name not in ("G1", "V0"):
            raise ValueError("Unknown G1 comparison model")
        if name not in self.models:
            if name == "G1":
                self.checked("G1.joblib")
                model = G1Detector(self.out)
            else:
                model = Detector("V0")
            if model.identity != self.seal["frozen"]["models"][name]:
                raise ValueError("G1 comparison model differs from validation freeze")
            self.models[name] = model
        return self.models[name]

    def case(self, sample, after, sigma, seed, provided=None):
        record, original, raw, _ = super().case(sample, after, sigma, seed, provided)
        for part in ("original", "trial"):
            a = record["outputs"]["V0"][part]
            b = record["outputs"]["G1"][part]
            if len({a["raw_input_hash"], b["raw_input_hash"], a["processed_input_hash"], b["processed_input_hash"]}) != 1:
                raise ValueError("Different inputs for V0 and G1")
        record.update(schema=2, experiment="surface_g1", preprocessing_policy="V0")
        return record, original, raw

    def export(self, case):
        record, original, raw = case
        buf = io.BytesIO()
        with ZipFile(buf, "w", ZIP_DEFLATED) as archive:
            archive.writestr("case.json", json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False))
            archive.writestr("raw_input.npy", numpy_bytes(raw))
            archive.writestr("original_preview.png", png(original))
            archive.writestr("noise_preview.png", png(raw))
        return buf.getvalue()


def finalize(out):
    out = out.resolve()
    snapshot = source_snapshot()
    if read_json(out / "baseline_inventory.json") != snapshot:
        raise ValueError("Original experiment changed during the G1 study")
    meta = pd.read_csv(RUN / "split_manifest.csv")
    seal = read_json(out / "selection.json")
    status = read_json(out / "test_status.json")
    if frozen(out) != seal["frozen"]:
        raise ValueError("Selection snapshot changed")
    splits = ["validation"]
    if seal["selected"] == "G1":
        identity = verify_selection(out, meta)
        if (status["phase"] != "complete" or status["evaluated"] is not True or status["opening_id"] != identity
                or status["models"] != seal["frozen"]["models"] or status["adopt"] != status["assessment"]["eligible"]):
            raise ValueError("Test unfinished or inconsistent with frozen selection")
        splits.append("test")
    elif status["phase"] != "not_selected" or status["evaluated"] or seal["assessment"]["eligible"]:
        raise ValueError("Validation and test state inconsistent")
    files = ["baseline_inventory.json", "protocol.json", "failure_conditions.json", "train_views.csv",
             "G1.joblib", "selection.json", "test_status.json"]
    files.extend(f"{s}_{suffix}" for s in splits for suffix in (
        "predictions.csv.gz", "pairs.csv.gz", "summary.csv", "classes.csv", "confusion.csv", "changes.csv", "by_parent.csv"))
    man = dict(experiment="surface_g1", task_id=TASK, sample_unit=UNIT, splits=splits,
               n_originals=meta.groupby("split").size().to_dict(), baseline_run_id=snapshot["run_id"],
               baseline_files={n:snapshot["files"][n] for n in BASE_FILES},
               files={n:file_sha(out / n) for n in files}, code_sha256={n:file_sha(ROOT / "src" / n) for n in RUN_CODE})
    man["run_id"] = stable_id(man)
    dest = out / "manifest.json"
    if dest.exists() and read_json(dest) != man:
        raise ValueError("Existing G1 manifest differs from finalized result")
    put_json(dest, man)
    run = G1Run(out)
    split = splits[-1]
    pairs = run.table(split, "pairs")
    folder = out / "examples"
    folder.mkdir(exist_ok=True)
    for kind in ("fixed", "new_error"):
        group = pairs[(pairs.change == kind) & (pairs.sigma > 0)].sort_values(["sigma", "seed", "sample_id"])
        if not group.empty:
            row = group.iloc[0]
            path = folder / (kind + ".zip")
            path.write_bytes(run.export(run.case(row.sample_id, "G1", row.sigma, int(row.seed))))
            run.replay(path)
    return man


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_G1)
    op = parser.add_mutually_exclusive_group(required=True)
    op.add_argument("--finalize", action="store_true")
    op.add_argument("--replay", type=Path)
    args = parser.parse_args()
    if args.finalize:
        print(finalize(args.run_dir)["run_id"])
    else:
        print(json.dumps(G1Run(args.run_dir).replay(args.replay), ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
