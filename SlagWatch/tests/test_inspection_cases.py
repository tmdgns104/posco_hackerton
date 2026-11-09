"""A saved result is reproducible history, but is not the oracle for a new model."""
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.inspection_cases import (build_case, inspect_case, inspect_inputs, list_cases,
                                  replay_case, retest_case, retest_collection, save_case)
from src.inspection_runner import run_project


class Adapter:
    def __init__(self, version, output):
        self.identity = {"model": version, "artifacts_sha256": hashlib.sha256(version.encode()).hexdigest(),
                         "device": "cpu", "threshold": 0.5}
        self.output = output
        self.delta = 0.0
        self.stop = False
        self.calls = []

    def predict(self, raws):
        if self.stop:
            raise RuntimeError("simulated adapter failure")
        self.calls.extend(hashlib.sha256(im.tobytes()).hexdigest() for im in raws)
        return [{"status": "ok", "predicted_label": self.output, "score_type": "decision",
                 "scores": {"defect": float((self.output == "defect") + self.delta)}} for im in raws]


class FixtureProject:
    id, kind, template_id = "user_fixture", "binary", "ksdd2"
    source_id = "ksdd2:validation:ksdd2-10123"
    classes = ["normal", "defect"]
    official = False
    protocol = {"score_atol": 0.0003}
    frozen_status = None

    def __init__(self):
        self.models = {"old": Adapter("v1", "normal"), "candidate": Adapter("v2", "defect")}
        self.arr = np.array([[0.5, 0.7, 0.9], [0.2, 0.6, 0.0]], np.float32)

    def check_scope(self, split):
        if split not in ("validation", "test"):
            raise ValueError("out of scope")

    def rows(self, split):
        return pd.DataFrame([dict(sample_id="001", group_id="012", source_id=self.source_id,
                                  split=split, true_label="defect", image_path="sample.png",
                                  file_sha256=hashlib.sha256(self.arr.tobytes()).hexdigest())])

    def load(self, row):
        return self.arr.copy()

    def change(self, raw, row, condition, seed):
        return raw if condition == "clean" else raw / np.float32(2)

    def conditions(self, split):
        return [("clean", 0), ("dark", 0)]


def replace(archive, key, data):
    out = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(archive)) as src, zipfile.ZipFile(out, "w") as dst:
        for p in src.namelist():
            dst.writestr(p, data if p == key else src.read(p))
    return out.getvalue()


class InspectionCasesTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.p = FixtureProject()
        self.run = run_project(self.p, "validation", self.root / "runs")
        self.blob = build_case(self.p, self.run, "001", "dark", 0)

    def test_history_replays_both_identities_but_new_trial_uses_only_target_and_truth(self):
        first = save_case(self.p, self.run, self.blob, self.root / "cases")
        self.assertTrue(first["created"])
        self.assertFalse(save_case(self.p, self.run, self.blob, self.root / "cases")["created"])
        meta, orig, actual = inspect_inputs(self.blob)
        self.assertEqual(meta["group_id"], "012")
        self.assertIn("license_url", meta["attribution"])
        np.testing.assert_array_equal(actual, orig / np.float32(2))
        self.p.models["old"].delta = 0.00029
        replay = replay_case(self.p, self.run, self.blob)
        self.assertTrue(replay["verified"])
        self.assertEqual(replay["atol"], 0.0003)
        for adapter in self.p.models.values():
            self.assertEqual(adapter.calls[-1], meta["input_sha256"])
        self.p.models["old"].delta = 0.00031
        with self.assertRaisesRegex(ValueError, "tolerance"):
            replay_case(self.p, self.run, self.blob)
        self.p.models["old"].identity["threshold"] = 0.2
        with self.assertRaisesRegex(ValueError, "identity"):
            replay_case(self.p, self.run, self.blob)
        # Simulate a future registry with none of the old adapters installed.
        self.p.models = {"future": Adapter("v3", "defect")}
        r = retest_case(self.p, self.run, first["path"], "future", self.root / "results")
        self.assertTrue(r["correct"])
        self.assertTrue(r["prediction_changed"])
        self.assertEqual(r["reference_prediction"], "normal")
        self.assertEqual(r["change_from_baseline"], "fixed")
        self.assertEqual(r["input_sha256"], meta["input_sha256"])
        self.assertEqual(json.loads(Path(r["result_path"]).read_bytes())["model_identity"]["model"], "v3")
        duplicate = retest_collection(self.p, self.run, [first["path"], first["path"]], "future", self.root / "results")
        self.assertEqual(len(duplicate), 1)
        self.assertNotEqual(duplicate[0]["result_id"], r["result_id"])
        self.p.models["future"].stop = True
        failed = retest_case(self.p, self.run, self.blob, "future", self.root / "results")
        self.assertEqual(failed["status"], "failed")
        self.assertIsNone(failed["correct"])
        self.assertEqual(failed["change_from_baseline"], "incomplete")

    def test_portable_archive_does_not_use_missing_original_or_changed_transform(self):
        def removed(*args):
            raise AssertionError("saved case must not consult a live image source or new transform")
        self.p.load = removed
        self.p.change = removed
        replay = replay_case(self.p, self.run, self.blob)
        self.assertTrue(replay["verified"])
        self.p.models = {"future": Adapter("v7", "defect")}
        self.assertTrue(retest_case(self.p, self.run, self.blob, "future", self.root / "results")["correct"])
        # Reshaping preserves the protocol's *pixel byte* hash but changes its meaning.
        # The trusted layout must reject it even with an updated self-declared NPY digest.
        meta, original, raw = inspect_inputs(self.blob)
        changed = io.BytesIO()
        np.save(changed, raw.reshape(3, 2))
        altered = replace(self.blob, "input.npy", changed.getvalue())
        forged = {**meta, "input_npy_sha256": hashlib.sha256(changed.getvalue()).hexdigest()}
        with self.assertRaisesRegex(ValueError, "layout"):
            retest_case(self.p, self.run, replace(altered, "case.json", json.dumps(forged).encode()),
                        "future", self.root / "results")

    def test_legacy_run_requires_both_stored_arrays_match_reconstructed_pixels_and_layout(self):
        # Write a genuinely shape-less historical-format fixture, including its old integrity seal.
        folder = Path(self.run)
        cfg = json.loads((folder / "manifest.json").read_bytes())
        del cfg["input_layout"]
        for n in ("pairs.csv", "predictions.csv"):
            f = folder / n
            rows = pd.read_csv(f, dtype=str, keep_default_na=False)
            rows.drop(columns=["original_input_shape", "raw_input_shape",
                               "original_input_dtype", "raw_input_dtype"]).to_csv(f, index=False)
            cfg["files"][n] = hashlib.sha256(f.read_bytes()).hexdigest()
        def stable(obj):
            return (json.dumps(obj, ensure_ascii=False, sort_keys=True, allow_nan=False, indent=2) + "\n").encode()
        del cfg["record_sha256"]
        cfg["record_sha256"] = hashlib.sha256(stable(cfg)).hexdigest()
        (folder / "manifest.json").write_bytes(stable(cfg))
        good = build_case(self.p, folder, "001", "dark", 0)
        self.assertTrue(replay_case(self.p, folder, good)["verified"])
        meta, original, trial = inspect_inputs(good)
        for member, field, altered in (("original.npy", "original_npy_sha256", original.reshape(3,2)),
                                      ("input.npy", "input_npy_sha256", trial + np.float32(.001))):
            changed = io.BytesIO()
            np.save(changed, altered)
            forged = {**meta, field: hashlib.sha256(changed.getvalue()).hexdigest()}
            blob = replace(replace(good, member, changed.getvalue()), "case.json", json.dumps(forged).encode())
            with self.assertRaisesRegex(ValueError, "trusted reconstruction"):
                replay_case(self.p, folder, blob)

    def test_freeform_provenance_does_not_license_unrelated_user_images(self):
        self.p.source_id = "user_csv:my_new_ksdd2_style_line"
        other = run_project(self.p, "validation", self.root / "runs")
        a = inspect_case(build_case(self.p, other, "001", "clean", 0))["attribution"]
        self.assertEqual(a["source_id"], self.p.source_id)
        self.assertNotIn("license", a)
        self.assertNotIn("license_url", a)

    def test_new_execution_is_a_separate_history_of_the_same_specimen(self):
        other = run_project(self.p, "validation", self.root / "runs")
        blob = build_case(self.p, other, "001", "dark", 0)
        self.assertEqual(inspect_case(blob)["case_id"], inspect_case(self.blob)["case_id"])
        self.assertNotEqual(inspect_case(blob)["history_id"], inspect_case(self.blob)["history_id"])
        with self.assertRaises(ValueError):
            replay_case(self.p, self.run, blob)
        for run, data in ((self.run, self.blob), (other, blob)):
            save_case(self.p, run, data, self.root / "cases")
        saved = list_cases(self.root / "cases", self.p.id)
        self.assertEqual(len(saved), 2)
        self.assertEqual(len({s["case_id"] for s in saved}), 1)

    def test_frozen_test_allows_historical_replay_only(self):
        run = run_project(self.p, "test", self.root / "runs")
        blob = build_case(self.p, run, "001", "clean", 0)
        self.assertTrue(replay_case(self.p, run, blob)["verified"])
        self.assertEqual(inspect_case(blob)["role"], "frozen_test_replay_only")
        calls = len(self.p.models["candidate"].calls)
        with self.assertRaisesRegex(ValueError, "validation"):
            retest_case(self.p, run, blob, "candidate", self.root / "results")
        self.assertEqual(calls, len(self.p.models["candidate"].calls))
        self.assertFalse((self.root / "results").exists())

    def test_reject_tampered_truth_predictions_ids_and_arrays_before_model_calls(self):
        metadata = inspect_case(self.blob)
        for key, value in (("true_label", "normal"), ("case_id", "a" * 64), ("run_id", "another-run"),
                           ("score_tolerance", {"atol": 1e9, "rtol": 2})):
            with self.subTest(key=key):
                altered = replace(self.blob, "case.json", json.dumps({**metadata, key: value}).encode())
                with self.assertRaises(ValueError):
                    replay_case(self.p, self.run, altered)
        wrong = io.BytesIO()
        np.save(wrong, np.ones_like(self.p.arr))
        for name in ("original.npy", "input.npy"):
            with self.assertRaisesRegex(ValueError, "hash|reconstruction"):
                retest_case(self.p, self.run, replace(self.blob, name, wrong.getvalue()),
                            "candidate", self.root / "results")
        altered = replace(self.blob, "input.npy", wrong.getvalue())
        forged = {**metadata, "input_npy_sha256": hashlib.sha256(wrong.getvalue()).hexdigest()}
        with self.assertRaisesRegex(ValueError, "trusted run|historical run"):
            replay_case(self.p, self.run, replace(altered, "case.json", json.dumps(forged).encode()))
        predictions = metadata["predictions"]
        predictions["old"]["scores"]["defect"] = 10.0
        with self.assertRaises(ValueError):
            replay_case(self.p, self.run, replace(self.blob, "case.json", json.dumps(metadata).encode()))
        self.assertFalse((self.root / "results").exists())
        # Tamper the local checked prediction file as well: the selected run detects it.
        path = Path(self.run) / "predictions.csv"
        path.write_bytes(path.read_bytes() + b"\n")
        with self.assertRaises(ValueError):
            replay_case(self.p, self.run, self.blob)

    def test_malformed_archive_and_oversized_npy_header_rejected_before_allocation(self):
        header = io.BytesIO()
        np.lib.format.write_array_header_1_0(header, {"descr": "<f4", "fortran_order": False, "shape": (2**40, 7)})
        with self.assertRaisesRegex(ValueError, "shape"):
            inspect_inputs(replace(self.blob, "input.npy", header.getvalue()))
        with self.assertRaises(ValueError):
            inspect_inputs(b"not a zip")
        out = io.BytesIO()
        with zipfile.ZipFile(out, "w") as z:
            z.writestr("../case.json", b"{}")
            z.writestr("original.npy", b"")
            z.writestr("input.npy", b"")
        with self.assertRaisesRegex(ValueError, "member"):
            inspect_inputs(out.getvalue())


if __name__ == "__main__":
    unittest.main()
