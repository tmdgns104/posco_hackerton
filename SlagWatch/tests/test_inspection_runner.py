"""Public runner contract tests; tiny new inputs and adapters require no stored prediction table."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.inspection_runner import hash_raw, load_run, recompute_decision, run_project


CELLS = [("clean", 0), ("noise_weak", 101), ("noise_weak", 102),
         ("noise_strong", 101), ("noise_strong", 102), ("dark", 0), ("bright", 0)]


def read_csv(folder, name):
    return pd.read_csv(folder/(name+".csv"), dtype={"sample_id": str, "group_id": str}, keep_default_na=False)


class ImageAdapter:
    def __init__(self, name, rule):
        self.identity = {"id": name, "source": "synthetic-test-v1", "device": "cpu"}
        self.rule, self.hashes = rule, []

    def predict(self, images):
        result = []
        for raw in images:
            result.append(self.rule(int(round(raw[0, 0]*255)), int(round(raw[1, 0]*10)), raw))
            self.hashes.append(hash_raw(raw))
        return result


def prediction(label):
    return {"status": "ok", "predicted_label": label, "score_type": "decision_function",
            "scores": {"defect": 1.01 if label == "defect" else -1.03}}


def binary_rule(misses, alarms):
    return lambda i, cell, raw: prediction(("defect" if i not in misses else "normal")
                                           if i < 10 else ("defect" if i in alarms else "normal"))


class Project:
    id = "new-inspection-project"
    kind = "binary"
    official = True
    classes = ["normal", "defect"]
    protocol = {"score_atol": 1e-5, "conditions": [dict(condition=c, seed=s) for c, s in CELLS]}

    def __init__(self):
        self.labels = ["defect"]*10 + ["normal"]*20
        self.models = {"previous-new": ImageAdapter("previous-new", binary_rule({0, 1, 2, 3}, {10, 11, 12})),
                       "candidate-new": ImageAdapter("candidate-new", binary_rule({2, 4}, {11, 13}))}
        self.loads, self.transforms = 0, 0

    def check_scope(self, split):
        if split != "validation":
            raise ValueError("test is locked")

    def rows(self, split):
        return pd.DataFrame([dict(sample_id=f"{i:03}", image_path=f"source/{i}.png", true_label=label,
                                  group_id=f"g{i}", split=split, source_id="new-dataset",
                                  file_sha256=hashlib.sha256(bytes([i])).hexdigest())
                             for i, label in enumerate(self.labels)])

    def conditions(self, split):
        return CELLS

    def load(self, row):
        self.loads += 1
        return np.full((2, 2), int(row.sample_id)/255, dtype=np.float32)

    def change(self, original, row, condition, seed):
        self.transforms += 1
        raw = original.copy()
        raw[1, 0] = CELLS.index((condition, seed)) / 10
        return raw


class RunnerContract(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_complete_new_inference_and_four_paired_changes(self):
        project = Project()
        steps = []
        run = run_project(project, "validation", self.root, progress=lambda done, total: steps.append((done, total)))
        manifest = load_run(run)
        decision = recompute_decision(run)
        self.assertEqual((manifest["n_originals"], manifest["planned_pairs"], manifest["completed_predictions"]),
                         (30, 210, 420))
        old, new = project.models.values()
        self.assertEqual(old.hashes, new.hashes)
        self.assertEqual(len(set(old.hashes)), 210)
        self.assertEqual((project.loads, project.transforms, steps[0], steps[-1]), (30, 210, (0, 210), (210, 210)))
        self.assertEqual(decision["status"], "pass")  # equality at both absolute limits qualifies
        self.assertEqual(decision["representative"]["models"]["candidate-new"]["weighted_fn"], 2)
        summary = read_csv(run, "summary").query("condition == 'clean'").set_index("model_id")
        for name, counts in (("previous-new", [6, 4, 3, 17]), ("candidate-new", [8, 2, 2, 18])):
            self.assertEqual(summary.loc[name, ["tp", "fn", "fp", "tn"]].tolist(), counts)
        changes = read_csv(run, "changes").query("scope == 'all'").iloc[0]
        self.assertEqual(changes[["fixed_fn", "new_fn", "fixed_fp", "new_fp", "both_wrong", "both_right"]].tolist(),
                         [21, 7, 14, 7, 14, 147])
        ref = read_csv(run, "baselines").query("scope == 'all'").set_index("model_id")
        self.assertEqual(ref.loc["all-normal", ["tp", "fn", "fp", "tn"]].tolist(), [0, 70, 0, 140])
        self.assertEqual(ref.loc["all-defect", ["tp", "fn", "fp", "tn"]].tolist(), [70, 0, 140, 0])
        shapes = read_csv(run, "pairs")[["original_input_shape", "raw_input_shape", "original_input_dtype", "raw_input_dtype"]]
        self.assertEqual(shapes.drop_duplicates().values.tolist(), [["[2,2]", "[2,2]", "float32", "float32"]])
        self.assertEqual(set(read_csv(run, "classes").query("condition == 'clean'").class_label), {"defect", "normal"})
        self.assertEqual(read_csv(run, "data_manifest").sample_id.iloc[0], "000")
        second = run_project(project, "validation", self.root)
        self.assertNotEqual(run, second)
        for file in ("predictions.csv", "pairs.csv", "changes.csv", "decision.json"):
            self.assertEqual((run/file).read_bytes(), (second/file).read_bytes())
        (run/"pairs.csv").write_bytes((run/"pairs.csv").read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "modified"):
            load_run(run)

    def test_failures_are_individual_and_denominators_do_not_disappear(self):
        project = Project()
        original_load = project.load
        def decode(row):
            if row.sample_id == "000":
                raise OSError("unreadable fixture")
            return original_load(row)
        project.load = decode
        original_transform = project.change
        def transform(raw, row, name, seed):
            if row.sample_id == "011" and name == "dark":
                return raw[:1, :]
            return original_transform(raw, row, name, seed)
        project.change = transform
        old, new = project.models.values()
        normal_candidate = new.rule
        def tricky(i, cell, raw):
            if i == 1:  # crashes a batch, but every unaffected sample must still run
                raise RuntimeError("image rejected")
            if i == 2:
                return dict(prediction("defect"), scores={"defect": float("nan")})
            if i == 3:
                return dict(prediction("defect"), status="deferred")
            if i == 4:
                raw.setflags(write=True)
                raw[:] = 0
            if i == 5:
                return dict(prediction("defect"), scores={})
            return normal_candidate(i, cell, raw)
        new.rule = tricky
        run = run_project(project, "validation", self.root)
        manifest, verdict = load_run(run), recompute_decision(run)
        self.assertEqual(verdict["status"], "incomplete")
        self.assertEqual((manifest["planned_pairs"], manifest["failed_pairs"], manifest["failed_predictions"]),
                         (210, 43, 51))
        summary = read_csv(run, "summary").query("scope == 'all'").set_index("model_id")
        self.assertEqual(summary.loc["candidate-new", ["planned", "completed", "n_defect", "failed_defect", "recall"]].tolist(),
                         [210, 167, 70, 42, ""])
        self.assertEqual(summary.loc["previous-new", "failed"], 8)
        pairs = read_csv(run, "pairs")
        self.assertTrue(pairs[pairs.sample_id == "000"].raw_input_hash.eq("").all())
        self.assertTrue(pairs[pairs.sample_id == "004"].before_status.eq("ok").all())
        self.assertTrue(pairs[pairs.sample_id == "004"].after_pred.eq("").all())
        self.assertTrue(pairs[pairs.sample_id == "009"].after_status.eq("ok").all())
        failures = read_csv(run, "failures")
        self.assertEqual(len(failures), 51)
        self.assertTrue({"load", "transform", "output", "predict"} <= set(failures.error.str.split(":").str[0]))
        for row in read_csv(run, "changes").itertuples():
            self.assertEqual(row.planned, row.failed+row.both_wrong+row.both_right+row.fixed_fn+row.new_fn+row.fixed_fp+row.new_fp)

    def test_first_model_cannot_change_the_other_models_input(self):
        project = Project()
        old, new = project.models.values()
        old_rule, new_rule = old.rule, new.rule
        seen = [{}, {}]
        def mutate(i, cell, raw):
            seen[0][i, cell] = id(raw)
            result = old_rule(i, cell, raw)
            if i == 7:
                raw.shape = (4,)  # metadata is mutable even though the pixels are not
            return result
        def verify(i, cell, raw):
            seen[1][i, cell] = id(raw)
            self.assertEqual(raw.shape, (2, 2))
            return new_rule(i, cell, raw)
        old.rule, new.rule = mutate, verify
        run = run_project(project, "validation", self.root)
        pairs = read_csv(run, "pairs")
        self.assertTrue(pairs.after_status.eq("ok").all())
        self.assertEqual(set(pairs[pairs.before_status == "failed"].sample_id), {"007"})
        self.assertTrue(all(seen[0][k] == v for k, v in seen[1].items() if k[0] != 7))
        self.assertEqual(load_run(run)["failed_pairs"], 7)

    def test_subset_even_when_all_ids_named_and_absent_truth_class(self):
        project = Project()
        ids = project.rows("validation").sample_id
        run = run_project(project, "validation", self.root, sample_ids=ids)
        self.assertEqual((load_run(run)["scope"], recompute_decision(run)["status"]), ("subset", "incomplete"))
        only_defects = run_project(project, "validation", self.root, sample_ids=ids[:3])
        s = read_csv(only_defects, "summary")
        self.assertTrue(s.fpr.eq("").all())
        self.assertTrue(s.n_normal.eq(0).all())
        with self.assertRaisesRegex(ValueError, "unknown"):
            run_project(project, "validation", self.root, sample_ids=["not-an-id"])

    def test_tradeoff_and_clean_noninferiority_are_checked_separately(self):
        project = Project()
        project.models["previous-new"].rule = binary_rule({0, 1, 2, 3}, set())
        new_rule = project.models["candidate-new"].rule
        project.models["candidate-new"].rule = lambda i, cell, raw: (
            prediction("defect") if i == 10 and cell == 0 else binary_rule({2}, set())(i, cell, raw))
        decision = recompute_decision(run_project(project, "validation", self.root))
        fails = [(r["id"], r["condition"]) for r in decision["criteria"] if r["passed"] is False]
        self.assertEqual(decision["status"], "fail")
        self.assertIn(("relative_fpr", "clean"), fails)
        self.assertIn(("representative_improvement", "ALL"), fails)
        self.assertFalse(any(name.startswith("absolute_") for name, _ in fails))
        project.models["candidate-new"].rule = new_rule

    def test_locked_scope_and_interrupt_cannot_publish(self):
        project = Project()
        with self.assertRaisesRegex(ValueError, "locked"):
            run_project(project, "test", self.root)
        self.assertEqual(project.loads, 0)
        self.assertFalse(list(self.root.iterdir()))
        def interrupt(done, total):
            if done:
                raise RuntimeError("disconnect")
        with self.assertRaisesRegex(RuntimeError, "disconnect"):
            run_project(project, "validation", self.root, progress=interrupt)
        stage, = list(self.root.iterdir())
        self.assertTrue(stage.name.startswith("."))
        self.assertEqual(json.loads((stage/"manifest.json").read_text())["status"], "incomplete")
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            load_run(stage)

    def test_multiclass_uses_own_macro_f1_and_never_binary_metrics(self):
        project = Project()
        project.kind, project.classes = "multiclass", list("abcdef")
        project.labels = list("abcdef")*2
        project.frozen_status = {"phase": "complete", "adopt": True}
        project.protocol = {"goals": dict(min_baseline_clean=.85, min_clean=.85, max_clean_loss=.02,
                                min_noise_gain=.03, min_noise_mean=.80, max_class_clean_recall_loss=.05,
                                max_per_sigma_loss=.02)}
        project.conditions = lambda split: [("clean", 0), ("gaussian_002", 1), ("gaussian_002", 2)]
        project.change = lambda raw, row, c, s: np.array([[raw[0, 0], 0], [s/10, 0]], dtype=np.float32)
        def multi(i, cell, old):
            label = project.classes[(i + int(cell > 0 and old)) % 6]
            return dict(predicted_label=label, scores={k: float(k == label) for k in project.classes},
                        score_type="decision_function", status="ok")
        project.models = {mid: ImageAdapter(mid, lambda i, cell, _, old=old: multi(i, cell, old))
                          for mid, old in (("original6", True), ("fresh6", False))}
        run = run_project(project, "validation", self.root)
        decision = recompute_decision(run)
        self.assertEqual((decision["status"], decision["frozen_status"]["adopt"]), ("pass", True))
        summary = read_csv(run, "summary")
        self.assertFalse({"tp", "fp", "fn", "tn", "fpr"} & set(summary))
        self.assertEqual(set(read_csv(run, "classes").class_label), set(project.classes))
        self.assertFalse((run/"baselines.csv").exists())
        changes = read_csv(run, "changes").query("scope == 'all'").iloc[0]
        self.assertEqual((changes.fixed, changes.new_error, changes.both_right, changes.both_wrong), (24, 0, 12, 0))
        subset = run_project(project, "validation", self.root, sample_ids=["000", "001", "002", "003", "004", "005"])
        self.assertEqual((recompute_decision(subset)["status"], load_run(subset)["frozen_status"]["adopt"]),
                         ("incomplete", True))


if __name__ == "__main__":
    unittest.main()
