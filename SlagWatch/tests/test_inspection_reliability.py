import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from src.inspection_governance import freeze_design, split_conflicts, verify_design
from src.inspection_reliability import classify_pairs, defect_size_summary, group_bootstrap, transition_summary


def pair(sample, truth, before, after, condition="clean", seed=0, before_status="ok", after_status="ok"):
    return dict(sample_id=sample, true_label=truth, before_pred=before, after_pred=after,
                condition=condition, seed=seed, before_status=before_status, after_status=after_status)


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.pairs = pd.DataFrame([
            pair("d1", "defect", "defect", "normal"),
            pair("d2", "defect", "normal", "defect"),
            pair("d3", "defect", "normal", "normal"),
            pair("d4", "defect", "defect", "defect"),
            pair("n1", "normal", "normal", "defect"),
            pair("n2", "normal", "defect", "normal"),
            pair("n3", "normal", "defect", "defect"),
            pair("n4", "normal", "normal", "normal"),
            pair("d1", "defect", "defect", "defect", "noise", 1),
        ])

    def test_eight_transitions_and_repeated_original(self):
        self.assertEqual(classify_pairs(self.pairs).transition.tolist(),
                         ["new_fn", "fixed_fn", "persistent_fn", "stable_tp", "new_fp",
                          "fixed_fp", "persistent_fp", "stable_tn", "stable_tp"])
        row = transition_summary(self.pairs).iloc[0]
        self.assertEqual((row.originals, row.decisions, row.new_fn, row.fixed_fn), (8, 9, 1, 1))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            classify_pairs(pd.concat([self.pairs, self.pairs.iloc[:1]]))

    def test_bootstrap_paired_reproducibility_and_weighting(self):
        one = group_bootstrap(self.pairs, repetitions=500, seed=7)
        pd.testing.assert_frame_equal(one, group_bootstrap(self.pairs, repetitions=500, seed=7))
        old = one.query("metric == 'fn_rate' and model == 'before'").iloc[0]
        self.assertAlmostEqual(old.estimate, 2 / 5)
        equal = group_bootstrap(self.pairs, repetitions=500, seed=7, weighting="originals")
        self.assertAlmostEqual(equal.query("metric == 'fn_rate' and model == 'before'").iloc[0].estimate, 2 / 4)
        delta = one.query("metric == 'fn_rate' and model == 'difference'").iloc[0]
        self.assertAlmostEqual(delta.estimate, 0)
        rare = self.pairs[self.pairs.sample_id.isin(["d1", "n1", "n2", "n3", "n4"])]
        rare_out = group_bootstrap(rare, repetitions=500, seed=7)
        self.assertLess(rare_out.query("metric == 'fn_rate' and model == 'difference'").iloc[0].valid_repetitions, 500)

    def test_missing_class_and_failed_prediction_are_unavailable(self):
        negatives = self.pairs[self.pairs.true_label.eq("normal")]
        out = group_bootstrap(negatives, repetitions=100, seed=1)
        recall = out.query("metric == 'recall'")
        self.assertTrue(recall.estimate.isna().all())
        self.assertTrue(recall.valid_repetitions.eq(0).all())
        broken = self.pairs.copy()
        broken.loc[0, "after_status"] = "failed"
        out = group_bootstrap(broken, repetitions=100, seed=1)
        rows = out[out.metric.eq("recall")].set_index("model")
        self.assertTrue(pd.isna(rows.loc["after", "estimate"]))
        self.assertTrue(pd.isna(rows.loc["difference", "estimate"]))
        self.assertFalse(pd.isna(rows.loc["before", "estimate"]))
        self.assertEqual(classify_pairs(broken).transition.iloc[0], "failed")

    def test_mask_area_thresholds_and_missing_annotations(self):
        index = pd.DataFrame([{"sample_id": f"d{i}", "defect_pixels": pixels, "width": 100, "height": 100}
                              for i, pixels in enumerate((5, 10, 100, 500), 1)])
        out = defect_size_summary(self.pairs, index, small_below=.001, medium_below=.01).set_index("size")
        self.assertEqual(out.originals.tolist(), [1, 1, 2])
        self.assertEqual(out.loc["small", "new_fn"], 1)
        with self.assertRaisesRegex(ValueError, "Missing"):
            defect_size_summary(self.pairs, index.iloc[:-1], small_below=.001, medium_below=.01)
        with self.assertRaisesRegex(ValueError, "boundaries"):
            defect_size_summary(self.pairs, index, small_below=.01, medium_below=.001)

    def test_real_existing_counts(self):
        run = Path(__file__).resolve().parents[1] / "inspection_runs" / \
            "20261108T193331597782Z-ca753bd910bc42fcb3b785eda802f8f9" / "pairs.csv"
        if not run.exists():
            self.skipTest("Bundled full validation run not available")
        row = transition_summary(pd.read_csv(run, keep_default_na=False)).iloc[0]
        self.assertEqual((row.originals, row.decisions, row.new_fn, row.fixed_fn, row.new_fp, row.fixed_fp),
                         (466, 3262, 34, 6, 71, 409))


class GovernanceTests(unittest.TestCase):
    def test_cross_split_leakage_and_frozen_threshold(self):
        class Adapter:
            identity = {"model_id": "K1", "threshold": 0.5}

        class Project:
            id = "test"
            index_hash = "index"
            protocol_file_sha256 = "protocol"
            models = {"K1": Adapter()}
            index = pd.DataFrame([
                {"sample_id": "a", "split": "train", "file_sha256": "hash1", "group_id": "g1"},
                {"sample_id": "b", "split": "validation", "file_sha256": "hash2", "group_id": "g2"},
            ])

            def check_scope(self, split):
                if split != "validation":
                    raise ValueError("locked")

            def conditions(self, split):
                return [("clean", 0)]

        project = Project()
        self.assertTrue(split_conflicts(project.index).empty)
        duplicate = pd.concat([project.index, project.index.iloc[[0]].assign(split="test")])
        self.assertEqual(set(split_conflicts(duplicate).field), {"sample_id", "file_sha256", "group_id"})
        with tempfile.TemporaryDirectory() as tmp:
            path = freeze_design(project, "validation", Path(tmp), actor="tester",
                                 hypothesis="hypothesis", criteria="criteria")
            self.assertEqual(verify_design(path, project, "validation")["actor_self_reported"], "tester")
            project.models["K1"].identity = {"model_id": "K1", "threshold": 0.6}
            with self.assertRaisesRegex(ValueError, "differs"):
                verify_design(path, project, "validation")
            record = json.loads(path.read_text(encoding="utf-8"))
            record["hypothesis"] = "altered"
            path.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "modified"):
                verify_design(path, project, "validation")
            with self.assertRaisesRegex(ValueError, "locked"):
                freeze_design(project, "test", Path(tmp), actor="tester", hypothesis="h", criteria="c")


if __name__ == "__main__":
    unittest.main()
