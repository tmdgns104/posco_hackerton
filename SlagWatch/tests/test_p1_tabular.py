import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src import p1_tabular as tabular
from src import p1_abstention as abstention
from src import p1_confidence
from src import security


class TabularTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        security_root = patch.object(security, "SECURITY_ROOT", self.root / "security")
        audit_root = patch.object(security, "AUDIT_ROOT", self.root / "audit")
        audit_env = patch.dict("os.environ", {"STEELTRUST_AUDIT_URL": "", "STEELTRUST_AUDIT_BEARER": ""})
        for item in (security_root, audit_root, audit_env):
            item.start()
            self.addCleanup(item.stop)
        security.bootstrap_admin("tester", "a sufficiently long password")
        self.token = security.login("tester", "a sufficiently long password")
        self.features = pd.DataFrame({"temperature": [-4., -3., -2., -1., 1., 2., 3., 4.],
                                      "pressure": [-3., -2., -1., -1., 1., 1., 2., 3.]})
        self.labels = ["normal"] * 4 + ["defect"] * 4
        self.dataset = self.features.copy()
        self.dataset.insert(0, "true_label", self.labels)
        self.dataset.insert(0, "sample_id", [f"s{i}" for i in range(len(self.labels))])
        self.payload = self.dataset.to_csv(index=False).encode()
        for model_id, model in (("linear", LogisticRegression(random_state=7)),
                                ("forest", RandomForestClassifier(n_estimators=20, random_state=7))):
            model.fit(self.features, self.labels)
            data = io.BytesIO()
            joblib.dump(model, data)
            tabular.register_model(self.root, data.getvalue(), "model.joblib", model_id, model_id,
                                    "1", list(self.features.columns), trusted=True, auth_token=self.token)

    def test_compare_report_and_atomic_local_replacement(self):
        adapters = [tabular.TabularAdapter(self.root, x) for x in ("linear", "forest")]
        tabular.validate_dataset(self.payload, adapters)
        report = tabular.run_comparison(self.root, "linear", "forest", self.payload,
                                        seed=42, noise_std=0, positive="defect")
        self.assertEqual((report["status"], report["decision"]), ("COMPLETED", "PASS"))
        self.assertEqual(len(report["metrics"]), 4)
        self.assertEqual(report["metrics"][0]["accuracy"], 1.)
        self.assertEqual(report["metrics"][0]["fn"], 0)
        self.assertEqual(report["metrics"][0]["fp"], 0)
        self.assertEqual(tabular.load_run(self.root, report["run_id"]), report)
        folder = self.root / "runs" / report["run_id"]
        rows = pd.read_csv(folder / "summary.csv")
        self.assertEqual(rows.iloc[0].accuracy, report["metrics"][0]["accuracy"])
        self.assertEqual(len(pd.read_csv(folder / "pairs.csv")), 16)
        with self.assertRaisesRegex(tabular.ContractError, "CONFIRMATION_REQUIRED"):
            tabular.replace_model(self.root, report["run_id"], actor="tester", confirmed=False)
        self.assertIsNone(tabular.active_model(self.root))
        tabular.replace_model(self.root, report["run_id"], actor="tester", confirmed=True, auth_token=self.token)
        self.assertEqual(tabular.active_model(self.root), "forest")
        with self.assertRaisesRegex(tabular.ContractError, "MODEL_ACTIVE"):
            tabular.deactivate_model(self.root, "forest", auth_token=self.token)
        tabular.deactivate_model(self.root, "linear", auth_token=self.token)
        self.assertEqual(tabular.load_run(self.root, report["run_id"]), report)
        self.assertEqual(tabular.list_models(self.root)[1]["status"], "inactive")

    def test_contract_mismatch_and_untrusted_files(self):
        adapters = [tabular.TabularAdapter(self.root, x) for x in ("linear", "forest")]
        with self.assertRaisesRegex(tabular.ContractError, "TRUST_REQUIRED"):
            tabular.register_model(self.root, b"unsafe", "x.joblib", "bad", "bad", "1",
                                   ["temperature"], trusted=False)
        with self.assertRaisesRegex(tabular.ContractError, "SCORE_TYPE_UNSUPPORTED"):
            adapters[0].normalize_output(["defect"], np.array([[2., -1.]]))
        for frame, error in (
            (self.dataset.drop(columns="pressure"), "FEATURE_MISSING"),
            (self.dataset[["sample_id", "true_label", "pressure", "temperature"]], "FEATURE_ORDER_MISMATCH"),
            (self.dataset.assign(temperature="unknown"), "FEATURE_TYPE_MISMATCH"),
            (self.dataset.assign(true_label="other"), "CLASS_MAPPING_MISMATCH"),
        ):
            with self.subTest(error=error), self.assertRaisesRegex(tabular.ContractError, error):
                tabular.validate_dataset(frame.to_csv(index=False).encode(), adapters)

    def test_unauthorized_or_audit_failure_cannot_mutate(self):
        payload = (self.root / "models" / "linear" / "artifact.joblib").read_bytes()
        with self.assertRaises(security.SecurityError):
            tabular.register_model(self.root, payload, "x.joblib", "another", "another", "1",
                                   list(self.features.columns), trusted=True)
        self.assertFalse((self.root / "models" / "another").exists())
        with patch.object(security, "send_audit", side_effect=security.SecurityError("audit failure")):
            with self.assertRaises(security.SecurityError):
                tabular.register_model(self.root, payload, "x.joblib", "another", "another", "1",
                                       list(self.features.columns), trusted=True, auth_token=self.token)
            with self.assertRaises(security.SecurityError):
                tabular.deactivate_model(self.root, "linear", auth_token=self.token)
        self.assertFalse((self.root / "models" / "another").exists())
        self.assertEqual(tabular.list_models(self.root)[1]["status"], "ready")
        report = tabular.run_comparison(self.root, "linear", "forest", self.payload, seed=42, noise_std=0)
        with patch.object(security, "send_audit", side_effect=security.SecurityError("audit failure")):
            with self.assertRaises(security.SecurityError):
                tabular.replace_model(self.root, report["run_id"], actor="tester", confirmed=True, auth_token=self.token)
        self.assertIsNone(tabular.active_model(self.root))

    def test_completion_audit_failure_reports_possible_partial_success(self):
        report = tabular.run_comparison(self.root, "linear", "forest", self.payload, seed=42, noise_std=0)
        original = security.send_audit
        def fail_result(event):
            if event["phase"] == "result":
                raise security.SecurityError("result unavailable")
            return original(event)
        with patch.object(security, "send_audit", side_effect=fail_result):
            with self.assertRaisesRegex(security.SecurityError, "이미 저장"):
                tabular.replace_model(self.root, report["run_id"], actor="tester", confirmed=True, auth_token=self.token)
        self.assertEqual(tabular.active_model(self.root), "forest")

    def test_partial_run_and_tamper_cannot_replace(self):
        original = tabular.TabularAdapter.predict
        def faulty(adapter, data):
            if adapter.model_id == "forest":
                raise tabular.ContractError("SCORE_TYPE_UNSUPPORTED", "test failure")
            return original(adapter, data)
        with patch.object(tabular.TabularAdapter, "predict", faulty):
            report = tabular.run_comparison(self.root, "linear", "forest", self.payload, seed=1, noise_std=0)
        self.assertEqual(report["status"], "PARTIAL")
        self.assertGreater(report["failed_predictions"], 0)
        with self.assertRaisesRegex(tabular.ContractError, "EVALUATION_NOT_PASSED"):
            tabular.replace_model(self.root, report["run_id"], actor="tester", confirmed=True, auth_token=self.token)
        self.assertIsNone(tabular.active_model(self.root))
        path = self.root / "runs" / report["run_id"] / "summary.csv"
        path.write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(tabular.ContractError, "RUN_INTEGRITY_ERROR"):
            tabular.load_run(self.root, report["run_id"])

    def test_ui_registry_and_swap_do_not_change_active_model(self):
        from streamlit.testing.v1 import AppTest

        with patch("src.p1_tabular_ui.ROOT", self.root):
            app = AppTest.from_string("from src.p1_tabular_ui import page\npage()", default_timeout=120).run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            before = next(x for x in app.selectbox if x.label == "기준 모델").value
            after = next(x for x in app.selectbox if x.label == "후보 모델").value
            next(x for x in app.button if x.label == "기준·후보 바꾸기").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(next(x for x in app.selectbox if x.label == "기준 모델").value, after)
            self.assertEqual(next(x for x in app.selectbox if x.label == "후보 모델").value, before)
            self.assertIsNone(tabular.active_model(self.root))
            self.assertTrue(next(x for x in app.button if x.label == "동일 조건 비교 실행").disabled)

    def test_ui_confirmation_changes_local_active_model(self):
        from streamlit.testing.v1 import AppTest

        report = tabular.run_comparison(self.root, "linear", "forest", self.payload,
                                        seed=42, noise_std=0, positive="defect")
        self.assertEqual(report["decision"], "PASS")
        with patch("src.p1_tabular_ui.ROOT", self.root):
            app = AppTest.from_string("from src.p1_tabular_ui import page\npage()", default_timeout=120)
            app.session_state["security-token"] = self.token
            app.run()
            next(x for x in app.checkbox if x.label.startswith("위 제한을 이해")).check()
            next(x for x in app.button if x.label == "로컬 모델 교체 확인").click().run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            self.assertEqual(tabular.active_model(self.root), "forest")
            self.assertEqual(next(x for x in app.selectbox if x.label == "기준 모델").value, "forest")

    def test_paired_bootstrap_and_incomplete_data(self):
        report = tabular.run_comparison(self.root, "linear", "forest", self.payload,
                                        seed=42, noise_std=0, positive="defect")
        first = p1_confidence.analyze_run(self.root, report["run_id"], repetitions=200, seed=7)
        again = p1_confidence.analyze_run(self.root, report["run_id"], repetitions=200, seed=7)
        pd.testing.assert_frame_equal(first, again)
        accuracy = first[(first.condition == "ALL") & (first.metric == "accuracy")]
        self.assertEqual(set(accuracy.model), {"before", "after", "difference"})
        self.assertTrue(accuracy.original_groups.eq(8).all())
        self.assertTrue(accuracy.decisions.eq(16).all())
        self.assertEqual(accuracy.set_index("model").loc["difference", "estimate"], 0)
        self.assertTrue(accuracy.valid_resamples.eq(200).all())
        pairs = pd.read_csv(self.root / "runs" / report["run_id"] / "pairs.csv")
        pairs["true_label"] = "normal"
        result = p1_confidence.paired_intervals(pairs, positive="defect", repetitions=20)
        self.assertTrue(result[result.metric.eq("fn_rate")].estimate.isna().all())
        pairs.loc[0, "status_before"] = "failed"
        with self.assertRaisesRegex(ValueError, "미완료"):
            p1_confidence.paired_intervals(pairs, positive="defect", repetitions=20)

    def test_frozen_policy_independent_evaluation_and_zero_denominators(self):
        dev = tabular.run_comparison(self.root, "linear", "forest", self.payload,
                                     seed=42, noise_std=0, positive="defect")
        policies = [abstention.freeze_policy(self.root, dev["run_id"], model, "original", .5,
                                             name="test", version="1") for model in ("linear", "forest")]
        self.assertTrue(all(x["development_coverage"] >= .5 for x in policies))
        with self.assertRaisesRegex(tabular.ContractError, "POLICY_EVALUATION_INVALID"):
            abstention.evaluate_policies(self.root, [policies[0]["policy_id"]], dev["run_id"])
        separate = self.dataset.copy()
        separate["sample_id"] = [f"evaluation{i}" for i in range(len(separate))]
        evaluation = tabular.run_comparison(self.root, "linear", "forest",
                                            separate.to_csv(index=False).encode(),
                                            seed=42, noise_std=0, positive="defect")
        result = abstention.evaluate_policies(self.root, [x["policy_id"] for x in policies], evaluation["run_id"])
        self.assertEqual(abstention.load_experiment(self.root, result["experiment_id"]), result)
        for row in result["metrics"]:
            self.assertEqual(row["automatic"] + row["held"], row["eligible"])
            self.assertAlmostEqual(row["coverage"] + row["abstention_rate"], 1.)
            self.assertAlmostEqual(row["coverage_gap"], row["coverage"] - .5)
        data = pd.DataFrame([{"status": "ok", "scores_json": '{"defect": 0.1}',
                              "predicted_class": "defect", "true_label": "defect"}])
        m, _ = abstention.metrics(data, {"threshold": .9, "policy_id": "p", "model_id": "m",
                                        "target_coverage": .5}, "defect")
        self.assertIsNone(m["automatic_error_rate"])
        self.assertIsNone(m["fn_rate"])
        self.assertIsNone(m["fp_rate"])
        self.assertEqual(m["unnecessary_abstentions"], 1)
        examples = pd.DataFrame([
            {"status": "ok", "scores_json": '{"normal": 0.9}', "predicted_class": "normal", "true_label": "defect"},
            {"status": "ok", "scores_json": '{"defect": 0.9}', "predicted_class": "defect", "true_label": "normal"},
            {"status": "ok", "scores_json": '{"normal": 0.1}', "predicted_class": "normal", "true_label": "normal"},
        ])
        m, _ = abstention.metrics(examples, {"threshold": .5, "policy_id": "p", "model_id": "m",
                                            "target_coverage": .5}, "defect")
        self.assertEqual((m["fn"], m["fn_denominator"], m["fp"], m["fp_denominator"]), (1, 1, 1, 1))
        self.assertEqual((m["automatic_errors"], m["unnecessary_abstentions"]), (2, 1))


if __name__ == "__main__":
    unittest.main()
