import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib

from src import neu_uploaded_models as uploaded, security
from src.inspection_projects import load_project
from src.surface_data import RUN


class UploadedModelsTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        self.root = root / "models"
        for item in (patch.object(security, "SECURITY_ROOT", root / "security"),
                     patch.object(security, "AUDIT_ROOT", root / "audit"),
                     patch.dict("os.environ", {"STEELTRUST_AUDIT_URL": "", "STEELTRUST_AUDIT_BEARER": ""})):
            item.start()
            self.addCleanup(item.stop)
        security.bootstrap_admin("tester", "a sufficiently long password")
        self.token = security.login("tester", "a sufficiently long password")
        self.bundle = joblib.load(RUN / "models" / "V1.joblib")

    def payload(self):
        buffer = io.BytesIO()
        joblib.dump(self.bundle, buffer)
        return buffer.getvalue()

    def test_registration_authorization_contract_and_tamper(self):
        payload = self.payload()
        with self.assertRaises(security.SecurityError):
            uploaded.register(self.root, "custom1", payload, "m.joblib", trusted=True, auth_token=None)
        with patch.object(security, "send_audit", side_effect=security.SecurityError("audit failure")):
            with self.assertRaises(security.SecurityError):
                uploaded.register(self.root, "custom1", payload, "m.joblib", trusted=True, auth_token=self.token)
        self.assertEqual(uploaded.list_models(self.root), [])
        with self.assertRaises(ValueError):
            uploaded.register(self.root, "custom1", payload, "m.joblib", trusted=False, auth_token=self.token)
        with self.assertRaisesRegex(ValueError, "로드하지"):
            uploaded.register(self.root, "custom1", b"invalid", "m.joblib", trusted=True, auth_token=self.token)
        self.bundle["encoder_sha"] = "wrong"
        with self.assertRaisesRegex(ValueError, "일치"):
            uploaded.register(self.root, "custom1", self.payload(), "m.joblib", trusted=True, auth_token=self.token)
        uploaded.register(self.root, "custom1", payload, "m.joblib", trusted=True, auth_token=self.token)
        path = self.root / "models" / "custom1" / "model.joblib"
        path.write_bytes(payload + b"modified")
        with self.assertRaisesRegex(ValueError, "변경"):
            uploaded.evaluate(self.root, "custom1", auth_token=self.token)

    def test_uploaded_candidate_is_actually_evaluated_and_persisted(self):
        classifier = self.bundle["head"][-1]
        classifier.coef_[:] = 0
        classifier.intercept_[:] = 0
        classifier.intercept_[0] = 100
        uploaded.register(self.root, "custom1", self.payload(), "custom.joblib",
                          trusted=True, auth_token=self.token)
        project = load_project("neu_cls6")
        ids = project.rows("validation").groupby("true_label").head(1).sample_id.tolist()
        path = uploaded.evaluate(self.root, "custom1", auth_token=self.token, sample_ids=ids)
        self.assertIn(path.name, uploaded.list_runs(self.root))
        cfg, pairs = uploaded.load_run(self.root, path.name)
        self.assertEqual(cfg["after_model_id"], "custom1")
        self.assertEqual(cfg["scope"], "subset")
        self.assertEqual(cfg["failed_pairs"], 0)
        self.assertEqual(len(pairs), len(ids) * len(project.conditions("validation")))
        self.assertTrue(pairs.after_pred.eq(project.classes[0]).all())
        self.assertGreater(pairs.before_pred.ne(pairs.after_pred).sum(), 0)
        from streamlit.testing.v1 import AppTest
        with patch.object(uploaded, "REGISTRY", self.root):
            app = AppTest.from_string("from src.neu_dashboard_ui import _uploaded_results\n_uploaded_results()",
                                      default_timeout=120).run()
            self.assertFalse(app.exception)
            self.assertTrue(any("custom1" in text.value for text in app.markdown))
            expected_before = int(pairs.before_pred.ne(pairs.true_label).sum())
            expected_after = int(pairs.after_pred.ne(pairs.true_label).sum())
            self.assertEqual(app.metric[0].value, f"{expected_before:,} → {expected_after:,}회")
        (path / "pairs.csv").write_text("modified", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "modified"):
            uploaded.load_run(self.root, path.name)

    def test_failed_predictions_are_visible_and_excluded(self):
        uploaded.register(self.root, "failure", self.payload(), "m.joblib", trusted=True, auth_token=self.token)
        project = load_project("neu_cls6")
        sample = project.rows("validation").iloc[0].sample_id
        with patch.object(uploaded.BundleAdapter, "predict", side_effect=RuntimeError("test inference failure")):
            path = uploaded.evaluate(self.root, "failure", auth_token=self.token, sample_ids=[sample])
        cfg, pairs = uploaded.load_run(self.root, path.name)
        self.assertEqual(cfg["failed_pairs"], len(pairs))
        from streamlit.testing.v1 import AppTest
        with patch.object(uploaded, "REGISTRY", self.root):
            app = AppTest.from_string("from src.neu_dashboard_ui import _uploaded_results\n_uploaded_results()",
                                      default_timeout=120).run()
            self.assertFalse(app.exception)
            self.assertEqual(app.metric[0].value, "0 → 0회")
            self.assertTrue(any("불완전" in warning.value for warning in app.warning))

    def test_model_comparison_page_selects_registered_models_and_preserves_existing_result(self):
        from streamlit.testing.v1 import AppTest

        payload = self.payload()
        for model in ("candidate_a", "candidate_b", "unevaluated"):
            uploaded.register(self.root, model, payload, "m.joblib", trusted=True, auth_token=self.token)
        project = load_project("neu_cls6")
        sample = project.rows("validation").iloc[0].sample_id
        first = uploaded.evaluate(self.root, "candidate_a", auth_token=self.token, sample_ids=[sample])
        second = uploaded.evaluate(self.root, "candidate_b", auth_token=self.token, sample_ids=[sample])
        cfg_a, pairs_a = uploaded.load_run(self.root, first.name)
        cfg_b, pairs_b = uploaded.load_run(self.root, second.name)
        with patch.object(uploaded, "REGISTRY", self.root):
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app_demo.py"), default_timeout=120).run()
            next(button for button in app.button if button.label == "모델 비교").click().run()
            self.assertFalse(app.exception)
            self.assertTrue(any(item.value == "기존 모델 비교" for item in app.subheader))
            self.assertTrue(any("전체 및 원본 성능" in item.value for item in app.subheader))
            self.assertEqual(next(item for item in app.selectbox if item.label == "비교할 등록 모델").value,
                             "candidate_b")
            self.assertEqual(next(item for item in app.selectbox if item.label == "업로드 모델 비교 기록").value,
                             second.name)
            self.assertTrue(any("V0 → candidate_b" in item.value for item in app.markdown))
            self.assertEqual(next(item for item in app.metric if item.label == "업로드 모델 오분류").value,
                             f"{pairs_b.before_pred.ne(pairs_b.true_label).sum():,} → "
                             f"{pairs_b.after_pred.ne(pairs_b.true_label).sum():,}회")
            next(item for item in app.selectbox if item.label == "비교할 등록 모델").select("candidate_a").run()
            self.assertFalse(app.exception)
            self.assertEqual(next(item for item in app.selectbox if item.label == "업로드 모델 비교 기록").value,
                             first.name)
            self.assertTrue(any("V0 → candidate_a" in item.value for item in app.markdown))
            next(item for item in app.selectbox if item.label == "비교할 등록 모델").select("unevaluated").run()
            self.assertFalse(app.exception)
            self.assertTrue(any("비교 기록이 없습니다" in item.value for item in app.info))
            self.assertTrue(any(item.label == "선택 모델과 V0 비교 실행" for item in app.button))
            app.session_state["security-token"] = self.token
            next(item for item in app.radio if item.label == "새 비교 표본").set_value("클래스별 시연 표본")
            with patch.object(type(project), "demo_ids", return_value=[sample]):
                next(item for item in app.button if item.label == "선택 모델과 V0 비교 실행").click().run()
            self.assertFalse(app.exception)
            self.assertTrue(any("V0 → unevaluated" in item.value for item in app.markdown))
            run_id = next(item for item in app.selectbox if item.label == "업로드 모델 비교 기록").value
            self.assertEqual(uploaded.load_run(self.root, run_id)[0]["after_model_id"], "unevaluated")
            self.assertEqual(cfg_a["after_model_id"], "candidate_a")
            self.assertEqual(cfg_b["after_model_id"], "candidate_b")


if __name__ == "__main__":
    unittest.main()
