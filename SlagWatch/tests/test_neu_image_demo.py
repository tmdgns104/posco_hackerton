import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import neu_image_demo as demo, security


class NeuImageDemoTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.root = root / "demo"
        for item in (patch.object(security, "SECURITY_ROOT", root / "security"),
                     patch.object(security, "AUDIT_ROOT", root / "audit"),
                     patch.dict("os.environ", {"STEELTRUST_AUDIT_URL": "", "STEELTRUST_AUDIT_BEARER": ""})):
            item.start()
            self.addCleanup(item.stop)
        security.bootstrap_admin("tester", "a sufficiently long password")
        self.token = security.login("tester", "a sufficiently long password")

    def test_registration_comparison_activation_and_retest(self):
        with self.assertRaises(security.SecurityError):
            demo.register_builtin(self.root, "V1", auth_token=None)
        for model in demo.MODEL_IDS:
            demo.register_builtin(self.root, model, auth_token=self.token)
        self.assertEqual([x["model_id"] for x in demo.registered_models(self.root)], ["V1", "V2"])
        report = demo.compare(self.root, "V1", "V2")
        self.assertEqual(report["n_samples"], 12)
        self.assertEqual(report["correct_before"], 12)
        self.assertEqual(report["correct_after"], 12)
        self.assertTrue(report["gate_passed"])
        with patch.object(security, "send_audit", side_effect=security.SecurityError("audit failure")):
            with self.assertRaises(security.SecurityError):
                demo.activate(self.root, report["run_id"], confirmed=True, auth_token=self.token)
        self.assertEqual(demo.active_model(self.root), "V1")
        state = demo.activate(self.root, report["run_id"], confirmed=True, auth_token=self.token)
        self.assertEqual(state["model_id"], "V2")
        self.assertEqual(demo.active_model(self.root), "V2")
        self.assertEqual(demo.evaluate_active(self.root), {"model_id": "V2", "samples": 12, "correct": 12})
        with self.assertRaisesRegex(ValueError, "변경"):
            demo.activate(self.root, report["run_id"], confirmed=True, auth_token=self.token)

    def test_tampered_comparison_is_not_accepted(self):
        for model in demo.MODEL_IDS:
            demo.register_builtin(self.root, model, auth_token=self.token)
        report = demo.compare(self.root, "V1", "V2")
        path = self.root / "runs" / (report["run_id"] + ".json")
        report["correct_after"] = 99
        path.write_text(json.dumps(report), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "변경"):
            demo.activate(self.root, report["run_id"], confirmed=True, auth_token=self.token)
        self.assertEqual(demo.active_model(self.root), "V1")

    def test_ui_selects_new_comparison_after_replacement(self):
        from streamlit.testing.v1 import AppTest

        for model in demo.MODEL_IDS:
            demo.register_builtin(self.root, model, auth_token=self.token)
        with patch("src.p1_tabular_ui.NEU_DEMO_ROOT", self.root), \
                patch("src.p1_tabular_ui.ROOT", self.root.parent / "tabular"):
            app = AppTest.from_string("from src.p1_tabular_ui import page\npage()", default_timeout=120)
            app.session_state["security-token"] = self.token
            app.run()
            next(x for x in app.button if x.label == "V1·V2 동일 이미지 비교 실행").click().run()
            next(x for x in app.checkbox if x.label.startswith("공식 승인이나 설비 배포")).check()
            next(x for x in app.button if x.label == "이미지 시험 활성 모델 교체").click().run()
            self.assertEqual(demo.active_model(self.root), "V2")
            next(x for x in app.button if x.label == "V1·V2 동일 이미지 비교 실행").click().run()
            self.assertFalse(app.exception)
            run_id = next(x for x in app.selectbox if x.label == "이미지 비교 기록").value
            report = demo.load_run(self.root, run_id)
            self.assertEqual((report["before_model_id"], report["after_model_id"]), ("V2", "V1"))


if __name__ == "__main__":
    unittest.main()
