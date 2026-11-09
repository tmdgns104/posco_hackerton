from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.inspection_activation import activate_model, activation_evidence, active_model, check_compatibility
from src.inspection_event_log import record_event
from src.inspection_model import ENCODER_SHA, PREP
from src.inspection_projects import load_project
from src.inspection_runner import run_project
from src.inspection_workspace import annotation_overlay
from src import security
from tests.test_inspection_runner import Project


class ActivationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for item in (patch.object(security, "SECURITY_ROOT", self.root / "security"),
                     patch.object(security, "AUDIT_ROOT", self.root / "audit"),
                     patch.dict("os.environ", {"STEELTRUST_AUDIT_URL": "", "STEELTRUST_AUDIT_BEARER": ""})):
            item.start()
            self.addCleanup(item.stop)
        security.bootstrap_admin("tester", "a sufficiently long password")
        self.token = security.login("tester", "a sufficiently long password")
        self.project = Project()
        self.project.template_id = "ksdd2"
        self.project.index_hash = "index"
        self.project.protocol_file_sha256 = "protocol"
        for name, adapter in self.project.models.items():
            adapter.identity.update(model_id=name, encoder_sha256=ENCODER_SHA, preprocessing=PREP,
                                    class_mapping={"0": "normal", "1": "defect"})
        self.before, self.after = list(self.project.models)

    def test_passed_full_run_confirms_and_atomically_switches_local_model(self):
        run = run_project(self.project, "validation", self.root / "runs")
        self.assertEqual(active_model(self.root / "state", self.project), self.before)
        self.assertEqual(activation_evidence(self.project, run, self.before, self.after, self.project.models)["status"], "pass")
        with self.assertRaisesRegex(ValueError, "confirmation"):
            activate_model(self.root / "state", self.project, run, self.before, self.after,
                           self.project.models, actor="tester", confirmed=False)
        self.assertEqual(active_model(self.root / "state", self.project), self.before)
        record = activate_model(self.root / "state", self.project, run, self.before, self.after,
                                self.project.models, actor="tester", confirmed=True, auth_token=self.token)
        self.assertEqual(record["run_id"], run.name)
        self.assertEqual(active_model(self.root / "state", self.project), self.after)
        with self.assertRaisesRegex(ValueError, "changed"):
            activate_model(self.root / "state", self.project, run, self.before, self.after,
                           self.project.models, actor="tester", confirmed=True, auth_token=self.token)

    def test_subset_wrong_identity_or_changed_dataset_never_changes_active_model(self):
        ids = self.project.rows("validation").sample_id.tolist()
        subset = run_project(self.project, "validation", self.root / "runs", sample_ids=ids)
        with self.assertRaisesRegex(ValueError, "complete validation"):
            activation_evidence(self.project, subset, self.before, self.after, self.project.models)
        run = run_project(self.project, "validation", self.root / "runs")
        self.project.models[self.after].identity["encoder_sha256"] = "changed"
        with self.assertRaisesRegex(ValueError, "incompatible"):
            check_compatibility(self.project, self.before, self.after, self.project.models)
        self.project.models[self.after].identity["encoder_sha256"] = ENCODER_SHA
        self.project.index_hash = "changed"
        with self.assertRaisesRegex(ValueError, "complete validation"):
            activate_model(self.root / "state", self.project, run, self.before, self.after,
                           self.project.models, actor="tester", confirmed=True, auth_token=self.token)
        self.assertEqual(active_model(self.root / "state", self.project), self.before)

    def test_authentication_and_audit_failure_block_activation(self):
        run = run_project(self.project, "validation", self.root / "runs")
        with self.assertRaises(security.SecurityError):
            activate_model(self.root / "state", self.project, run, self.before, self.after,
                           self.project.models, actor="tester", confirmed=True)
        with patch.object(security, "send_audit", side_effect=security.SecurityError("audit failure")):
            with self.assertRaises(security.SecurityError):
                activate_model(self.root / "state", self.project, run, self.before, self.after,
                               self.project.models, actor="tester", confirmed=True, auth_token=self.token)
        self.assertEqual(active_model(self.root / "state", self.project), self.before)

    def test_actual_k1_failing_validation_cannot_activate(self):
        project = load_project("ksdd2")
        run = Path(__file__).resolve().parents[1] / "inspection_runs" / \
            "20261108T193331597782Z-ca753bd910bc42fcb3b785eda802f8f9"
        with self.assertRaisesRegex(ValueError, "did not pass"):
            activate_model(self.root, project, run, "K0", "K1", project.models,
                           actor="tester", confirmed=True, auth_token=self.token)
        self.assertEqual(active_model(self.root, project), "K0")

    def test_original_annotation_overlay_matches_verified_run(self):
        project = load_project("ksdd2")
        run = Path(__file__).resolve().parents[1] / "inspection_runs" / \
            "20261108T193331597782Z-ca753bd910bc42fcb3b785eda802f8f9"
        from src.inspection_runner import load_run
        manifest = load_run(run)
        row = project.index[(project.index.split == "validation") & (project.index.true_label == "defect")].iloc[0]
        original = project.load(row)
        overlay = annotation_overlay(project, manifest, row.sample_id, original)
        self.assertEqual(overlay.shape, original.shape)
        self.assertTrue((overlay != original).any())
        manifest["source_manifest_sha256"] = "changed"
        with self.assertRaisesRegex(ValueError, "unavailable"):
            annotation_overlay(project, manifest, row.sample_id, original)

    def test_ui_confirms_passed_run_and_updates_workspace_default(self):
        from streamlit.testing.v1 import AppTest

        project = self.project
        project.title = "Synthetic test"
        project.source = "Synthetic test data"
        project.index = project.rows("validation")
        project.available_splits = lambda: ["validation"]
        project.is_full_split = lambda split: True
        project.demo_ids = lambda split: project.rows(split).sample_id.tolist()[:4]
        runs = self.root / "runs"
        run_project(project, "validation", runs)
        def event(pid, event_name, outcome, details=None):
            return record_event(self.root / "events", event_name, project_id=pid, outcome=outcome, details=details)

        with patch("src.inspection_projects.list_projects", return_value={project.id: project.title}), \
                patch("src.inspection_projects.load_project", return_value=project), \
                patch("src.inspection_ui._steel_projects", return_value={project.id: project.title}), \
                patch("src.inspection_projects.RUNS", runs), \
                patch("src.inspection_workspace.ACTIVE", self.root / "state"), \
                patch("src.inspection_ui._event", side_effect=event):
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app_demo.py"), default_timeout=120)
            app.session_state["security-token"] = self.token
            app.run()
            next(x for x in app.button if x.label == "교체 판단").click().run()
            next(x for x in app.button if x.label == "호환성 및 전체 검증 확인").click().run()
            self.assertFalse(app.error)
            next(x for x in app.checkbox if x.label.startswith("독립 시험")).check()
            next(x for x in app.button if x.label == "로컬 활성 모델 변경").click().run()
            self.assertFalse(app.error)
            self.assertEqual(active_model(self.root / "state", project), self.after)
            next(x for x in app.button if x.label == "새 검증 실행").click().run()
            self.assertEqual(next(x for x in app.selectbox if x.label == "기준 모델").value, self.after)


if __name__ == "__main__":
    unittest.main()
