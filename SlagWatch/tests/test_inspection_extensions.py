import concurrent.futures
import base64
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.inspection_activation import active_model
from src.inspection_brightness import brightness_summary, original_brightness
from src.inspection_cases import build_case, replay_case, retest_case
from src.inspection_event_log import read_events, record_event
from src.inspection_model_registry import ModelView, list_models, load_registered, register_model
from src.inspection_projects import load_project
from src.inspection_runner import load_run, run_project


ROOT = Path(__file__).resolve().parents[1]
K1 = ROOT / "experiments" / "inspection_binary" / "models" / "K1.joblib"


class ExtensionTests(unittest.TestCase):
    def test_project_introduction_uses_selected_record_and_works_without_one(self):
        from streamlit.testing.v1 import AppTest
        from src.inspection_projects import RUNS

        app = AppTest.from_file(str(ROOT / "app_demo.py"), default_timeout=120).run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        buttons = [x.label for x in app.button]
        self.assertLess(buttons.index("프로젝트 소개"), buttons.index("개요"))
        hero = next(str(x.value) for x in app.get("html") if 'class="steel-hero"' in str(x.value))
        self.assertIn("철강 표면 결함 AI의 오류를 찾고", hero)
        self.assertEqual(hero.count("<figure>"), 6)
        self.assertTrue(any("학습 원본의 결함 종류별 예시" in x.value for x in app.caption))
        story = next(str(x.value) for x in app.get("html") if 'class="project-story"' in str(x.value))
        stages = ("문제발견", "문제정의", "해결방법 모색", "해결 방법")
        self.assertEqual(story.count('class="project-story-step"'), 4)
        self.assertEqual([story.index(stage) for stage in stages], sorted(story.index(stage) for stage in stages))
        self.assertIn("같은 원본과 seed", story)
        rid = next(x for x in app.selectbox if x.label == "저장된 실행 기록").value
        manifest = load_run(RUNS / rid)
        summary = pd.read_csv(RUNS / rid / "summary.csv")
        models = summary[summary.scope.eq("all")].set_index("model_id")
        cards = next(str(x.value) for x in app.get("html") if 'class="steel-cards"' in str(x.value))
        for model in (manifest["before_model_id"], manifest["after_model_id"]):
            self.assertIn(model, cards)
            self.assertIn(f"{int(models.loc[model, 'completed'] - models.loc[model, 'correct']):,}", cards)
        next(x for x in app.button if x.label == "검증 개요 보기").click().run()
        self.assertTrue(any(x.value == "전체 및 원본 성능" for x in app.subheader))
        self.assertFalse(app.exception)
        with tempfile.TemporaryDirectory() as temp, patch("src.inspection_projects.RUNS", Path(temp)):
            empty = AppTest.from_file(str(ROOT / "app_demo.py"), default_timeout=120).run()
            self.assertFalse(empty.exception)
            self.assertTrue(any("저장된 평가가 없습니다" in x.value for x in empty.info))
            self.assertTrue(any('class="steel-hero"' in str(x.value) for x in empty.get("html")))
            self.assertTrue(any('class="project-story"' in str(x.value) for x in empty.get("html")))

    def test_neu_workspace_restores_selected_run_charts_and_details(self):
        from streamlit.testing.v1 import AppTest
        from src.inspection_projects import RUNS

        app = AppTest.from_file(str(ROOT / "app_demo.py"), default_timeout=120).run()
        rid = next(x for x in app.selectbox if x.label == "저장된 실행 기록").value
        manifest = load_run(RUNS / rid)
        summary = pd.read_csv(RUNS / rid / "summary.csv")
        cells = summary[summary.scope.eq("cell")]
        for section in ("개요", "모델 비교", "교체 판단", "전이·통계 분석"):
            next(x for x in app.button if x.label == section).click().run(timeout=120)
            self.assertFalse(app.exception, section)
            self.assertFalse(app.error, section)
            charts = app.get("plotly_chart")
            self.assertEqual(len(charts), 1, section)
            spec = json.loads(charts[0].proto.spec)
            self.assertEqual({trace["name"] for trace in spec["data"]},
                             {manifest["before_model_id"], manifest["after_model_id"]})
            for trace in spec["data"]:
                values = trace["y"]
                if isinstance(values, dict):
                    values = np.frombuffer(base64.b64decode(values["bdata"]), dtype=values["dtype"])
                expected = cells[cells.model_id.eq(trace["name"])]["accuracy"].to_numpy()
                np.testing.assert_allclose(values, expected, equal_nan=True)
            if section != "전이·통계 분석":
                graph_panel = next(x for x in app.expander if x.label == "조건·seed별 전후 비교")
                self.assertTrue(graph_panel.proto.expanded, section)
                self.assertTrue(any(x.value == "전체 및 원본 성능" for x in app.subheader))
                self.assertTrue(any(x.label.startswith("사전 기준별 근거") for x in app.expander))
                cards = next(str(x.value) for x in app.get("html") if 'class="steel-cards"' in str(x.value))
                self.assertIn("기존 모델의 오분류", cards)
                self.assertIn("새 오분류", cards)
            else:
                self.assertTrue(any(x.value == "조건별 오분류와 모델 성능" for x in app.subheader))
                self.assertTrue(any(x.label == "결함 종류별 재현율과 오분류" for x in app.expander))
                next(x for x in app.radio if x.label == "그래프 지표").set_value("macro_f1").run()
                self.assertFalse(app.exception)
                for trace in json.loads(app.get("plotly_chart")[0].proto.spec)["data"]:
                    values = trace["y"]
                    if isinstance(values, dict):
                        values = np.frombuffer(base64.b64decode(values["bdata"]), dtype=values["dtype"])
                    expected = cells[cells.model_id.eq(trace["name"])]["macro_f1"].to_numpy()
                    np.testing.assert_allclose(values, expected, equal_nan=True)
                condition = next(x for x in app.selectbox if x.label == "사례를 확인할 조건")
                condition.select_index(len(condition.options) - 1).run()
                expected_condition = next(x for x in app.selectbox if x.label == "사례를 확인할 조건").value
                next(x for x in app.button if x.label == "이 조건의 실패 사례 보기").click().run()
                self.assertFalse(app.exception)
                actual_condition = next(x for x in app.selectbox if x.label == "사례 입력 조건").value
                self.assertEqual(actual_condition, expected_condition)

    def test_workspace_shows_only_steel_projects_and_restores_overview(self):
        from streamlit.testing.v1 import AppTest

        app = AppTest.from_file(str(ROOT / "app_demo.py"), default_timeout=120).run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        selector = next(x for x in app.selectbox if x.label == "검사 프로젝트")
        self.assertEqual(selector.value, "neu_cls6")
        self.assertNotIn("KolektorSDD2 · 정상 / 불량 검사", selector.options)
        next(x for x in app.button if x.label == "개요").click().run()
        self.assertTrue(any(x.value == "전체 및 원본 성능" for x in app.subheader))
        self.assertTrue(any(x.label.startswith("사전 기준별 근거") for x in app.expander))
        self.assertTrue(any(x.label == "조건·seed별 전후 비교" for x in app.expander))
        cards = next(str(x.value) for x in app.get("html") if 'class="steel-cards"' in str(x.value))
        self.assertIn("기존 모델의 오분류", cards)
        self.assertIn("후보 모델의 오분류", cards)
        self.assertNotIn("K0", cards)
        self.assertFalse(any("슬래그" in x.label for x in app.expander))
        next(x for x in app.button if x.label == "프로젝트 설정").click().run()
        self.assertFalse(app.exception)
        choices = next(x for x in app.selectbox if x.label == "등록할 모델 조합")
        self.assertEqual(choices.value, "neu_cls6")
        self.assertEqual(len(choices.options), 1)

    def test_steel_project_filter_preserves_registered_neu_only(self):
        from src.inspection_ui import _steel_projects

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for pid, template in (("user_steel01", "neu_cls6"), ("user_other01", "ksdd2")):
                (root / pid).mkdir()
                (root / pid / "project.json").write_text(json.dumps({"template_id": template}), encoding="utf-8")
            projects = {"ksdd2": "binary", "neu_cls6": "steel",
                        "user_steel01": "steel user", "user_other01": "binary user"}
            with patch("src.inspection_projects.list_projects", return_value=projects), \
                    patch("src.inspection_projects.DATASETS", root):
                self.assertEqual(_steel_projects(), {"neu_cls6": "steel", "user_steel01": "steel user"})

    def test_registration_guards_integrity_and_real_paired_run(self):
        project = load_project("ksdd2")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            registry = root / "models"
            from src import security
            security_root = patch.object(security, "SECURITY_ROOT", root / "security")
            audit_root = patch.object(security, "AUDIT_ROOT", root / "audit")
            audit_env = patch.dict("os.environ", {"STEELTRUST_AUDIT_URL": "", "STEELTRUST_AUDIT_BEARER": ""})
            for item in (security_root, audit_root, audit_env):
                item.start()
                self.addCleanup(item.stop)
            security.bootstrap_admin("tester", "a sufficiently long password")
            token = security.login("tester", "a sufficiently long password")
            with self.assertRaisesRegex(ValueError, "trust"):
                register_model(registry, K1, "K2", trusted=False)
            metadata = register_model(registry, K1, "K2", trusted=True, auth_token=token)
            self.assertEqual(metadata["embedded_model_id"], "K1")
            self.assertEqual(list_models(registry), ["K2"])
            with self.assertRaisesRegex(ValueError, "already"):
                register_model(registry, K1, "k2", trusted=True, auth_token=token)
            candidate = load_registered(registry, "K2")
            view = ModelView(project, {"K0": project.models["K0"], "K2": candidate}, development_only=True)
            with self.assertRaisesRegex(ValueError, "validation"):
                view.check_scope("test")
            ids = project.demo_ids("validation")[:2]
            run = run_project(view, "validation", root / "runs", sample_ids=ids)
            manifest = load_run(run)
            self.assertFalse(manifest["official"])
            self.assertEqual((manifest["before_model_id"], manifest["after_model_id"]), ("K0", "K2"))
            self.assertEqual(manifest["failed_pairs"], 0)
            case = build_case(project, run, ids[0], *manifest["conditions"][0])
            replay_view = ModelView(project, {"K0": project.models["K0"], "K2": load_registered(registry, "K2")})
            self.assertTrue(replay_case(replay_view, run, case)["verified"])
            result = retest_case(replay_view, run, case, "K2", root / "retests")
            self.assertEqual(result["status"], "ok")
            (registry / "K2.joblib").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "hash"):
                load_registered(registry, "K2")

    def test_brightness_uses_original_rgb_and_counts_repeated_conditions(self):
        class Project:
            def rows(self, split):
                return pd.DataFrame([
                    dict(sample_id="a", file_sha256="hash_a", true_label="defect", split=split),
                    dict(sample_id="b", file_sha256="hash_b", true_label="normal", split=split)])

            def load(self, row):
                rgb = [1, 0, 0] if row.sample_id == "a" else [0, 1, 0]
                return np.array([[rgb]], dtype=np.float32)

        base = [
            dict(sample_id="a", file_sha256="hash_a", true_label="defect", before_pred="defect", after_pred="normal"),
            dict(sample_id="b", file_sha256="hash_b", true_label="normal", before_pred="normal", after_pred="defect")]
        pairs = pd.DataFrame([{**row, "split": "validation", "condition": condition, "seed": 0,
                               "before_status": "ok", "after_status": "ok"}
                              for row in base for condition in ("clean", "dark")])
        measurements = original_brightness(Project(), "validation", pairs).set_index("sample_id")
        self.assertAlmostEqual(measurements.loc["a", "brightness"], .2126)
        self.assertAlmostEqual(measurements.loc["b", "brightness"], .7152)
        result = brightness_summary(pairs, measurements.reset_index(), low_below=.3, high_from=.7).set_index("brightness_group")
        self.assertEqual((result.loc["low", "originals"], result.loc["low", "decisions"], result.loc["low", "new_fn"]), (1, 2, 2))
        self.assertEqual((result.loc["high", "originals"], result.loc["high", "new_fp"]), (1, 2))
        bad = pairs.copy()
        bad.loc[0, "file_sha256"] = "changed"
        with self.assertRaisesRegex(ValueError, "lineage"):
            original_brightness(Project(), "validation", bad)

    def test_events_concurrent_and_corruption_visible(self):
        with tempfile.TemporaryDirectory() as temp:
            def emit(number):
                record_event(temp, "evaluation", project_id="p", outcome="completed", details={"number": number})
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(emit, range(40)))
            events = read_events(temp, project_id="p")
            self.assertEqual(len(events), 40)
            self.assertEqual(len({entry["id"] for entry in events}), 40)
            with (Path(temp) / "events.jsonl").open("a", encoding="utf-8") as stream:
                stream.write("{broken\n")
            with self.assertRaisesRegex(ValueError, "line"):
                read_events(temp, project_id="p")

    def test_ui_register_evaluate_and_measure_brightness(self):
        from streamlit.testing.v1 import AppTest
        from src import inspection_ui
        from src.inspection_projects import InspectionProject

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            from src import security
            for item in (patch.object(security, "SECURITY_ROOT", root / "security"),
                         patch.object(security, "AUDIT_ROOT", root / "audit"),
                         patch.dict("os.environ", {"STEELTRUST_AUDIT_URL": "", "STEELTRUST_AUDIT_BEARER": ""})):
                item.start()
                self.addCleanup(item.stop)
            security.bootstrap_admin("tester", "a sufficiently long password")
            token = security.login("tester", "a sufficiently long password")
            original_demo = InspectionProject.demo_ids
            def event(project_id, event_name, outcome, details=None):
                return record_event(root / "events", event_name, project_id=project_id,
                                    outcome=outcome, details=details)

            with patch.object(inspection_ui, "_model_root", return_value=root / "models"), \
                    patch.object(inspection_ui, "_event", side_effect=event), \
                    patch.object(inspection_ui, "_steel_projects", return_value={"ksdd2": "KSDD2"}), \
                    patch("src.inspection_projects.RUNS", root / "runs"), \
                    patch.object(InspectionProject, "demo_ids", lambda self, split: original_demo(self, split)[:2]):
                app = AppTest.from_file(str(ROOT / "app_demo.py"), default_timeout=120)
                app.session_state["security-token"] = token
                app.run()
                next(x for x in app.button if x.label == "프로젝트 설정").click().run()
                next(x for x in app.text_input if x.label.startswith("새 모델 ID")).set_value("K2")
                next(x for x in app.text_input if x.label.startswith("신뢰한 로컬")).set_value(str(K1))
                app.checkbox[0].check()
                next(x for x in app.button if x.label == "모델 등록").click().run()
                self.assertFalse(app.exception)
                self.assertEqual(list_models(root / "models"), ["K2"])
                next(x for x in app.button if x.label == "새 검증 실행").click().run()
                next(x for x in app.selectbox if x.label == "후보 모델").select("K2").run()
                next(x for x in app.button if x.label == "기준·후보 바꾸기").click().run()
                self.assertEqual(next(x for x in app.selectbox if x.label == "기준 모델").value, "K2")
                self.assertEqual(next(x for x in app.selectbox if x.label == "후보 모델").value, "K0")
                next(x for x in app.button if x.label == "기준·후보 바꾸기").click().run()
                next(x for x in app.button if x.label == "새 추론 실행").click().run(timeout=120)
                self.assertFalse(app.exception)
                manifests = list((root / "runs").glob("*/manifest.json"))
                self.assertEqual(len(manifests), 1)
                manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
                self.assertEqual((manifest["after_model_id"], manifest["failed_pairs"]), ("K2", 0))
                next(x for x in app.button if x.label == "전이·통계 분석").click().run()
                next(x for x in app.button if x.label == "원본 밝기 측정").click().run()
                self.assertFalse(app.exception)
                self.assertFalse(app.error)
                outcomes = [(e["event"], e["outcome"]) for e in read_events(root / "events", project_id="ksdd2")]
                self.assertIn(("model_registration", "completed"), outcomes)
                self.assertIn(("evaluation", "completed"), outcomes)

    def test_workspace_navigation_and_real_failed_decision(self):
        from streamlit.testing.v1 import AppTest

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch("src.inspection_workspace.ACTIVE", root / "state"), \
                    patch("src.inspection_workspace.EVENTS", root / "events"), \
                    patch("src.inspection_workspace.MODELS", root / "models"), \
                    patch("src.inspection_ui._steel_projects", return_value={"ksdd2": "KSDD2", "neu_cls6": "NEU"}):
                app = AppTest.from_file(str(ROOT / "app_demo.py"), default_timeout=120).run()
                for section in ("모델 비교", "실패 사례", "모델 레지스트리", "새 검증 실행",
                                "전이·통계 분석", "사례 모음·재시험", "교체 판단",
                                "평가 설계 기록", "보고서", "이벤트 기록", "프로젝트 설정", "개요"):
                    next(x for x in app.button if x.label == section).click().run(timeout=120)
                    self.assertFalse(app.exception, section)
                    self.assertFalse(app.error, section)
                    if section == "실패 사례":
                        next(x for x in app.checkbox if x.label == "원본 정답 마스크 표시").check().run()
                        self.assertFalse(app.exception)
                        self.assertFalse(app.error)
                    if section == "교체 판단":
                        next(x for x in app.button if x.label == "호환성 및 전체 검증 확인").click().run(timeout=120)
                        self.assertTrue(app.error)
                        self.assertEqual(active_model(root / "state", load_project("ksdd2")), "K0")
                next(x for x in app.selectbox if x.label == "검사 프로젝트").select("neu_cls6").run(timeout=120)
                self.assertFalse(app.exception)
                for section in ("개요", "모델 비교", "실패 사례", "모델 레지스트리", "교체 판단", "보고서"):
                    next(x for x in app.button if x.label == section).click().run(timeout=120)
                    self.assertFalse(app.exception, "NEU: " + section)
                    self.assertFalse(app.error, "NEU: " + section + " " + repr([e.value for e in app.error]))


if __name__ == "__main__":
    unittest.main()
