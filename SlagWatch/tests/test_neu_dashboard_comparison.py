import unittest
from pathlib import Path
from unittest.mock import patch

from src import neu_dashboard_comparison as comparison


class NeuDashboardComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pairs = comparison.load_comparison()

    def test_paired_results_for_both_models_and_each_scope(self):
        expected = {
            "원본": [(360, 1, 0, 1, 0), (360, 1, 1, 1, 1)],
            "잡음 전체": [(3240, 447, 492, 133, 178), (3240, 447, 530, 137, 220)],
            "전체 조건": [(3600, 448, 492, 134, 178), (3600, 448, 531, 138, 221)],
        }
        for scope, values in expected.items():
            with self.subTest(scope=scope):
                result = comparison.summarize(comparison.select_scope(self.pairs, scope))
                self.assertEqual(list(result["후보"]), ["V1", "V2"])
                self.assertEqual(list(result["원본 이미지"]), [360, 360])
                actual = list(result[["짝지은 판정", "V0 오류", "후보 오류", "고친 오류", "새 오류"]]
                              .itertuples(index=False, name=None))
                self.assertEqual(actual, values)
                self.assertEqual(list(result["순 변화"]), list(result["새 오류"] - result["고친 오류"]))
        for scope in comparison.SCOPES[3:]:
            result = comparison.summarize(comparison.select_scope(self.pairs, scope))
            self.assertEqual(list(result["짝지은 판정"]), [1080, 1080])
            self.assertEqual(list(result["원본 이미지"]), [360, 360])

    def test_incomplete_or_inconsistent_paired_data_is_rejected(self):
        class FakeRun:
            def __init__(self, pairs, real):
                self.pairs = pairs
                self.meta = real.meta
                self.detector = real.detector

            def table(self, split, name):
                return self.pairs

        from src.surface_replay import SurfaceRun
        real = SurfaceRun()
        with self.assertRaisesRegex(ValueError, "완전하지"):
            comparison.validated_pairs(FakeRun(self.pairs.iloc[1:].copy(), real))
        bad = self.pairs.copy()
        bad.loc[bad.index[0], "raw_input_hash_after"] = "changed"
        with self.assertRaisesRegex(ValueError, "같은 시험 입력"):
            comparison.validated_pairs(FakeRun(bad, real))
        bad = self.pairs.copy()
        bad.loc[bad.index[0], "change"] = "new_error"
        with self.assertRaisesRegex(ValueError, "오류 변화"):
            comparison.validated_pairs(FakeRun(bad, real))
        with self.assertRaises(ValueError):
            comparison.select_scope(self.pairs, "unsupported")

    def test_comparisons_moved_to_model_comparison_and_overview_restored(self):
        from streamlit.testing.v1 import AppTest

        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app_demo.py"), default_timeout=120).run()
        next(x for x in app.button if x.label == "개요").click().run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        self.assertFalse(any(x.value == "기본 모델 V0와 추가 모델 비교" for x in app.subheader))
        self.assertFalse(any(x.label == "V0·V1·V2 비교 조건" for x in app.selectbox))
        self.assertTrue(any(x.value == "전체 및 원본 성능" for x in app.subheader))
        self.assertEqual(len(app.get("plotly_chart")), 1)
        next(x for x in app.button if x.label == "모델 비교").click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any(x.value == "최근 등록 모델과 V0 비교" for x in app.subheader))
        self.assertTrue(any(x.value == "기본 모델 V0와 추가 모델 비교" for x in app.subheader))
        self.assertTrue(any(x.value == "기존 모델 비교" for x in app.subheader))
        metrics = [x.value for x in app.metric if x.label == "오분류"]
        self.assertEqual(metrics, ["448 → 492회", "448 → 531회"])
        next(x for x in app.selectbox if x.label == "V0·V1·V2 비교 조건").select("잡음 전체").run()
        self.assertFalse(app.exception)
        self.assertEqual([x.value for x in app.metric if x.label == "오분류"],
                         ["447 → 492회", "447 → 530회"])
        self.assertTrue(any("133회" in x.value and "178회" in x.value for x in app.markdown))
        self.assertTrue(any("137회" in x.value and "220회" in x.value for x in app.markdown))
        next(x for x in app.selectbox if x.label == "V0·V1·V2 비교 조건").select("원본").run()
        self.assertEqual([x.value for x in app.metric if x.label == "오분류"],
                         ["1 → 0회", "1 → 1회"])

    def test_invalid_comparison_does_not_remove_existing_model_comparison(self):
        from streamlit.testing.v1 import AppTest

        with patch("src.neu_dashboard_comparison.load_comparison", side_effect=ValueError("test mismatch")):
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app_demo.py"), default_timeout=120).run()
            next(x for x in app.button if x.label == "모델 비교").click().run()
            self.assertTrue(any("test mismatch" in x.value for x in app.error))
            self.assertTrue(any(x.value == "전체 및 원본 성능" for x in app.subheader))
            self.assertEqual(len(app.get("plotly_chart")), 1)


if __name__ == "__main__":
    unittest.main()
