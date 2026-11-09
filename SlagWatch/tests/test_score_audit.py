import unittest

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from src.hi_study import DEST, ROOT
from src.safety_eval import SAFETY, priority
from src.score_audit import AUDIT, inner_diagnostics, paired_slags


class TestScoreAudit(unittest.TestCase):
    def test_threshold_relative_priority_and_class_only_disagreement(self):
        # Both detectors may use the same labels with completely different score scales.
        score = np.array([-.3, .79, .8, 1.6])
        peer_class = np.array([0, 1, 0, 1])
        risk = priority(score, .8, peer_class)
        distance_priority = 1 / (1 + np.array([1.1, .01, 0., .8]))
        np.testing.assert_allclose(risk["margin"], distance_priority)
        np.testing.assert_allclose(risk["proposed"], distance_priority + [0, 1, 1, 0])
        shifted = priority(score + 3, .8 + 3, peer_class)
        for policy, values in risk.items():
            np.testing.assert_allclose(values, shifted[policy])

    def test_paired_breakdown_and_inner_operating_points(self):
        def read(path):
            return pd.read_csv(path, float_precision="round_trip")
        raw = paired_slags(read(DEST/"predictions.csv"), read(SAFETY/"cases.csv.gz"))
        pd.testing.assert_frame_equal(raw, read(AUDIT/"paired_s.csv"))
        total = raw.groupby(["scenario", "mode"]).n.sum()
        self.assertTrue(total.eq(160).all())
        current = raw.query("scenario=='clean' and mode=='matched'")
        np.testing.assert_array_equal(current[["n", "margin_held", "proposed_held"]],
                                      [[17, 10, 17], [6, 3, 6], [11, 8, 5], [126, 21, 16]])
        totals = read(SAFETY/"policies.csv").query("model_id=='I'")
        for (scenario, mode), rows in raw.groupby(["scenario", "mode"]):
            for policy in ["margin", "proposed"]:
                held = rows.groupby("I")[policy+"_held"].sum()
                ref = totals[(totals.scenario==scenario) & (totals["mode"]==mode) & (totals.policy==policy)].iloc[0]
                self.assertEqual((held[1], held[0]), (ref.H_TP, ref.H_FN))
        inner = inner_diagnostics(read(DEST/"inner_oof_clean.csv"))
        pd.testing.assert_frame_equal(inner, read(AUDIT/"inner_points.csv"))
        self.assertTrue(inner.n_s.eq(120).all() and inner.n_nonS.eq(240).all())
        self.assertTrue(inner.query("point=='selected'").fpr.le(.10).all())

    def test_no_hold_default_and_recovered_alarm_visible(self):
        app = AppTest.from_file(str(ROOT/"app.py"), default_timeout=30).run()
        self.assertFalse(list(app.exception))
        app.radio(key="hi_workspace").set_value("실패 재검증").run()
        table = app.dataframe[0].value
        self.assertTrue(table["보류 여부"].eq("보류 없음").all())
        self.assertIn("원래 판정", table)
        app.radio(key="hi_workspace").set_value("안전 지원 검증").run()
        self.assertEqual(app.selectbox(key="case_policy").value, "none")
        app.radio(key="safety_case_type").set_value("A1이 놓치고 I가 검출").run()
        self.assertIn("정책 처리 전달", " ".join(m.value for m in app.markdown))
        app.selectbox(key="case_policy").set_value("proposed").run()
        self.assertFalse(list(app.exception))
        message = " ".join(m.value for m in app.markdown)
        self.assertIn("원래 예측 슬래그로 분류", message)
        self.assertIn("정책 처리 검토 대상으로 보류", message)
        self.assertTrue(any("A1 원래 예측 · 비슬래그로 분류" in x.value for x in app.caption))
