from pathlib import Path
import json
import unittest

import joblib
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from src.data import load_dataset
from src.features import NAMES, SPECTRAL_NAMES, band_powers
from src.faults import SCENARIOS, seed_for
from src.inference import classify_window
from src.quality import quality


ROOT = Path(__file__).resolve().parents[1]


class TestSavedRun(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.meta, cls.windows, cls.manifest = load_dataset(ROOT/"data/raw")
        cls.pred = pd.read_csv(ROOT/"reports/predictions.csv")
        cls.fold_map = json.loads((ROOT/"artifacts/fold_map.json").read_text())

    def test_split_and_coverage(self):
        self.assertEqual(len(self.manifest),48)
        self.assertEqual(len(self.windows),480)
        self.assertEqual(len(self.pred),2400)
        self.assertFalse(self.pred.duplicated(["file_id","window_idx","scenario"]).any())
        self.assertEqual(set(self.pred.scenario),set(SCENARIOS))
        for fold in range(4):
            b=joblib.load(ROOT/f"artifacts/fold_{fold}.joblib")
            self.assertTrue(set(b["train_groups"]).isdisjoint(b["test_groups"]))
            self.assertEqual((len(b["train_groups"]),len(b["test_groups"])),(12,4))
            self.assertEqual(set(self.pred.query("fold_id == @fold").group_id),set(b["test_groups"]))
            self.assertEqual(b["feature_names"],NAMES)
            self.assertEqual(b["spectral_feature_names"], NAMES + SPECTRAL_NAMES)

    def test_replay(self):
        for g in range(1,17):
            file=f"B-{g}.csv"
            index=g%10
            window=self.windows[self.meta.index[(self.meta.file_id==file)&(self.meta.window_idx==index)][0]]
            original=window.copy()
            b=joblib.load(ROOT/f'artifacts/fold_{self.fold_map[str(g)]}.joblib')
            for s in SCENARIOS:
                seed=seed_for(g,"B",index,s)
                changed,out=classify_window(window,b,s,index,seed)
                repeated,_=classify_window(window,b,s,index,seed)
                np.testing.assert_array_equal(changed,repeated)
                row=self.pred.query("file_id==@file and window_idx==@index and scenario==@s").iloc[0]
                self.assertEqual(out["input_status"],row.input_status)
                for key in ["rf_score","rf_pred","baseline_pred","spectral_score","spectral_pred"]:
                    if out[key] is None:self.assertTrue(pd.isna(row[key]))
                    else:self.assertAlmostEqual(out[key],row[key])
            np.testing.assert_array_equal(original,window)

    def test_denominators_and_metrics(self):
        metrics=json.loads((ROOT/"reports/metrics.json").read_text(encoding="utf-8"))["metrics"]
        for m in metrics:
            rows=self.pred[self.pred.scenario==m["scenario"]]
            pred=rows[m["model"]+"_pred"]
            s=rows.y_true.eq(1)
            held=pred.isna()
            self.assertEqual(m["held"],int(held.sum()))
            self.assertEqual(m["detection_rate"],(s&pred.eq(1)).sum()/160)
            self.assertEqual(m["false_alarm_rate"],(~s&pred.eq(1)).sum()/320)
            self.assertEqual(m["fn"],int((s&pred.eq(0)).sum()))
            if m["scenario"] in ("missing","freeze"):
                self.assertTrue((held == rows.window_idx.mod(2).eq(0)).all())
                self.assertEqual(m["held"],240)
        self.assertEqual(quality(np.ones((100,3))),"BAD_SHAPE")
        w=self.windows[0].copy()
        w[1,0]=np.inf
        self.assertEqual(quality(w),"NONFINITE")

    def test_band_power_order_and_units(self):
        t = np.arange(3200) / 6400
        signal = np.column_stack([np.sin(2*np.pi*f*t) for f in (20, 250, 1800)])
        bands = band_powers(signal).reshape(3, 8)
        np.testing.assert_allclose(bands.sum(axis=1), np.ones(3))
        np.testing.assert_array_equal(bands.argmax(axis=1), [0,4,7])
        self.assertTrue(np.all(bands.max(axis=1) > 0.99))
        np.testing.assert_allclose(bands,band_powers(signal*7+3).reshape(3,8),atol=1e-12)

    def test_app_replays_and_restores(self):
        app=AppTest.from_file(str(ROOT/"app.py"),default_timeout=20).run()
        app.radio(key="hi_workspace").set_value("검증 도구").run()
        app.selectbox(key="record_id").set_value("B-12.csv")
        app.slider[0].set_value(1).run()
        self.assertEqual(app.table[0].value.loc["진폭 RF","시험 입력"],"유동 미감지 (0.355)")
        app.selectbox(key="scenario_id").set_value("spike").run()
        self.assertEqual(app.table[0].value.loc["진폭 RF","시험 입력"],"유동 감지 (0.531)")
        app.slider[0].set_value(0)
        for scenario in ("missing","freeze"):
            app.selectbox(key="scenario_id").set_value(scenario).run()
            self.assertIn("입력 확인 필요",[s.value for s in app.subheader])
            self.assertTrue(app.warning)
        app.selectbox(key="scenario_id").set_value("clean")
        app.slider[0].set_value(1).run()
        self.assertEqual(app.table[0].value.loc["진폭 RF","시험 입력"],"유동 미감지 (0.355)")
        self.assertTrue({"선택 기록 CSV","전체 예측 CSV","평가 결과 JSON"}.issubset({x.label for x in app.get("download_button")}))
        self.assertFalse(list(app.exception))
