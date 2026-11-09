import json
import unittest

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from streamlit.testing.v1 import AppTest

from src.data import load_dataset
from src.detector import RUN, Encoder, REP, predict_record
from src.detector_study import ORDER, model_bundle, encode
from src.hi_study import ROOT, read_config, make_wave, select_cut


class TestDetector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = read_config()
        cls.meta, cls.waves, _ = load_dataset(ROOT/"data/raw")
        cls.p = pd.read_csv(RUN/"predictions.csv",float_precision="round_trip")
        cls.inner = pd.read_csv(RUN/"inner_oof.csv.gz",float_precision="round_trip")

    def test_train_split_threshold_selection_and_scaler(self):
        for split in self.cfg["folds"]:
            f = split["fold_id"]
            b = joblib.load(RUN/f"models/fold_{f}.joblib")
            self.assertFalse(set(b["train_groups"]) & set(b["test_groups"]))
            training = self.meta[self.meta.group_id.isin(b["train_groups"])]
            x,_ = encode(training,self.waves,self.cfg["features_I"])
            sub = self.inner[self.inner.fold_id==f]
            for model in ["H","I","J","K"]:
                est = b["estimators"][model]
                np.testing.assert_allclose(est[0].mean_,x[REP[model]].mean(0),atol=1e-12)
            self.assertEqual(set(sub.group_id),set(b["train_groups"]))
            for rate in [.10,.15]:
                ranks = []
                for model in ORDER:
                    g = sub[sub.model_id==model]
                    self.assertEqual(len(g),360)
                    self.assertFalse(g.duplicated(["file_id","window_idx"]).any())
                    c = select_cut(g.y_true,g.score,rate)
                    self.assertEqual(c["threshold"],b["cuts"][rate][model])
                    ranks.append((-c["tp"],c["fp"],ORDER.index(model),model))
                winner = min(ranks)[-1]
                self.assertEqual(winner,b["winners"][rate])
                outer = self.p[(self.p.fold_id==f) & (self.p.budget==rate)]
                self.assertEqual(set(outer.group_id),set(b["test_groups"]))
                np.testing.assert_array_equal(outer.query("model_id=='selected'").configuration.unique(),[winner])
                for model in ORDER:
                    self.assertTrue(outer[outer.model_id==model].threshold.eq(b["cuts"][rate][model]).all())

    def test_context_is_causal_and_invalid_input_clears_it(self):
        b = model_bundle(joblib.load(RUN/"models/fold_0.joblib"),"K",.15)
        w = self.waves[:10]
        original = list(predict_record(w,b))
        changed = list(predict_record(w[:4]+[x*-3 for x in w[4:]],b))
        self.assertEqual(original[:4],changed[:4])
        prefix = list(predict_record(w[:3],b))
        self.assertEqual(prefix,original[:3])
        self.assertEqual(list(predict_record(w[3:7],b))[-1],original[6])
        for invalid in [None,np.full((3200,3),np.nan),np.ones((3200,3)),np.ones((12,3))]:
            _,bad,after = list(predict_record([w[0],invalid,w[1]],b))
            self.assertIsNone(bad["score"]); self.assertIsNone(bad["prediction"])
            self.assertEqual(after,list(predict_record([w[1]],b))[0])
        x = w[0].copy()
        Encoder(REP.values(),b["settings"]).step(x)
        np.testing.assert_array_equal(x,w[0])

    def test_each_fold_replays_one_record_in_every_condition(self):
        for split in self.cfg["folds"]:
            f = split["fold_id"]
            b = joblib.load(RUN/f"models/fold_{f}.joblib")
            sample = self.meta[self.meta.file_id==f"S-{split['test'][0]}.csv"]
            for scenario in self.cfg["conditions"]:
                w = [make_wave(self.waves[r.Index],b["scale"],r,scenario) for r in sample.itertuples()]
                saved = self.p[(self.p.file_id==sample.file_id.iloc[0]) & (self.p.fold_id==f) &
                               (self.p.scenario==scenario) & (self.p.budget==.15)]
                for name in ORDER:
                    rows = saved[saved.model_id==name].sort_values("window_idx")
                    cases = list(predict_record(w,model_bundle(b,name,.15)))
                    np.testing.assert_allclose(rows.score,[r["score"] for r in cases],atol=1e-12,rtol=0)
                    np.testing.assert_array_equal(rows.prediction,[r["prediction"] for r in cases])

    def test_csv_denominators_and_rank_metrics(self):
        key=["budget","model_id","scenario"]
        self.assertEqual(len(self.p),33600)
        self.assertFalse(self.p.duplicated(key+["sample_id"]).any())
        self.assertTrue((self.p.groupby(["sample_id","scenario"]).input_sha256.nunique()==1).all())
        s = pd.read_csv(RUN/"summary.csv").set_index(key)
        for k,r in self.p.groupby(key):
            a,b = r.y_true==1,r.prediction==1
            got = [int((a&b).sum()),int((a&~b).sum()),int((~a&b).sum()),int((~a&~b).sum())]
            np.testing.assert_array_equal(got,s.loc[k,["tp","fn","fp","tn"]].to_numpy())
            self.assertEqual(sum(got),480); self.assertEqual(sum(got[:2]),160)
            folds = r.groupby("fold_id")
            self.assertAlmostEqual(s.loc[k,"mean_fold_roc_auc"],np.mean([roc_auc_score(g.y_true,g.score) for _,g in folds]))
            self.assertAlmostEqual(s.loc[k,"mean_fold_ap"],np.mean([average_precision_score(g.y_true,g.score) for _,g in folds]))
        groups = pd.read_csv(RUN/"by_group.csv")
        self.assertTrue(groups.n_s.eq(10).all() and groups.n_nonS.eq(20).all())

    def test_app_cases_and_full_model_separation(self):
        app = AppTest.from_file(str(ROOT/"app.py"),default_timeout=30).run()
        app.radio(key="hi_workspace").set_value("슬래그 유동 판정").run()
        app.selectbox(key="hi_model").set_value("new:J").run()
        for label,truth,old,new in [("회복한 미검출",1,0,1),("새로 놓친 S",1,1,0),("새 오경보",0,0,1)]:
            app.button(key="det_jump_"+label).click().run()
            self.assertFalse(list(app.exception)); self.assertFalse(list(app.error))
            row = self.p[(self.p.model_id=="J") & (self.p.budget==.15) & (self.p.scenario=="clean") &
                         (self.p.file_id==app.selectbox(key="hi_file").value) &
                         (self.p.window_idx==app.slider(key="hi_index").value)].iloc[0]
            self.assertEqual((row.y_true,row.A1,row.prediction),(truth,old,new))
            view=app.dataframe[0].value
            self.assertTrue(view["보류 여부"].eq("보류 없음").all())
            self.assertEqual(view.iloc[-1]["모델 점수"],f"{row.score:.4f}")
        summary=json.loads((RUN/"selection.json").read_text())["full"]
        full=joblib.load(RUN/"models/full.joblib")
        self.assertEqual(set(full["train_groups"]),set(range(1,17)))
        self.assertEqual(full["threshold"],summary["threshold"])
        self.assertEqual(full["model_id"],summary["model_id"])
        g=pd.read_csv(RUN/"full_inner_summary.csv",float_precision="round_trip").query("budget==0.15")
        best=g.assign(i=g.model_id.map(ORDER.index)).sort_values(["tp","fp","i"],ascending=[False,True,True]).iloc[0]
        self.assertEqual((best.model_id,best.threshold),(full["model_id"],full["threshold"]))
        self.assertEqual(len(list(predict_record(self.waves[:10],full))),10)
