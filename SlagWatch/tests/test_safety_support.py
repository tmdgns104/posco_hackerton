import unittest
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest
from src.safety_eval import ROOT,OUT,DEST,SAFETY,SEEDS,KEY,priority,random_decisions,with_rates
from src.safety_ui import case_masks

class TestSafetySupport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r=pd.read_csv(SAFETY/"cases.csv.gz",float_precision="round_trip")
        cls.t=pd.read_csv(SAFETY/"all_runs.csv.gz")
        cls.g=pd.read_csv(SAFETY/"group_runs.csv.gz")
        cls.rnd=pd.read_csv(SAFETY/"random_summary.csv")

    def test_freeze_outputs_and_legacy_masks(self):
        cols=KEY+["scenario"]
        h=pd.read_csv(DEST/"predictions.csv",float_precision="round_trip")
        for name in ("margin","proposed"):
            actual=self.r[(self.r.policy==name)&self.r.model_id.isin(["A0","A1","H","I"])]
            pair=h.merge(actual,on=cols+["model_id"],validate="one_to_one")
            self.assertEqual(len(pair),len(h))
            for col in ("score","threshold","prediction"):
                np.testing.assert_array_equal(pair[col+"_x"],pair[col+"_y"])
        for target,suf in [("E",""),("G","_G")]:
            orig=pd.read_csv(OUT/f"gate_coverage{suf}.csv",float_precision="round_trip")
            orig["policy"]=orig.policy.map({"margin80":"margin","risk80":"proposed"})
            pair=orig.merge(self.r[self.r.model_id==target],on=cols+["policy"],validate="one_to_one")
            self.assertEqual(len(pair),len(orig))
            np.testing.assert_array_equal(pair.score,pair.model_score)
            for col in ("pass_matched","pass_calibrated"):np.testing.assert_array_equal(pair[col+"_x"],pair[col+"_y"])
        self.assertEqual(len(self.r),6*5*480*2)

    def test_conservation_and_denominators(self):
        for table in (self.t,self.g):
            np.testing.assert_array_equal(table.N_S,table.TP_pass+table.FN_pass+table.H_TP+table.H_FN)
            np.testing.assert_array_equal(table.N_N,table.TN_pass+table.FP_pass+table.H_TN+table.H_FP)
            np.testing.assert_array_equal(table.pass_count,table.TP_pass+table.FN_pass+table.FP_pass+table.TN_pass)
            np.testing.assert_array_equal(table.N,table.pass_count+table.hold_count)
            np.testing.assert_array_equal(table.N,table.N_E+table.N_B+table.N_S)
            np.testing.assert_allclose(table.coverage_nonS,(table.E_pass+table.B_pass)/table.N_N,atol=1e-12)
            for part,den,num in [("tp_pass_rate","N_S","TP_pass"),("fn_pass_rate","N_S","FN_pass"),
                                 ("fp_pass_rate","N_N","FP_pass"),("coverage","N","pass_count")]:
                np.testing.assert_allclose(table[part],table[num]/table[den],atol=1e-12)
        empty=self.t.iloc[:1].copy()
        empty.loc[:,"N_S"]=0
        self.assertTrue(with_rates(empty).tp_pass_rate.isna().all())
        matched=self.t[self.t["mode"]=="matched"]
        self.assertTrue((matched.pass_count==384).all())
        self.assertTrue((matched.hold_count==96).all())
        branches=self.t[self.t.policy=="none"]
        self.assertTrue((branches.H_FN+branches.H_TP==0).all())

    def test_cases_match_random_counts_and_legacy_classes(self):
        random=self.t[self.t.policy=="random"]
        self.assertTrue((random.groupby(["model_id","scenario","mode"]).seed.nunique()==len(SEEDS)).all())
        for model,cond in [("I","clean"),("E","new_z_spike"),("G","noise"),("A0","spike")]:
            r=self.r.query("model_id==@model and scenario==@cond")
            for policy,mode,seed in [("random","matched",0),("random","matched",31),("random","calibrated",13),
                                     ("proposed","calibrated",-1),("proposed","matched",-1),("margin","matched",-1)]:
                p=case_masks(r,policy,mode,seed)
                row=self.t[(self.t.model_id==model)&(self.t.scenario==cond)&(self.t.policy==policy)&(self.t["mode"]==mode)&(self.t.seed==seed)].iloc[0]
                np.testing.assert_array_equal([row.TP_pass,row.FN_pass,row.H_TP,row.H_FN,row.FP_pass],
                    [((p.y_true==y)&(p.prediction==pred)&(p.passed==passed)).sum() for y,pred,passed in
                     [(1,1,True),(1,0,True),(1,1,False),(1,0,False),(0,1,True)]])
                self.assertTrue((p.groupby("fold_id").passed.sum()==96).all() if mode=="matched" else True)
                perm=r.copy();perm["y_true"]=1-perm.y_true;perm["scenario"]="unknown"
                np.testing.assert_array_equal(p.passed,case_masks(perm,policy,mode,seed).passed)
        sel=random.query("model_id=='I' and scenario=='clean' and mode=='matched'")
        r=self.rnd.query("model_id=='I' and scenario=='clean' and mode=='matched'").iloc[0]
        self.assertAlmostEqual(r.H_TP_mean,sel.H_TP.mean());self.assertAlmostEqual(r.H_TP_std,sel.H_TP.std())

    def test_ui_changes_policy_and_truth_remains_visible(self):
        app=AppTest.from_file(str(ROOT/"app.py"),default_timeout=30).run()
        app.radio(key="hi_workspace").set_value("안전 지원 검증").run()
        self.assertFalse(list(app.exception))
        self.assertEqual(app.title[0].value,"안전 지원 검증")
        self.assertIn("22건 보류"," ".join(x.value for x in app.markdown))
        app.selectbox(key="safety_mode").set_value("calibrated").run()
        app.selectbox(key="safety_model").set_value("G")
        app.selectbox(key="safety_condition").set_value("spike").run()
        self.assertFalse(list(app.exception))
        self.assertIn("원래 예측"," ".join(x.value for x in app.markdown))
