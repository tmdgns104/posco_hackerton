import json
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from src.improve import ROOT, OUT, CONDITIONS
from src.impulse import repair
from src.comparison_ui import replay_comparison


class PairedRegression(unittest.TestCase):
    def test_original_models_and_clean_split(self):
        p=pd.read_csv(OUT/"paired_scores.csv")
        v3=pd.read_csv(ROOT/"reports/predictions.csv")
        k=["file_id","window_idx","scenario","fold_id","y_true"]
        for family in ("rf","spectral"):
            test=p[(p.family==family)&(p.variant=="A")&p.scenario.isin(["clean","noise","spike"])]
            j=test.merge(v3,on=k,validate="one_to_one")
            self.assertEqual(len(j),1440)
            np.testing.assert_allclose(j.score,j[family+"_score"],atol=1e-12,rtol=0)
        for fold in json.loads((OUT/"folds.json").read_text()):
            self.assertTrue(set(fold["train_groups"]).isdisjoint(fold["test_groups"]))
            self.assertEqual(len(fold["train_groups"]),12)
            self.assertEqual(len(fold["test_groups"]),4)
        for _,g in p.groupby(["scenario","family","variant"]):
            self.assertEqual((len(g),int(g.y_true.sum())),(480,160))
            self.assertFalse(g.duplicated(["file_id","window_idx"]).any())

    def test_same_scenario_ids_and_replay(self):
        suite=pd.read_csv(OUT/"regression_2000.csv")
        original=pd.read_csv(ROOT/"reports/search_results.csv")
        self.assertEqual(suite.scenario_id.tolist(),original.scenario_id.tolist())
        a=suite.A.ge(.5)!=suite.y_true
        e=suite.E.ge(.5)!=suite.y_true
        self.assertEqual(int(a.sum()-e.sum()),int((a&~e).sum()-(~a&e).sum()))
        for mask in [a&~e,~a&e,a&e,~a&~e]:
            r=suite[mask].iloc[0]
            w,test,corrected,x,y=replay_comparison(r,"rf","E",True)
            self.assertFalse(np.shares_memory(test,corrected))
        for filename,var in [("shape_scores.csv","F"),("multi_scores.csv","G")]:
            new=pd.read_csv(OUT/filename);a0=pd.read_csv(OUT/"paired_scores.csv")
            for f in range(4):
                t=new[(new.variant==var)&(new.fold_id==f)&(new.scenario=="spike")].iloc[0].copy()
                t[var]=t.score
                t["A"]=a0[(a0.family=="spectral")&(a0.variant=="A")&(a0.file_id==t.file_id)&(a0.window_idx==t.window_idx)&(a0.scenario==t.scenario)].score.item()
                replay_comparison(t,"spectral",var,False)

    def test_gate_calibration_and_matched_denominators(self):
        for suffix in ("","_G"):
            records=pd.read_csv(OUT/f"gate_coverage{suffix}.csv")
            summary=pd.read_csv(OUT/f"gate_summary{suffix}.csv")
            for r in summary.itertuples():
                data=records[(records.policy==r.policy)&(records.scenario==r.scenario)]
                passed=data["pass_"+r.mode]
                self.assertEqual(r.passed,int(passed.sum()))
                self.assertEqual(r.errors_pass,r.fp+r.fn)
                self.assertEqual(int((((data.model_score>=.5)!=data.y_true)&passed).sum()),r.errors_pass)
                if r.mode=="matched":self.assertEqual(r.passed,384)
            for row in json.loads((OUT/f"gate_folds{suffix}.json").read_text()):
                self.assertTrue(set(row["training_groups"]).isdisjoint(row["tested_groups"]))
