import hashlib
import json
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from src.search import ROOT,load_context,replay,search_failures
from src.gate import apply_gate,evaluate_window
from src.metrics_gate import evaluate_rows,compute_gate_metrics


class TestSteel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ctx=load_context()
        cls.rows=pd.read_csv(ROOT/"reports/search_results.csv")

    def test_search_population_and_replay(self):
        r=self.rows
        self.assertEqual(len(r),r.scenario_id.nunique())
        self.assertTrue((r.clean_pred==r.y_true).all())
        self.assertTrue(((r.perturbed_pred!=r.y_true)==(r.failure_type!="none")).all())
        self.assertTrue((r.loc[r.failure_type=="induced_false_positive","y_true"]==0).all())
        self.assertTrue((r.loc[r.failure_type=="induced_false_negative","y_true"]==1).all())
        for _,subset in r.groupby(["fault_type","gate_outcome","fold_id"]):
            row=subset.iloc[0]
            original,changed,result=replay(self.ctx,row)
            self.assertFalse(np.shares_memory(original,changed))
            self.assertFalse(original.flags.writeable)
            self.assertEqual(result["pred"],row.perturbed_pred)
            self.assertEqual(result["decision"],row.gate_decision)
        source=pd.read_csv(ROOT/"reports/manifest.csv")
        for row in source.itertuples():
            path=next((ROOT/"data/raw").rglob(row.file_id))
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),row.sha256)

    def test_limits_and_repeat(self):
        a=search_failures(self.ctx,max_trials=4,max_seconds=60)
        b=search_failures(self.ctx,max_trials=4,max_seconds=60)
        self.assertEqual(len(a),4)
        self.assertEqual(a.scenario_id.tolist(),b.scenario_id.tolist())
        self.assertEqual(a.scenario_id.tolist(),self.rows.head(4).scenario_id.tolist())
        self.assertTrue(search_failures(self.ctx,max_trials=100,max_seconds=0).empty)

    def test_gate_order_and_edges(self):
        for w in (np.full((3200,3),np.nan),np.full((3200,3),np.inf),np.ones((3200,3)),np.zeros((12,3))):
            for pol in ("G0","G1","G2","G3"):
                with patch("src.inference.features",side_effect=AssertionError):
                    r=evaluate_window(w,{},policy=pol)
                self.assertIsNone(r["score"])
                self.assertEqual(r["decision"],"NO_PREDICTION" if pol=="G0" else "HOLD")
        self.assertEqual(apply_gate("",.5,0,"G2",0)[0],"PASS")
        self.assertEqual(apply_gate("",.4,1,"G2",.1)[0],"PASS")
        self.assertEqual(apply_gate("",.49,1,"G3",.1)[1],["LOW_MARGIN","MODEL_DISAGREEMENT"])

    def test_metrics_and_null_denominators(self):
        p=self.ctx["clean"].iloc[:4].copy()
        p["rf_score"]=[.65,.51,.49,np.nan];p["rf_pred"]=[1,1,0,np.nan]
        p["y_true"]=[0,0,0,1];p["reason_codes"]=["","","","NONFINITE"]
        m=compute_gate_metrics(evaluate_rows(p,"rf","G2",.10),"rf")
        self.assertEqual((m["FP"],m["caught"],m["unnecessary_hold"],m["quality_held"]),(1,1,1,1))
        self.assertEqual((m["block_rate"],m["coverage"],m["conditional_error"]),(.5,.25,1))
        m=compute_gate_metrics(evaluate_rows(p.tail(1),"rf","G2",.10),"rf")
        self.assertIsNone(m["block_rate"]);self.assertIsNone(m["conditional_error"])
        for m in json.loads((ROOT/"reports/gate_metrics.json").read_text()):
            self.assertEqual(sum(m[k] for k in ("FP","FN","TP","TN","HOLD_S","HOLD_nonS","no_prediction")),480)
