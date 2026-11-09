import copy
import json
import statistics
import unittest

import numpy as np
import pandas as pd
import torch
from streamlit.testing.v1 import AppTest

from src.cnn_eval import paired
from src.hi_study import ROOT, make_wave, select_cut
from src.pair_detector import detector, digest
from src.pair_search import ReplaySet
from src.raw_cnn import CFG, RUN, tensor_waves


class TestRawPair(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ctx=ReplaySet()

    def test_split_statistics_epochs_threshold_and_patches(self):
        ctx=self.ctx
        for fold in ["0","1","2","3","full"]:
            data=json.loads((RUN/f"fit_{fold}.json").read_text())
            points=pd.read_csv(RUN/f"inner_{fold}.csv",float_precision="round_trip")
            ids=ctx.meta.index[ctx.meta.group_id.isin(data["train"])]
            tr=np.stack([ctx.waves[i] for i in ids])
            for name in ["C0","C1"]:
                d=detector(name,fold)
                self.assertEqual(set(d.train),set(data["train"]))
                self.assertFalse(set(d.test)&set(d.train))
                g=points[points.model_id==name]
                self.assertTrue(not g.duplicated(["file_id","window_idx"]).any() and set(g.group_id)==set(d.train))
                self.assertEqual(select_cut(g.y_true,g.score,.15)["threshold"],d.threshold)
                hist=data["models"][name]["history"]
                self.assertEqual(int(statistics.median(r["epoch"] for r in hist)),d.bundle["epoch"])
                for i,r in enumerate(hist):
                    self.assertEqual(set(r["val"]),set(g[g.inner_id==i].group_id))
                    self.assertFalse(set(r["val"])&set(r["train"]))
                    self.assertEqual(set(r["val"]+r["train"]),set(d.train))
                    self.assertEqual(1+np.argmin(r["validation_bce"]),r["epoch"])
                np.testing.assert_allclose(d.net.axis_mean,tr.mean((0,1)),atol=5e-7)
                np.testing.assert_allclose(d.net.axis_std,tr.std((0,1)),atol=5e-7)
        raw=tensor_waves(ctx.waves[:2])
        for name in ["C0","C1"]:
            net=detector(name,0).net
            with torch.inference_mode():
                x=(raw-net.axis_mean)/net.axis_std
                if name=="C1":
                    x=x/x.square().mean((1,2),keepdim=True).sqrt().clamp_min(1e-6)
                parts=[net.encoder(x[:,i:i+512].transpose(1,2)).squeeze(1) for i in CFG["patch_starts"]]
                torch.testing.assert_close(torch.stack(parts).mean(0),net(raw),rtol=0,atol=1e-6)

    def test_batch_and_single_match_on_entire_cnn_population(self):
        ctx=self.ctx
        for f in range(4):
            base=detector("J",f)
            te=ctx.meta[ctx.meta.group_id.isin(base.test)]
            for scenario in ctx.conditions:
                raw=[make_wave(ctx.waves[r.Index],base.legacy["scale"],r,scenario) for r in te.itertuples()]
                for name in ["C0","C1"]:
                    d=detector(name,f)
                    g=ctx.pred[(ctx.pred.fold_id==f)&(ctx.pred.scenario==scenario)&(ctx.pred.model_id==name)].set_index(["file_id","window_idx"])
                    saved=g.loc[pd.MultiIndex.from_frame(te[["file_id","window_idx"]])]
                    single=[d.infer([w])[0] for w in raw]
                    np.testing.assert_array_equal(saved.input_sha256,[digest(w) for w in raw])
                    np.testing.assert_array_equal(saved.prediction,[r["prediction"] for r in single])
                    np.testing.assert_allclose(saved.score,[r["score"] for r in single],atol=CFG["score_atol"],rtol=0)
        for name in ["A1","J","C0","C1"]:
            invalid=detector(name,0).infer([None,np.full((3200,3),np.inf),np.full((3200,3),np.nan),np.zeros((3200,3)),np.zeros((512,3))])
            self.assertTrue(all(r["score"] is None and r["prediction"] is None and r["input_status"]!="VALID" for r in invalid))

    def test_replay_detects_mismatches_and_pair_ids(self):
        run=json.loads((RUN/"search_J_C1.json").read_text())
        changed=[r for r in run["records"] if r["before_induced"]]
        self.assertEqual(run["n_clean"],480)
        for r in changed[:3]:
            _,w,case=self.ctx.replay(r)
            self.assertEqual(digest(w),r["input_sha256"])
            _,_,other=self.ctx.assess("J","C0",r["file_id"],r["window_idx"],r["fault"])
            self.assertEqual(case["case_id"],other["case_id"])
            self.assertNotEqual(case["execution_id"],other["execution_id"])
        for mutate in ("input_sha256","checkpoint","threshold","score","prediction"):
            bad=copy.deepcopy(changed[0])
            if mutate=="input_sha256":bad[mutate]="ffffffff"
            elif mutate=="checkpoint":bad["outputs"]["C1"]["model_sha256"]="ffffffff"
            elif mutate=="threshold":bad["outputs"]["C1"][mutate]+=.001
            elif mutate=="score":bad["outputs"]["C1"]["test"][mutate]+=.01
            else:bad["outputs"]["C1"]["test"][mutate]^=1
            with self.assertRaises(ValueError):self.ctx.replay(bad)

    def test_population_and_new_ui(self):
        summary=pd.read_csv(RUN/"summary.csv")
        changes=pd.read_csv(RUN/"changes.csv")
        p=self.ctx.pred
        for r in summary.itertuples():
            g=p[(p.model_id==r.model_id)&(p.scenario==r.scenario)]
            self.assertEqual(len(g),480); self.assertEqual(r.tp+r.fn,160);self.assertEqual(r.fp+r.tn,320)
            self.assertEqual(r.tp,((g.y_true==1)&(g.prediction==1)).sum())
            self.assertEqual(r.fp,((g.y_true==0)&(g.prediction==1)).sum())
        for row in changes.itertuples():
            g=paired(p[p.scenario==row.scenario],row.before,row.after)
            for k in ["rescued_FN","new_FN","removed_FP","new_FP"]:
                self.assertEqual(g.change.eq(k).sum(),getattr(row,k))
        app=AppTest.from_file(str(ROOT/"app.py"),default_timeout=35).run()
        app.radio(key="hi_workspace").set_value("실패 재검증").run()
        app.selectbox(key="pair_study").set_value("cnn_v1").run()
        self.assertEqual(app.selectbox(key="pair_before_cnn_v1").value,"J")
        self.assertEqual(app.selectbox(key="pair_after_cnn_v1_J").value,"C1")
        app.radio(key="pair_view_cnn_v1_J_C1").set_value("전체 회귀 사례").run()
        for k in ["rescued_FN","new_FN","new_FP"]:
            app.selectbox(key="pair_change_cnn_v1_J_C1").set_value(k).run()
            self.assertFalse(list(app.error) or list(app.exception))
            self.assertTrue(app.dataframe[0].value["보류 여부"].eq("보류 없음").all())
