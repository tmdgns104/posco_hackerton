import copy
import json
import unittest

import joblib
import numpy as np
import pandas as pd
from aeon.transformations.collection.convolution_based import MiniRocket
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from src.cnn_eval import paired
from src.faults import inject_fault
from src.hi_study import make_wave, select_cut
from src.pair_detector import detector, digest, file_hash
from src.pair_runs import CNN, ROCKET, ROCKET_PAIRS
from src.pair_search import ReplaySet
from src.rocket_detector import channel_first, decisions, representation_hash, transform
from src.rocket_eval import adoption
from src.rocket_study import changed_windows


class TestRocket(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hash=file_hash(ROCKET/"manifest.json")
        cls.ctx=ReplaySet(ROCKET,cls.hash)
        cls.cfg=json.loads((ROCKET/"protocol.json").read_text())
        cls.meta=cls.ctx.meta
        cls.raw=cls.ctx.waves

    def test_each_fit_and_threshold(self):
        self.assertEqual(self.cfg["target_fp_rate"],.15)
        for split in self.cfg["splits"]:
            f=split["fold_id"]
            folder=ROCKET/f"fit_{f}"
            outer=joblib.load(ROCKET/f"models/{f}.joblib")
            self.assertEqual(outer["n_features"],9996)
            self.assertFalse(set(outer["train_groups"])&set(outer["test_groups"]))
            self.assertEqual(set(outer["train_groups"]),set(split["train"]))
            for name in ("R0","R1"):
                raw=pd.read_csv(folder/(name.lower()+"_inner.csv.gz"),float_precision="round_trip")
                clean=raw.query("scenario=='clean'")
                self.assertEqual(set(clean.group_id),set(split["train"]))
                self.assertFalse(clean.duplicated(["file_id","window_idx"]).any())
                cut=select_cut(clean.y_true,clean.score,.15)["threshold"]
                self.assertEqual(cut,outer["thresholds"][name])
                for i,part in enumerate(split["inner"]):
                    b=joblib.load(folder/f"inner_{i}.joblib")
                    self.assertEqual(set(b["train_groups"]),set(part["train"]))
                    self.assertFalse(set(b["train_groups"]) & (set(split.get("test",[]))|set(part["val"])))
                    self.assertEqual(set(clean[clean.inner_id==i].group_id),set(part["val"]))
                    self.assertEqual(b["feature_hash"],representation_hash(b["rocket"],b["scaler"]))
                    self.assertEqual(b["scaler"].n_samples_seen_,len(part["train"])*30)
                    np.testing.assert_array_equal(b["heads"][name].classes_,[0,1])
                    self.assertEqual(b["heads"][name].alpha,1.)

    def test_genuine_inner_fit_and_weighted_ridge(self):
        b=joblib.load(ROCKET/"fit_0/inner_0.joblib")
        train=self.meta[self.meta.group_id.isin(b["train_groups"])]
        w=[self.raw[i] for i in train.index]
        with threadpool_limits(limits=2,user_api="blas"):
            fresh=MiniRocket(n_kernels=10000,n_jobs=4,random_state=42).fit(channel_first(w))
            x=fresh.transform(channel_first(w))
            norm=StandardScaler(with_mean=False).fit(x)
            self.assertEqual(representation_hash(fresh,norm),b["feature_hash"])
            np.testing.assert_allclose(b["scale"],np.median(np.std(w,axis=1),axis=0),atol=1e-12)
            parts=[norm.transform(x).astype(float)]
            for cond in ("noise","impulse"):
                log=[]
                changed=changed_windows(train,self.raw,b["scale"],"train_aug",0,0,cond,log)
                self.assertTrue(all(r["seed"]>>32==313 and r["group_id"] in b["train_groups"] for r in log))
                parts.append(transform(b,changed))
            weights=np.repeat([.5,.25,.25],len(train))
            np.testing.assert_array_equal(weights.reshape(3,-1).sum(0),np.ones(len(train)))
            head=RidgeClassifier(alpha=1).fit(np.concatenate(parts),np.tile(train.y_true,3),sample_weight=weights)
            np.testing.assert_allclose(head.coef_,b["heads"]["R1"].coef_,rtol=0,atol=1e-8)
        ex=self.ctx.evidence["0"]["inner_example"]
        g=self.ctx.lookup[ex["file_id"],ex["window_idx"]]
        config=pd.read_csv(ROCKET/"fit_0/perturbations.csv.gz",float_precision="round_trip")
        r=config[(config.source=="inner_validation")&(config.input_sha256==ex["input_sha256"])].iloc[0]
        inner=joblib.load(ROCKET/f"fit_0/inner_{r.inner_id}.joblib")
        self.assertIn(g.group_id,inner["test_groups"]); self.assertNotIn(g.group_id,inner["train_groups"])
        self.assertEqual(int(r.seed)>>32,719)
        fault="noise" if r.condition=="noise" else "spike"
        changed,_,_=inject_fault(self.raw[g.Index],fault,r.axis,r.scale,int(r["count"]),int(r.seed),inner["scale"],
                                  json.loads(r.positions),json.loads(r.signs))
        self.assertEqual(digest(changed),ex["input_sha256"])
        value=decisions(inner["heads"]["R0"],transform(inner,[changed]))[0]
        self.assertAlmostEqual(value,ex["perturbed_score"],places=6)

    def test_held_out_replay_and_quality(self):
        p=self.ctx.pred
        self.assertEqual(len(p),9600)
        self.assertTrue(p.groupby(["file_id","window_idx","scenario"]).input_sha256.nunique().eq(1).all())
        for f in range(4):
            j=detector("J",f)
            te=self.meta[self.meta.group_id.isin(j.test)]
            for scenario in self.ctx.conditions:
                waves=[make_wave(self.raw[r.Index],j.legacy["scale"],r,scenario) for r in te.itertuples()]
                hashes=[digest(w) for w in waves]
                ids=pd.MultiIndex.from_frame(te[["file_id","window_idx"]])
                for name in ("R0","R1"):
                    d=self.ctx.model(name,f)
                    saved=p[(p.fold_id==f)&(p.scenario==scenario)&(p.model_id==name)].set_index(["file_id","window_idx"]).loc[ids]
                    out=d.infer(waves)
                    np.testing.assert_array_equal(saved.input_sha256,hashes)
                    np.testing.assert_allclose(saved.score,[r["score"] for r in out],atol=1e-5,rtol=0)
                    np.testing.assert_array_equal(saved.prediction,[r["prediction"] for r in out])
                    positions=set(np.flatnonzero(te.window_idx.eq(0)))|{int(np.argmin(abs(saved.score-d.threshold)))}
                    for i in positions:
                        r=d.infer([waves[i]])[0]
                        self.assertEqual(r["prediction"],saved.prediction.iloc[i])
                        self.assertAlmostEqual(r["score"],saved.score.iloc[i],places=5)
        for name in ("R0","R1"):
            d=self.ctx.model(name,0)
            for v in (None,np.ones((3,2)),np.ones((3200,3)),np.full((3200,3),np.inf),np.full((3200,3),np.nan)):
                r=d.infer([v])[0]
                self.assertIsNone(r["prediction"]); self.assertIsNone(r["score"])

    def test_conservation_and_eligibility(self):
        s=pd.read_csv(ROCKET/"summary.csv").set_index(["model_id","scenario"])
        c=pd.read_csv(ROCKET/"changes.csv")
        for row in c.itertuples():
            a=s.loc[row.before,row.scenario];b=s.loc[row.after,row.scenario]
            self.assertEqual(b.tp-a.tp,row.rescued_FN-row.new_FN)
            self.assertEqual(b.fp-a.fp,row.new_FP-row.removed_FP)
            self.assertEqual((b.tp+b.fn,b.fp+b.tn,b.fp_E+b.fp_B),(160,320,b.fp))
            cases=paired(self.ctx.pred.query("scenario==@row.scenario"),row.before,row.after)
            for k in ("rescued_FN","new_FN","removed_FP","new_FP"):
                self.assertEqual((cases.change==k).sum(),getattr(row,k))
        syn=s.reset_index().copy()
        syn.loc[syn.model_id=='R0',['tp','fp']]=[112,48]
        v=adoption(syn,self.cfg["goals"])["R0"]
        self.assertTrue(v["all_pass"]); self.assertFalse(v["adopt"] or v["comparison_eligible"])

    def test_run_and_payload_binding(self):
        for before,after in ROCKET_PAIRS:
            r=json.loads((ROCKET/f"examples/{before}_{after}_rescued_FN.json").read_text())
            self.ctx.replay(r)
            for change in ("model_sha256","bundle_sha256","feature_sha256","threshold","input_sha256","manifest_sha256","study","fault"):
                bad=copy.deepcopy(r)
                if change=="fault":bad["fault"]["scenario"]="noise"
                elif change in ("input_sha256","manifest_sha256","study"):bad[change]="tampered"
                else:bad["outputs"][after][change]=0
                with self.assertRaises(ValueError):self.ctx.replay(bad)
        with self.assertRaises(ValueError):ReplaySet(ROCKET,"tampered")
        # The signed CNN_v1 exports retain their original schema and checkpoints.
        old=json.loads((CNN/"examples/recovered.json").read_text())
        ReplaySet(CNN,file_hash(CNN/"manifest.json")).replay(old)
