import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from src.data import load_dataset
from src.hi_study import ROOT,read_config,select_cut
from src.surface_data import RUN,CLASSES,CFG,array_sha
from src.surface_detector import Detector,extract,read_raw,stable_id,prepare
from src.surface_study import assess
from src.surface_replay import SurfaceRun,numpy_bytes


class TestSurface(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ctx=SurfaceRun()

    def test_grouped_file_task_and_stop(self):
        directory=ROOT/"experiments/temporal5s"
        pred=pd.read_csv(directory/"predictions.csv",float_precision="round_trip")
        iv=pd.read_csv(directory/"inner_oof.csv",float_precision="round_trip")
        self.assertEqual(pred.file_id.nunique(),48)
        self.assertEqual(set(pred.scenario),{"clean"})
        self.assertEqual(pred.groupby(["file_id","scenario"]).input_sha256.nunique().max(),1)
        for f in read_config()["folds"]:
            b=joblib.load(directory/f"models/fold_{f['fold_id']}.joblib")
            test=pred[pred.fold_id==f["fold_id"]]
            self.assertEqual(set(test.group_id),set(b["test"]))
            for name,(_,n,cut) in b["models"].items():
                g=iv[(iv.fold_id==f["fold_id"])&(iv.model_id==name)]
                self.assertEqual(set(g.group_id),set(b["train"]))
                self.assertFalse(set(g.group_id)&set(test.group_id))
                self.assertEqual(select_cut(g.y_true,g.score,.15)["threshold"],cut)
                self.assertEqual(n,{"J_file5s":0,"T1":6,"T2":24}[name])
        self.assertFalse(json.loads((directory/"decision.json").read_text())["passed"])

    def test_source_parent_splits_and_train_only_scaling(self):
        meta=self.ctx.meta
        self.assertEqual(meta.groupby("parent_image_id").split.nunique().max(),1)
        self.assertEqual(meta.groupby("pixels_sha256").split.nunique().max(),1)
        self.assertEqual(meta.groupby("canonical_sha256").split.nunique().max(),1)
        sizes=meta.groupby(["split","label"]).size()
        for stage,n in [("train",180),("validation",60),("test",60)]:
            self.assertTrue(sizes.loc[stage].eq(n).all())
        tr=meta[meta.split=="train"]
        self.assertEqual(self.ctx.detector("V0").identity["train_id"],stable_id(sorted(tr.sample_id)))
        for name in ["V0","V1","V2"]:
            b=self.ctx.detector(name)
            self.assertEqual(b.head[0].n_samples_seen_,len(tr))
            self.assertEqual(b.head[-1].C,1)
            self.assertEqual(b.identity["train_id"],self.ctx.detector("V0").identity["train_id"])
            self.assertEqual(b.identity["encoder_sha256"],CFG["weight_sha256"])

    def test_selection_and_all_denominators(self):
        ctx=self.ctx
        for split in ctx.man["splits"]:
            preds=ctx.table(split,"predictions");metrics=ctx.table(split,"summary");changed=ctx.table(split,"changes")
            for (name,sigma,seed),g in preds.groupby(["model_id","sigma","seed"]):
                m=metrics[(metrics.model_id==name)&(metrics.sigma==sigma)&(metrics.seed==seed)].iloc[0]
                self.assertEqual(g.sample_id.nunique(),360);self.assertEqual(len(g),360)
                self.assertAlmostEqual(m.macro_f1,f1_score(g.label,g.prediction,labels=CLASSES,average="macro"))
                self.assertEqual(m.correct,int(g.prediction.eq(g.label).sum()))
            for (after,sigma,seed),g in ctx.table(split,"pairs").groupby(["after","sigma","seed"]):
                self.assertEqual(len(g),360); self.assertTrue(g.raw_input_hash_before.eq(g.raw_input_hash_after).all())
                fixed=g.label.ne(g.prediction_before)&g.label.eq(g.prediction_after)
                new=g.label.eq(g.prediction_before)&g.label.ne(g.prediction_after)
                m=changed[(changed.after==after)&(changed.sigma==sigma)&(changed.seed==seed)].iloc[0]
                self.assertEqual((fixed.sum(),new.sum()),(m.fixed,m.new_error))
            if split=="validation":
                classes=ctx.table(split,"classes")
                for name in ("V1","V2"):
                    self.assertEqual(assess(metrics,classes,name)["checks"],ctx.seal["results"][name]["checks"])
        if ctx.seal["selected"] is None:
            self.assertFalse(ctx.status["evaluated"])
            self.assertFalse((RUN/"test_predictions.csv.gz").exists())
            with self.assertRaises(ValueError):ctx.pair_rows("test","V1",0,0)

    def test_lossless_pair_replay_and_tamper(self):
        examples=[RUN/"examples/fixed.zip",RUN/"examples/new_error.zip"]
        cases=[self.ctx.replay(path) for path in examples]
        self.assertEqual([r["change"] for r in cases],["fixed","new_error"])
        self.assertNotEqual(cases[0]["execution_id"],cases[1]["execution_id"])
        with ZipFile(examples[0]) as z:
            files={n:z.read(n) for n in z.namelist()}
        arr=np.load(io.BytesIO(files["raw_input.npy"]),allow_pickle=False)
        self.assertEqual(arr.dtype,np.float32)
        record=cases[0]
        for name in ["V0",record["after"]]:
            self.assertEqual(record["outputs"][name]["trial"]["raw_input_hash"],array_sha(arr))
            self.assertEqual(record["outputs"][name]["trial"]["processed_input_hash"],array_sha(prepare(arr,name)))
        self.assertFalse(np.array_equal(arr,(arr*255).round()/255))
        for mode in ("npy","model","class"):
            edited=files.copy()
            if mode=="npy":
                bad=arr.copy();bad[0,0]=1-bad[0,0];edited["raw_input.npy"]=numpy_bytes(bad)
            else:
                bad=copy.deepcopy(record)
                if mode=="model":bad["outputs"]["V0"]["identity"]["head_sha256"]="changed"
                else:bad["label"]="unknown"
                edited["case.json"]=json.dumps(bad).encode()
            data=io.BytesIO()
            with ZipFile(data,"w") as z:
                for k,v in edited.items():z.writestr(k,v)
            with self.assertRaises(ValueError):self.ctx.replay(io.BytesIO(data.getvalue()))
