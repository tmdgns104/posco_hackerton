import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.surface_data import CFG, CLASSES, RUN, array_sha
from src.surface_detector import case_seed, disturb, read_raw
from src.surface_g1 import DEFAULT_G1, VIEWS, fingerprint, put_json, read_json, source_snapshot
from src.surface_g1_replay import G1Run
from src.surface_g1_study import open_test, require_population
from src.surface_replay import SurfaceRun, numpy_bytes


class TestSurfaceG1(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ctx = G1Run()

    def test_view_lineage_and_immutable_source(self):
        ctx = self.ctx
        self.assertEqual(source_snapshot(), read_json(DEFAULT_G1 / "baseline_inventory.json"))
        views = pd.read_csv(ctx.checked("train_views.csv"), dtype={"derived_seed":str})
        train = ctx.meta.query("split == 'train'")
        self.assertEqual(set(views.sample_id), set(train.sample_id))
        self.assertEqual(ctx.meta.groupby("parent_image_id").split.nunique().max(), 1)
        self.assertEqual((len(train), len(views), set(views.split)), (1080,4320,{"train"}))
        self.assertAlmostEqual(views.weight.sum(), 1080)
        np.testing.assert_allclose(views.groupby("sample_id").weight.sum(), 1)
        for view, sigma, seed, weight in VIEWS:
            part = views[views.view == view]
            self.assertEqual(part.sample_id.tolist(), train.sample_id.tolist())
            self.assertEqual(part.label.tolist(), train.label.tolist())
            self.assertEqual(part.parent_image_id.tolist(), train.parent_image_id.tolist())
            self.assertTrue(part.sigma.eq(sigma).all() and part.seed.eq(seed).all())
            np.testing.assert_allclose(part.weight, weight)
            for r in part.iloc[[0, 540, -1]].itertuples():
                image = read_raw(train.loc[r.sample_id])
                self.assertEqual(array_sha(image), r.original_input_hash)
                self.assertEqual(str(case_seed(r.parent_image_id,"train",sigma,seed)), r.derived_seed)
                self.assertEqual(array_sha(disturb(image,r.parent_image_id,"train",sigma,seed)),r.raw_input_hash)

    def test_same_scaler_and_representation_distinct_head(self):
        old, new = self.ctx.detector("V0"), self.ctx.detector("G1")
        for attr in ("mean_","var_","scale_","n_features_in_","n_samples_seen_"):
            np.testing.assert_array_equal(getattr(old.head[0],attr),getattr(new.head[0],attr))
        self.assertEqual(new.identity["scaler_fingerprint"], fingerprint(old.head[0]))
        self.assertEqual(old.identity["encoder_sha256"],new.identity["encoder_sha256"])
        self.assertEqual(old.head[-1].get_params(),new.head[-1].get_params())
        self.assertFalse(np.array_equal(old.head[-1].coef_,new.head[-1].coef_))
        self.assertEqual((new.model_id,new.policy),("G1","V0"))
        self.assertNotEqual(old.run_id,new.run_id)
        for split in self.ctx.man["splits"]:
            pairs = self.ctx.table(split,"pairs")
            pred = self.ctx.table(split,"predictions")
            meta = self.ctx.meta[self.ctx.meta.split == split]
            require_population(pred,meta,split,self.ctx.protocol)
            for broken in (pred.iloc[1:],pd.concat([pred,pred.iloc[:1]]),pred.assign(sigma=.12)):
                with self.assertRaises(ValueError):require_population(broken,meta,split,self.ctx.protocol)
            self.assertTrue((pairs.raw_input_hash_before==pairs.raw_input_hash_after).all())
            self.assertTrue((pairs.processed_input_hash_before==pairs.processed_input_hash_after).all())
            info = self.ctx.table(split,"summary").set_index(["model_id","sigma","seed"])
            for key,g in pred.groupby(["model_id","sigma","seed"]):
                self.assertEqual(len(g),360)
                self.assertAlmostEqual(info.loc[key].macro_f1, f1_score(g.label,g.prediction,labels=CLASSES,average="macro"))
            delta = self.ctx.table(split,"changes").set_index(["sigma","seed"])
            for key,g in pairs.groupby(["sigma","seed"]):
                old_ok = g.label == g.prediction_before
                new_ok = g.label == g.prediction_after
                self.assertEqual(tuple(delta.loc[key,["fixed","new_error","both_wrong","both_correct"]]),
                                 ((~old_ok & new_ok).sum(),(old_ok & ~new_ok).sum(),(~old_ok & ~new_ok).sum(),(old_ok & new_ok).sum()))
                self.assertEqual(int(info.loc[("G1",*key)].correct-info.loc[("V0",*key)].correct),
                                 int(delta.loc[key].fixed-delta.loc[key].new_error))

    def test_scoped_test_opening_and_interruption(self):
        ctx = self.ctx
        with patch.dict(ctx.seal,selected=None):
            with self.assertRaises(ValueError):ctx.pair_rows("test","G1",0,0)
        with patch.dict(ctx.status,evaluated=False):
            with self.assertRaises(ValueError):ctx.pair_rows("test","G1",0,0)
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            put_json(out/"selection.json",{"selected":None})
            put_json(out/"test_status.json",{"phase":"unopened","evaluated":False})
            with patch("src.surface_g1_study.verify_selection",side_effect=ValueError("failed selection")), patch("src.surface_g1_study.split_eval") as evaluate:
                with self.assertRaises(ValueError):open_test(out,ctx.meta)
                evaluate.assert_not_called()
            def interrupt(*_args):
                state=read_json(out/"test_status.json")
                self.assertEqual((state["phase"],state["opening_id"]),("opening","frozen"))
                raise RuntimeError("interrupted")
            with patch("src.surface_g1_study.verify_selection",return_value="frozen"), patch("src.surface_g1_study.identities",return_value={}), patch("src.surface_g1_study.G1Detector"), patch("src.surface_g1_study.split_eval",side_effect=interrupt):
                with self.assertRaises(RuntimeError):open_test(out,ctx.meta)
            self.assertEqual(read_json(out/"test_status.json")["phase"],"opened_failed")
            with patch("src.surface_g1_study.verify_selection",return_value="different"), patch("src.surface_g1_study.split_eval") as evaluate:
                with self.assertRaises(ValueError):open_test(out,ctx.meta)
                evaluate.assert_not_called()

    def test_replay_rejects_changed_pixels_scaler_identity_and_head(self):
        previous = SurfaceRun()
        for kind in ("fixed","new_error"):
            self.assertEqual(previous.replay(RUN/f"examples/{kind}.zip")["change"],kind)
            path = DEFAULT_G1/f"examples/{kind}.zip"
            self.assertEqual(self.ctx.replay(path)["change"],kind)
        with ZipFile(path) as archive:
            files = {n:archive.read(n) for n in archive.namelist()}
        for key in ("scaler_fingerprint","head_sha256","pixels","score"):
            edited = files.copy()
            if key == "pixels":
                arr = np.load(io.BytesIO(edited["raw_input.npy"]),allow_pickle=False)
                arr[0,0] = 1 - arr[0,0]
                edited["raw_input.npy"] = numpy_bytes(arr)
            else:
                record = json.loads(edited["case.json"])
                output = record["outputs"]["G1"]
                if key == "score": output["trial"]["scores"][0] += CFG["score_atol"]*3
                else: output["identity"][key] = "changed"
                edited["case.json"] = json.dumps(record).encode()
            buf = io.BytesIO()
            with ZipFile(buf,"w") as archive:
                for name,data in edited.items():archive.writestr(name,data)
            with self.assertRaises(ValueError):self.ctx.replay(io.BytesIO(buf.getvalue()))
