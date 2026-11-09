import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
import pandas as pd

import start_demo
from src.demo_case import MAX_ZIP, replay_upload
from src.demo_results import aggregate, counts, criteria
from src.surface_data import array_sha
from src.surface_g1 import DEFAULT_G1
from src.surface_g1_replay import G1Run
from src.surface_replay import numpy_bytes


class DemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.run_ctx = G1Run()

    def test_overview_and_criteria_against_pairs(self):
        for split in self.run_ctx.man["splits"]:
            pairs = self.run_ctx.table(split, "pairs")
            data = counts(self.run_ctx,split)
            for mask,rows in ((pairs.sigma>0,data[data.sigma>0]),(pairs.sigma==0,data[data.sigma==0])):
                group = pairs[mask]
                a = group.label == group.prediction_before
                b = group.label == group.prediction_after
                overall = aggregate(rows)
                self.assertEqual((overall["V0_wrong"],overall["fixed"],overall["new_error"],overall["both_wrong"],overall["G1_wrong"]),
                                 ((~a).sum(),(~a & b).sum(),(a & ~b).sum(),(~a & ~b).sum(),(~b).sum()))
                self.assertEqual(len(group), len(rows)*360)
            report=criteria(self.run_ctx,split)
            self.assertEqual(len(report),6)
            self.assertEqual(set(report["판정"]),{"통과"})

    def test_upload_round_trip_and_refusal(self):
        run=self.run_ctx
        for kind in ("fixed","new_error"):
            data=(DEFAULT_G1/"examples"/(kind+".zip")).read_bytes()
            r, _, raw = replay_upload(run,data)
            self.assertEqual(kind,r["change"])
            self.assertEqual(array_sha(raw),r["raw_input_hash"])
        with ZipFile(io.BytesIO(data)) as z:
            files={k:z.read(k) for k in z.namelist()}
        def archive(**members):
            out=io.BytesIO()
            with ZipFile(out,"w") as z:
                for k,v in members.items():z.writestr(k,v)
            return out.getvalue()
        raw[0,0]=1-raw[0,0]
        wrong_model=json.loads(files["case.json"])
        wrong_model["outputs"]["G1"]["identity"]["head_sha256"]="wrong"
        wrong_run={**wrong_model,"run_id":"old"}
        for payload in (b"not zip", b"x"*(MAX_ZIP+1), archive(**{"case.json":b"{}"}),
                        archive(**{**files,"../model.pkl":b"x"}),
                        archive(**{**files,"raw_input.npy":numpy_bytes(raw)}),
                        archive(**{**files,"case.json":json.dumps(wrong_run)}),
                        archive(**{**files,"case.json":json.dumps(wrong_model)})):
            with self.assertRaises(ValueError):replay_upload(run,payload)

    def test_startup_preflight_and_no_repeat_install(self):
        with self.assertRaises(RuntimeError):start_demo.check_python((3,12),64)
        with self.assertRaises(RuntimeError):start_demo.check_python((3,11),32)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):start_demo.check_files(Path(tmp))
            with patch.object(start_demo.sys,"prefix",tmp), patch.object(start_demo,"check_dependencies") as check, patch.object(start_demo,"pip") as pip:
                for _ in range(2):
                    start_demo.prepare(start_demo.ROOT,True)
                self.assertEqual(check.call_count,2)
                pip.assert_not_called()
                identity=json.loads((Path(tmp)/"steeltrust-demo-install.json").read_text())
                self.assertEqual(identity,start_demo.install_identity(start_demo.ROOT))
                check.side_effect=RuntimeError("dependency conflict")
                with self.assertRaisesRegex(RuntimeError,"dependency conflict"):
                    start_demo.prepare(start_demo.ROOT,False)
