"""One new Ridge-free logistic head. Image preprocessing and encoder stay in surface_detector."""
import copy
import json
import os
from importlib.metadata import version
from pathlib import Path
from tempfile import NamedTemporaryFile

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from .surface_data import ROOT,DEFAULT_RUN,RUN,CFG,CLASSES,TASK,UNIT,array_sha,sha,stable_id
from .surface_detector import Detector,WEIGHTS,read_raw,disturb,extract,case_seed
from .surface_study import conditions

DEFAULT_G1=ROOT/"experiments/surface_g1"
VIEWS=(("clean",0.,0,.5),("noise_002",.02,301,1/6),("noise_005",.05,301,1/6),("noise_010",.10,301,1/6))
CORE=("surface_data.py","surface_detector.py","surface_study.py")
TRAIN_CODE=("surface_g1.py","surface_g1_study.py")


def file_sha(path):
    import hashlib
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf8"))


def put_json(path,obj):
    with NamedTemporaryFile("w",encoding="utf8",dir=path.parent,delete=False) as tmp:
        json.dump(obj,tmp,ensure_ascii=False,indent=2,allow_nan=False)
        tmp.flush();os.fsync(tmp.fileno())
    os.replace(tmp.name,path)


def source_snapshot():
    if RUN!=DEFAULT_RUN.resolve():
        raise ValueError("G1 requires the original surface_v1 source")
    manifest=read_json(RUN/"manifest.json")
    if stable_id({k:v for k,v in manifest.items() if k!="run_id"})!=manifest["run_id"]:
        raise ValueError("Baseline manifest identity mismatch")
    for key,h in manifest["code_sha256"].items():
        if file_sha(ROOT/"src"/key)!=h:raise ValueError("Baseline code hash mismatch: "+key)
    for key,h in manifest["files"].items():
        if file_sha(RUN/key)!=h:raise ValueError("Baseline artifact hash mismatch: "+key)
    if file_sha(WEIGHTS)!=manifest["encoder_sha256"]:raise ValueError("Encoder hash mismatch")
    files={p.relative_to(RUN).as_posix():file_sha(p) for p in sorted(RUN.rglob("*")) if p.is_file()}
    return dict(run_id=manifest["run_id"],code_sha256=manifest["code_sha256"],files=files)


def fingerprint(scaler):
    values={}
    for name in ("mean_","var_","scale_","n_features_in_","n_samples_seen_"):
        x=np.asarray(getattr(scaler,name))
        values[name]=[x.dtype.str,list(x.shape),array_sha(x)]
    return stable_id(values)


def new_protocol(snapshot):
    v=Detector("V0")
    return dict(task_id=TASK,sample_unit=UNIT,experiment="surface_g1",model_id="G1",preprocessing_policy="V0",
                baseline_run_id=snapshot["run_id"],baseline=snapshot["files"]["models/V0.joblib"],
                baseline_manifest_sha256=snapshot["files"]["manifest.json"],baseline_protocol_sha256=snapshot["files"]["protocol.json"],
                split_sha256=snapshot["files"]["split_manifest.csv"],encoder_sha256=CFG["weight_sha256"],
                scaler_fingerprint=fingerprint(v.head[0]),views=[list(v) for v in VIEWS],
                source_train_ids=v.identity["train_id"],logistic_params=v.head[-1].get_params(),
                conditions={s:[list(k) for k in conditions(s)] for s in ("validation","test")},goals=CFG["goals"],
                score_atol=CFG["score_atol"],core_sha256=snapshot["code_sha256"],
                code_sha256={k:file_sha(ROOT/"src"/k) for k in TRAIN_CODE},
                versions={p:version(p) for p in ("torch","torchvision","opencv-python-headless","numpy","scikit-learn","Pillow","streamlit","filelock")})


def check_protocol(out):
    proto=read_json(out/"protocol.json")
    if RUN!=DEFAULT_RUN.resolve() or proto["baseline_manifest_sha256"]!=file_sha(RUN/"manifest.json"):
        raise ValueError("Wrong baseline run")
    if proto["baseline_protocol_sha256"]!=file_sha(RUN/"protocol.json") or proto["split_sha256"]!=file_sha(RUN/"split_manifest.csv"):
        raise ValueError("Baseline split/protocol changed")
    for f,h in {**proto["core_sha256"],**proto["code_sha256"]}.items():
        if file_sha(ROOT/"src"/f)!=h:raise ValueError("Code changed: "+f)
    if proto["conditions"]!={s:[list(k) for k in conditions(s)] for s in ("validation","test")} or proto["goals"]!=CFG["goals"]:
        raise ValueError("Evaluation conditions changed")
    if proto["baseline"]!=file_sha(RUN/"models/V0.joblib") or proto["encoder_sha256"]!=file_sha(WEIGHTS):
        raise ValueError("Baseline model or encoder changed")
    return proto


def train(out,meta):
    proto=check_protocol(out)
    tr=meta[meta.split=="train"]
    if len(tr)!=1080 or meta.groupby("parent_image_id").split.nunique().max()!=1 or stable_id(sorted(tr.sample_id))!=proto["source_train_ids"]:
        raise ValueError("Unexpected training population")
    base=Detector("V0");scaler=copy.deepcopy(base.head[0]);classifier=clone(base.head[-1])
    if fingerprint(scaler)!=proto["scaler_fingerprint"] or classifier.get_params()!=base.head[-1].get_params():raise ValueError("Invalid clone")
    records=[];features=[]
    for name,sigma,seed,weight in VIEWS:
        for start in range(0,len(tr),32):
            part=tr.iloc[start:start+32];rs=list(part.itertuples())
            original=[read_raw(r) for r in rs]
            raw=[disturb(a,r.parent_image_id,"train",sigma,seed) for a,r in zip(original,rs,strict=True)]
            x,h=extract(raw,"V0")
            if x.dtype!=np.float64:raise ValueError("Unexpected feature dtype")
            features.append(x)
            for r,a,b,processed in zip(rs,original,raw,h,strict=True):
                raw_hash=array_sha(b)
                if raw_hash!=processed:raise ValueError("Unexpected training filter")
                records.append(dict(sample_id=r.sample_id,parent_image_id=r.parent_image_id,label=r.label,split="train",view=name,
                                    sigma=sigma,seed=seed,derived_seed=str(case_seed(r.parent_image_id,"train",sigma,seed)),
                                    weight=weight,original_file_sha256=r.file_sha256,original_input_hash=array_sha(a),raw_input_hash=raw_hash))
        print("train view complete",name,flush=True)
    rows=pd.DataFrame(records);w=rows.weight.to_numpy()
    if len(rows)!=4320 or not rows.groupby("sample_id").size().eq(4).all() or not np.isclose(w.sum(),1080) or not np.allclose(rows.groupby("sample_id").weight.sum(),1):
        raise ValueError("Invalid augmentation weights or counts")
    rows.to_csv(out/"train_views.csv",index=False)
    with threadpool_limits(limits=2,user_api="blas"):
        classifier.fit(scaler.transform(np.concatenate(features)),rows.label.to_numpy(),sample_weight=w)
    if list(classifier.classes_)!=CLASSES or classifier.n_iter_.max()>=classifier.max_iter or fingerprint(scaler)!=fingerprint(base.head[0]):
        raise ValueError("Classifier convergence, classes or scaler mismatch")
    bundle=dict(model_id="G1",preprocessing_policy="V0",baseline_identity=base.identity,encoder_sha256=proto["encoder_sha256"],
                baseline_protocol_sha256=proto["baseline_protocol_sha256"],g1_protocol_sha256=file_sha(out/"protocol.json"),
                train_id=proto["source_train_ids"],scaler_fingerprint=fingerprint(scaler),train_views_sha256=file_sha(out/"train_views.csv"),
                head=Pipeline([("standardscaler",scaler),("logisticregression",classifier)]))
    joblib.dump(bundle,out/"G1.joblib",compress=3)
    print("G1 fitted",len(rows),tr.parent_image_id.nunique(),"parents; iterations",classifier.n_iter_,flush=True)


class G1Detector(Detector):
    """Inherited infer() uses self.policy for extract(); self.model_id names the fitted head."""
    model_id="G1"
    def __init__(self,out):
        self.policy="V0"
        proto=check_protocol(out);base=Detector("V0");file=out/"G1.joblib"
        b=joblib.load(file)
        self.head=b["head"]
        expected=dict(model_id=self.model_id,preprocessing_policy=self.policy,baseline_identity=base.identity,
                      encoder_sha256=proto["encoder_sha256"],scaler_fingerprint=fingerprint(base.head[0]),train_id=proto["source_train_ids"],
                      baseline_protocol_sha256=proto["baseline_protocol_sha256"],g1_protocol_sha256=file_sha(out/"protocol.json"),
                      train_views_sha256=file_sha(out/"train_views.csv"))
        if any(b[k]!=v for k,v in expected.items()) or fingerprint(self.head[0])!=expected["scaler_fingerprint"] or list(self.head.classes_)!=CLASSES:
            raise ValueError("Frozen G1 model identity mismatch")
        self.identity=dict(**expected,head_sha256=file_sha(file),preprocessing_sha256=base.identity["preprocessing_sha256"])
        self.run_id=stable_id(self.identity)
