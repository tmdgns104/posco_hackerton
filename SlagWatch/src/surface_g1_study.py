"""Single G1 comparison; only a sealed validation pass can open test."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from filelock import FileLock

from .surface_data import CFG,CLASSES,RUN,TASK,UNIT,array_sha,stable_id
from .surface_detector import Detector,read_raw
from .surface_study import conditions,input_batch,score_batch,tables,compare,assess
from .surface_g1 import DEFAULT_G1,G1Detector,check_protocol,file_sha,new_protocol,put_json,read_json,source_snapshot,train


def require_population(pred,rows,split,protocol):
    expected=set(map(tuple,protocol["conditions"][split]));ids=set(rows.sample_id)
    if len(ids)!=360 or not rows.split.eq(split).all() or set(pred.model_id)!={"V0","G1"}:
        raise ValueError("Wrong evaluation population")
    if pred.duplicated(["sample_id","sigma","seed","model_id"]).any() or not pred.evaluation.eq(split).all():
        raise ValueError("Repeated or wrong evaluation observations")
    for name,g in pred.groupby("model_id"):
        grouped=g.groupby(["sigma","seed"])
        if set(grouped.groups)!=expected or any(set(part.sample_id)!=ids for _,part in grouped):
            raise ValueError("Missing/extra conditions or images for "+name)
    if not pred.raw_input_hash.eq(pred.processed_input_hash).all():raise ValueError("G1 must use unfiltered V0 input")


def split_eval(out,meta,split,models):
    proto=check_protocol(out);rows=meta[meta.split==split]
    legacy=None
    if split=="validation":
        old=pd.read_csv(RUN/"validation_predictions.csv.gz").query("model_id=='V0'")
        legacy=old.set_index(["sigma","seed","sample_id"],verify_integrity=True)
    original=np.stack([read_raw(r) for r in rows.itertuples()])
    frames=[]
    for sigma,seed in conditions(split):
        raw=input_batch(original,rows,split,sigma,seed)
        for name,model in models.items():
            f=score_batch(model,raw,rows,split,sigma,seed).assign(model_id=name,preprocessing_policy="V0")
            if legacy is not None and name=="V0":
                expected=legacy.loc[(sigma,seed)].loc[f.sample_id]
                for k in ("model_run_id","prediction","raw_input_hash","processed_input_hash","label","parent_image_id"):
                    np.testing.assert_array_equal(expected[k],f[k])
                np.testing.assert_allclose(np.stack(expected.scores.map(json.loads)),np.stack(f.scores.map(json.loads)),
                                           rtol=0,atol=CFG["score_atol"])
            frames.append(f)
        print("evaluated",split,sigma,seed,flush=True)
    pred=pd.concat(frames,ignore_index=True)
    require_population(pred,rows,split,proto)
    pred.to_csv(out/f"{split}_predictions.csv.gz",index=False)
    summary,classes,cm=tables(pred)
    pairs=compare(pred,"G1")
    if not pairs.processed_input_hash_before.eq(pairs.processed_input_hash_after).all():raise ValueError("Preprocessing mismatch")
    pairs.to_csv(out/f"{split}_pairs.csv.gz",index=False)
    keys=["task_id","sample_unit","evaluation","n_originals","before","after","sigma","seed"]
    changes=pairs.groupby(keys+["change"]).size().unstack(fill_value=0).reindex(columns=["fixed","new_error","both_wrong","both_correct"],fill_value=0)
    by_parent=pairs.groupby(keys+["parent_image_id","label","change"]).size().unstack(fill_value=0)
    for suffix,t in [("summary",summary),("classes",classes),("confusion",cm),("changes",changes.reset_index()),("by_parent",by_parent.reset_index())]:
        t.to_csv(out/f"{split}_{suffix}.csv",index=False)
    return assess(summary,classes,"G1")


def files_for(out,split):
    return {p.name:file_sha(p) for p in sorted(out.glob(split+"_*.csv*"))}


def identities(out):
    return {"V0":Detector("V0").identity,"G1":G1Detector(out).identity}


def frozen(out):
    proto=check_protocol(out)
    return dict(models=identities(out),g1_protocol_sha=file_sha(out/"protocol.json"),baseline_manifest_sha=proto["baseline_manifest_sha256"],
                split_sha=proto["split_sha256"],train_views_sha=file_sha(out/"train_views.csv"),
                validation_files=files_for(out,"validation"),failure_sources_sha=file_sha(out/"failure_conditions.json"))


def verify_selection(out,meta):
    seal=read_json(out/"selection.json")
    if seal["frozen"]!=frozen(out):raise ValueError("Selection-time model, split, training record or evaluation changed")
    pred=pd.read_csv(out/"validation_predictions.csv.gz")
    require_population(pred,meta[meta.split=="validation"],"validation",read_json(out/"protocol.json"))
    summary,cl,_=tables(pred);assessment=assess(summary,cl,"G1")
    if (not assessment["eligible"] or seal["selected"]!="G1" or assessment!=seal["assessment"]):
        raise ValueError("Validation did not authorize test opening")
    return stable_id([file_sha(out/"selection.json"),seal["frozen"]])


def open_test(out,meta):
    identity=verify_selection(out,meta)
    status=read_json(out/"test_status.json")
    if status["phase"] not in ("unopened","opening","opened_failed") or status.get("opening_id",identity)!=identity:
        raise ValueError("Test cannot be opened with this seal/status")
    opening=dict(phase="opening",evaluated=False,adopt=None,opening_id=identity,selection_sha=file_sha(out/"selection.json"),
                 models=identities(out),attempt=status.get("attempt",0)+1)
    put_json(out/"test_status.json",opening)
    try:
        result=split_eval(out,meta,"test",{"V0":Detector("V0"),"G1":G1Detector(out)})
        put_json(out/"test_status.json",dict(**{**opening,"phase":"complete","evaluated":True,"adopt":result["eligible"]},assessment=result))
    except BaseException as e:
        put_json(out/"test_status.json",{**opening,"phase":"opened_failed","error":type(e).__name__})
        raise


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,default=DEFAULT_G1)
    parser.add_argument("--resume-test",action="store_true")
    args=parser.parse_args();out=args.output.resolve()
    if out==RUN or RUN in out.parents:raise ValueError("G1 requires a separate directory")
    out.mkdir(parents=True,exist_ok=True)
    with FileLock(out/".study.lock",timeout=0):
        meta=pd.read_csv(RUN/"split_manifest.csv")
        if args.resume_test:
            open_test(out,meta);return
        if any(p.name!=".study.lock" for p in out.iterdir()):raise ValueError("Use a fresh empty output directory")
        snap=source_snapshot();put_json(out/"baseline_inventory.json",snap)
        proto=new_protocol(snap);put_json(out/"protocol.json",proto)
        disc=pd.read_csv(RUN/"validation_discoveries.csv.gz")
        evidence=dict(baseline_run=snap["run_id"],csv_sha=snap["files"]["validation_discoveries.csv.gz"],
                      inputs_zip_sha=snap["files"]["validation_discovered_inputs.zip"],source_split="validation",
                      definition="V0 correct on clean, wrong on exactly the same source under Gaussian noise",
                      affected_originals=disc.sample_id.nunique(),trials=len(disc),by_sigma=disc.groupby("sigma").size().to_dict(),
                      example_ids=disc.groupby("sigma").first().reset_index()[["sample_id","sigma","seed","raw_input_hash","npy_member"]].to_dict("records"),
                      response="Same sigma levels on train parents only, training seed 301; no validation originals are used in training")
        put_json(out/"failure_conditions.json",evidence)
        put_json(out/"test_status.json",dict(phase="unopened",evaluated=False,adopt=None))
        train(out,meta)
        if source_snapshot()!=snap:raise ValueError("Baseline artifact modified before evaluation")
        result=split_eval(out,meta,"validation",{"V0":Detector("V0"),"G1":G1Detector(out)})
        seal=dict(selected="G1" if result["eligible"] else None,assessment=result,selection_source="validation",frozen=frozen(out))
        put_json(out/"selection.json",seal)
        print("sealed validation",seal["selected"],result,flush=True)
        if seal["selected"]:
            open_test(out,meta)
        else:
            put_json(out/"test_status.json",dict(phase="not_selected",evaluated=False,adopt=False,reason="validation checks failed"))
        if source_snapshot()!=snap:raise ValueError("Baseline artifact modified")
    print("Evaluation finished",flush=True)


if __name__=="__main__":
    main()
