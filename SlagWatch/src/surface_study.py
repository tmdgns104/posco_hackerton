"""Validation-only selection and one locked test comparison for NEU-CLS originals."""
import io
import json
import tempfile
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix,precision_recall_fscore_support,accuracy_score,f1_score

from .surface_data import CFG,CLASSES,RUN,TASK,UNIT,sha,stable_id,write_json
from .surface_detector import Detector,read_raw,disturb,fit_head


def conditions(split):
    return [(0.,0)]+[(s,k) for s in CFG["sigmas"] for k in CFG[split+"_seeds"]]


def input_batch(base,rows,split,sigma,seed):
    return np.stack([disturb(x,r.parent_image_id,split,sigma,seed) for x,r in zip(base,rows.itertuples(),strict=True)])


def score_batch(model,raw,rows,split,sigma,seed):
    result=pd.DataFrame(model.infer(raw))
    frame=rows[["sample_id","parent_image_id","label"]].reset_index(drop=True).join(result)
    frame["scores"]=frame.scores.map(lambda v:json.dumps(v,separators=(",",":")))
    return frame.assign(task_id=TASK,sample_unit=UNIT,evaluation=split,n_originals=len(rows),sigma=sigma,seed=seed,
                        model_id=model.policy,model_run_id=model.run_id)


def tables(pred):
    summary=[];classes=[];confusions=[]
    for (split,policy,sigma,seed),g in pred.groupby(["evaluation","model_id","sigma","seed"]):
        if g.sample_id.nunique()!=len(g):raise ValueError("Duplicated denominator")
        meta=dict(task_id=TASK,sample_unit=UNIT,evaluation=split,model_id=policy,sigma=sigma,seed=seed,n_originals=len(g))
        y,p=g.label,g.prediction
        summary.append(dict(**meta,accuracy=accuracy_score(y,p),macro_f1=f1_score(y,p,labels=CLASSES,average="macro"),
                            correct=int((p==y).sum()),wrong=int((p!=y).sum())))
        _,rec,f1,n=precision_recall_fscore_support(y,p,labels=CLASSES,zero_division=0)
        cm=confusion_matrix(y,p,labels=CLASSES)
        for i,label in enumerate(CLASSES):
            classes.append(dict(**meta,label=label,support=int(n[i]),recall=float(rec[i]),f1=float(f1[i])))
            for j,to in enumerate(CLASSES):
                confusions.append(dict(**meta,label=label,predicted=to,count=int(cm[i,j])))
    return pd.DataFrame(summary),pd.DataFrame(classes),pd.DataFrame(confusions)


def compare(pred,after):
    keys=["task_id","sample_unit","evaluation","n_originals","sample_id","parent_image_id","sigma","seed","label"]
    fields=keys+["prediction","raw_input_hash","processed_input_hash","model_run_id"]
    a=pred[pred.model_id=="V0"][fields];b=pred[pred.model_id==after][fields]
    pair=a.merge(b,on=keys,validate="one_to_one",suffixes=("_before","_after"))
    if len(pair)!=len(a) or len(pair)!=len(b) or not pair.raw_input_hash_before.eq(pair.raw_input_hash_after).all():
        raise ValueError("Mismatched comparison population or raw input")
    old=pair.label==pair.prediction_before;new=pair.label==pair.prediction_after
    return pair.assign(before="V0",after=after,
        change=np.select([~old&new,old&~new,~old&~new],["fixed","new_error","both_wrong"],default="both_correct"))


def assess(summary,classes,policy):
    base=summary[summary.model_id=="V0"];cand=summary[summary.model_id==policy]
    c=float(cand[cand.sigma==0].macro_f1.item()); ref=float(base[base.sigma==0].macro_f1.item())
    noise=cand[cand.sigma>0].groupby("sigma").macro_f1.mean()
    dif=noise-base[base.sigma>0].groupby("sigma").macro_f1.mean()
    recall=classes[classes.sigma==0].pivot(index="label",columns="model_id",values="recall")
    recall_delta=recall[policy]-recall.V0
    clean_loss=ref-c
    values=dict(clean_macro_f1=c,baseline_clean=ref,clean_delta=c-ref,noise_macro_f1=float(noise.mean()),
                noise_gain=float(dif.mean()),clean_recall_delta=recall_delta.to_dict(),
                noise_delta_by_sigma={str(k):float(v) for k,v in dif.items()})
    checks=dict(clean_f1=c>=.85,clean_drop=clean_loss<=.02+1e-12,avg_gain=dif.mean()>=.03-1e-12,
                avg_noise_f1=noise.mean()>=.80-1e-12,class_recall=recall_delta.min()>=-.05-1e-12,
                per_sigma=dif.min()>=-.02-1e-12)
    checks={k:bool(v) for k,v in checks.items()}
    return dict(**values,checks=checks,eligible=all(checks.values()) and ref>=.85)


def save_tables(pred,split,policies):
    pred.to_csv(RUN/f"{split}_predictions.csv.gz",index=False)
    summary,classes,cm=tables(pred)
    for name,t in [("summary",summary),("classes",classes),("confusion",cm)]:t.to_csv(RUN/f"{split}_{name}.csv",index=False)
    pairs=pd.concat([compare(pred,m) for m in policies],ignore_index=True)
    pairs.to_csv(RUN/f"{split}_pairs.csv.gz",index=False)
    keys=["task_id","sample_unit","evaluation","n_originals","before","after","sigma","seed"]
    changes=pairs.groupby(keys+["change"]).size().unstack(fill_value=0).reindex(columns=["fixed","new_error","both_wrong","both_correct"],fill_value=0)
    changes.reset_index().to_csv(RUN/f"{split}_changes.csv",index=False)
    return {m:assess(summary,classes,m) for m in policies}


def originals(rows):
    return np.stack([read_raw(r) for r in rows.itertuples()])


def validate(meta):
    tr=meta[meta.split=="train"];va=meta[meta.split=="validation"]
    train=originals(tr);base=originals(va)
    source_id=stable_id(sorted(tr.sample_id))
    frames=[];models={}
    with tempfile.TemporaryDirectory() as tmp, ZipFile(RUN/"validation_discovered_inputs.zip","w",ZIP_DEFLATED,compresslevel=2) as archive:
        cache=Path(tmp);found=[]
        for policy in CFG["preprocessing"]:
            fit_head(train,tr.label,policy,source_id)
            d=Detector(policy);models[policy]=d.identity
            for sigma,seed in conditions("validation"):
                path=cache/f"{sigma}_{seed}.npy"
                if not sigma:arr=base
                elif policy=="V0":
                    arr=input_batch(base,va,"validation",sigma,seed)
                    np.save(path,arr,allow_pickle=False)
                else:arr=np.load(path,mmap_mode="r",allow_pickle=False)
                t=score_batch(d,arr,va,"validation",sigma,seed);frames.append(t)
                if policy=="V0" and not sigma:
                    clean=t.copy()
                    val=f1_score(t.label,t.prediction,average="macro")
                    print("Baseline clean validation macro F1",val,flush=True)
                    if val<CFG["goals"]["min_baseline_clean"]:raise ValueError("Baseline clean diagnostic required")
                elif policy=="V0":
                    for j,r in enumerate(t.itertuples()):
                        ref=clean.iloc[j]
                        if r.prediction!=r.label and ref.prediction==r.label:
                            name=f"{r.sample_id}_{sigma:.2f}_{seed}.npy";buf=io.BytesIO();np.save(buf,arr[j],allow_pickle=False)
                            archive.writestr(name,buf.getvalue())
                            found.append(dict(sample_id=r.sample_id,parent_image_id=r.parent_image_id,label=r.label,sigma=sigma,seed=seed,
                                clean_prediction=ref.prediction,clean_scores=ref.scores,noise_prediction=r.prediction,noise_scores=r.scores,
                                raw_input_hash=r.raw_input_hash,model_run_id=d.run_id,npy_member=name))
                print("validation",policy,sigma,seed,f1_score(t.label,t.prediction,average="macro"),flush=True)
        pd.DataFrame(found).assign(task_id=TASK,sample_unit=UNIT,evaluation="validation",n_originals=len(va)).to_csv(RUN/"validation_discoveries.csv.gz",index=False)
    results=save_tables(pd.concat(frames,ignore_index=True),"validation",["V1","V2"])
    eligible=[p for p,r in results.items() if r["eligible"]]
    choice=min(eligible,key=lambda p:(-results[p]["noise_macro_f1"],p)) if eligible else None
    seal=dict(task_id=TASK,selection_source="validation",selected=choice,results=results,models=models,
              protocol_sha=sha((RUN/"protocol.json").read_bytes()),split_sha=sha((RUN/"split_manifest.csv").read_bytes()),
              test_seeds=CFG["test_seeds"])
    write_json(RUN/"selection.json",seal);return seal


def test_once(meta,seal):
    candidate=seal["selected"]
    if candidate is None:
        write_json(RUN/"test_status.json",dict(evaluated=False,reason="No validation-eligible candidate",adopt=False))
        return
    te=meta[meta.split=="test"];base=originals(te)
    models=[Detector(n) for n in ("V0",candidate)]
    if any(d.identity!=seal["models"][d.policy] for d in models):raise ValueError("Frozen model mismatch")
    if seal["protocol_sha"]!=sha((RUN/"protocol.json").read_bytes()) or seal["split_sha"]!=sha((RUN/"split_manifest.csv").read_bytes()):
        raise ValueError("Split or protocol changed after validation")
    frames=[]
    for sigma,seed in conditions("test"):
        arr=input_batch(base,te,"test",sigma,seed)
        for d in models:frames.append(score_batch(d,arr,te,"test",sigma,seed))
        print("test evaluated",sigma,seed,flush=True)
    result=save_tables(pd.concat(frames,ignore_index=True),"test",[candidate])[candidate]
    write_json(RUN/"test_status.json",dict(evaluated=True,candidate=candidate,selection_sha=sha((RUN/"selection.json").read_bytes()),
               assessment=result,adopt=result["eligible"]))


def reproduce(path):
    import os
    import shutil
    import subprocess
    import sys
    from .surface_data import DEFAULT_RUN
    dest=path.resolve()
    if dest.exists() and any(dest.iterdir()):
        raise ValueError("Choose an empty output directory to preserve the frozen run")
    (dest/"models").mkdir(parents=True,exist_ok=True)
    shutil.copy2(DEFAULT_RUN/"protocol.json",dest/"protocol.json")
    shutil.copy2(DEFAULT_RUN/"models/resnet18-f37072fd.pth",dest/"models/resnet18-f37072fd.pth")
    env={**os.environ,"STEELTRUST_SURFACE_RUN":str(dest)}
    for args in (["src.surface_data"],["src.surface_study"],["src.surface_replay","--finalize"]):
        subprocess.run([sys.executable,"-m",*args],env=env,check=True)


def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--reproduce",type=Path,help="Train in a separate empty run directory")
    args=parser.parse_args()
    if args.reproduce:
        reproduce(args.reproduce);return
    started=perf_counter()
    if (RUN/"test_status.json").exists():raise SystemExit("This run is complete. Use the frozen outputs, or reproduce in a fresh result directory.")
    meta=pd.read_csv(RUN/"split_manifest.csv")
    seal=json.loads((RUN/"selection.json").read_text()) if (RUN/"selection.json").exists() else validate(meta)
    print("FROZEN VALIDATION DECISION",seal["selected"],flush=True)
    test_once(meta,seal);print("Finished in",round(perf_counter()-started,1),"seconds",flush=True)

if __name__=="__main__":main()
