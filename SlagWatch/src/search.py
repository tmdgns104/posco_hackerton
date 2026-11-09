import argparse
import hashlib
import itertools
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from .data import load_dataset
from .faults import inject_fault
from .gate import evaluate_window, MARGIN

ROOT=Path(__file__).resolve().parents[1]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=True)


def load_context(root=ROOT):
    meta,windows,manifest=load_dataset(root/"data/raw")
    dataset_hash=digest(canonical(dict(zip(manifest.file_id,manifest.sha256))).encode())
    folds=json.loads((root/"artifacts/fold_map.json").read_text())
    bundles,versions={},{}
    for f in range(4):
        path=root/f"artifacts/fold_{f}.joblib"
        bundles[f]=joblib.load(path)
        versions[f]=digest(path.read_bytes())
        b=bundles[f]
        if set(b["train_groups"])&set(b["test_groups"]): raise ValueError("Fold overlap")
    items={}
    for i,row in meta.iterrows():
        fold=folds[str(row.group_id)]
        if row.group_id not in bundles[fold]["test_groups"]: raise ValueError("OOF mismatch")
        w=windows[i]
        w.flags.writeable=False
        items[(row.file_id,row.window_idx)]=(w,fold)
    return dict(items=items,bundles=bundles,versions=versions,dataset_hash=dataset_hash,
                clean=pd.read_csv(root/"reports/predictions.csv").query("scenario=='clean'").copy())


def grid(faults):
    configs=[]
    for fault in faults:
        scales=(.05,.1,.2,.4,.8) if fault=="noise" else (1,2,3,5,8)
        counts=(0,) if fault=="noise" else (1,4,16)
        configs.extend((fault,axis,s,c,seed) for s,c,axis,seed in itertools.product(scales,counts,("x","all"),(42,43)))
    rng=np.random.default_rng(20261107)
    return [configs[i] for i in rng.permutation(len(configs))]


def scenario_key(ctx,file,idx,model,fold,params,pos,sign):
    return digest(canonical([ctx["dataset_hash"],ctx["versions"][fold],model,fold,file,int(idx),params,pos,sign]).encode())


def search_failures(ctx,model="rf",direction="both",faults=("noise","spike"),max_trials=2000,max_seconds=90,progress=None):
    start=time.monotonic()
    clean=ctx["clean"]
    pred=model+"_pred"
    correct=clean[clean[pred]==clean.y_true]
    selected=[]
    for truth in (0,1):
        if direction!="both" and truth!=int(direction):continue
        choices=correct[correct.y_true==truth].sort_values(["group_id","window_idx","file_id"])
        selected.append(choices.sample(n=min(20,len(choices)),random_state=42).to_dict("records"))
    targets=[r for pair in itertools.zip_longest(*selected) for r in pair if r]
    rows=[]
    run_id=digest(canonical([ctx["dataset_hash"],ctx["versions"],model,direction,list(faults),max_trials,max_seconds,MARGIN]).encode())[:16]
    hits=0
    for params in grid(faults):
        for r in targets:
            elapsed=time.monotonic()-start
            if len(rows)>=max_trials or elapsed>=max_seconds:break
            file,idx=r["file_id"],int(r["window_idx"])
            w,fold=ctx["items"][file,idx]
            b=ctx["bundles"][fold]
            fault,axis,scale,count,seed=params
            changed,pos,sign=inject_fault(w,*params[:2],scale,count,seed,b["scale"])
            result=evaluate_window(changed,b,model)
            wrong=result["pred"]!=r["y_true"]
            failure=("induced_false_negative" if r["y_true"] else "induced_false_positive") if wrong else "none"
            hits+=int(wrong)
            sid=scenario_key(ctx,file,idx,model,fold,params,pos,sign)
            outcome=("gate_caught_error" if result["decision"]=="HOLD" else "gate_missed_error") if wrong else ("unnecessary_abstention" if result["decision"]=="HOLD" else "correct_pass")
            rows.append(dict(scenario_id=sid,dataset_hash=ctx["dataset_hash"],file_id=file,group_id=r["group_id"],phase=r["phase"],window_idx=idx,fold_id=fold,
                model_id=model,model_sha256=ctx["versions"][fold],y_true=r["y_true"],clean_pred=int(r[pred]),clean_score=r[model+"_score"],
                fault_type=fault,axis=axis,scale=float(scale),count=count,seed=seed,positions_json=canonical(pos),signs_json=canonical(sign),
                perturbed_pred=result["pred"],perturbed_score=result["score"],input_status=result["input_status"],gate_policy="G2",margin=MARGIN,
                gate_decision=result["decision"],reason_codes_json=canonical(result["reasons"]),failure_type=failure,gate_outcome=outcome,
                search_run_id=run_id,feature_version="v2-12" if model=="rf" else "v2-36",trial_order=len(rows)+1,
                elapsed_ms=round((time.monotonic()-start)*1000),source_window_hash=digest(w.tobytes()),changed_window_hash=digest(changed.tobytes())))
            if progress and (len(rows)%40==0):progress(len(rows),hits,time.monotonic()-start)
        if len(rows)>=max_trials or time.monotonic()-start>=max_seconds:break
    if progress:progress(len(rows),hits,time.monotonic()-start)
    return pd.DataFrame(rows)


def replay(ctx,r):
    file,idx,model=r["file_id"],int(r["window_idx"]),r["model_id"]
    w,fold=ctx["items"][file,idx]
    pos,sign=json.loads(r["positions_json"]),json.loads(r["signs_json"])
    fault=r["fault_type"]
    scale=float(r["scale"]) if fault=="noise" else int(r["scale"])
    params=(fault,r["axis"],scale,int(r["count"]),int(r["seed"]))
    changed,_,_=inject_fault(w,*params,ctx["bundles"][fold]["scale"],positions=pos,signs=sign)
    if (fold!=int(r["fold_id"]) or ctx["versions"][fold]!=r["model_sha256"] or
        scenario_key(ctx,file,idx,model,fold,params,pos,sign)!=r["scenario_id"] or
        digest(changed.tobytes())!=r["changed_window_hash"] or digest(w.tobytes())!=r["source_window_hash"]):
        raise ValueError("원본·교란·모델의 재현 식별자가 다릅니다.")
    result=evaluate_window(changed,ctx["bundles"][fold],model,r["gate_policy"],float(r["margin"]))
    if (not np.isclose(result["score"],r["perturbed_score"],atol=1e-12,rtol=0) or
        result["decision"]!=r["gate_decision"] or result["reasons"]!=json.loads(r["reason_codes_json"])):
        raise ValueError("저장된 점수 또는 보류 사유와 재현 결과가 다릅니다.")
    return w,changed,result


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--max-trials",type=int,default=2000)
    p.add_argument("--max-seconds",type=float,default=90)
    args=p.parse_args()
    ctx=load_context()
    result=search_failures(ctx,max_trials=args.max_trials,max_seconds=args.max_seconds)
    result.to_csv(ROOT/"reports/search_results.csv",index=False)
    prior=ctx["clean"].query("rf_pred!=y_true").assign(failure_type="preexisting_error")
    prior.to_csv(ROOT/"reports/preexisting_errors.csv",index=False)
    print(result.groupby(["failure_type","gate_outcome"]).size())
