"""Separate v3-baseline repair experiment. Same trained RF, label-free input transformation."""
import json
import joblib
import numpy as np
import pandas as pd
from .improve import ROOT,OUT,FAMILIES,CONDITIONS,vector,signal,design,summary
from .data import load_dataset
from .faults import inject_fault


def repair(w):
    out=w.copy()
    center=w[1:-1]
    interp=(w[:-2]+w[2:])/2
    residual=center-interp
    offset=np.median(residual,axis=0)
    sigma=1.4826*np.median(np.abs(residual-offset),axis=0)
    spikes=((center-w[:-2])*(center-w[2:])>0)&(np.abs(residual-offset)>np.maximum(4*sigma,1e-12))
    out[1:-1]=np.where(spikes,interp,center)
    return out


def run():
    meta,windows,_=load_dataset(ROOT/"data/raw")
    lookup={(r.file_id,r.window_idx):windows[r.Index] for r in meta.itertuples()}
    suite=pd.read_csv(OUT/"regression_2000.csv")
    records=[]
    for fold in range(4):
        b=joblib.load(ROOT/f"artifacts/fold_{fold}.joblib")
        test=meta[meta.group_id.isin(b["test_groups"])]
        for cond in CONDITIONS:
            x=np.stack([vector(repair(signal(windows[r.Index],b["scale"],r,cond))) for r in test.itertuples()])
            for fam in FAMILIES:
                scores=b[fam].predict_proba(design(x,fam,"A"))[:,1]
                records.append(test.assign(fold_id=fold,family=fam,variant="E",scenario=cond,score=scores))
        part=suite[suite.fold_id==fold]
        x=[]
        for r in part.itertuples():
            w=lookup[r.file_id,r.window_idx]
            changed,_,_=inject_fault(w,r.fault_type,r.axis,r.scale,r.count,r.seed,b["scale"],json.loads(r.positions_json),json.loads(r.signs_json))
            x.append(vector(repair(changed)))
        suite.loc[part.index,"E"]=b["rf"].predict_proba(design(np.stack(x),"rf","A"))[:,1]
    table=pd.concat(records,ignore_index=True)
    table.to_csv(OUT/"impulse_scores.csv",index=False)
    suite.to_csv(OUT/"regression_2000.csv",index=False)
    pd.DataFrame([dict(scenario=c,family=f,variant="E",**summary(g)) for (c,f),g in table.groupby(["scenario","family"])]).to_csv(OUT/"impulse_summary.csv",index=False)
    for key,g in table.groupby(["scenario","family"]):
        print(key,summary(g))
    a=(suite.A>=.5)!=suite.y_true;e=(suite.E>=.5)!=suite.y_true
    print("E wrong/fixed/regressed",e.sum(),(a&~e).sum(),(~a&e).sum())

if __name__=="__main__":run()
