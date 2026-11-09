"""Offline gate comparison on E / amplitude RF. Each outer fold calibrates on inner OOF only."""
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from .data import load_dataset
from .features import features
from .improve import ROOT,OUT,CONDITIONS,RF_CONFIG,signal
from .impulse import repair
from .shape import shape
from .multi_axis import train_impulse, z_check
from sklearn.base import clone


def gate_inputs(raw_model,windows,shape_model=None):
    x=np.stack([features(w) for w in windows])
    fixed=np.stack([features(repair(w)) for w in windows])
    p=raw_model.predict_proba(x)[:,1]
    q=(raw_model.predict_proba(fixed)[:,1] if shape_model is None else
       shape_model.predict_proba(np.stack([shape(w) for w in windows]))[:,1])
    crest=np.max(x[:,3::4]/np.maximum(x[:,0::4],1e-12),axis=1)
    return q,np.column_stack([np.abs(q-.5),q,np.abs(q-p),crest])


def evaluate(target="E"):
    suffix="_G" if target=="G" else ""
    meta,w,_=load_dataset(ROOT/"data/raw")
    outer=[];settings=[]
    for fold in range(4):
        b=joblib.load(ROOT/f"artifacts/fold_{fold}.joblib")
        tr=meta[meta.group_id.isin(b["train_groups"])]
        te=meta[meta.group_id.isin(b["test_groups"])]
        g_model=joblib.load(OUT/f"models/multi_{fold}.joblib") if target=="G" else None
        inner_x=[];inner_wrong=[]
        for a,z in GroupKFold(3).split(tr,groups=tr.group_id):
            train=tr.iloc[a]; val=tr.iloc[z]
            assert set(train.group_id).isdisjoint(val.group_id) and set(tr.group_id).isdisjoint(te.group_id)
            model=RandomForestClassifier(**RF_CONFIG).fit(np.stack([features(w[i]) for i in train.index]),train.y_true)
            model.n_jobs=1
            scale=np.median([w[i].std(axis=0) for i in train.index],axis=0)
            inner_shape=None
            if g_model is not None:
                x_train=[shape(train_impulse(w[r.Index],scale,r) if c=="spike" else signal(w[r.Index],scale,r,c,True))
                         for c in ("clean","noise","spike") for r in train.itertuples()]
                inner_shape=clone(g_model).fit(x_train,np.tile(train.y_true,3))
            for cond in ("clean","noise","spike"):
                q,h=gate_inputs(model,[signal(w[r.Index],scale,r,cond,training=True) for r in val.itertuples()],inner_shape)
                inner_x.extend(h);inner_wrong.extend((q>=.5)!=val.y_true.to_numpy())
        h=np.stack(inner_x)
        policy=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=1000)).fit(h,inner_wrong)
        # Both scores are oriented so that smaller means lower estimated error.
        cuts={"margin80":float(np.quantile(-h[:,0],.8)),"risk80":float(np.quantile(policy.predict_proba(h)[:,1],.8))}
        settings.append(dict(fold=fold,training_groups=b["train_groups"],tested_groups=b["test_groups"],thresholds=cuts))
        joblib.dump(dict(policy=policy,cuts=cuts),OUT/f"models/gate{suffix}_{fold}.joblib",compress=3)
        for cond in ((*CONDITIONS,"new_z_spike") if g_model is not None else CONDITIONS):
            q,h=gate_inputs(b["rf"],[z_check(w[r.Index],b["scale"],r) if cond=="new_z_spike" else signal(w[r.Index],b["scale"],r,cond) for r in te.itertuples()],g_model)
            for name,risk in [("margin80",-h[:,0]),("risk80",policy.predict_proba(h)[:,1])]:
                # Fixed-coverage audit is offline ranking within a fold/scenario. It never uses labels.
                top=np.argsort(risk,kind="stable")[:96]
                matched=np.zeros(len(te),dtype=bool);matched[top]=True
                outer.append(te.assign(fold_id=fold,scenario=cond,policy=name,model_score=q,risk_score=risk,
                                       calibration_cut=cuts[name],pass_calibrated=risk<=cuts[name],pass_matched=matched))
    table=pd.concat(outer,ignore_index=True)
    table.to_csv(OUT/f"gate_coverage{suffix}.csv",index=False)
    (OUT/f"gate_folds{suffix}.json").write_text(json.dumps(settings,indent=2))
    rows=[]
    for (cond,policy),g in table.groupby(["scenario","policy"]):
        wrong=(g.model_score>=.5)!=g.y_true
        for mode in ("calibrated","matched"):
            passed=g["pass_"+mode]
            rows.append(dict(scenario=cond,policy=policy,mode=mode,n=len(g),passed=int(passed.sum()),
                             errors_pass=int((passed&wrong).sum()),fp=int((passed&wrong&(g.y_true==0)).sum()),
                             fn=int((passed&wrong&(g.y_true==1)).sum()),correct_hold=int((~passed&~wrong).sum()),
                             coverage=float(passed.mean()),conditional_error=float(wrong[passed].mean()) if passed.any() else None))
    out=pd.DataFrame(rows)
    out.to_csv(OUT/f"gate_summary{suffix}.csv",index=False)
    print(out.to_string(index=False))

if __name__=="__main__":
    import sys
    evaluate(sys.argv[1] if len(sys.argv)>1 else "E")
