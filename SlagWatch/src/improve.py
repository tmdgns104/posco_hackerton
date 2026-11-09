"""Paired, group-held-out robustness experiment on the v3 RF classifiers."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score

from .data import load_dataset
from .features import features, band_powers
from .faults import inject, seed_for, inject_fault
from .train_eval import RF_CONFIG

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments"
CONDITIONS = ("clean","noise","spike","new_y_spike")
FAMILIES = ("rf","spectral")
VARIANTS = ("A","B","C","D")


def vector(w):
    q05,q25,med,q75,q95=np.quantile(w,[.05,.25,.5,.75,.95],axis=0)
    mad=np.median(np.abs(w-med),axis=0)
    central=(w>=q05)&(w<=q95)
    trimmed_rms=np.sqrt(np.where(central,w*w,0).sum(axis=0)/central.sum(axis=0))
    return np.r_[features(w),band_powers(w),np.stack([mad,q75-q25,trimmed_rms],axis=1).ravel()]


def design(matrix,family,variant):
    basic = matrix[:,:12 if family=="rf" else 36]
    return np.column_stack([basic,matrix[:,36:]]) if variant in ("C","D") else basic


def signal(w,scale,r,condition,training=False):
    group,phase,idx=int(r.group_id),r.phase,int(r.window_idx)
    if condition=="new_y_spike":
        rng=np.random.default_rng(np.random.SeedSequence([9107,group,"EBS".index(phase),idx]))
        out=w.copy()
        out[rng.choice(len(w),24,replace=False),1]+=rng.choice([-1,1],24)*4*scale[1]
        return out
    # Disjoint seed namespaces for augmentation and fixed v3 evaluation.
    seed=seed_for(group,phase,idx,condition)
    if training: seed+=2**32
    return inject(w,scale,condition,idx,seed)


def summary(df):
    y=df.y_true.to_numpy()
    p=(df.score.to_numpy()>=.5)
    return dict(n=len(y),tp=int((p & (y==1)).sum()),fn=int((~p & (y==1)).sum()),
                fp=int((p & (y==0)).sum()),tn=int((~p & (y==0)).sum()),
                balanced_accuracy=balanced_accuracy_score(y,p),macro_f1=f1_score(y,p,average="macro"))


def fit():
    OUT.mkdir(exist_ok=True)
    meta,w,_=load_dataset(ROOT/"data/raw")
    raw=np.stack([vector(a) for a in w])
    models_dir=OUT/"models";models_dir.mkdir(exist_ok=True)
    rows=[]; manifest=[]
    for fold in range(4):
        b=joblib.load(ROOT/f"artifacts/fold_{fold}.joblib")
        train=meta[meta.group_id.isin(b["train_groups"])]
        test=meta[meta.group_id.isin(b["test_groups"])]
        assert len(train)==360 and len(test)==120 and set(train.group_id).isdisjoint(test.group_id)
        scale=np.median([w[i].std(axis=0) for i in train.index],axis=0)
        np.testing.assert_allclose(scale,b["scale"],rtol=1e-12)
        aug=np.concatenate([raw[train.index]]+[np.stack([vector(signal(w[r.Index],scale,r,c,True))
                        for r in train.itertuples()]) for c in ("noise","spike")])
        inputs={c:raw[test.index] if c=="clean" else np.stack([vector(signal(w[r.Index],scale,r,c))
                                    for r in test.itertuples()]) for c in CONDITIONS}
        learned={}
        for family in FAMILIES:
            for variant in VARIANTS:
                if variant=="A": model=b[family]
                else:
                    boosted=variant in ("B","D")
                    x=aug if boosted else raw[train.index]
                    y=np.tile(train.y_true,3 if boosted else 1)
                    model=RandomForestClassifier(**RF_CONFIG).fit(design(x,family,variant),y)
                    model.n_jobs=1
                    learned[(family,variant)]=model
                for c,matrix in inputs.items():
                    score=model.predict_proba(design(matrix,family,variant))[:,1]
                    rows.append(test.assign(fold_id=fold,family=family,variant=variant,scenario=c,score=score))
        manifest.append(dict(fold=fold,train_groups=b["train_groups"],test_groups=b["test_groups"],
                             raw_windows=360,augmented_inputs=1080,train_scale=scale.tolist()))
        joblib.dump(learned,models_dir/f"fold_{fold}.joblib",compress=3)
    table=pd.concat(rows,ignore_index=True)
    table.to_csv(OUT/"paired_scores.csv",index=False)
    results=[]
    for key,group in table.groupby(["family","variant","scenario"],sort=False):
        results.append(dict(zip(["family","variant","scenario"],key),**summary(group)))
    metrics=pd.DataFrame(results)
    metrics.to_csv(OUT/"scores_summary.csv",index=False)
    (OUT/"folds.json").write_text(json.dumps(manifest,indent=2))
    print(metrics[["family","variant","scenario","tp","fp","balanced_accuracy"]].to_string(index=False))


def replay_suite():
    ctx=pd.read_csv(ROOT/"reports/search_results.csv")
    meta,windows,_=load_dataset(ROOT/"data/raw")
    by_id={(r.file_id,r.window_idx):windows[r.Index] for r in meta.itertuples()}
    all_rows=[]
    for fold,part in ctx.groupby("fold_id"):
        old=joblib.load(ROOT/f"artifacts/fold_{fold}.joblib")
        learned=joblib.load(OUT/f"models/fold_{fold}.joblib")
        vectors=[]
        for r in part.itertuples():
            w=by_id[r.file_id,r.window_idx]
            changed,_,_=inject_fault(w,r.fault_type,r.axis,r.scale,r.count,r.seed,old["scale"],
                                     json.loads(r.positions_json),json.loads(r.signs_json))
            assert hashlib.sha256(changed.tobytes()).hexdigest()==r.changed_window_hash
            vectors.append(vector(changed))
        vectors=np.stack(vectors)
        part=part.copy()
        for v in VARIANTS:
            model=old['rf'] if v=="A" else learned['rf',v]
            part[v]=model.predict_proba(design(vectors,"rf",v))[:,1]
        np.testing.assert_allclose(part.A,part.perturbed_score,atol=1e-12,rtol=0)
        all_rows.append(part)
    pd.concat(all_rows).sort_values("trial_order").to_csv(OUT/"regression_2000.csv",index=False)

if __name__=="__main__":
    fit()
    replay_suite()
