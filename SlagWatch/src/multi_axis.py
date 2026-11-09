"""Test whether spreading training impulses across axes helps the shape classifier."""
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from .improve import ROOT,OUT,CONDITIONS,signal,summary
from .data import load_dataset
from .shape import shape


def train_impulse(w,scale,r):
    rng=np.random.default_rng(np.random.SeedSequence([67027,int(r.group_id),"EBS".index(r.phase),int(r.window_idx)]))
    axis=int(rng.integers(0,3))
    size=int(rng.choice([4,8,16,24,32]));amplitude=rng.uniform(2,8)
    pos=rng.choice(len(w),size,replace=False)
    out=w.copy()
    out[pos,axis]+=rng.choice([-1,1],size)*amplitude*scale[axis]
    return out


def z_check(w,scale,r):
    rng=np.random.default_rng(np.random.SeedSequence([57011,int(r.group_id),"EBS".index(r.phase),int(r.window_idx)]))
    pos=rng.choice(len(w),12,replace=False)
    out=w.copy()
    out[pos,2]+=rng.choice([-1,1],12)*6*scale[2]
    return out


def build():
    from .improve import vector,design
    meta,w,_=load_dataset(ROOT/"data/raw")
    rows=[]
    for fold in range(4):
        b=joblib.load(ROOT/f"artifacts/fold_{fold}.joblib")
        f=joblib.load(OUT/f"models/shape_{fold}.joblib")
        tr=meta[meta.group_id.isin(b["train_groups"])];te=meta[meta.group_id.isin(b["test_groups"])]
        X=[shape(train_impulse(w[r.Index],b["scale"],r) if c=="spike" else signal(w[r.Index],b["scale"],r,c,True))
           for c in ("clean","noise","spike") for r in tr.itertuples()]
        g=clone(f).fit(X,np.tile(tr.y_true,3))
        joblib.dump(g,OUT/f"models/multi_{fold}.joblib",compress=3)
        for c in (*CONDITIONS,"new_z_spike"):
            waves=[z_check(w[r.Index],b["scale"],r) if c=="new_z_spike" else signal(w[r.Index],b["scale"],r,c) for r in te.itertuples()]
            x=np.stack([shape(v) for v in waves])
            rows.append(te.assign(fold_id=fold,family="spectral",variant="G",scenario=c,score=g.predict_proba(x)[:,1]))
            if c=="new_z_spike":
                rows.append(te.assign(fold_id=fold,family="spectral",variant="F",scenario=c,score=f.predict_proba(x)[:,1]))
                old=b["spectral"].predict_proba(design(np.stack([vector(v) for v in waves]),"spectral","A"))[:,1]
                rows.append(te.assign(fold_id=fold,family="spectral",variant="A",scenario=c,score=old))
    t=pd.concat(rows,ignore_index=True)
    t.to_csv(OUT/"multi_scores.csv",index=False)
    stats=pd.DataFrame([dict(scenario=c,family="spectral",variant=v,**summary(p)) for (c,v),p in t.groupby(["scenario","variant"])])
    stats.to_csv(OUT/"multi_summary.csv",index=False)
    print(stats.to_string(index=False))

if __name__=="__main__":build()
