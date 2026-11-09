"""Verify input and v4 operating counts before fitting candidates. No model is trained here."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from .data import load_dataset,recording_paths,HZ
from .improve import ROOT,OUT

DEST=OUT/"hi_v1"

def counts(y,p):
    y=np.asarray(y,dtype=int);p=np.asarray(p,dtype=int)
    return dict(tp=int(((y==1)&(p==1)).sum()),fn=int(((y==1)&(p==0)).sum()),
                fp=int(((y==0)&(p==1)).sum()),tn=int(((y==0)&(p==0)).sum()))

def run():
    DEST.mkdir(exist_ok=True)
    meta,w,manifest=load_dataset(ROOT/"data/raw")
    assert len(w)==480 and len(manifest)==48 and set(meta.group_id)==set(range(1,17))
    timing=[]
    for name,path in recording_paths(ROOT/"data/raw").items():
        t=pd.to_timedelta(pd.read_csv(path,usecols=["time"]).time).to_numpy().astype('timedelta64[ns]').astype(np.int64)
        dt=np.diff(t)
        timing.append(dict(file=name,span_s=(t[-1]-t[0])/1e9,median_us=np.median(dt)/1e3,
                           min_us=dt.min()/1e3,max_us=dt.max()/1e3,nonincreasing=int((dt<=0).sum())))
    pd.DataFrame(timing).to_csv(DEST/"timestamp_audit.csv",index=False)
    spectra=pd.read_csv(OUT/"paired_scores.csv").query("variant=='A'")
    e=pd.read_csv(OUT/"impulse_scores.csv")
    fs=pd.concat([pd.read_csv(OUT/"shape_scores.csv"),pd.read_csv(OUT/"multi_scores.csv").query("variant=='F'")])
    details=[]
    for key,tab in [('A',spectra),('E',e),('F',fs)]:
        for (cond,fam),g in tab.groupby(['scenario','family']):
            details.append(dict(candidate=key,scenario=cond,family=fam,**counts(g.y_true,g.score>=.5),
                                fp_E=int(((g.phase=='E')&(g.score>=.5)).sum()),fp_B=int(((g.phase=='B')&(g.score>=.5)).sum())))
    df=pd.DataFrame(details);df.to_csv(DEST/"v4_audit.csv",index=False)
    r=pd.read_csv(OUT/"regression_2000.csv")
    assert len(r)==2000 and len(r[['file_id','window_idx']].drop_duplicates())==40 and (r.clean_pred==r.y_true).all()
    print("RAW 2000 / E errors:",int(((r.A>=.5)!=r.y_true).sum()),int(((r.E>=.5)!=r.y_true).sum()))
    for f in range(4):
        b=joblib.load(ROOT/f"artifacts/fold_{f}.joblib");tr=meta.index[meta.group_id.isin(b["train_groups"])]
        assert not set(b["train_groups"])&set(b["test_groups"]) and len(tr)==360
        np.testing.assert_allclose(b["scale"],np.median([w[i].std(axis=0) for i in tr],axis=0),rtol=1e-12)
    print(df.query("(family=='spectral' and scenario=='clean' and candidate=='A') or (family=='rf' and scenario=='spike') or candidate=='F'").to_string(index=False))
    print(pd.DataFrame(timing)[['span_s','median_us','min_us','max_us','nonincreasing']].agg(['min','max']))
    clean=spectra.query("family=='spectral' and scenario=='clean'").copy()
    clean['miss_s']=(clean.y_true==1)&(clean.score<.5)
    clean['fp_E']=(clean.phase=='E')&(clean.score>=.5)
    clean['fp_B']=(clean.phase=='B')&(clean.score>=.5)
    g=clean.groupby('group_id')[['miss_s','fp_E','fp_B']].sum()
    print(g.to_string());g.to_csv(DEST/"v4_clean_groups.csv")
if __name__=="__main__":run()
