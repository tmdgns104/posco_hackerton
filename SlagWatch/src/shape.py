"""One exploratory classifier for the remaining clean-record misses; frozen settings."""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from .data import HZ, load_dataset
from .improve import ROOT,OUT,CONDITIONS,signal,summary,design,vector
from .impulse import repair


def shape(w):
    # Relative energy and coupling, from the median of short overlapping spectra.
    w=repair(w)
    seg=np.stack([w[i:i+512] for i in range(0,len(w)-511,256)])
    z=np.fft.rfft((seg-seg.mean(axis=1,keepdims=True))*np.hanning(512)[None,:,None],axis=1)[:,1:,:]
    power=np.abs(z)**2
    bands=power.reshape(len(z),32,8,3).sum(axis=2)
    energy=np.median(bands,axis=0)
    rel=np.log(np.maximum(energy/np.maximum(energy.sum(axis=0),1e-20),1e-12))
    coherence=[]
    for a,b in [(0,1),(1,2),(2,0)]:
        cross=(z[:,:,a]*z[:,:,b].conjugate()).reshape(len(z),16,16).mean(axis=(0,2))
        pa=power[:,:,a].reshape(len(z),16,16).mean(axis=(0,2))
        pb=power[:,:,b].reshape(len(z),16,16).mean(axis=(0,2))
        coherence.extend(np.abs(cross)**2/np.maximum(pa*pb,1e-20))
    return np.r_[rel.ravel(),coherence]


def build():
    meta,windows,_=load_dataset(ROOT/"data/raw")
    rows=[]
    for fold in range(4):
        b=joblib.load(ROOT/f"artifacts/fold_{fold}.joblib")
        tr=meta[meta.group_id.isin(b["train_groups"])];te=meta[meta.group_id.isin(b["test_groups"])]
        X=[shape(signal(windows[r.Index],b["scale"],r,c,True)) for c in ("clean","noise","spike") for r in tr.itertuples()]
        m=make_pipeline(StandardScaler(),LogisticRegression(C=.03,class_weight="balanced",max_iter=1000))
        m.fit(X,np.tile(tr.y_true,3))
        joblib.dump(m,OUT/f"models/shape_{fold}.joblib",compress=3)
        for c in CONDITIONS:
            x=[shape(signal(windows[r.Index],b["scale"],r,c)) for r in te.itertuples()]
            rows.append(te.assign(fold_id=fold,family="spectral",variant="F",scenario=c,score=m.predict_proba(x)[:,1]))
    result=pd.concat(rows,ignore_index=True)
    result.to_csv(OUT/"shape_scores.csv",index=False)
    scores=pd.DataFrame([dict(scenario=c,family="spectral",variant="F",**summary(g)) for c,g in result.groupby("scenario")])
    scores.to_csv(OUT/"shape_summary.csv",index=False)
    print(scores.to_string(index=False))
if __name__=="__main__":build()
